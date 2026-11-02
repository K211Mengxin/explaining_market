"""Offline tests for the agent in predict.py: no network, no API keys.

    uv run --with langchain-core --with langchain-openai pytest scripts/tests


Covers the point-in-time rules (the part that would silently inflate a
backtest if it were wrong), the peer grouping, the LangChain chain's parsing,
and predict.py's guarantee to always return a number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import predict as predict_module  # noqa: E402
from predict import (  # noqa: E402
    AgentData, AgentSettings, EventHistory, MarketData, PastEvent, PriceSeries, SecData, Store,
    _blocks, build_chain, build_context, build_peers, derive_cutoff, knn_peers, parse_prediction, sic_peers,
    snapshot,
)

FakeListChatModel = pytest.importorskip(
    "langchain_core.language_models.fake_chat_models",
    reason="run with: uv run --with langchain-core --with langchain-openai pytest scripts/tests",
).FakeListChatModel


# -- fundamentals ------------------------------------------------------------------

# A company with fiscal quarters ending Jan/Apr/Jul/Oct; Q2 FY27 is filed AFTER the
# event date used below, exactly like a real 10-Q that follows the earnings call.
FACTS = {
    "revenue": [
        ["2025-02-01", "2025-04-30", 100.0, "2025-05-30", 0],
        ["2025-05-01", "2025-07-31", 110.0, "2025-08-29", 0],
        ["2025-08-01", "2025-10-31", 120.0, "2025-11-26", 0],
        ["2025-02-01", "2026-01-31", 460.0, "2026-03-03", 0],
        ["2026-02-01", "2026-04-30", 125.0, "2026-05-29", 0],
        ["2026-05-01", "2026-07-31", 140.0, "2026-08-28", 0],  # filed after the event
    ],
    "net_income": [
        ["2025-02-01", "2026-01-31", 46.0, "2026-03-03", 0],
        ["2026-02-01", "2026-04-30", 15.0, "2026-05-29", 0],
        ["2025-02-01", "2025-04-30", 10.0, "2025-05-30", 0],
        ["2026-05-01", "2026-07-31", 99.0, "2026-08-28", 0],
    ],
    "assets": [[None, "2026-04-30", 1000.0, "2026-05-29", 0], [None, "2026-07-31", 5000.0, "2026-08-28", 0]],
    "liabilities": [[None, "2026-04-30", 400.0, "2026-05-29", 0]],
    "shares": [[None, "2026-05-20", 10.0, "2026-05-29", 0]],
}


def test_snapshot_ignores_filings_made_after_the_cutoff() -> None:
    s = snapshot(FACTS, "2026-08-27", last_close=50.0)
    assert s["period_end"] == "2026-04-30"          # not the Q2 filed on 08-28
    assert s["rev_yoy"] == pytest.approx(125 / 100 - 1)
    assert s["revenue_ttm"] == pytest.approx(460 - 100 + 125)  # FY + YTD - prior YTD
    assert s["roa"] == pytest.approx((46 - 10 + 15) / 1000)
    assert s["leverage"] == pytest.approx(0.4)
    assert s["market_cap"] == pytest.approx(500.0)


def test_snapshot_sees_the_new_quarter_once_filed() -> None:
    s = snapshot(FACTS, "2026-08-29")
    assert s["period_end"] == "2026-07-31"
    assert s["rev_yoy"] == pytest.approx(140 / 110 - 1)


def test_snapshot_without_facts_is_none() -> None:
    assert snapshot(None, "2026-08-27") is None
    assert snapshot({"revenue": [["2026-05-01", "2026-07-31", 1.0, "2026-09-01", 0]]}, "2026-08-27") is None


# -- event history -----------------------------------------------------------------


def _ev(eid, t, start, end, car):
    return PastEvent(eid, t, f"{start}T21:00:00Z", start, end, car, "archive")


def test_known_before_requires_the_window_to_have_closed() -> None:
    h = EventHistory([
        _ev("a", "AAA", "2026-08-20", "2026-08-21", 0.05),
        _ev("b", "BBB", "2026-08-26", "2026-08-27", -0.04),  # closes ON the cutoff date: excluded
        _ev("c", "CCC", "2026-08-27", "2026-08-28", 0.10),   # the future: excluded
        PastEvent("d", "DDD", "", "2026-08-10", None, None, "calendar"),  # unresolved
    ])
    known = h.known_before("2026-08-27")
    assert [e.event_id for e in known] == ["a"]
    assert [e.event_id for e in h.pending_before("2026-08-27", tickers={"DDD"}, since="2026-06-01")] == ["d"]


def test_resolved_events_are_not_overwritten_by_calendar_rows() -> None:
    h = EventHistory([_ev("a", "AAA", "2026-08-20", "2026-08-21", 0.05)])
    h.add_calendar([{"event_id": "a", "knowledge_cutoff": "2026-08-20T20:00:00Z",
                     "focal_assets": [{"identifier_value": "AAA"}]}])
    assert h.known_before("2026-09-01")[0].car1 == 0.05


@pytest.mark.parametrize("when, cutoff", [
    ("2026-08-27T21:00:00Z", "2026-08-27T20:00:00Z"),  # after the close -> same day
    ("2026-08-07T14:00:00Z", "2026-08-06T20:00:00Z"),  # during the session -> previous close
    ("2026-08-03T12:30:00Z", "2026-07-31T20:00:00Z"),  # Monday pre-market -> Friday
])
def test_derive_cutoff_matches_the_archive_convention(when, cutoff) -> None:
    assert derive_cutoff(when) == cutoff


# -- market data -------------------------------------------------------------------


def _series(prices: dict[str, float]) -> PriceSeries:
    days = sorted(prices)
    return PriceSeries(tuple(days), tuple(prices[d] for d in days), tuple(1000.0 for _ in days))


def test_event_car_and_pre_event_features(tmp_path) -> None:
    days = [f"2026-07-{d:02d}" for d in range(1, 31)] + ["2026-08-03", "2026-08-04"]
    spy = {d: 100.0 + i * 0.1 for i, d in enumerate(days)}
    stock = {d: 50.0 + i for i, d in enumerate(days)}
    stock["2026-08-04"] = stock["2026-08-03"] * 1.10
    data = {"SPY": _series(spy), "XYZ": _series(stock)}
    md = MarketData(Store(tmp_path), online=True, fetch=lambda t: data.get(t))

    car = md.event_car("XYZ", "2026-08-03")
    assert car["window_end"] == "2026-08-04"
    assert car["r_i"] == pytest.approx(0.10)
    assert car["car1"] == pytest.approx(0.10 - (spy["2026-08-04"] / spy["2026-08-03"] - 1))

    pf = md.pre_event_features("XYZ", "2026-08-03")
    assert pf["as_of"] == "2026-07-30"  # strictly before the cutoff date
    assert pf["ret_5d"] == pytest.approx(stock["2026-07-30"] / stock["2026-07-25"] - 1)


def test_split_days_are_not_returns(tmp_path) -> None:
    days = [f"2026-07-{d:02d}" for d in range(1, 31)]
    stock = {d: 100.0 for d in days}
    for d in days[15:]:
        stock[d] = 50.0  # 2-for-1 split, no economic move
    data = {"SPY": _series({d: 100.0 for d in days}), "XYZ": _series(stock)}
    md = MarketData(Store(tmp_path), online=True, fetch=lambda t: data.get(t))
    pf = md.pre_event_features("XYZ", "2026-07-31")
    assert pf["ret_20d"] == pytest.approx(0.0)
    assert "pos_52w" not in pf


# -- peers -------------------------------------------------------------------------

PROFILES = {
    "AAA": {"sic": "7372"}, "BBB": {"sic": "7372"}, "CCC": {"sic": "7371"},
    "DDD": {"sic": "7374"}, "EEE": {"sic": "2834"},
}


def _snap(growth, margin, size):
    return {"rev_yoy": growth, "gross_margin": margin, "op_margin": margin / 2, "net_margin": margin / 3,
            "roa": margin / 10, "leverage": 0.5, "accruals": 0.0, "log_assets": size,
            "days_since_period_end": 60}


SNAPS = {"AAA": _snap(0.20, 0.8, 22), "BBB": _snap(0.21, 0.79, 22.1), "CCC": _snap(0.5, 0.3, 18),
         "DDD": _snap(0.19, 0.81, 21.9), "EEE": _snap(-0.1, 0.6, 24)}


def test_sic_peers_widen_until_big_enough() -> None:
    assert sic_peers("AAA", PROFILES, 1, SNAPS).members == ["BBB"]
    g = sic_peers("AAA", PROFILES, 3, SNAPS)
    assert g.description.startswith("SIC 737x") and set(g.members) == {"BBB", "CCC", "DDD"}


def test_knn_finds_the_fundamentally_closest() -> None:
    g = knn_peers("AAA", SNAPS, 2)
    assert set(g.members) == {"BBB", "DDD"}
    assert build_peers("hybrid", "AAA", PROFILES, SNAPS, 2).members


# -- chain -------------------------------------------------------------------------


def test_parse_prediction_accepts_fenced_json_and_rejects_out_of_range() -> None:
    assert parse_prediction('```json\n{"predicted_percentile": 0.61, "reason": "x"}\n```').predicted_percentile == 0.61
    with pytest.raises(Exception):
        parse_prediction('{"predicted_percentile": 1.4}')
    with pytest.raises(Exception):
        parse_prediction("")


def test_chain_renders_the_prompt_and_parses_the_reply() -> None:
    llm = FakeListChatModel(responses=['{"predicted_percentile": 0.42, "reason": "guide cut"}'])
    out = build_chain(AgentSettings(), llm=llm).invoke(
        {"event_type": "EARNINGS_RELEASE", "ticker": "AAA", "materials": "m {not a var}", "context": "c"})
    assert out.predicted_percentile == 0.42 and out.reason == "guide cut"


# -- predict.py --------------------------------------------------------------------

EVENT = {"event_id": "e1", "event_type": "EARNINGS_RELEASE", "event_datetime": "2026-08-27T21:00:00Z",
         "knowledge_cutoff": "2026-08-27T20:00:00Z", "information_url": "https://example.test/x",
         "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "AAA"}]}


class _Resp:
    def raise_for_status(self) -> None: ...
    def json(self) -> dict: return {"items": [{"id": "earnings-call-facts", "kind": "facts", "content": ["Up."]}]}


def _offline_settings(tmp_path, **kw) -> AgentSettings:
    return AgentSettings(deepseek_api_key="test", store_dir=tmp_path, online=False, use_calendar=False, **kw)


def test_predict_uses_the_chain_and_the_context(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(predict_module.httpx, "get", lambda *a, **k: _Resp())
    seen = {}
    llm = FakeListChatModel(responses=['{"predicted_percentile": 0.7, "reason": "beat"}'])
    chain = build_chain(AgentSettings(), llm=llm)
    monkeypatch.setattr(predict_module, "_chain", lambda s: chain.with_listeners(
        on_start=lambda run: seen.update(inputs=run.inputs)))
    preds, traces = predict_module.predict_with_trace(EVENT, _offline_settings(tmp_path))
    assert preds == [{"identifier_value": "AAA", "predicted_percentile": 0.7}]
    assert traces[0]["reason"] == "beat"
    assert "1. Up." in seen["inputs"]["materials"]


def test_predict_falls_back_to_half_when_the_chain_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(predict_module.httpx, "get", lambda *a, **k: _Resp())
    chain = build_chain(AgentSettings(), llm=FakeListChatModel(responses=["not json"]))
    monkeypatch.setattr(predict_module, "_chain", lambda s: chain)
    preds, traces = predict_module.predict_with_trace(EVENT, _offline_settings(tmp_path))
    assert preds[0]["predicted_percentile"] == 0.5
    assert "fallback" in traces[0]


def test_predict_survives_a_missing_bundle(monkeypatch, tmp_path) -> None:
    def boom(*a, **k):
        raise OSError("expired url")

    monkeypatch.setattr(predict_module.httpx, "get", boom)
    chain = build_chain(AgentSettings(), llm=FakeListChatModel(responses=['{"predicted_percentile": 0.3}']))
    monkeypatch.setattr(predict_module, "_chain", lambda s: chain)
    preds, traces = predict_module.predict_with_trace(EVENT, _offline_settings(tmp_path))
    assert preds[0]["predicted_percentile"] == 0.3
    assert "materials_error" in traces[0]


# -- features ----------------------------------------------------------------------


def test_build_context_with_local_data_only(tmp_path) -> None:
    store = Store(tmp_path)
    store.write_json({"AAA": {"cik": 1, "sic": "7372", "sic_desc": "Software"},
                      "BBB": {"cik": 2, "sic": "7372"}}, "sec", "profiles.json.gz")
    store.write_json({"1": FACTS, "2": FACTS}, "sec", "facts_all.json.gz")
    settings = _offline_settings(tmp_path, n_peers=3)
    data = AgentData(settings)
    data._history = EventHistory([_ev("p1", "BBB", "2026-08-10", "2026-08-11", -0.06),
                                  _ev("o1", "AAA", "2026-05-28", "2026-05-29", 0.03)])
    data._history_loaded = 1e18
    ctx = build_context(EVENT, "AAA", settings, data)
    text = ctx.to_prompt()
    assert "Software" in text
    assert "1 of 1 peers already reported" in text and "-6.0%" in text
    assert "mean absolute reaction +3.0%" not in text and "3.0%" in text
    assert "prices" not in ctx.sections  # no price data offline: block skipped, not failed

    none = build_context(EVENT, "AAA", _offline_settings(tmp_path, blocks=frozenset()), data)
    assert none.to_prompt() == "(no additional context available)"


def test_blocks_setting_parses_and_validates() -> None:
    assert _blocks("none") == frozenset()
    assert _blocks("prices, history") == {"prices", "history"}
    with pytest.raises(ValueError):
        _blocks("prices,typo")


def test_store_round_trip(tmp_path) -> None:
    s = Store(tmp_path)
    s.write_json({"a": [1, 2]}, "x", "y.json.gz")
    assert s.read_json("x", "y.json.gz") == {"a": [1, 2]}
    assert s.read_json("x", "missing.json", default=3) == 3
    assert json.loads(json.dumps(s.age_days("x", "y.json.gz") >= 0))


# -- SEC (mocked HTTP, real Autodesk XBRL) -------------------------------------------


def test_sec_data_fetches_compacts_and_caches(tmp_path) -> None:
    import httpx

    adsk = json.loads(Path(__file__).with_name("data_adsk_companyfacts.json").read_text())
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        assert "@" in req.headers["User-Agent"]
        if req.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 769397, "ticker": "ADSK", "title": "Autodesk"}})
        if "/submissions/" in req.url.path:
            return httpx.Response(200, json={"name": "Autodesk, Inc.", "sic": "7372",
                                             "sicDescription": "Services-Prepackaged Software"})
        return httpx.Response(200, json=adsk)

    sec = SecData(Store(tmp_path), "Test User test@example.org", min_interval_s=0,
                  client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert sec.profile("ADSK")["sic"] == "7372"
    facts = sec.facts("ADSK")
    s = snapshot(facts, "2026-08-27")
    assert s["period_end"] == "2026-04-30"          # Q2 (filed 2026-08-28) is not yet known
    assert s["rev_yoy"] == pytest.approx(1934 / 1633 - 1, abs=0.02)
    n = len(calls)
    assert sec.facts("ADSK") == facts and len(calls) == n  # served from the cache


# -- deployed case: bundle from a Modal Volume, LangChain fallback -----------------


def test_store_syncs_the_bundle_from_a_modal_volume(monkeypatch, tmp_path) -> None:
    import types

    src = Store(tmp_path / "src")
    src.write_json({"built_at": "2026-10-25T22:00:00Z", "files": []}, "bundle", "manifest.json")
    src.write_json({"AAA": {"cik": 1, "sic": "7372"}}, "bundle", "profiles.json.gz")
    src.write_json({"version": 1, "signature": [], "events": [
        ["e1", "AAA", "2026-08-20T21:00:00Z", "2026-08-20", "2026-08-21", 0.05, "archive"]]},
        "bundle", "history.json.gz")
    src.write_json({"built": "2026-10-25", "series": {"AAA": {"dates": ["2026-10-23", "2026-10-24"],
                                                               "close": [10.0, 11.0], "volume": [1, 1]}}},
                   "bundle", "prices.json.gz")
    files = {f"bundle/{p.name}": p.read_bytes() for p in (tmp_path / "src" / "bundle").iterdir()}

    class FakeVolume:
        def listdir(self, path, recursive=False):
            return [types.SimpleNamespace(path=f) for f in files]

        def read_file(self, path):
            yield files[path]

    fake = types.SimpleNamespace(Volume=types.SimpleNamespace(from_name=lambda name: FakeVolume()),
                                 is_local=lambda: False)
    monkeypatch.setitem(sys.modules, "modal", fake)

    settings = AgentSettings(deepseek_api_key="k", store_dir=tmp_path / "run", online=False, use_calendar=False)
    data = AgentData(settings)
    assert data.ensure_bundle().startswith("synced")
    assert data.sec.profiles()["AAA"]["sic"] == "7372"
    assert len(data.history) == 1
    assert data.market.series("AAA").close[-1] == 11.0
    age = data.store.age_days("sec", "profiles.json.gz")
    assert age > 5  # mtime = bundle build time, so freshness checks see the data's age
    assert Store(tmp_path / "run").sync_from_volume("x") == "cached"


def test_openai_fallback_chain_uses_the_same_prompt(monkeypatch) -> None:
    import types

    sent = {}

    class FakeCompletions:
        def create(self, **kw):
            sent.update(kw)
            msg = types.SimpleNamespace(content='{"predicted_percentile": 0.44, "reason": "r"}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    chain = predict_module.OpenAIFallbackChain.__new__(predict_module.OpenAIFallbackChain)
    chain.settings = AgentSettings(deepseek_api_key="k")
    chain.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
    out = chain.invoke({"event_type": "E", "ticker": "AAA", "materials": "m {x}", "context": "c"})
    assert out.predicted_percentile == 0.44
    assert sent["response_format"] == {"type": "json_object"}
    assert '{"predicted_percentile": 0.55' in sent["messages"][1]["content"]
    assert "m {x}" in sent["messages"][1]["content"]


def test_chain_falls_back_to_the_openai_sdk_without_langchain(monkeypatch) -> None:
    monkeypatch.setattr(predict_module, "ensure_langchain", lambda *a, **k: False)
    predict_module._chains.clear()
    chain = predict_module._chain(AgentSettings(deepseek_api_key="k"))
    predict_module._chains.clear()
    assert isinstance(chain, predict_module.OpenAIFallbackChain)


@pytest.mark.parametrize("when, cutoff", [
    ("2026-11-03T21:00:00Z", "2026-11-03T21:00:00Z"),  # winter time: the close is 21:00 UTC
    ("2026-03-09T13:00:00Z", "2026-03-06T21:00:00Z"),  # Monday 08:00 EDT -> Friday close (EST)
])
def test_derive_cutoff_handles_daylight_saving(when, cutoff) -> None:
    assert derive_cutoff(when) == cutoff
