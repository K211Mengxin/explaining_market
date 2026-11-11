"""Offline tests for the questionnaire + combiner path (predict.py sections 12-14)
and its training script. No network, no API keys, no LangChain needed:

    uv run pytest scripts/tests/test_combiner.py
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import predict as P  # noqa: E402
import train_combiner as T  # noqa: E402
from predict import (  # noqa: E402
    QUESTION_KEYS, AgentSettings, Combiner, anonymize, average_answers, company_core_name, feature_row,
    load_combiner, magnitude_inputs, option_stats_values, parse_questionnaire, run_questionnaire,
)


# -- questionnaire -----------------------------------------------------------------


def test_parse_questionnaire_clips_and_keeps_nulls() -> None:
    reply = '```json\n{"eps_vs_expectation": 7, "guidance_change": null, "overall": "-2", "margin_trend": true}\n```'
    out = parse_questionnaire(reply)
    assert set(out) == set(QUESTION_KEYS) | set(P.NUMBER_KEYS) | {"guidance_metric"}
    assert all(out[k] is None for k in P.NUMBER_KEYS) and out["guidance_metric"] is None
    assert out["eps_vs_expectation"] == 2.0          # clipped to the item's scale
    assert out["overall"] == -2.0                    # numeric string accepted
    assert out["guidance_change"] is None
    assert out["margin_trend"] is None               # booleans are not answers
    assert out["revenue_vs_expectation"] is None     # missing key -> null


@pytest.mark.parametrize("reply", ["not json", "[1, 2]", '{"unrelated": 1}', '{"overall": null}'])
def test_parse_questionnaire_rejects_unusable_replies(reply) -> None:
    with pytest.raises(ValueError):
        parse_questionnaire(reply)


def test_parse_questionnaire_reads_the_copied_numbers() -> None:
    out = parse_questionnaire('{"overall": 1, "revenue_actual": 5.1, "revenue_expected": "4.9", '
                              '"eps_actual": null, "guidance_metric": "Revenue", "guidance_new_mid": 1e20}')
    assert out["revenue_actual"] == 5.1 and out["revenue_expected"] == 4.9
    assert out["eps_actual"] is None and out["guidance_new_mid"] is None   # absurd number dropped
    assert out["guidance_metric"] == "revenue"
    assert parse_questionnaire('{"overall": 1, "guidance_metric": "margin"}')["guidance_metric"] is None


def test_prompt_asks_for_numbers_without_computing() -> None:
    prompt = P.questionnaire_prompt("materials")
    assert "Part B" in prompt and all(f'"{k}"' in prompt for k in P.NUMBER_KEYS)
    assert "Do not compute" in P.QUESTIONNAIRE_SYSTEM


def test_questionnaire_reads_a_shorter_text_with_the_fixed_part_first() -> None:
    bundle = {"items": [
        {"id": "earnings-call-facts", "kind": "facts", "content": ["Revenue up 12%."]},
        {"id": "option-implied-stats", "kind": "stats", "content": {
            "implied_earnings_volatility": {"value": 0.07, "status": "ok"}}},
        {"id": "earnings-preview", "kind": "preview", "content": "P" * 9000},
    ]}
    full, short = P.format_materials(bundle), P.questionnaire_materials(bundle, 3000)
    assert "Implied earnings volatility" in full and "Implied earnings volatility" not in short
    assert "P" * 3000 in short and "P" * 3001 not in short
    assert "P" * 8000 in P.questionnaire_materials(bundle)             # default: the chain's cut
    assert "Revenue up 12%." in short and len(short) < len(full) / 2
    assert P.questionnaire_materials(None) == ""
    # every prompt starts with the same text (cache-friendly); materials come last
    a, b = P.questionnaire_prompt("event A"), P.questionnaire_prompt("event B")
    prefix = a[: a.index("=== EVENT MATERIALS ===")]
    assert b.startswith(prefix) and a.rstrip().endswith("=== END ===")


def test_sampling_is_nested_and_independent_of_other_quarters() -> None:
    rows = [{"event_id": f"e{i}", "ticker": "T"} for i in range(500)]
    small, large = T.sample_rows(rows, "2026Q3", 100, 0), T.sample_rows(rows, "2026Q3", 250, 0)
    assert len(small) == 100 and large[:100] == small                   # a bigger run reuses the smaller one
    assert T.sample_rows(list(reversed(rows)), "2026Q3", 100, 0) == small   # input order does not matter
    assert T.sample_rows(rows, "2026Q2", 100, 0) != small and T.sample_rows(rows, "2026Q3", 100, 1) != small
    assert T.sample_rows(rows, "2026Q3", 0, 0) == rows and T.sample_rows(rows[:50], "2026Q3", 100, 0) == rows[:50]


def test_average_answers_takes_the_median_of_numbers() -> None:
    blank = {k: None for k in (*QUESTION_KEYS, *P.NUMBER_KEYS)} | {"guidance_metric": None}
    replies = [blank | {"revenue_actual": 5.1, "guidance_metric": "revenue"},
               blank | {"revenue_actual": 5100.0, "guidance_metric": "revenue"},   # wrong units
               blank | {"revenue_actual": 5.2, "guidance_metric": "eps"}]
    out = average_answers(replies)
    assert out["revenue_actual"] == 5.2 and out["guidance_metric"] == "revenue"


def test_average_answers_ignores_nulls() -> None:
    a = {k: None for k in QUESTION_KEYS} | {"overall": 2.0, "guidance_change": 1.0}
    b = {k: None for k in QUESTION_KEYS} | {"overall": 4.0}
    out = average_answers([a, b])
    assert out["overall"] == 3.0 and out["guidance_change"] == 1.0 and out["margin_trend"] is None


@pytest.mark.parametrize("name, core", [("Autodesk, Inc.", "Autodesk"), ("NVIDIA CORP", "NVIDIA"),
                                        ("Meta Platforms, Inc.", "Meta Platforms"), ("XY INC", None), (None, None)])
def test_company_core_name(name, core) -> None:
    assert company_core_name(name) == core


def test_anonymize_masks_ticker_and_name_but_not_ordinary_words() -> None:
    text = "# Autodesk (ADSK) preview. ADSK beat; ADSKX is another symbol. AUTODESK raised guidance."
    out = anonymize(text, "ADSK", "Autodesk, Inc.")
    assert "ADSK " not in out and "Autodesk" not in out and "AUTODESK" not in out
    assert "ADSKX" in out
    short = anonymize("A strong quarter for Agilent (A). NYSE: A shares rose.", "A")
    assert short.startswith("A strong quarter") and "(A)" not in short and "NYSE: A" not in short


class FakeClient:
    """Stands in for openai.OpenAI: records prompts, returns canned replies in turn."""

    def __init__(self, replies: list[str]) -> None:
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_run_questionnaire_hides_the_company_and_uses_json_mode() -> None:
    client = FakeClient(['{"overall": 3, "guidance_change": 2}'])
    out = run_questionnaire("ADSK (Autodesk) raised full-year guidance.", "ADSK", AgentSettings(),
                            company_name="Autodesk, Inc.", client=client)
    assert out["overall"] == 3.0 and out["guidance_change"] == 2.0
    call = client.calls[0]
    prompt = call["messages"][1]["content"]
    assert "ADSK" not in prompt and "Autodesk" not in prompt and "raised full-year guidance" in prompt
    assert call["response_format"] == {"type": "json_object"}


def test_questionnaire_turns_thinking_off_by_default_and_counts_tokens() -> None:
    client = FakeClient(['{"overall": 1}'])
    usage = SimpleNamespace(prompt_tokens=1200, prompt_cache_hit_tokens=900, completion_tokens=240,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
    original = client._create
    client.chat.completions.create = lambda **kw: SimpleNamespace(choices=original(**kw).choices, usage=usage)
    before = dict(P.QUESTIONNAIRE_USAGE)
    run_questionnaire("text", "AAA", AgentSettings(), client=client)
    call = client.calls[0]
    assert call["extra_body"] == {"thinking": {"type": "disabled"}} and call["max_tokens"] == P.QUESTIONNAIRE_MAX_TOKENS
    assert "reasoning_effort" not in call
    assert P.QUESTIONNAIRE_USAGE["calls"] == before["calls"] + 1
    assert P.QUESTIONNAIRE_USAGE["prompt"] == before["prompt"] + 1200
    assert P.QUESTIONNAIRE_USAGE["cache_hit"] == before["cache_hit"] + 900
    assert P.QUESTIONNAIRE_USAGE["completion"] == before["completion"] + 240

    client = FakeClient(['{"overall": 1}'])
    run_questionnaire("text", "AAA", AgentSettings(questionnaire_thinking="low"), client=client)
    call = client.calls[0]
    assert call["extra_body"] == {"thinking": {"type": "enabled"}} and call["reasoning_effort"] == "low"
    assert "max_tokens" not in call


def test_thinking_level_parsing(monkeypatch) -> None:
    assert [P._thinking_level(v) for v in (None, "OFF", "0", "on", "Low", "max")] == \
        ["off", "off", "off", "high", "low", "max"]
    with pytest.raises(ValueError):
        P._thinking_level("sometimes")
    monkeypatch.setenv("AGENT_Q_THINKING", "low")
    assert AgentSettings.from_env().questionnaire_thinking == "low"


def test_run_questionnaire_averages_samples_and_survives_one_failure() -> None:
    client = FakeClient(['{"overall": 2}', "garbage", '{"overall": 4}'])
    out = run_questionnaire("text", "AAA", AgentSettings(questionnaire_samples=3), client=client)
    assert out["overall"] == 3.0 and len(client.calls) == 3


# -- features -----------------------------------------------------------------------

BUNDLE = {"items": [
    {"id": "earnings-call-facts", "kind": "facts", "content": ["Revenue up 12%.", "Guidance raised."]},
    {"id": "option-implied-stats", "kind": "stats", "content": {
        "implied_absolute_earnings_move": {"value": 0.06, "status": "ok"},
        "implied_earnings_volatility": {"value": 0.075, "status": "ok"},
        "skew_25_delta": {"value": None, "status": "insufficient_liquidity"}}},
]}


def test_option_stats_and_feature_row() -> None:
    opts = option_stats_values(BUNDLE)
    assert opts == {"implied_move": 0.06, "implied_vol": 0.075, "skew_25d": None}
    assert option_stats_values({"items": []})["implied_move"] is None
    row = feature_row({"overall": 2.0}, {"abn_ret_20d": 0.05, "market_cap": 1e10, "vol_60d": 0.3}, opts)
    assert set(row) == set(P.FEATURE_NAMES)
    assert row["q_overall"] == 2.0 and row["q_guidance_change"] is None
    assert row["abn_ret_20d"] == 0.05 and row["log_market_cap"] == pytest.approx(np.log(1e10))
    assert row["has_options"] == 1.0 and row["skew_25d"] is None
    assert feature_row(None, None, None)["has_options"] == 0.0
    mags = magnitude_inputs({"own_mean_abs_car": 0.04, "vol_60d": 0.3}, opts)
    assert mags == {"implied_move": 0.06, "own_mean_abs_car": 0.04, "peer_mean_abs_car": None, "vol_60d": 0.3}


def test_same_revenue_beat_scores_higher_for_the_steadier_company() -> None:
    answers = {"revenue_actual": 102.0, "revenue_expected": 100.0}           # a 2% beat for both
    steady = P.surprise_features(answers, {"rev_growth_vol": 0.01})
    volatile = P.surprise_features(answers, {"rev_growth_vol": 0.10})
    assert steady["revenue_surprise_pct"] == volatile["revenue_surprise_pct"] == pytest.approx(0.02)
    assert steady["revenue_surprise_z"] == pytest.approx(2.0)                # 2 standard deviations
    assert volatile["revenue_surprise_z"] == pytest.approx(0.2)
    # a tiny company scale is floored, and z is capped
    assert P.surprise_features(answers, {"rev_growth_vol": 0.0001})["revenue_surprise_z"] == pytest.approx(2.0)
    assert P.surprise_features({"revenue_actual": 140.0, "revenue_expected": 100.0},
                               {"rev_growth_vol": 0.01})["revenue_surprise_z"] == 10.0


def test_surprise_features_guard_against_units_and_metric_mixups() -> None:
    out = P.surprise_features({"revenue_actual": 5100.0, "revenue_expected": 4.9,        # millions vs billions
                               "guidance_new_mid": 1.10, "guidance_ref_mid": 1.00, "guidance_metric": "eps",
                               "eps_actual": 1.32, "eps_expected": 1.20}, {"last_close": 60.0, "rev_growth_vol": 0.05})
    assert out["revenue_surprise_pct"] is None and out["revenue_surprise_z"] is None
    assert out["guidance_change_pct"] is None                                # EPS guidance is not revenue
    assert out["eps_surprise_px"] == pytest.approx(0.12 / 60)
    rev_guide = P.surprise_features({"guidance_new_mid": 20.2, "guidance_ref_mid": 19.5, "guidance_metric": "revenue"},
                                    {"rev_growth_vol": 0.05})
    assert rev_guide["guidance_change_pct"] == pytest.approx(0.7 / 19.5)
    assert rev_guide["guidance_change_z"] == pytest.approx(0.7 / 19.5 / 0.05)
    assert P.surprise_features(None, None) == {k: None for k in P.SURPRISE_FEATURES}


def test_revenue_growth_volatility_from_quarterly_filings() -> None:
    growth = [0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.07, 0.10]                # y/y growth, newest last
    rows, base = [], {}
    quarter_ends = ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
                    "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                    "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    starts = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]
    for i, end in enumerate(quarter_ends):
        start = f"{int(end[:4])}-{starts[i % 4][5:]}"
        value = 100.0 if i < 4 else base[i - 4] * (1 + growth[i - 4])
        base[i] = value
        rows.append([start, end, value, end, 0])
    vol = P.revenue_growth_volatility(P.Flow(rows), "2025-12-31")
    assert vol == pytest.approx(float(np.std(growth)))
    assert P.revenue_growth_volatility(P.Flow(rows[:6]), "2024-06-30") is None  # too little history


# -- combiner -----------------------------------------------------------------------

COMBINER = Combiner(features=("q_overall", "abn_ret_20d"), means=(0.0, 0.01), stds=(2.0, 0.05),
                    weights=(0.04, -0.01), scale=2.0, center=0.5,
                    magnitude={"base": "implied_move", "ratios": {"own_mean_abs_car": 1.5}, "ref": 0.05, "power": 0.5})


def test_combiner_predict_math() -> None:
    row = {"q_overall": 4.0, "abn_ret_20d": 0.11}
    z = 0.04 * (4 - 0) / 2 + -0.01 * (0.11 - 0.01) / 0.05          # 0.02 - 0.02 = 0
    p, info = COMBINER.predict(row, {})
    assert info["z"] == pytest.approx(z) and p == pytest.approx(0.5 + 2.0 * z)
    p, info = COMBINER.predict({"q_overall": 4.0}, {})              # missing feature adds nothing
    assert p == pytest.approx(0.5 + 2 * 0.08) and info["features_present"] == 1
    # magnitude: implied move 4x the reference doubles the deviation (sqrt)
    p, info = COMBINER.predict({"q_overall": 4.0}, {"implied_move": 0.20})
    assert info["magnitude"] == 0.20 and p == pytest.approx(0.5 + 0.16 * 2)
    # falls back to a proxy measure, converted by its ratio
    _, info = COMBINER.predict({"q_overall": 4.0}, {"own_mean_abs_car": 0.04})
    assert info["magnitude"] == pytest.approx(0.06)
    # always inside [0, 1]
    assert COMBINER.predict({"q_overall": 1e6}, {})[0] == 0.999


def test_combiner_round_trip_and_validation(tmp_path) -> None:
    again = Combiner.from_dict(json.loads(json.dumps(COMBINER.to_dict())))
    assert again.predict({"q_overall": 3.0}, {"implied_move": 0.1}) == COMBINER.predict({"q_overall": 3.0},
                                                                                         {"implied_move": 0.1})
    bad = COMBINER.to_dict() | {"format": 99}
    with pytest.raises(ValueError):
        Combiner.from_dict(bad)
    d = COMBINER.to_dict()
    d["features"][0]["name"] = "not_a_feature"
    with pytest.raises(ValueError):
        Combiner.from_dict(d)

    path = tmp_path / "combiner.json"
    assert load_combiner(path) is None                               # missing file
    path.write_text(json.dumps(COMBINER.to_dict() | {"questionnaire_version": "q0"}))
    assert load_combiner(path) is None                               # trained on another questionnaire
    broken = tmp_path / "broken.json"
    broken.write_text("{broken")
    assert load_combiner(broken) is None                             # unreadable file
    other = tmp_path / "ok.json"
    other.write_text(json.dumps(COMBINER.to_dict()))
    assert load_combiner(other) is not None


# -- training -----------------------------------------------------------------------


def test_delta_r2_matches_a_direct_least_squares_fit() -> None:
    rng = np.random.default_rng(0)
    s, p = rng.random(200), rng.random(200)
    y = 0.4 * s + 0.2 * p + 0.3 * rng.random(200)

    def r2(cols):
        A = np.column_stack([np.ones(200), *cols])
        res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        return 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))

    assert T.delta_r2(p, s, y) == pytest.approx(r2([p, s]) - r2([s]))
    assert T.delta_r2(np.full(200, 0.65), s, y) == 0.0                  # constant predictions earn nothing


def synthetic_rows(n_per_quarter=300, quarters=("2026Q1", "2026Q2", "2026Q3"), seed=0) -> list[dict]:
    """y depends on the surprise, on q_guidance_change, and (as size) on implied_move."""
    rng = random.Random(seed)
    rows = []
    for q in quarters:
        raw = []
        for _ in range(n_per_quarter):
            surprise, guide, noise_feat = rng.gauss(0, 1), rng.choice([-2, -1, 0, 1, 2]), rng.gauss(0, 1)
            move = rng.uniform(0.02, 0.12)
            car = move * (0.8 * surprise + 0.6 * guide + rng.gauss(0, 1.2))
            feats = {n: None for n in P.FEATURE_NAMES} | {
                "q_guidance_change": float(guide), "q_overall": float(rng.gauss(0, 1)) + 0.3 * surprise,
                "abn_ret_20d": noise_feat, "has_options": 1.0}
            raw.append({"quarter": q, "car1": car, "surprise": surprise, "features": feats,
                        "mags": {"implied_move": move, "own_mean_abs_car": move * 0.7,
                                 "peer_mean_abs_car": None, "vol_60d": None}})
        for r, y in zip(raw, T.percentile_ranks([r["car1"] for r in raw])):
            r["y"] = y
        for r, s in zip(raw, T.percentile_ranks([r["surprise"] for r in raw])):
            r["s"] = s
        rows.extend(raw)
    return rows


def test_training_recovers_the_signal_beyond_the_surprise() -> None:
    rows = synthetic_rows()
    names, coverage = T.choose_features(rows, 0.05)
    assert "q_guidance_change" in names and "has_options" not in names  # constant -> dropped
    assert "q_eps_vs_expectation" not in names                         # never answered -> dropped
    cv = T.cross_validate(rows, names, [0.01, 1.0, 100.0], seed=0)
    assert cv["folds"] == ["2026Q1", "2026Q2", "2026Q3"]                # leave one quarter out
    no_scaling = cv["table"][T._cv_label(cv["lambda"], 0.0, True)]["mean"]
    assert no_scaling > 0.02                                            # signal even without scaling
    assert cv["power"] > 0                                              # size really matters here
    assert cv["table"][T._cv_label(cv["lambda"], cv["power"], cv["joint"])]["mean"] > no_scaling
    assert {k.split("|")[2] for k in cv["table"]} == {"none", "joint", "after"}
    c = T.fit(rows, names, cv["lambda"], power=cv["power"], joint=cv["joint"])
    w = dict(zip(c.features, c.weights))
    assert w["q_guidance_change"] > 5 * abs(w["abn_ret_20d"])
    assert c.magnitude["base"] == "implied_move"
    assert c.magnitude["ratios"]["own_mean_abs_car"] == pytest.approx(1 / 0.7)
    # the saved file is exactly what the live pipeline loads
    assert Combiner.from_dict(json.loads(json.dumps(c.to_dict()))).weights == c.weights


def test_both_fits_match_the_live_formula() -> None:
    rows = synthetic_rows(n_per_quarter=400, seed=3)
    names, _ = T.choose_features(rows, 0.05)
    c = T.fit(rows, names, 0.1, power=0.5)
    assert c.magnitude["power"] == 0.5
    # the training formula is exactly what predict.Combiner computes live
    g = T.magnitude_factors(rows, c.magnitude)
    z = np.array([c.linear(r["features"])[0] for r in rows])
    expected = np.clip(0.5 + c.scale * g * z, 0.001, 0.999)
    assert np.allclose(T.predict_rows(c, rows), expected)
    # rows without any magnitude measure are left unscaled
    assert T.magnitude_factors([{"mags": {}}], c.magnitude).tolist() == [1.0]
    # joint=False = the old recipe: unscaled weights, stretched afterwards
    after = T.fit(rows, names, 0.1, power=0.5, joint=False)
    plain = T.fit(rows, names, 0.1, power=0.0)
    assert np.allclose(after.weights, plain.weights)
    assert after.magnitude == c.magnitude
    assert not np.allclose(c.weights, plain.weights)                    # joint fit really differs


def test_cross_validation_can_choose_no_scaling() -> None:
    rows = synthetic_rows(n_per_quarter=200, seed=5)
    for r in rows:                                    # magnitude now unrelated to the outcome
        r["mags"] = {"implied_move": random.Random(hash(r["car1"])).uniform(0.02, 0.12)}
    names, _ = T.choose_features(rows, 0.05)
    cv = T.cross_validate(rows, names, [1.0], seed=0, powers=(0.0, 0.5))
    assert set(cv["table"]) == {"1.0|p=0.0|none", "1.0|p=0.5|joint", "1.0|p=0.5|after"}
    assert cv["power"] in (0.0, 0.5) and isinstance(cv["joint"], bool)


def test_fold_without_a_feature_still_gives_finite_combiner() -> None:
    rows = synthetic_rows(n_per_quarter=60)
    for r in rows:
        if r["quarter"] != "2026Q3":
            r["features"]["skew_25d"] = None
        else:
            r["features"]["skew_25d"] = random.random()
    names, _ = T.choose_features(rows, 0.05)
    train = [r for r in rows if r["quarter"] != "2026Q3"]
    c = T.fit(train, names, 1.0, power=0.0)
    Combiner.from_dict(c.to_dict())  # finite means/stds, valid
    assert np.all(np.isfinite(T.predict_rows(c, rows)))


def _archive_file(path: Path, quarter_events: list[tuple[str, int]], seed: int = 1) -> None:
    rng = random.Random(seed)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for i in range(quarter_events[0][1]):
            guide = rng.choice([-2, -1, 0, 1, 2])
            surprise = rng.gauss(0, 1)
            car = 0.02 * surprise + 0.015 * guide + rng.gauss(0, 0.03)
            ticker = f"T{i:03d}"
            fh.write(json.dumps({
                "event_id": f"{quarter_events[0][0]}-{i}", "event_type": "EARNINGS_RELEASE",
                "event_datetime": "2026-08-20T21:00:00Z", "knowledge_cutoff": "2026-08-20T20:00:00Z",
                "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
                "event_returns": {ticker: {"car1": car, "window_start_date": "2026-08-20",
                                           "window_end_date": "2026-08-21"}},
                "metrics": {"earnings_surprise": {"surprise": surprise, "surprise_status": "ok"}},
                "disclosure": {"items": [{"id": "earnings-call-facts", "kind": "facts",
                                          "content": [f"GUIDE={guide}", f"{ticker} reported."]}]},
            }) + "\n")


def test_train_main_end_to_end_writes_a_loadable_combiner(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(T, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(T, "MODEL_PATH", tmp_path / "model" / "combiner.json")
    monkeypatch.setattr(T, "BUNDLE_COPY", tmp_path / "bundle" / "combiner.json")
    monkeypatch.setenv("AGENT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("AGENT_ONLINE", "0")          # main() sets these; monkeypatch restores them
    monkeypatch.setenv("AGENT_USE_CALENDAR", "0")
    files = []
    for q in ("2026Q2", "2026Q3"):
        f = tmp_path / f"EARNINGS_RELEASE_{q}.jsonl.gz"
        _archive_file(f, [(q, 80)], seed=hash(q) % 100)
        files.append(f)

    seen = []

    def fake_questionnaire(materials, ticker, settings, company_name=None):
        seen.append(materials)
        guide = int(materials.split("GUIDE=")[1].split()[0])
        return {k: None for k in QUESTION_KEYS} | {"guidance_change": float(guide), "overall": 0.0}

    out = tmp_path / "model" / "combiner.json"
    args = ["--files", *map(str, files), "--offline", "--workers", "2", "--lambdas", "0.1,1", "--out", str(out)]
    c = T.main(args, questionnaire_fn=fake_questionnaire)
    assert c is not None and out.exists() and (tmp_path / "bundle" / "combiner.json").exists()
    assert len(seen) == 160
    loaded = load_combiner(out)
    assert loaded is not None and dict(zip(loaded.features, loaded.weights))["q_guidance_change"] > 0
    assert loaded.meta["quarters"] == ["2026Q2", "2026Q3"]

    # second run: everything from the cache, no new questionnaire calls
    seen.clear()
    T.main([*args, "--fit-only"], questionnaire_fn=fake_questionnaire)
    assert seen == []


def test_holdout_quarter_is_scored_but_never_trained_on(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(T, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(T, "MODEL_PATH", tmp_path / "model" / "combiner.json")
    monkeypatch.setattr(T, "BUNDLE_COPY", tmp_path / "bundle" / "combiner.json")
    monkeypatch.setenv("AGENT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("AGENT_ONLINE", "0")
    monkeypatch.setenv("AGENT_USE_CALENDAR", "0")
    files = []
    for i, q in enumerate(("2026Q1", "2026Q2", "2026Q3")):
        f = tmp_path / f"EARNINGS_RELEASE_{q}.jsonl.gz"
        _archive_file(f, [(q, 90)], seed=i + 1)
        files.append(f)
    seen = []

    def fake_questionnaire(materials, ticker, settings, company_name=None):
        seen.append(materials)
        guide = int(materials.split("GUIDE=")[1].split()[0])
        return {k: None for k in QUESTION_KEYS} | {"guidance_change": float(guide), "overall": 0.0}

    out = tmp_path / "model" / "combiner.json"
    c = T.main(["--files", *map(str, files), "--offline", "--lambdas", "0.1,1", "--limit", "60",
                "--holdout", "2026Q3", "--holdout-limit", "50", "--out", str(out)],
               questionnaire_fn=fake_questionnaire)
    assert len(seen) == 60 + 60 + 50
    assert c.meta["quarters"] == ["2026Q1", "2026Q2"] and c.meta["n_rows"] == 120   # Q3 not trained on
    assert c.meta["cv"]["folds"] == ["2026Q1", "2026Q2"]                             # nor tuned on
    hold = c.meta["holdout"]
    assert hold["quarter"] == "2026Q3" and hold["n"] == 50 and hold["delta_r2"] > 0
    assert hold["ci90"][0] <= hold["delta_r2"] <= hold["ci90"][1]
    assert json.loads(out.read_text())["meta"]["holdout"]["n"] == 50
    assert "HOLDOUT 2026Q3" in capsys.readouterr().out


# -- live pipeline ------------------------------------------------------------------

EVENT = {"event_id": "e1", "event_type": "EARNINGS_RELEASE", "event_datetime": "2026-08-27T21:00:00Z",
         "knowledge_cutoff": "2026-08-27T20:00:00Z", "information_url": "https://example.test/x",
         "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "AAA"}]}


class _Resp:
    def raise_for_status(self) -> None: ...
    def json(self) -> dict: return BUNDLE


def _settings_with_combiner(tmp_path) -> AgentSettings:
    path = tmp_path / "model" / "combiner.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(COMBINER.to_dict()))
    return AgentSettings(deepseek_api_key="test", store_dir=tmp_path, online=False, use_calendar=False)


def test_predict_uses_questionnaire_and_combiner(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(P.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(P, "run_questionnaire",
                        lambda materials, ticker, settings, company_name=None: {"overall": 4.0})
    monkeypatch.setattr(P, "_chain", lambda s: pytest.fail("the chain must not run"))
    monkeypatch.setattr(P, "start_langchain_install", lambda: pytest.fail("no pip install in combiner mode"))
    preds, traces = P.predict_with_trace(EVENT, _settings_with_combiner(tmp_path))
    expected = COMBINER.predict({"q_overall": 4.0}, {"implied_move": 0.06})[0]
    assert preds == [{"identifier_value": "AAA", "predicted_percentile": pytest.approx(expected)}]
    assert traces[0]["mode"] == "combiner" and traces[0]["questionnaire"] == {"overall": 4.0}
    assert traces[0]["combiner"]["magnitude"] == 0.06 and "_context" not in traces[0]


def test_predict_keeps_the_combiner_when_the_questionnaire_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(P.httpx, "get", lambda *a, **k: _Resp())

    def boom(*a, **k):
        raise TimeoutError("deepseek down")

    monkeypatch.setattr(P, "run_questionnaire", boom)
    preds, traces = P.predict_with_trace(EVENT, _settings_with_combiner(tmp_path))
    assert 0.0 <= preds[0]["predicted_percentile"] <= 1.0
    assert "deepseek down" in traces[0]["questionnaire_error"] and traces[0]["mode"] == "combiner"


def test_predict_falls_back_to_the_chain_when_the_combiner_step_breaks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(P.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(P, "run_questionnaire", lambda *a, **k: {"overall": 1.0})
    monkeypatch.setattr(P, "feature_row", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug")))

    class Chain:
        def invoke(self, inputs):
            return P.Prediction(predicted_percentile=0.61, reason="fallback chain")

    monkeypatch.setattr(P, "_chain_without_install", lambda s: Chain())
    preds, traces = P.predict_with_trace(EVENT, _settings_with_combiner(tmp_path))
    assert preds[0]["predicted_percentile"] == 0.61
    assert "bug" in traces[0]["combiner_error"] and traces[0]["mode"] == "chain (combiner failed)"


def test_without_a_combiner_the_chain_path_is_unchanged(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(P.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(P, "start_langchain_install", lambda: None)

    class Chain:
        def invoke(self, inputs):
            return P.Prediction(predicted_percentile=0.42, reason="chain")

    monkeypatch.setattr(P, "_chain", lambda s: Chain())
    settings = AgentSettings(deepseek_api_key="test", store_dir=tmp_path, online=False, use_calendar=False)
    preds, traces = P.predict_with_trace(EVENT, settings)
    assert preds[0]["predicted_percentile"] == 0.42 and traces[0]["mode"] == "chain"