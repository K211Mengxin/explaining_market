"""★ THIS IS THE ONLY FILE YOU NEED TO EDIT. ★

`predict(event)` is called once per competition event, after the webhook has
already been verified for you. Return one prediction per focal asset. Everything
else in this repo (webhook verification, dedupe, submission) is plumbing.

This file is the whole agent, in numbered sections:

    0. materials       GET information_url -> format_materials()
    1. settings        configuration from the environment / .env
    2. store           on-disk cache; deployed, filled from the Modal Volume
    3. em_api          competition archive + events calendar
    4. market_data     daily prices, pre-event price features, event CARs
    5. sec_data        SEC EDGAR: SIC industry and XBRL statements
    6. fundamentals    point-in-time ratios (TTM, growth, margins, accruals ...)
    7. event_history   every past event and its realized reaction
    8. peers           SIC / fundamental-nearest-neighbour / hybrid peer groups
    9. prompts         system + user prompt
   10. chain           LangChain LCEL: prompt | ChatOpenAI(DeepSeek, JSON) | parser
   11. features        the "market and peer context", block by block
   12. questionnaire   the LLM fills in a fixed questionnaire about the text
   13. combiner        trained weights: questionnaire + context -> percentile
   14. pipeline        predict(): materials -> combiner (or chain) -> 0.5 fallback

It runs inside the official modal_app.py unchanged: that image only has
fastapi/httpx/openai/pydantic, so LangChain is installed on a container's first
call (section 10) and the prebuilt data store is read from the Modal Volume
`em-agent-store` through the Modal API (section 2). Build and upload the store
from your machine with scripts/build_store.py.

If `DEEPSEEK_API_KEY` is not set it returns a 0.5 baseline without touching the
network, so the full deploy → receive → submit round-trip still works without
burning credits.
"""

from __future__ import annotations

import bisect
import gzip
import importlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

# Time budget, sized against the 5-minute prediction window that opens when your
# handler ACKs the webhook. Worst case is 15 (materials) + 60 (context, see
# AGENT_FEATURE_BUDGET) + 90 x 2 (LLM call + one retry) = 255s, leaving ~45s to
# submit. Nothing upstream retries a failed prediction — once the delivery is
# ACKed the platform considers it done — so the one retry here is the only one
# you get. Raising any of these can push you past the deadline.
SUMMARY_TIMEOUT_SECONDS = 15.0

_HTTP: httpx.Client | None = None
_HTTP_LOCK = threading.Lock()


def http() -> httpx.Client:
    """One shared, thread-safe client for the agent's data calls.

    The event materials are still fetched with plain `httpx.get` (stage 0), so
    tests that stub `predict.httpx.get` only ever replace that one call.
    """
    global _HTTP
    with _HTTP_LOCK:
        if _HTTP is None:
            _HTTP = httpx.Client(timeout=30, follow_redirects=True)
        return _HTTP


# ============================================================================
# 0. materials
# ============================================================================

# ----------------------------------------------------------------------
# Reading the event's materials.
#
# The document behind `information_url` is a bundle: a list of `items`, each
# with an `id`, a `kind`, and a `content` whose JSON type follows the kind.
# Today an earnings event can carry three:
#
#   earnings-call-facts   kind "facts"  a list of strings     always present
#   earnings-preview      kind "text"   one markdown string   may be absent
#   option-implied-stats  kind "stats"  an OBJECT of numbers  may be absent
#
# Select items by `id`, never by position, and expect kinds you have not seen:
# new ones can be added at any time.
# ----------------------------------------------------------------------

FACTS_ID = "earnings-call-facts"
PREVIEW_ID = "earnings-preview"
OPTION_STATS_ID = "option-implied-stats"

# The preview is by far the largest item (often ~8,000 characters). It is the
# only one that gets cut, and it goes LAST in the prompt so a cut never costs
# you the facts or the option statistics.
PREVIEW_MAX_CHARS = 8000
OTHER_ITEM_MAX_CHARS = 2000


def _percent(block: object) -> str | None:
    """A `{value, status}` statistic as a percentage, or None if unavailable."""
    if not isinstance(block, dict) or block.get("status") != "ok":
        return None
    value = block.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return f"{value * 100:+.1f}" if value < 0 else f"{value * 100:.1f}"


def _format_option_stats(stats: dict) -> str | None:
    volatility = _percent(stats.get("implied_earnings_volatility"))
    if volatility is None:
        return None
    lines = [
        f"- Implied earnings volatility: {volatility}% (the standard deviation of the "
        "stock's move on this release that option prices imply)",
    ]
    move = _percent(stats.get("implied_absolute_earnings_move"))
    if move is not None:
        lines.append(
            f"- Implied absolute move: {move}% (the size of move options price in; "
            "it says nothing about direction)"
        )
    skew = _percent(stats.get("skew_25_delta"))
    if skew is not None:
        lines.append(
            f"- 25-delta skew: {skew} volatility points (call minus put implied "
            "volatility; negative means downside protection is priced richer)"
        )
    else:
        lines.append("- 25-delta skew: unavailable (options too thinly traded)")
    as_of = stats.get("as_of")
    header = "Option-market expectations, measured before the release"
    if isinstance(as_of, str) and as_of:
        header += f" (as of {as_of})"
    return header + ":\n" + "\n".join(lines)


def format_materials(bundle: object) -> str:
    """Turn the event's bundle into the text the model reads.

    Each item gets its own labelled section. Items are found by `id`, so the
    order they arrive in does not matter, and an item this code has never heard
    of is included if it is text and skipped otherwise, never an error.
    """
    if not isinstance(bundle, dict):
        return ""
    raw_items = bundle.get("items")
    items = [i for i in raw_items if isinstance(i, dict)] if isinstance(raw_items, list) else []
    by_id = {i.get("id"): i for i in items}
    sections: list[str] = []

    facts = (by_id.get(FACTS_ID) or {}).get("content")
    if isinstance(facts, list) and facts:
        sections.append(
            "Facts from the earnings call:\n"
            + "\n".join(f"{n}. {fact}" for n, fact in enumerate(facts, start=1))
        )

    stats = (by_id.get(OPTION_STATS_ID) or {}).get("content")
    if isinstance(stats, dict):
        formatted = _format_option_stats(stats)
        if formatted:
            sections.append(formatted)

    for item in items:
        if item.get("id") in (FACTS_ID, PREVIEW_ID, OPTION_STATS_ID):
            continue
        content = item.get("content")
        if isinstance(content, list) and all(isinstance(c, str) for c in content):
            content = "\n".join(content)
        if isinstance(content, str) and content.strip():
            label = item.get("id") or item.get("kind") or "item"
            sections.append(f"Additional material ({label}):\n{content[:OTHER_ITEM_MAX_CHARS]}")

    preview = (by_id.get(PREVIEW_ID) or {}).get("content")
    if isinstance(preview, str) and preview.strip():
        text = preview[:PREVIEW_MAX_CHARS]
        if len(preview) > PREVIEW_MAX_CHARS:
            text += "\n[preview truncated]"
        sections.append(
            "Research note written BEFORE the release (expectations, not results):\n" + text
        )

    return "\n\n".join(sections)


# ============================================================================
# 1. settings
# ----------------------------------------------------------------------------
# Agent configuration, read from the environment (and therefore from .env).
#
# Every knob has a default, so an empty environment still works; only
# DEEPSEEK_API_KEY is needed for real predictions.
#
#   DEEPSEEK_API_KEY        enables the LLM call (unset -> 0.5 baseline)
#   DEEPSEEK_MODEL          default deepseek-flash
#   DEEPSEEK_TEMPERATURE    optional; unset leaves the provider default
#   AGENT_BLOCKS            comma list of context blocks, "all" (default) or "none"
#                           -> prices,fundamentals,peers,peer_reactions,history,relative
#   AGENT_PEER_METHOD       sic | knn | hybrid (default): how peer groups are formed
#   AGENT_N_PEERS           peers per group (default 10)
#   AGENT_FEATURE_BUDGET    seconds allowed for building the context (default 60)
#   AGENT_USE_CALENDAR      1 (default) / 0: use GET /events for reactions not yet archived
#   AGENT_ONLINE            1 (default) / 0: allow network fetches for missing data
#   AGENT_STORE_DIR         data store (default scripts/data locally; a temp dir on Modal,
#                           filled from the Modal Volume AGENT_VOLUME, default em-agent-store)
#   AGENT_LIVE_PRICES       1 (default) / 0: fetch fresh daily prices from Yahoo when the
#                           store is older than the event
#   SEC_USER_AGENT          required by SEC EDGAR: "Your Name your@email"
#   EM_API_KEY / EM_API_BASE_URL   competition API (archive, calendar)
#   AGENT_USE_COMBINER      1 (default) / 0: questionnaire + trained combiner (section 13)
#                           when a combiner file exists; else the single-score chain
#   AGENT_COMBINER          path of combiner.json (default <store>/model/combiner.json)
#   AGENT_Q_SAMPLES         questionnaire calls averaged per event (default 1); keep it
#                           equal to the value used when training the combiner
# ============================================================================

ALL_BLOCKS: tuple[str, ...] = (
    "prices",          # the stock's own run-up, volatility and liquidity before the cutoff
    "fundamentals",    # latest filed statements: growth, margins, accruals, size, valuation
    "peers",           # who the peer group is (SIC / fundamental neighbours)
    "peer_reactions",  # how peers that already reported this season reacted
    "history",         # how big this stock's and its peers' earnings reactions usually are
    "relative",        # where the stock sits within its peer group on fundamentals
)
PEER_METHODS = ("sic", "knn", "hybrid")

_REPO_ROOT = Path(__file__).resolve().parent


def _blocks(raw: str | None) -> frozenset[str]:
    raw = (raw or "all").strip().lower()
    if raw == "all":
        return frozenset(ALL_BLOCKS)
    if raw in ("none", ""):
        return frozenset()
    chosen = {b.strip() for b in raw.split(",") if b.strip()}
    unknown = chosen - set(ALL_BLOCKS)
    if unknown:
        raise ValueError(f"AGENT_BLOCKS has unknown block(s) {sorted(unknown)}; "
                         f"choose from {list(ALL_BLOCKS)}")
    return frozenset(chosen)


def _store_dir(raw: str | None) -> Path:
    """AGENT_STORE_DIR, else the local store (scripts/data), else a temp dir that the
    data bundle is copied into from the Modal Volume (the deployed case)."""
    if raw:
        root = Path(raw)
    elif (_REPO_ROOT / "scripts" / "data").is_dir():
        root = _REPO_ROOT / "scripts" / "data"
    else:
        return Path(tempfile.gettempdir()) / "em_store"
    # `modal volume put <vol> scripts/data /` may land the files under /data/data.
    nested = root / "data"
    if not (root / "archive").exists() and (nested / "archive").exists():
        return nested
    return root


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AgentSettings:
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_temperature: float | None = None
    llm_timeout_s: float = 90.0
    llm_max_retries: int = 1

    blocks: frozenset[str] = field(default_factory=lambda: frozenset(ALL_BLOCKS))
    peer_method: str = "hybrid"
    n_peers: int = 10
    feature_budget_s: float = 60.0
    use_calendar: bool = True
    online: bool = True

    store_dir: Path = Path(tempfile.gettempdir()) / "em_store"
    volume_name: str = "em-agent-store"
    live_prices: bool = True
    sec_user_agent: str = ""
    em_api_key: str | None = None
    em_api_base_url: str = "https://api.explainingmarkets.ai/v1"

    use_combiner: bool = True
    combiner_path: Path | None = None   # None -> <store_dir>/model/combiner.json
    questionnaire_samples: int = 1
    questionnaire_timeout_s: float = 60.0

    @property
    def combiner_file(self) -> Path:
        return self.combiner_path or self.store_dir / "model" / "combiner.json"

    @classmethod
    def from_env(cls) -> AgentSettings:
        env = os.environ
        method = env.get("AGENT_PEER_METHOD", "hybrid").strip().lower()
        if method not in PEER_METHODS:
            raise ValueError(f"AGENT_PEER_METHOD must be one of {PEER_METHODS}, got {method!r}")
        temp = env.get("DEEPSEEK_TEMPERATURE")
        return cls(
            deepseek_api_key=env.get("DEEPSEEK_API_KEY") or None,
            deepseek_model=env.get("DEEPSEEK_MODEL", "deepseek-flash"),
            deepseek_base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_temperature=float(temp) if temp else None,
            blocks=_blocks(env.get("AGENT_BLOCKS")),
            peer_method=method,
            n_peers=int(env.get("AGENT_N_PEERS", "10")),
            feature_budget_s=float(env.get("AGENT_FEATURE_BUDGET", "60")),
            use_calendar=_flag("AGENT_USE_CALENDAR", True),
            online=_flag("AGENT_ONLINE", True),
            store_dir=_store_dir(env.get("AGENT_STORE_DIR")),
            volume_name=env.get("AGENT_VOLUME", "em-agent-store"),
            live_prices=_flag("AGENT_LIVE_PRICES", True),
            sec_user_agent=env.get("SEC_USER_AGENT", "").strip(),
            em_api_key=env.get("EM_API_KEY") or None,
            em_api_base_url=env.get("EM_API_BASE_URL", "https://api.explainingmarkets.ai/v1").rstrip("/"),
            use_combiner=_flag("AGENT_USE_COMBINER", True),
            combiner_path=Path(env["AGENT_COMBINER"]) if env.get("AGENT_COMBINER") else None,
            questionnaire_samples=max(1, int(env.get("AGENT_Q_SAMPLES", "1"))),
        )

    def describe(self) -> str:
        blocks = ",".join(b for b in ALL_BLOCKS if b in self.blocks) or "none"
        return f"model={self.deepseek_model} blocks={blocks} peers={self.peer_method}x{self.n_peers}"


# ============================================================================
# 2. store
# ----------------------------------------------------------------------------
# A tiny on-disk JSON cache shared by every data source.
#
# Layout under the store root (scripts/data locally, /data on Modal):
#
#     archive/<EVENT_TYPE>_<QUARTER>.jsonl.gz   competition archive files
#     history/index.json.gz                     compact index of past events + reactions
#     calendar/<YYYY-MM-DD>.json                GET /events snapshots
#     prices/<TICKER>.json.gz                   daily closes / volumes
#     sec/company_tickers.json                  SEC ticker -> CIK map
#     sec/profiles.json.gz                      ticker -> {cik, name, sic, sic_desc}
#     sec/facts/<CIK>.json.gz                   compact XBRL series per company
#     sec/facts_all.json.gz                     every company's series in one file (fast load)
#
# Writes are atomic (write to a temp file, then rename), so a reader never sees a
# half-written file even when several threads fetch at once.
# ============================================================================

class Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def lock(self, key: str) -> threading.Lock:
        """A per-key lock, so two threads never fetch the same thing twice."""
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def read_json(self, *parts: str, default: Any = None) -> Any:
        p = self.path(*parts)
        try:
            if p.suffix == ".gz":
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    return json.load(fh)
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (OSError, ValueError):
            return default  # a corrupt cache entry behaves like a miss

    def write_json(self, obj: Any, *parts: str) -> Path:
        p = self.path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as raw:
                data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
                raw.write(gzip.compress(data) if p.suffix == ".gz" else data)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return p

    def age_days(self, *parts: str) -> float | None:
        """Days since the file was written, or None if it does not exist."""
        try:
            return (time.time() - self.path(*parts).stat().st_mtime) / 86400
        except FileNotFoundError:
            return None

    # -- the deployed case: copy the data bundle out of a Modal Volume ----------

    def sync_from_volume(self, volume_name: str) -> str:
        """Copy the bundle (see BUNDLE_FILES) from a Modal Volume into this store, once.

        The official modal_app.py does not mount a volume, so the files are read
        through the Modal API instead. Each file's mtime is set to the bundle's
        build time, so freshness checks (e.g. "refetch SEC facts older than 7
        days") measure the data's age, not the copy's.
        """
        with _SYNC_LOCK:
            if self.path(BUNDLE_MANIFEST).exists():
                return "cached"
            import modal  # present in every Modal container (and locally via the project deps)

            vol = modal.Volume.from_name(volume_name)
            found: dict[str, str] = {}
            for entry in vol.listdir("/", recursive=True):
                path = str(getattr(entry, "path", entry)).lstrip("/")
                base = path.rsplit("/", 1)[-1]
                if base in BUNDLE_FILES and (base not in found or "bundle/" in path):
                    found[base] = path
            if "manifest.json" not in found:
                return "no bundle in volume"
            built_ts = time.time()
            for base in ["manifest.json", *[b for b in BUNDLE_FILES if b != "manifest.json"]]:
                if base not in found:
                    continue
                target = self.path(*BUNDLE_FILES[base])
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in vol.read_file(found[base]):
                        fh.write(chunk)
                tmp.replace(target)
                if base == "manifest.json":
                    built = (self.read_json(*BUNDLE_FILES[base]) or {}).get("built_at")
                    if built:
                        built_ts = datetime.fromisoformat(built.replace("Z", "+00:00")).timestamp()
                os.utime(target, (built_ts, built_ts))
            self.write_json({"synced_from": volume_name, "files": sorted(found)}, BUNDLE_MANIFEST)
            return f"synced {len(found)} files"


# bundle file name -> where it lives in the store. scripts/build_store.py writes
# these into scripts/data/bundle/; upload that folder to the Modal Volume.
BUNDLE_FILES: dict[str, tuple[str, ...]] = {
    "manifest.json": ("bundle", "manifest.json"),
    "profiles.json.gz": ("sec", "profiles.json.gz"),
    "facts_all.json.gz": ("sec", "facts_all.json.gz"),
    "history.json.gz": ("history", "index.json.gz"),
    "prices.json.gz": ("prices", "_bundle.json.gz"),
    "combiner.json": ("model", "combiner.json"),   # written by scripts/train_combiner.py
}
BUNDLE_MANIFEST = "bundle_synced.json"
_SYNC_LOCK = threading.Lock()


# ============================================================================
# 3. em_api
# ----------------------------------------------------------------------------
# The competition API endpoints the agent reads from (never /predictions).
#
#     GET /archive                          manifest of per-quarter archive files
#     GET /archive/{event_type}/{quarter}   a fresh signed URL for one file
#     GET /events?start_date&end_date       the events calendar (with knowledge_cutoff)
#
# All calls authenticate with X-API-Key (EM_API_KEY).
# ============================================================================

class EmApi:
    def __init__(self, api_key: str | None, base_url: str, *, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, **params) -> object:
        if not self.api_key:
            raise RuntimeError("EM_API_KEY is not set")
        r = http().get(f"{self.base_url}{path}", headers={"X-API-Key": self.api_key},
                      params={k: v for k, v in params.items() if v}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # -- archive -------------------------------------------------------------

    def archive_files(self) -> list[dict]:
        files = self._get("/archive").get("files", [])  # type: ignore[union-attr]
        return sorted(files, key=lambda f: (f["event_type"], f["quarter"]))

    def download_archive(self, event_type: str, quarter: str, dest_dir: Path,
                         *, force: bool = False) -> Path:
        """Download one quarter file; skipped when the local copy has the reported size."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f"{event_type}_{quarter}.jsonl.gz"
        meta = self._get(f"/archive/{event_type}/{quarter}")
        size = meta.get("bytes")  # type: ignore[union-attr]
        if not force and target.exists() and size and target.stat().st_size == size:
            return target
        tmp = target.with_suffix(".part")
        with http().stream("GET", meta["url"], follow_redirects=True, timeout=180) as r:  # type: ignore[index]
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 16):
                    fh.write(chunk)
        tmp.replace(target)
        return target

    # -- calendar ------------------------------------------------------------

    def events(self, start: str, end: str) -> list[dict]:
        return list(self._get("/events", start_date=start, end_date=end))  # type: ignore[arg-type]

    def events_cached(self, store: Store, start: str, end: str, *, max_age_h: float = 2.0) -> list[dict]:
        """GET /events, cached on disk for a couple of hours (keyed by the window)."""
        key = f"{start[:10]}_{end[:10]}.json"
        age = store.age_days("calendar", key)
        if age is not None and age * 24 <= max_age_h:
            cached = store.read_json("calendar", key)
            if cached is not None:
                return cached
        rows = self.events(start, end)
        store.write_json(rows, "calendar", key)
        return rows


def utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================
# 4. market_data
# ----------------------------------------------------------------------------
# Daily prices with an on-disk cache: AKShare (Sina) when the store is built
# locally (scripts/build_store.py), the Yahoo chart API (plain httpx, no extra
# packages) for the days the store does not have yet. Both give the same raw
# closes (checked on ADSK).
#
# Two jobs:
#   * pre-event price features for a stock, using only trading days strictly before
#     the knowledge-cutoff date (run-up, abnormal run-up vs SPY, volatility, liquidity);
#   * the realized next-day reaction of an event (car1 = r_i - r_m), which matches
#     the competition's own definition: close(window_end) / close(window_start) - 1,
#     with SPY standing in for the market (median gap to the archive's r_m ~5bp).
#
# Prices are unadjusted (they reproduce the archive's r_i exactly). Split days are
# detected and dropped from return calculations so a 2-for-1 split does not look
# like a -50% day.
# ============================================================================

MARKET = "SPY"
MAX_STALE_DAYS = 7  # last price more than a week before the cutoff -> treat as missing
_SPLIT_RATIOS = (2, 3, 4, 5, 8, 10, 15, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 15, 1 / 20, 3 / 2, 2 / 3)

Fetcher = Callable[[str], "PriceSeries | None"]


@dataclass(frozen=True)
class PriceSeries:
    dates: tuple[str, ...]      # 'YYYY-MM-DD', ascending
    close: tuple[float, ...]
    volume: tuple[float, ...]

    def index_before(self, day: str) -> int:
        """Index of the last trading day strictly before `day` (-1 if none)."""
        return bisect.bisect_left(self.dates, day) - 1

    def index_of(self, day: str) -> int | None:
        i = bisect.bisect_left(self.dates, day)
        return i if i < len(self.dates) and self.dates[i] == day else None

    @property
    def last_date(self) -> str | None:
        return self.dates[-1] if self.dates else None


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def akshare_fetch(ticker: str) -> PriceSeries | None:
    """Full daily history for one US ticker from Sina, with retries."""
    import akshare as ak  # heavy import: only when a fetch is actually needed

    last_error: Exception | None = None
    for symbol in dict.fromkeys([ticker, ticker.replace(".", "-"), ticker.replace(".", "")]):
        for attempt in range(4):
            try:
                df = ak.stock_us_daily(symbol=symbol, adjust="")
            except Exception as exc:  # Sina resets connections now and then
                last_error = exc
                if "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__:
                    time.sleep(1.5 * 2**attempt)
                    continue
                break  # unknown symbol / parse error: try the next spelling
            if df is None or df.empty:
                break
            dates = tuple(str(d)[:10] for d in df["date"])
            close = tuple(float(c) for c in df["close"])
            volume = tuple(float(v) if v == v else 0.0 for v in df["volume"])
            return PriceSeries(dates, close, volume)
    if last_error is not None:
        raise last_error
    return None


def yahoo_fetch(ticker: str) -> PriceSeries | None:
    """About two years of daily closes from Yahoo's chart API (no key, plain httpx)."""
    symbol = ticker.replace(".", "-")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            r = http().get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                          params={"range": "2y", "interval": "1d"},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            res = (r.json().get("chart") or {}).get("result") or []
            if not res:
                return None
            res = res[0]
            offset = int((res.get("meta") or {}).get("gmtoffset") or -18000)
            quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
            rows = [(datetime.fromtimestamp(ts + offset, timezone.utc).strftime("%Y-%m-%d"), c, v or 0.0)
                    for ts, c, v in zip(res.get("timestamp") or [], quote.get("close") or [],
                                        quote.get("volume") or [])
                    if c is not None]
            if not rows:
                return None
            rows = sorted(dict((d, (d, c, v)) for d, c, v in rows).values())  # one row per day
            return PriceSeries(tuple(r[0] for r in rows), tuple(float(r[1]) for r in rows),
                               tuple(float(r[2]) for r in rows))
        except Exception as exc:
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return None


class MarketData:
    def __init__(self, store: Store, *, online: bool = True, fetch: Fetcher | None = None) -> None:
        self.store = store
        self.online = online
        self._fetch = fetch or yahoo_fetch
        self._mem: dict[str, tuple[PriceSeries, str]] = {}  # ticker -> (series, fetched day)
        self._bundle: dict | None = None

    def _from_bundle(self, key: str) -> tuple[PriceSeries | None, str]:
        """The consolidated recent-prices file of the data bundle (deployed case)."""
        if self._bundle is None:
            self._bundle = self.store.read_json(*BUNDLE_FILES["prices.json.gz"], default={}) or {}
        row = (self._bundle.get("series") or {}).get(key)
        if not row:
            return None, ""
        return PriceSeries(tuple(row["dates"]), tuple(row["close"]), tuple(row["volume"])), \
            self._bundle.get("built", "")

    # -- loading -------------------------------------------------------------

    def series(self, ticker: str, need_through: str | None = None) -> PriceSeries | None:
        """Daily series for `ticker`, refreshed when the cache ends before `need_through`.

        A ticker is fetched at most once per UTC day; if the data still ends before
        `need_through` after that, the newer days simply are not published yet.
        """
        key = ticker.upper()
        with self.store.lock(f"prices:{key}"):
            if key in self._mem:
                cached, fetched = self._mem[key]
            else:
                raw = self.store.read_json("prices", f"{key}.json.gz")
                cached, fetched = None, ""
                if raw:
                    cached = PriceSeries(tuple(raw["dates"]), tuple(raw["close"]), tuple(raw["volume"]))
                    fetched = raw.get("fetched", "")
                else:
                    cached, fetched = self._from_bundle(key)
                if cached is not None:
                    self._mem[key] = (cached, fetched)

            fresh_enough = cached is not None and (
                need_through is None or (cached.last_date or "") >= need_through or fetched >= _today()
            )
            if fresh_enough or not self.online:
                return cached
            series = self._fetch(key)
            if series is None:
                return cached
            self.store.write_json(
                {"fetched": _today(), "dates": list(series.dates), "close": list(series.close),
                 "volume": list(series.volume)},
                "prices", f"{key}.json.gz",
            )
            self._mem[key] = (series, _today())
            return series

    def market(self, need_through: str | None = None) -> PriceSeries | None:
        return self.series(MARKET, need_through)

    # -- calendar ------------------------------------------------------------

    def next_trading_day(self, day: str) -> str | None:
        spy = self.market(day)
        if not spy:
            return None
        i = bisect.bisect_right(spy.dates, day)
        return spy.dates[i] if i < len(spy.dates) else None

    # -- features ------------------------------------------------------------

    def pre_event_features(self, ticker: str, cutoff_date: str) -> dict | None:
        """Price features from trading days strictly before the cutoff date."""
        s = self.series(ticker, need_through=_prev_day(cutoff_date))
        m = self.market(need_through=_prev_day(cutoff_date))
        if not s:
            return None
        last = s.index_before(cutoff_date)
        if last < 5:
            return None
        gap = (_date(cutoff_date) - _date(s.dates[last])).days
        if gap > MAX_STALE_DAYS:
            return None  # the series stops well before the event (halted / incomplete data)
        rets = _daily_returns(s, max(0, last - 252), last)
        out: dict[str, float | str | int] = {"as_of": s.dates[last], "last_close": s.close[last]}
        for n in (5, 20, 60):
            if last - n < 0:
                continue
            r = _compound(rets, s.dates[last - n], s.dates[last])
            out[f"ret_{n}d"] = r
            if m:
                rm = _window_return(m, s.dates[last - n], s.dates[last])
                if rm is not None:
                    out[f"abn_ret_{n}d"] = r - rm
        window = [r for d, r in rets if d > s.dates[max(0, last - 60)]]
        if len(window) >= 20:
            out["vol_60d"] = statistics.pstdev(window) * math.sqrt(252)
        dv = [s.close[i] * s.volume[i] for i in range(max(0, last - 19), last + 1)]
        if dv:
            out["dollar_vol_20d"] = sum(dv) / len(dv)
        i_year = max(0, last - 251)
        hi, lo = max(s.close[i_year: last + 1]), min(s.close[i_year: last + 1])
        split_in_year = any(_is_split(s.close[i] / s.close[i - 1])
                            for i in range(i_year + 1, last + 1) if s.close[i - 1] > 0)
        if hi > lo and not split_in_year:  # price levels are meaningless across a split
            out["pos_52w"] = (s.close[last] - lo) / (hi - lo)
        out["n_days"] = last + 1
        return out

    def event_car(self, ticker: str, window_start: str) -> dict | None:
        """Realized reaction of an event whose return window opens at `window_start`'s close."""
        end = self.next_trading_day(window_start)
        if end is None:
            return None
        s = self.series(ticker, need_through=end)
        m = self.market(need_through=end)
        if not s or not m:
            return None
        i0, i1 = s.index_of(window_start), s.index_of(end)
        j0, j1 = m.index_of(window_start), m.index_of(end)
        if None in (i0, i1, j0, j1):
            return None
        r_i = s.close[i1] / s.close[i0] - 1  # type: ignore[index]
        r_m = m.close[j1] / m.close[j0] - 1  # type: ignore[index]
        return {"window_start": window_start, "window_end": end, "r_i": r_i, "r_m": r_m, "car1": r_i - r_m}


# -- helpers -------------------------------------------------------------------


def _date(day: str):
    from datetime import date

    return date.fromisoformat(day)


def _prev_day(day: str) -> str:
    from datetime import timedelta

    return (_date(day) - timedelta(days=1)).isoformat()


def _is_split(ratio: float) -> bool:
    return any(abs(ratio / r - 1) < 0.03 for r in _SPLIT_RATIOS) and abs(ratio - 1) > 0.4


def _daily_returns(s: PriceSeries, i0: int, i1: int) -> list[tuple[str, float]]:
    """(date, simple return) for days i0+1..i1, dropping suspected split days."""
    out = []
    for i in range(i0 + 1, i1 + 1):
        prev, cur = s.close[i - 1], s.close[i]
        if prev <= 0 or cur <= 0:
            continue
        ratio = cur / prev
        if _is_split(ratio):
            continue
        out.append((s.dates[i], ratio - 1))
    return out


def _compound(rets: list[tuple[str, float]], start: str, end: str) -> float:
    total = 1.0
    for d, r in rets:
        if start < d <= end:
            total *= 1 + r
    return total - 1


def _window_return(s: PriceSeries, start: str, end: str) -> float | None:
    i0, i1 = s.index_of(start), s.index_of(end)
    if i0 is None or i1 is None:
        return None
    return _compound(_daily_returns(s, i0, i1), start, end)


# ============================================================================
# 5. sec_data
# ----------------------------------------------------------------------------
# SEC EDGAR: ticker -> CIK, SIC industry and XBRL financial-statement facts.
#
# Free, official, no API key. SEC asks every client to send a User-Agent with a
# name and e-mail (SEC_USER_AGENT) and to stay under ~10 requests per second.
#
# Each XBRL fact carries `filed` (the date the 10-Q/10-K reached the SEC), which is
# what makes point-in-time use possible: a quarter's statements usually arrive a
# few days AFTER the earnings release, so for that event only earlier filings may
# be used. `fundamentals.snapshot` applies that rule; this module only fetches,
# compacts and caches.
#
# Compact facts (per company):
#     {"revenue": [[start, end, value, filed, priority], ...], "assets": [[null, end, ...]], ...}
# `start` is null for balance-sheet (instant) items; `priority` is the position of
# the XBRL concept in FIELDS (lower = preferred when two concepts report the same
# period).
# ============================================================================

# logical field -> (taxonomy, unit, [concepts in priority order])
FIELDS: dict[str, tuple[str, str, list[str]]] = {
    "revenue": ("us-gaap", "USD", [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ]),
    "gross_profit": ("us-gaap", "USD", ["GrossProfit"]),
    "operating_income": ("us-gaap", "USD", ["OperatingIncomeLoss"]),
    "net_income": ("us-gaap", "USD", [
        "NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
    ]),
    "cfo": ("us-gaap", "USD", [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]),
    "assets": ("us-gaap", "USD", ["Assets"]),
    "liabilities": ("us-gaap", "USD", ["Liabilities"]),
    "equity": ("us-gaap", "USD", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
    "cash": ("us-gaap", "USD", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ]),
    "shares": ("dei", "shares", ["EntityCommonStockSharesOutstanding"]),
}
EARLIEST_END = "2016-01-01"  # older history is not needed for TTM / y-o-y as of 2026+

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def compact_facts(companyfacts: dict) -> dict[str, list[list]]:
    """Reduce a companyfacts JSON (often MBs) to the handful of fields we use."""
    facts = companyfacts.get("facts") or {}
    out: dict[str, list[list]] = {}
    for name, (taxonomy, unit, concepts) in FIELDS.items():
        seen: dict[tuple, list] = {}
        for prio, concept in enumerate(concepts):
            rows = (((facts.get(taxonomy) or {}).get(concept) or {}).get("units") or {}).get(unit) or []
            for r in rows:
                end, filed, val = r.get("end"), r.get("filed"), r.get("val")
                if not end or not filed or val is None or end < EARLIEST_END:
                    continue
                key = (r.get("start"), end, filed)
                if key not in seen or prio < seen[key][4]:
                    seen[key] = [r.get("start"), end, float(val), filed, prio]
        if seen:
            out[name] = sorted(seen.values(), key=lambda x: (x[1], x[3]))
    return out


class SecData:
    def __init__(self, store: Store, user_agent: str, *, online: bool = True,
                 min_interval_s: float = 0.12, client: httpx.Client | None = None) -> None:
        self.store = store
        self.user_agent = user_agent
        self.online = online
        self._min_interval = min_interval_s
        self._last_call = 0.0
        self._rate = threading.Lock()
        self._client = client
        self._tickers: dict[str, int] | None = None
        self._profiles: dict[str, dict] | None = None
        self._facts_all: dict[str, dict] | None = None   # cik (str) -> compact facts

    # -- HTTP ------------------------------------------------------------------

    def _get(self, url: str) -> dict:
        if not self.user_agent:
            raise RuntimeError('SEC_USER_AGENT is not set (SEC requires "Your Name your@email")')
        with self._rate:  # global throttle: SEC allows ~10 requests/second
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
        client = self._client or http()
        for attempt in range(3):
            r = client.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
            if r.status_code == 404:
                return {}
            if r.status_code in (429, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        return {}

    # -- ticker -> CIK -----------------------------------------------------------

    def ticker_map(self) -> dict[str, int]:
        if self._tickers is None:
            raw = self.store.read_json("sec", "company_tickers.json")
            age = self.store.age_days("sec", "company_tickers.json")
            if self.online and self.user_agent and (raw is None or (age or 0) > 7):
                try:
                    raw = self._get(TICKERS_URL)
                    self.store.write_json(raw, "sec", "company_tickers.json")
                except Exception:
                    pass  # keep the stale copy, if any
            self._tickers = {v["ticker"].upper(): int(v["cik_str"]) for v in (raw or {}).values()}
        return self._tickers

    def cik(self, ticker: str) -> int | None:
        m = self.ticker_map()
        t = ticker.upper()
        return m.get(t) or m.get(t.replace(".", "-")) or m.get(t.replace("-", "."))

    # -- profiles (name, SIC) ------------------------------------------------

    def profiles(self) -> dict[str, dict]:
        """ticker -> {cik, name, sic, sic_desc}: the consolidated file plus any fetched since."""
        if self._profiles is None:
            self._profiles = dict(self.store.read_json("sec", "profiles.json.gz", default={}) or {})
            extra = self.store.path("sec", "profiles")
            if extra.is_dir():
                for p in extra.glob("*.json"):
                    data = self.store.read_json("sec", "profiles", p.name)
                    if data:
                        self._profiles.setdefault(p.stem, data)
        return self._profiles

    def profile(self, ticker: str) -> dict | None:
        t = ticker.upper()
        prof = self.profiles().get(t)
        if prof is not None or not self.online:
            return prof
        cik = self.cik(t)
        if cik is None:
            return None
        with self.store.lock(f"profile:{t}"):
            sub = self._get(SUBMISSIONS_URL.format(cik=cik))
            if not sub:
                return None
            prof = {"cik": cik, "name": sub.get("name"), "sic": str(sub.get("sic") or ""),
                    "sic_desc": sub.get("sicDescription") or ""}
            self.store.write_json(prof, "sec", "profiles", f"{t}.json")
            self.profiles()[t] = prof
            return prof

    # -- financial statement facts -------------------------------------------

    def _all(self) -> dict[str, dict]:
        if self._facts_all is None:
            self._facts_all = dict(self.store.read_json("sec", "facts_all.json.gz", default={}) or {})
        return self._facts_all

    def facts(self, ticker: str, *, max_age_days: float = 7.0) -> dict[str, list[list]] | None:
        """Compact facts for one company: consolidated file, per-company cache, or a fetch."""
        cik = (self.profile(ticker) or {}).get("cik") or self.cik(ticker)
        if cik is None:
            return None
        key = str(cik)
        name = f"{int(cik):010d}.json.gz"
        cached = self.store.read_json("sec", "facts", name)
        age = self.store.age_days("sec", "facts", name)
        if cached is None and key in self._all():
            cached, age = self._all()[key], self.store.age_days("sec", "facts_all.json.gz")
        if cached is not None and (not self.online or (age is not None and age <= max_age_days)):
            return cached
        if not self.online:
            return cached
        with self.store.lock(f"facts:{key}"):
            try:
                compact = compact_facts(self._get(FACTS_URL.format(cik=int(cik))))
            except Exception:
                return cached if cached is not None else self._all().get(key)
            self.store.write_json(compact, "sec", "facts", name)
            self._all()[key] = compact
            return compact

    def universe_facts(self) -> dict[str, dict[str, list[list]]]:
        """ticker -> compact facts for every company in the store (no network)."""
        by_cik = dict(self._all())
        facts_dir = self.store.path("sec", "facts")
        if facts_dir.is_dir():
            for p in facts_dir.glob("*.json.gz"):
                key = str(int(p.name.split(".")[0]))
                if key not in by_cik:
                    data = self.store.read_json("sec", "facts", p.name)
                    if data:
                        by_cik[key] = data
        out = {}
        for ticker, prof in self.profiles().items():
            data = by_cik.get(str(prof.get("cik")))
            if data:
                out[ticker] = data
        return out


# ============================================================================
# 6. fundamentals
# ----------------------------------------------------------------------------
# Point-in-time ratios from compact XBRL facts (see sec_data.compact_facts).
#
# `snapshot(facts, as_of)` uses only filings with `filed < as_of`, and for every
# period the latest such filing (so a restatement counts only once it was filed).
#
# Flows (revenue, income, cash flow) are reported as 3-month quarters, as
# year-to-date figures (10-Q) and as fiscal years (10-K). Quarterly values and
# trailing-twelve-month (TTM) values are reconstructed from whichever is present:
#
#     quarter(E) = 3-month value ending E, else YTD(E) - YTD(previous quarter end)
#     TTM(E)     = annual ending E, else the sum of four quarters,
#                  else annual(E - 1y) + YTD(E) - YTD(E - 1y)
# ============================================================================

# The features used for fundamental similarity (peers.knn_peers).
SIMILARITY_FEATURES = (
    "rev_yoy", "gross_margin", "op_margin", "net_margin", "roa", "leverage", "accruals", "log_assets",
)


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _days(start: str, end: str) -> int:
    return (_d(end) - _d(start)).days


def _shift(end: str, days: int) -> str:
    return (_d(end) + timedelta(days=days)).isoformat()


def point_in_time(rows: list[list], as_of: str) -> list[list]:
    """For each (start, end) the latest value filed strictly before `as_of`."""
    best: dict[tuple, list] = {}
    for start, end, val, filed, prio in rows:
        if filed >= as_of:
            continue
        key = (start, end)
        cur = best.get(key)
        if cur is None or (filed, -prio) > (cur[3], -cur[4]):
            best[key] = [start, end, val, filed, prio]
    return sorted(best.values(), key=lambda r: r[1])


class Flow:
    """A duration item (e.g. revenue) as known at one point in time."""

    def __init__(self, rows: list[list]) -> None:
        self.rows = [r for r in rows if r[0]]  # flows always have a start date
        self.ends = sorted({r[1] for r in self.rows})

    def _find(self, end: str, dur: int, *, tol_end: int = 7, tol_dur: int = 12) -> list | None:
        best = None
        for r in self.rows:
            if abs(_days(r[1], end)) <= tol_end and abs(_days(r[0], r[1]) - dur) <= tol_dur:
                if best is None or abs(_days(r[1], end)) < abs(_days(best[1], end)):
                    best = r
        return best

    def _ytd(self, end: str) -> list | None:
        """The longest YTD-style value (91-365 days) ending at `end`."""
        cands = [r for r in self.rows if abs(_days(r[1], end)) <= 7 and 80 <= _days(r[0], r[1]) <= 380]
        return max(cands, key=lambda r: _days(r[0], r[1])) if cands else None

    def quarter(self, end: str) -> float | None:
        q = self._find(end, 91)
        if q:
            return q[2]
        ytd = self._ytd(end)
        if not ytd:
            return None
        dur = _days(ytd[0], ytd[1])
        prev = self._find(_shift(ytd[1], -91), dur - 91)
        if prev and abs(_days(prev[0], ytd[0])) <= 7:
            return ytd[2] - prev[2]
        return None

    def ttm(self, end: str) -> float | None:
        annual = self._find(end, 364)
        if annual:
            return annual[2]
        qs = [self.quarter(_shift(end, -91 * k)) for k in range(4)]
        if all(q is not None for q in qs):
            return float(sum(qs))  # type: ignore[arg-type]
        ytd = self._ytd(end)
        if ytd:
            dur = _days(ytd[0], ytd[1])
            prev_annual = self._find(_shift(ytd[0], -1), 364, tol_end=10)
            prev_ytd = self._find(_shift(end, -364), dur)
            if prev_annual and prev_ytd:
                return prev_annual[2] + ytd[2] - prev_ytd[2]
        return None

    def latest_end(self) -> str | None:
        return self.ends[-1] if self.ends else None


def _instant(rows: list[list]) -> tuple[float, str] | None:
    inst = [r for r in rows if not r[0]] or rows
    return (inst[-1][2], inst[-1][1]) if inst else None


def _ratio(a: float | None, b: float | None, *, positive_denominator: bool = True) -> float | None:
    if a is None or b is None or b == 0 or (positive_denominator and b < 0):
        return None
    return a / b


def revenue_growth_volatility(rev: Flow, end: str, quarters: int = 8, min_quarters: int = 4) -> float | None:
    """Standard deviation of the y/y growth of the last `quarters` quarterly revenues up to `end`.

    How much this company's revenue normally moves: a 2% revenue surprise is large
    for a firm whose growth barely varies and small for one whose growth swings by
    10 points. Used to put surprises on each company's own scale (section 13).
    """
    growth, taken = [], []
    for e in sorted((x for x in rev.ends if x <= end and _days(x, end) <= 91 * quarters + 45), reverse=True):
        if any(abs(_days(e, t)) < 45 for t in taken):
            continue  # same quarter reported with a slightly different end date
        taken.append(e)
        q, q_ly = rev.quarter(e), rev.quarter(_shift(e, -364))
        if q is not None and q_ly and q_ly > 0:
            growth.append(max(-5.0, min(5.0, q / q_ly - 1)))
        if len(taken) >= quarters:
            break
    return statistics.pstdev(growth) if len(growth) >= min_quarters else None


def snapshot(facts: dict[str, list[list]] | None, as_of: str,
             last_close: float | None = None) -> dict | None:
    """Ratios as they could have been computed on `as_of` (exclusive)."""
    if not facts:
        return None
    pit = {k: point_in_time(v, as_of) for k, v in facts.items()}
    rev, ni = Flow(pit.get("revenue", [])), Flow(pit.get("net_income", []))
    end = max(filter(None, (rev.latest_end(), ni.latest_end())), default=None)
    if end is None:
        return None
    filed = max((r[3] for rows in pit.values() for r in rows if r[1] == end), default=None)
    flows = {k: Flow(pit.get(k, [])) for k in ("gross_profit", "operating_income", "cfo")}

    rev_q, rev_q_ly = rev.quarter(end), rev.quarter(_shift(end, -364))
    rev_ttm, rev_ttm_ly = rev.ttm(end), rev.ttm(_shift(end, -364))
    ni_ttm = ni.ttm(end)
    gp_ttm, oi_ttm, cfo_ttm = (flows[k].ttm(end) for k in ("gross_profit", "operating_income", "cfo"))

    def inst(name: str) -> float | None:
        v = _instant(pit.get(name, []))
        return v[0] if v else None

    assets, liabilities, equity, cash, shares = (inst(k) for k in ("assets", "liabilities", "equity", "cash", "shares"))
    if liabilities is None and assets is not None and equity is not None:
        liabilities = assets - equity

    out: dict[str, float | str | None] = {
        "period_end": end,
        "filed": filed,
        "days_since_period_end": _days(end, as_of),
        "revenue_ttm": rev_ttm,
        "rev_yoy": _ratio(rev_q, rev_q_ly) - 1 if _ratio(rev_q, rev_q_ly) is not None else None,
        "rev_ttm_growth": _ratio(rev_ttm, rev_ttm_ly) - 1 if _ratio(rev_ttm, rev_ttm_ly) is not None else None,
        "gross_margin": _ratio(gp_ttm, rev_ttm),
        "op_margin": _ratio(oi_ttm, rev_ttm),
        "net_margin": _ratio(ni_ttm, rev_ttm),
        "roa": _ratio(ni_ttm, assets),
        "leverage": _ratio(liabilities, assets),
        "accruals": _ratio(None if ni_ttm is None or cfo_ttm is None else ni_ttm - cfo_ttm, assets),
        "cash_assets": _ratio(cash, assets),
        "log_assets": math.log(assets) if assets and assets > 0 else None,
        "rev_growth_vol": revenue_growth_volatility(rev, end),
    }
    if last_close and shares and shares > 0:
        mcap = last_close * shares
        out["market_cap"] = mcap
        out["earnings_yield"] = _ratio(ni_ttm, mcap)
        out["sales_yield"] = _ratio(rev_ttm, mcap)
        out["book_to_market"] = _ratio(equity, mcap)
    # Cap absurd values (tiny denominators) so one company cannot dominate distances.
    for k in ("rev_yoy", "rev_ttm_growth", "gross_margin", "op_margin", "net_margin", "roa", "accruals"):
        v = out.get(k)
        if isinstance(v, float):
            out[k] = max(-5.0, min(5.0, v))
    return out


# ============================================================================
# 7. event_history
# ----------------------------------------------------------------------------
# Every past earnings event the agent knows about, with its realized reaction.
#
# Sources:
#   * the competition archive (scripts/data/archive/*.jsonl.gz): event dates,
#     return windows and the official car1 for every scored event;
#   * the live events calendar (GET /events) for events that are not archived yet;
#     their car1 is computed from prices (market_data.event_car) when needed.
#
# The point-in-time rule lives in `known_before`: an event's reaction is usable for
# another event only if its return window closed BEFORE that event's cutoff date.
# The return window of an event opens at the close of its knowledge-cutoff day and
# ends at the next close, so `window_start` == the cutoff date.
# ============================================================================

INDEX_VERSION = 1


def _ny_offset_hours(utc: datetime) -> int:
    """New York's UTC offset (-4 in daylight time, -5 otherwise), without tzdata.

    US daylight time runs from 2:00 local on the second Sunday of March to 2:00
    local on the first Sunday of November.
    """
    y = utc.year

    def nth_sunday(month: int, n: int) -> datetime:
        d = datetime(y, month, 1, tzinfo=timezone.utc)
        d += timedelta(days=(6 - d.weekday()) % 7 + 7 * (n - 1))
        return d

    start = nth_sunday(3, 2) + timedelta(hours=7)   # 02:00 EST = 07:00 UTC
    end = nth_sunday(11, 1) + timedelta(hours=6)    # 02:00 EDT = 06:00 UTC
    return -4 if start <= utc < end else -5


@dataclass(frozen=True)
class PastEvent:
    event_id: str
    ticker: str
    event_datetime: str
    window_start: str            # = knowledge-cutoff date
    window_end: str | None       # None until known
    car1: float | None           # None until known
    source: str                  # "archive" | "calendar" | "prices"

    @property
    def resolved(self) -> bool:
        return self.car1 is not None and self.window_end is not None


def derive_cutoff(event_datetime: str) -> str:
    """Last regular US close (16:00 New York) at or before the event, as ISO UTC.

    Fallback when neither the payload nor the calendar gives a knowledge_cutoff.
    Weekends are skipped; exchange holidays are not (rare, and only a fallback).
    """
    utc = datetime.fromisoformat(event_datetime.replace("Z", "+00:00")).astimezone(timezone.utc)
    ts = utc + timedelta(hours=_ny_offset_hours(utc))  # New York wall-clock time
    day = ts.date()
    if ts.time() < dtime(16, 0) or day.weekday() >= 5:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    noon = datetime.combine(day, dtime(12, 0), tzinfo=timezone.utc)
    close = datetime.combine(day, dtime(16, 0), tzinfo=timezone.utc) - timedelta(hours=_ny_offset_hours(noon))
    return close.strftime("%Y-%m-%dT%H:%M:%SZ")


def _archive_events(path) -> list[PastEvent]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            cutoff = (rec.get("knowledge_cutoff") or "")[:10]
            for asset in rec.get("focal_assets") or []:
                t = asset.get("identifier_value")
                leg = (rec.get("event_returns") or {}).get(t) or {}
                ok = leg.get("car1") is not None and leg.get("window_end_date")
                start = leg.get("window_start_date") or cutoff
                if not start or not t:
                    continue
                out.append(PastEvent(
                    event_id=rec["event_id"], ticker=t.upper(),
                    event_datetime=rec.get("event_datetime", ""),
                    window_start=start,
                    window_end=leg.get("window_end_date") if ok else None,
                    car1=float(leg["car1"]) if ok else None,
                    source="archive",
                ))
    return out


class EventHistory:
    def __init__(self, events: list[PastEvent] | None = None) -> None:
        self._by_key: dict[tuple[str, str], PastEvent] = {}
        self.add(events or [])

    # -- building ------------------------------------------------------------

    def add(self, events: list[PastEvent]) -> None:
        """Add events; a resolved entry is never replaced by an unresolved one."""
        for ev in events:
            key = (ev.event_id, ev.ticker)
            cur = self._by_key.get(key)
            if cur is None or (ev.resolved and not cur.resolved):
                self._by_key[key] = ev

    def add_calendar(self, rows: list[dict]) -> None:
        events = []
        for row in rows:
            cutoff = (row.get("knowledge_cutoff") or "")[:10]
            if not cutoff:
                continue
            for asset in row.get("focal_assets") or []:
                t = (asset.get("identifier_value") or "").upper()
                if t:
                    events.append(PastEvent(row["event_id"], t, row.get("event_datetime", ""),
                                            cutoff, None, None, "calendar"))
        self.add(events)

    def resolve(self, ev: PastEvent, car: dict) -> PastEvent:
        new = replace(ev, window_end=car["window_end"], car1=car["car1"], source="prices")
        self._by_key[(ev.event_id, ev.ticker)] = new
        return new

    @classmethod
    def from_store(cls, store: Store) -> EventHistory:
        """Load the compact index, rebuilding it when the archive files changed."""
        files = sorted(store.path("archive").glob("*.jsonl.gz")) if store.path("archive").is_dir() else []
        signature = [[p.name, p.stat().st_size, int(p.stat().st_mtime)] for p in files]
        index = store.read_json("history", "index.json.gz")
        if index and index.get("version") == INDEX_VERSION and (
                index.get("signature") == signature or not files):  # no archive: the bundle's index
            return cls([PastEvent(*row) for row in index["events"]])
        events = [ev for p in files for ev in _archive_events(p)]
        if files:
            store.write_json({"version": INDEX_VERSION, "signature": signature,
                              "events": [list(ev.__dict__.values()) for ev in events]},
                             "history", "index.json.gz")
        return cls(events)

    # -- queries -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_key)

    def events(self) -> list[PastEvent]:
        return list(self._by_key.values())

    def known_before(self, cutoff_date: str, *, tickers: set[str] | None = None,
                     since: str | None = None, exclude_event: str | None = None) -> list[PastEvent]:
        """Resolved events whose return window closed strictly before `cutoff_date`."""
        out = []
        for ev in self._by_key.values():
            if not ev.resolved or ev.window_end >= cutoff_date:  # type: ignore[operator]
                continue
            if tickers is not None and ev.ticker not in tickers:
                continue
            if since is not None and ev.window_start < since:
                continue
            if exclude_event is not None and ev.event_id == exclude_event:
                continue
            out.append(ev)
        return sorted(out, key=lambda e: e.window_start)

    def pending_before(self, cutoff_date: str, *, tickers: set[str], since: str) -> list[PastEvent]:
        """Unresolved events that MAY have closed before the cutoff (resolve via prices)."""
        return [ev for ev in self._by_key.values()
                if not ev.resolved and ev.ticker in tickers and since <= ev.window_start < cutoff_date]


def season_start(cutoff_date: str, days: int = 75) -> str:
    """Earliest window start counted as 'this reporting season' for an event."""
    return (date.fromisoformat(cutoff_date) - timedelta(days=days)).isoformat()


# ============================================================================
# 8. peers
# ----------------------------------------------------------------------------
# Peer groups for one stock.
#
#   sic     same 4-digit SIC industry; widened to 3, then 2 digits until there are
#           enough companies; ordered by closeness in size (log assets)
#   knn     k nearest neighbours on point-in-time fundamentals
#           (fundamentals.SIMILARITY_FEATURES), each feature rank-transformed to
#           [0, 1] over the candidate pool so no single ratio dominates
#   hybrid  knn within the narrowest SIC level (4, 3, then 2 digits) with enough
#           companies, else knn over everything (the default)
#
# Only companies whose latest statements are recent (an active filer) are
# candidates, so delisted names drop out.
# ============================================================================

MIN_COMMON_FEATURES = 4
MAX_STALENESS_DAYS = 200


@dataclass
class PeerGroup:
    method: str
    description: str
    members: list[str]
    distance: dict[str, float] = field(default_factory=dict)


def _active(snap: dict | None) -> bool:
    return bool(snap) and (snap.get("days_since_period_end") or 10**6) <= MAX_STALENESS_DAYS


def sic_peers(ticker: str, profiles: dict[str, dict], k: int,
              snapshots: dict[str, dict] | None = None) -> PeerGroup | None:
    sic = (profiles.get(ticker) or {}).get("sic") or ""
    if len(sic) < 2:
        return None
    snapshots = snapshots or {}
    pool: list[str] = []
    level = 4
    for level in (4, 3, 2):
        pool = [t for t, p in profiles.items()
                if t != ticker and (p.get("sic") or "")[:level] == sic[:level]
                and (not snapshots or _active(snapshots.get(t)))]
        if len(pool) >= k:
            break
    own_size = (snapshots.get(ticker) or {}).get("log_assets")
    if own_size is not None:
        pool.sort(key=lambda t: abs(((snapshots.get(t) or {}).get("log_assets") or own_size + 99) - own_size))
    else:
        pool.sort()
    desc = (profiles.get(ticker) or {}).get("sic_desc") or ""
    label = f"SIC {sic[:level]}{'' if level == 4 else 'x' * (4 - level)}" + (f" ({desc})" if level == 4 and desc else "")
    return PeerGroup("sic", label, pool[:k])


def _rank01(values: dict[str, float]) -> dict[str, float]:
    """Percentile ranks in [0, 1]; ties share their average rank (so results never
    depend on dict/set iteration order)."""
    order = sorted(values, key=lambda t: (values[t], t))
    n = len(order)
    if n < 2:
        return {t: 0.5 for t in order}
    out: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 / (n - 1)
        i = j + 1
    return out


def knn_peers(ticker: str, snapshots: dict[str, dict], k: int,
              candidates: set[str] | None = None, label: str = "fundamental neighbours") -> PeerGroup | None:
    own = snapshots.get(ticker)
    if not own:
        return None
    pool = {t for t, s in snapshots.items() if t != ticker and _active(s)}
    if candidates is not None:
        pool &= candidates
    if not pool:
        return None
    ranked: dict[str, dict[str, float]] = {}
    for feat in SIMILARITY_FEATURES:
        vals = {t: snapshots[t][feat] for t in pool | {ticker}
                if isinstance(snapshots[t].get(feat), (int, float))}
        if ticker in vals and len(vals) > 2:
            ranked[feat] = _rank01(vals)
    if len(ranked) < MIN_COMMON_FEATURES:
        return None
    dist = {}
    for t in pool:
        diffs = [(r[t] - r[ticker]) ** 2 for r in ranked.values() if t in r]
        if len(diffs) >= MIN_COMMON_FEATURES:
            dist[t] = math.sqrt(sum(diffs) / len(diffs))
    members = sorted(dist, key=lambda t: (dist[t], t))[:k]
    return PeerGroup("knn", label, members, {t: dist[t] for t in members})


def hybrid_peers(ticker: str, profiles: dict[str, dict], snapshots: dict[str, dict], k: int) -> PeerGroup | None:
    """knn within the NARROWEST SIC level (4, 3, then 2 digits) that has at least
    2k active companies; knn over everything if none does."""
    sic = (profiles.get(ticker) or {}).get("sic") or ""
    active = {t for t, s in snapshots.items() if _active(s)}
    for level in (4, 3, 2):
        if len(sic) < level:
            continue
        sector = {t for t, p in profiles.items() if (p.get("sic") or "")[:level] == sic[:level]}
        if len((sector & active) - {ticker}) >= 2 * k:
            label = f"fundamental neighbours within SIC {sic[:level]}{'x' * (4 - level)}"
            if level == 4 and (profiles.get(ticker) or {}).get("sic_desc"):
                label += f" ({profiles[ticker]['sic_desc']})"
            group = knn_peers(ticker, snapshots, k, sector, label)
            if group and len(group.members) >= max(3, k // 2):
                group.method = "hybrid"
                return group
    group = knn_peers(ticker, snapshots, k)
    if group:
        group.method = "hybrid"
        return group
    return sic_peers(ticker, profiles, k, snapshots)


def build_peers(method: str, ticker: str, profiles: dict[str, dict],
                snapshots: dict[str, dict], k: int) -> PeerGroup | None:
    if method == "sic":
        return sic_peers(ticker, profiles, k, snapshots)
    if method == "knn":
        return knn_peers(ticker, snapshots, k) or sic_peers(ticker, profiles, k, snapshots)
    return hybrid_peers(ticker, profiles, snapshots, k)


# ============================================================================
# 9. prompts
# ----------------------------------------------------------------------------
# Prompts for the prediction chain (LangChain ChatPromptTemplate placeholders in {braces}).
# ============================================================================

SYSTEM_PROMPT = """\
You are a senior equity analyst predicting how a stock will react to an event.

Predict a single percentile in [0, 1] for how the focal asset's next-day
abnormal return will rank across all of the quarter's event outcomes:
0 = the quarter's most negative reaction, 0.50 = median, 1 = its most positive.
The relevant return is the *unexpected*, market-adjusted return — a
great-but-fully-priced-in beat is not a top-decile event.

How to read the inputs:
- The earnings-call facts are the news. The research note and the option
  statistics describe what was expected BEFORE the release; the reaction is
  driven by the gap between the two, not by the news alone.
- A strong run-up into the print raises the bar; a large implied move or a
  history of large reactions means the stock tends to land in the tails.
- Peers that already reported this season tell you how the market is treating
  this kind of news right now (sector-wide guidance cuts, demand signals,
  read-through to this company).
- Fundamentals and the peer-relative position are background: use them to judge
  whether the reported numbers are good *for this company*, not as a signal on
  their own.

Calibration discipline:
- Long-run base rates: about 25% of events land "up" (>0.75), 50% "neutral"
  (0.25-0.75), 25% "down" (<0.25). Default toward 0.40-0.60 when signals are
  mixed or modest.
- Reserve values above 0.80 or below 0.20 for cases with unambiguous,
  multi-signal evidence. Do not exceed 0.90 or fall below 0.10 without
  overwhelming, lopsided evidence.
- Tone alone (confident vs hedging language) should move you no more than
  ~0.03 absent quantitative confirmation.
- Use the full precision of the scale (e.g. 0.57, 0.38), not round numbers.

Respond with JSON only.
"""

HUMAN_PROMPT = """\
Event type: {event_type}
Ticker: {ticker}

Event materials:
{materials}

Market and peer context (computed only from data available before the knowledge cutoff):
{context}

Weigh, in roughly this order:
  1. Results and guidance versus what was expected — the research note's consensus,
     the options-implied move, and the stock's run-up into the print.
  2. What peers that already reported this season revealed, and how the market reacted.
  3. Strategic shifts — product launches, M&A, capital allocation, leadership.
  4. Risks called out — regulatory, supply chain, demand, competition.
  5. Tone and confidence in management commentary (small weight).

Predict the next-day unexpected-return percentile for {ticker}.
Return JSON in exactly this format:
{{"predicted_percentile": 0.55, "reason": "at most 25 words"}}
"""


# ============================================================================
# 10. chain
# ----------------------------------------------------------------------------
# The LangChain LCEL chain: prompt | DeepSeek (JSON mode) | parser.
#
#     inputs  {"event_type", "ticker", "materials", "context"}
#     output  Prediction(predicted_percentile in [0, 1], reason)
#
# DeepSeek speaks the OpenAI API, so `ChatOpenAI` is pointed at its base URL.
# DeepSeek supports JSON mode (`response_format={"type": "json_object"}`) but not
# OpenAI's strict json_schema, so the [0, 1] bound is enforced by the parser:
# an out-of-range or malformed reply raises, and predict() falls back to 0.5.
#
# LangChain is not in the official Modal image: `ensure_langchain` installs it on
# a cold start, and `OpenAIFallbackChain` runs the identical prompt if that fails.
# ============================================================================

class Prediction(BaseModel):
    predicted_percentile: float = Field(ge=0.0, le=1.0)
    reason: str = ""


def parse_prediction(text: str) -> Prediction:
    """JSON (optionally wrapped in a ```json fence) -> Prediction; raises on anything else."""
    text = (text or "").strip()
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    if fenced:
        text = fenced.group(0)
    return Prediction.model_validate(json.loads(text))


LANGCHAIN_PACKAGES = ("langchain-core>=1.0,<2", "langchain-openai>=1.0,<2")
_LC_DIR = Path(tempfile.gettempdir()) / "em_pydeps"
_LC_LOCK = threading.Lock()
_LC_STATE: dict[str, object] = {}


def _langchain_importable() -> bool:
    try:
        import langchain_core  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_langchain(timeout_s: float = 120.0) -> bool:
    """Make LangChain importable, installing it into a temp dir on first use.

    The official Modal image only has fastapi/httpx/openai/pydantic, so a cold
    container pip-installs langchain-core + langchain-openai (~10-20 s, once per
    container). The target dir goes at the END of sys.path, so the image's own
    openai/pydantic keep precedence over the copies pip pulls in as dependencies.
    Locally, `uv run --with langchain-openai ...` (or an existing install) skips
    all of this. Returns False if it cannot be installed; the caller then runs
    the same prompt through the openai SDK instead.
    """
    with _LC_LOCK:
        if "ok" in _LC_STATE:
            return bool(_LC_STATE["ok"])
        if str(_LC_DIR) not in sys.path:
            sys.path.append(str(_LC_DIR))
        ok = _langchain_importable()
        if not ok:
            cmds = [[sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                     "--target", str(_LC_DIR), *LANGCHAIN_PACKAGES]]
            if shutil.which("uv"):
                cmds.append(["uv", "pip", "install", "--quiet", "--target", str(_LC_DIR),
                             "--python", sys.executable, *LANGCHAIN_PACKAGES])
            for cmd in cmds:
                try:
                    done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    _LC_STATE["error"] = f"{type(exc).__name__}: {exc}"
                    continue
                if done.returncode == 0:
                    importlib.invalidate_caches()
                    ok = _langchain_importable()
                    if ok:
                        break
                _LC_STATE["error"] = (done.stderr or done.stdout)[-300:]
        _LC_STATE["ok"] = ok
        return ok


def start_langchain_install() -> threading.Thread:
    """Install in the background while the context is being built."""
    t = threading.Thread(target=ensure_langchain, name="langchain-install", daemon=True)
    t.start()
    return t


def build_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])


def build_llm(settings: AgentSettings):
    from langchain_openai import ChatOpenAI

    kwargs = {}
    if settings.deepseek_temperature is not None:
        kwargs["temperature"] = settings.deepseek_temperature
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        # Keep httpx's normal proxy handling (HTTPS_PROXY etc.); langchain-openai's
        # keep-alive socket options would otherwise bypass a configured proxy.
        http_socket_options=(),
        **kwargs,
    )


def build_chain(settings: AgentSettings, llm=None):
    """The LCEL chain: prompt | llm(JSON mode) | text | Prediction."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    model = (llm or build_llm(settings)).bind(response_format={"type": "json_object"})
    return build_prompt() | model | StrOutputParser() | RunnableLambda(parse_prediction)


class OpenAIFallbackChain:
    """Same prompts, same JSON mode, same parser, via the openai SDK in the image.

    Used only when LangChain could not be installed, so a failed pip install
    never costs a prediction. Exposes `.invoke(inputs)` like the LCEL chain.
    """

    def __init__(self, settings: AgentSettings) -> None:
        from openai import OpenAI

        self.settings = settings
        self.client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
                             timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries)

    def invoke(self, inputs: dict) -> Prediction:
        kwargs = {}
        if self.settings.deepseek_temperature is not None:
            kwargs["temperature"] = self.settings.deepseek_temperature
        resp = self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": HUMAN_PROMPT.format(**inputs)}],
            response_format={"type": "json_object"},
            **kwargs,
        )
        return parse_prediction(resp.choices[0].message.content or "")


# ============================================================================
# 11. features
# ----------------------------------------------------------------------------
# Build the "market and peer context" for one event, block by block.
#
#     stage A (parallel)  fundamentals    own point-in-time statements (+ SIC, market cap)
#                         prices          own run-up / volatility / liquidity before the cutoff
#                         calendar        GET /events, so not-yet-archived peers can be found
#     stage B             peers           the peer group (SIC / knn / hybrid)
#     stage C (parallel)  peer_reactions  how peers that already reported this season reacted
#                         history         how large this stock's and its peers' reactions usually are
#                         relative        the stock's percentile within its peers on fundamentals
#
# Every block is optional (AgentSettings.blocks), isolated (an exception only
# drops that block) and bounded by one overall time budget, so a slow data source
# can never cost the prediction. Each block uses only data that existed before
# the event's knowledge cutoff.
# ============================================================================

TITLES = {
    "fundamentals": "Fundamentals (latest statements filed before the cutoff)",
    "prices": "Price action into the release",
    "peers": "Peer group",
    "peer_reactions": "Peers that already reported this season",
    "history": "Typical size of earnings reactions",
    "relative": "Position within the peer group",
}


# -- formatting -----------------------------------------------------------------


def pct(x: float | None, digits: int = 1, sign: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def money(x: float | None) -> str:
    if x is None:
        return "n/a"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"${x / div:.1f}{unit}"
    return f"${x:.0f}"


# -- context ----------------------------------------------------------------------


@dataclass
class EventContext:
    ticker: str
    cutoff: str
    sections: dict[str, list[str]] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def cutoff_date(self) -> str:
        return self.cutoff[:10]

    def to_prompt(self) -> str:
        parts = []
        for block in ALL_BLOCKS:
            lines = self.sections.get(block)
            if lines:
                parts.append(f"{TITLES[block]}:\n" + "\n".join(f"- {line}" for line in lines))
        return "\n\n".join(parts) if parts else "(no additional context available)"


class AgentData:
    """Data sources shared by every event in this process (warm containers reuse it)."""

    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        self.store = Store(settings.store_dir)
        self.market = MarketData(self.store, online=settings.online and settings.live_prices)
        self.sec = SecData(self.store, settings.sec_user_agent,
                           online=settings.online and bool(settings.sec_user_agent))
        self.api = EmApi(settings.em_api_key, settings.em_api_base_url)
        self._history: EventHistory | None = None
        self._history_loaded = 0.0
        self._universe: dict[str, dict[str, dict]] = {}
        self._lock = threading.Lock()
        self.bundle_status: str | None = None

    def ensure_bundle(self) -> str:
        """Locally the store is already on disk; deployed, copy it from the Modal Volume."""
        if self.bundle_status is None:
            if self.store.path("sec", "profiles.json.gz").exists():
                self.bundle_status = "local store"
            elif _on_modal():
                try:
                    self.bundle_status = self.store.sync_from_volume(self.settings.volume_name)
                except Exception as exc:  # no volume / no bundle: run on live data only
                    self.bundle_status = f"no bundle ({type(exc).__name__}: {str(exc)[:120]})"
            else:
                self.bundle_status = "no store (run scripts/build_store.py)"
        return self.bundle_status

    @property
    def history(self) -> EventHistory:
        with self._lock:
            if self._history is None or time.time() - self._history_loaded > 3600:
                self._history = EventHistory.from_store(self.store)
                self._history_loaded = time.time()
            return self._history

    def universe_snapshots(self, cutoff_date: str) -> dict[str, dict]:
        """Every stored company's snapshot as of the Monday on/before the cutoff (cached)."""
        d = date.fromisoformat(cutoff_date)
        as_of = (d - timedelta(days=d.weekday())).isoformat()
        with self._lock:
            if as_of not in self._universe:
                snaps = {}
                for t, facts in self.sec.universe_facts().items():
                    s = snapshot(facts, as_of)
                    if s:
                        snaps[t] = s
                self._universe[as_of] = snaps
                if len(self._universe) > 16:
                    self._universe.pop(next(iter(self._universe)))
            return self._universe[as_of]


def _on_modal() -> bool:
    try:
        import modal

        return not modal.is_local()
    except Exception:
        return False


_DATA: dict[tuple, AgentData] = {}


def get_data(settings: AgentSettings) -> AgentData:
    key = (str(settings.store_dir), settings.online, settings.sec_user_agent, settings.em_api_key)
    if key not in _DATA:
        _DATA[key] = AgentData(settings)
    return _DATA[key]


def resolve_cutoff(event: dict, data: AgentData | None = None) -> str:
    """knowledge_cutoff from the payload, else the calendar, else derived from the time."""
    if event.get("knowledge_cutoff"):
        return event["knowledge_cutoff"]
    when = event.get("event_datetime")
    if data is not None and data.api.enabled and when and data.settings.use_calendar:
        try:
            day = date.fromisoformat(when[:10])
            rows = data.api.events_cached(data.store, (day - timedelta(days=4)).isoformat(),
                                          (day + timedelta(days=1)).isoformat())
            for row in rows:
                if row.get("event_id") == event.get("event_id") and row.get("knowledge_cutoff"):
                    return row["knowledge_cutoff"]
        except Exception:
            pass
    if when:
        return derive_cutoff(when)
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- blocks -----------------------------------------------------------------------


@dataclass
class _State:
    ticker: str
    cutoff_date: str
    event_id: str | None
    profile: dict | None = None
    own: dict | None = None            # own fundamentals snapshot
    prices: dict | None = None
    peers: PeerGroup | None = None
    snaps: dict[str, dict] = field(default_factory=dict)


def block_prices(data: AgentData, st: _State, ctx: EventContext) -> None:
    pf = data.market.pre_event_features(st.ticker, st.cutoff_date)
    st.prices = pf
    if not pf:
        return
    parts = []
    for n in (5, 20, 60):
        if f"ret_{n}d" in pf:
            abn = pf.get(f"abn_ret_{n}d")
            parts.append(f"{n}d {pct(pf[f'ret_{n}d'])}" + (f" ({pct(abn)} vs SPY)" if abn is not None else ""))
    lines = [f"Returns to the {pf['as_of']} close: " + ", ".join(parts)]
    extra = []
    if "vol_60d" in pf:
        extra.append(f"60-day volatility {pct(pf['vol_60d'], 0, sign=False)} annualised")
    if "pos_52w" in pf:
        extra.append(f"52-week range position {pf['pos_52w']:.2f} (0 = low, 1 = high)")
    if "dollar_vol_20d" in pf:
        extra.append(f"avg daily dollar volume {money(pf['dollar_vol_20d'])}")
    if extra:
        lines.append("; ".join(extra))
    ctx.sections["prices"] = lines
    ctx.data.update({k: pf.get(k) for k in ("abn_ret_5d", "abn_ret_20d", "abn_ret_60d", "vol_60d", "pos_52w",
                                            "last_close")})


def block_fundamentals(data: AgentData, st: _State, ctx: EventContext) -> None:
    st.profile = data.sec.profile(st.ticker)
    facts = data.sec.facts(st.ticker)
    pf = st.prices or data.market.pre_event_features(st.ticker, st.cutoff_date)
    st.own = snapshot(facts, st.cutoff_date, (pf or {}).get("last_close"))
    lines = []
    if st.profile and st.profile.get("sic"):
        lines.append(f"Industry: SIC {st.profile['sic']} {st.profile.get('sic_desc', '')}".rstrip())
    s = st.own
    if s:
        lines.append(f"Period ended {s['period_end']} (filed {s['filed']}); "
                     f"TTM revenue {money(s.get('revenue_ttm'))}")
        lines.append(
            f"Revenue growth y/y {pct(s.get('rev_yoy'))} (TTM {pct(s.get('rev_ttm_growth'))}); "
            f"gross margin {pct(s.get('gross_margin'), 0, False)}, operating margin {pct(s.get('op_margin'), 0, False)}, "
            f"net margin {pct(s.get('net_margin'), 0, False)}; ROA {pct(s.get('roa'))}")
        bits = [f"accruals {pct(s.get('accruals'))} of assets",
                f"liabilities/assets {s['leverage']:.2f}" if s.get("leverage") is not None else None,
                f"market cap {money(s.get('market_cap'))}" if s.get("market_cap") else None,
                f"earnings yield {pct(s.get('earnings_yield'))}" if s.get("earnings_yield") is not None else None,
                f"book-to-market {s['book_to_market']:.2f}" if s.get("book_to_market") is not None else None]
        lines.append("; ".join(b for b in bits if b))
        ctx.data.update({k: s.get(k) for k in ("rev_yoy", "op_margin", "net_margin", "roa", "accruals",
                                               "market_cap", "earnings_yield", "rev_growth_vol")})
    if lines:
        ctx.sections["fundamentals"] = lines


def block_calendar(data: AgentData, st: _State, ctx: EventContext) -> None:
    """Pull not-yet-archived events of this season into the history (no prompt text)."""
    start = season_start(st.cutoff_date, 80)
    rows = data.api.events_cached(data.store, start, st.cutoff_date)
    data.history.add_calendar(rows)
    ctx.data["calendar_events"] = len(rows)


def block_peers(data: AgentData, st: _State, ctx: EventContext) -> None:
    st.profile = st.profile or data.sec.profile(st.ticker)
    snaps = dict(data.universe_snapshots(st.cutoff_date))
    if st.own:
        snaps[st.ticker] = st.own
    profiles = data.sec.profiles()
    st.snaps = snaps
    st.peers = build_peers(data.settings.peer_method, st.ticker, profiles, snaps, data.settings.n_peers)
    if st.peers and st.peers.members:
        ctx.sections["peers"] = [f"{st.peers.description}: {', '.join(st.peers.members)}"]
        ctx.data["peers"] = " ".join(st.peers.members)
        ctx.data["peer_method"] = st.peers.method


def _summary(cars: list[float]) -> str:
    up = sum(1 for c in cars if c > 0)
    return (f"mean next-day abnormal return {pct(statistics.fmean(cars))}, median {pct(statistics.median(cars))}, "
            f"{up} up / {len(cars) - up} down")


def block_peer_reactions(data: AgentData, st: _State, ctx: EventContext, deadline: float) -> None:
    hist = data.history
    since = season_start(st.cutoff_date)
    peers = set(st.peers.members) if st.peers else set()

    # Peers whose reaction is not in the archive yet: compute it from prices.
    if peers and data.settings.online:
        pending = hist.pending_before(st.cutoff_date, tickers=peers, since=since)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(data.market.event_car, ev.ticker, ev.window_start): ev for ev in pending}
            done, _ = wait(futs, timeout=max(0.0, deadline - time.monotonic()))
            for f in done:
                car = f.exception() is None and f.result()
                if car and car["window_end"] < st.cutoff_date:
                    hist.resolve(futs[f], car)

    known = hist.known_before(st.cutoff_date, since=since, exclude_event=st.event_id)
    lines = []
    if peers:
        mine = [e for e in known if e.ticker in peers]
        if mine:
            lines.append(f"{len({e.ticker for e in mine})} of {len(peers)} peers already reported: "
                         + _summary([e.car1 for e in mine]))  # type: ignore[misc]
            recent = sorted(mine, key=lambda e: e.window_start, reverse=True)[:6]
            lines.append("Most recent: " + "; ".join(f"{e.ticker} {e.window_start[5:]} {pct(e.car1)}" for e in recent))
            ctx.data["peers_reported"] = len({e.ticker for e in mine})
            ctx.data["peer_mean_car"] = statistics.fmean(e.car1 for e in mine)  # type: ignore[misc]
        else:
            lines.append(f"None of the {len(peers)} peers has reported yet this season")
            ctx.data["peers_reported"] = 0
    sic2 = ((st.profile or {}).get("sic") or "")[:2]
    if sic2:
        profiles = data.sec.profiles()
        sector = [e for e in known if (profiles.get(e.ticker) or {}).get("sic", "")[:2] == sic2]
        if len(sector) >= 3:
            lines.append(f"Whole SIC {sic2}xx sector so far: {len(sector)} reports, "
                         + _summary([e.car1 for e in sector]))  # type: ignore[misc]
            ctx.data["sector_mean_car"] = statistics.fmean(e.car1 for e in sector)  # type: ignore[misc]
    if len(known) >= 20:
        cars = sorted(e.car1 for e in known)  # type: ignore[type-var]
        lines.append(f"All companies so far this season: {len(cars)} reports, mean {pct(statistics.fmean(cars))}, "
                     f"10th/90th percentile {pct(cars[len(cars) // 10])} / {pct(cars[9 * len(cars) // 10])}")
    if lines:
        ctx.sections["peer_reactions"] = lines


def block_history(data: AgentData, st: _State, ctx: EventContext) -> None:
    hist = data.history
    own = hist.known_before(st.cutoff_date, tickers={st.ticker}, exclude_event=st.event_id)[-8:]
    lines = []
    if own:
        abs_mean = statistics.fmean(abs(e.car1) for e in own)  # type: ignore[arg-type]
        lines.append(f"This stock, last {len(own)} releases: mean absolute reaction {pct(abs_mean, sign=False)}; "
                     "most recent " + ", ".join(f"{e.window_start[:7]} {pct(e.car1)}" for e in own[-4:][::-1]))
        ctx.data["own_mean_abs_car"] = abs_mean
    else:
        lines.append("No earlier releases of this stock in the archive")
    if st.peers and st.peers.members:
        past = [e for e in hist.known_before(season_start(st.cutoff_date), tickers=set(st.peers.members))][-40:]
        if len(past) >= 3:
            v = statistics.fmean(abs(e.car1) for e in past)  # type: ignore[arg-type]
            lines.append(f"Peers, earlier seasons: mean absolute reaction {pct(v, sign=False)} over {len(past)} releases")
            ctx.data["peer_mean_abs_car"] = v
    ctx.sections["history"] = lines


def block_relative(data: AgentData, st: _State, ctx: EventContext) -> None:
    if not st.own or not st.peers or not st.peers.members:
        return
    feats = {"rev_yoy": "revenue growth", "gross_margin": "gross margin", "op_margin": "operating margin",
             "net_margin": "net margin", "roa": "ROA", "accruals": "accruals", "leverage": "leverage"}
    parts = []
    for key, label in feats.items():
        mine = st.own.get(key)
        vals = [st.snaps[t][key] for t in st.peers.members
                if isinstance((st.snaps.get(t) or {}).get(key), (int, float))]
        if isinstance(mine, (int, float)) and len(vals) >= 3:
            rank = sum(v < mine for v in vals) / len(vals)
            parts.append(f"{label} {rank:.2f}")
            ctx.data[f"rel_{key}"] = rank
    if parts:
        ctx.sections["relative"] = ["Share of peers below this stock (1.00 = highest in group): " + ", ".join(parts)]


# -- orchestration ----------------------------------------------------------------


def build_context(event: dict, ticker: str, settings: AgentSettings,
                  data: AgentData | None = None) -> EventContext:
    """Never raises: failures are recorded in ctx.errors and the block is skipped."""
    t0 = time.monotonic()
    deadline = t0 + settings.feature_budget_s
    data = data or get_data(settings)
    cutoff = resolve_cutoff(event, data)
    ctx = EventContext(ticker=ticker.upper(), cutoff=cutoff)
    st = _State(ticker=ticker.upper(), cutoff_date=cutoff[:10], event_id=event.get("event_id"))
    want = settings.blocks
    if not want:
        return ctx
    ctx.data["store"] = data.ensure_bundle()

    def run(name: str, fn, *args) -> None:
        start = time.monotonic()
        try:
            fn(data, st, ctx, *args)
        except Exception as exc:
            ctx.errors[name] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            ctx.timings[name] = round(time.monotonic() - start, 2)

    need_peers = bool(want & {"peers", "peer_reactions", "history", "relative"})
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        stage_a = [pool.submit(run, "prices", block_prices)]
        if want & {"fundamentals", "relative"} or need_peers:
            stage_a.append(pool.submit(run, "fundamentals", block_fundamentals))
        if settings.use_calendar and data.api.enabled and settings.online and want & {"peer_reactions"}:
            stage_a.append(pool.submit(run, "calendar", block_calendar))
        _wait(stage_a, deadline, ctx, "stage A")

        if need_peers and time.monotonic() < deadline:
            _wait([pool.submit(run, "peers", block_peers)], deadline, ctx, "peers")

        stage_c = []
        if "peer_reactions" in want:
            stage_c.append(pool.submit(run, "peer_reactions", block_peer_reactions, deadline - 2))
        if "history" in want:
            stage_c.append(pool.submit(run, "history", block_history))
        if "relative" in want:
            stage_c.append(pool.submit(run, "relative", block_relative))
        _wait(stage_c, deadline, ctx, "stage C")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    for block in list(ctx.sections):
        if block not in want:
            del ctx.sections[block]  # computed as a dependency, but switched off for the prompt
    ctx.timings["total"] = round(time.monotonic() - t0, 2)
    return ctx


def _wait(futures, deadline: float, ctx: EventContext, label: str) -> None:
    _, not_done = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
    if not_done:
        ctx.errors[label] = f"time budget exhausted ({len(not_done)} block(s) skipped)"


# ============================================================================
# 12. questionnaire
# ----------------------------------------------------------------------------
# The LLM reads the event's text and fills in a FIXED questionnaire: every item
# asks one specific thing on an anchored scale, so an answer means the same on
# every event. The LLM does not weigh the items or produce the final number —
# section 13 does that with weights learned from the archive.
#
# The model sees the materials only (no ticker, no date), and the ticker and the
# company name are masked in the text, which makes it harder for it to recall how
# the stock actually reacted. Recall would inflate a backtest and would not carry
# over to live events.
#
# Besides the scored items, the model COPIES a few numbers from the text (actual
# vs expected revenue and EPS, new vs reference guidance). It never computes
# anything: section 13 turns the pairs into surprises and puts them on each
# company's own scale (a 2% revenue beat means more for a steady company than for
# one whose growth swings by 10 points).
#
# Bump QUESTIONNAIRE_VERSION whenever an item or the prompt changes: cached
# answers and a trained combiner are tied to it.
# ============================================================================

QUESTIONNAIRE_VERSION = "q2"


@dataclass(frozen=True)
class Question:
    key: str
    lo: int
    hi: int
    text: str


QUESTIONS: tuple[Question, ...] = (
    Question("eps_vs_expectation", -2, 2,
             "EPS versus the expectation stated in the materials (consensus, prior guidance or the "
             "research note): +2 clearly above, +1 slightly above, 0 in line, -1 slightly below, "
             "-2 clearly below."),
    Question("revenue_vs_expectation", -2, 2,
             "Revenue versus the expectation stated in the materials, same scale as above."),
    Question("guidance_change", -2, 2,
             "Forward guidance versus the company's previous guidance or the stated expectation: "
             "+2 clearly raised, +1 slightly raised, 0 maintained, -1 slightly lowered, "
             "-2 clearly cut or withdrawn. null if no guidance is mentioned."),
    Question("earnings_quality", -1, 1,
             "Where the EPS result comes from: +1 operating improvement (revenue growth, margins), "
             "0 mixed or unclear, -1 mainly one-offs (lower tax rate, asset sales, reserve releases, "
             "share buybacks)."),
    Question("margin_trend", -1, 1,
             "Gross or operating margin versus a year earlier: +1 expanding, 0 flat, -1 contracting."),
    Question("demand_outlook", -2, 2,
             "Management's forward-looking statements about demand (orders, bookings, backlog, "
             "pipeline, customer spending): +2 clearly strengthening, +1 somewhat stronger, 0 stable, "
             "-1 somewhat weaker, -2 clearly weakening."),
    Question("vs_preview", -2, 2,
             "Against the research note written BEFORE the release (its expectations and the key "
             "debates it names): +2 results clearly better than the note expected, 0 roughly as "
             "expected, -2 clearly worse. null if there is no research note."),
    Question("new_risks", 0, 3,
             "Material new negatives the materials reveal (regulatory, legal, supply chain, customer "
             "loss, management departure, accounting issues): 0 none, 1 minor, 2 notable, 3 severe."),
    Question("overall", -5, 5,
             "Overall, how much better or worse than what the market expected was this release: "
             "+5 far better, 0 as expected, -5 far worse."),
)
QUESTION_KEYS: tuple[str, ...] = tuple(q.key for q in QUESTIONS)

# Numbers copied from the materials (null when not stated). Units only need to match
# WITHIN a pair; section 13 uses ratios.
NUMBER_FIELDS: tuple[tuple[str, str], ...] = (
    ("revenue_actual", "reported revenue for the quarter"),
    ("revenue_expected", "the expectation for that revenue stated in the materials (consensus, else the "
                         "midpoint of the company's prior guidance), in the same units as revenue_actual"),
    ("eps_actual", "reported EPS for the quarter"),
    ("eps_expected", "the expectation for that EPS stated in the materials, on the SAME basis "
                     "(adjusted vs GAAP) as eps_actual"),
    ("guidance_new_mid", "midpoint of the newly issued guidance for the main guided metric "
                         "(prefer revenue; a single number counts as its own midpoint)"),
    ("guidance_ref_mid", "the reference for that same metric and period: the company's previous guidance "
                         "midpoint, else the consensus stated in the materials; same units"),
)
NUMBER_KEYS: tuple[str, ...] = tuple(k for k, _ in NUMBER_FIELDS)
GUIDANCE_METRICS = ("revenue", "eps", "other")

QUESTIONNAIRE_SYSTEM = """\
You are an equity analyst filling in a fixed questionnaire about one company's earnings release.

Rules:
- Part A: answer every item strictly on its own scale, as an integer.
- Judge results RELATIVE TO EXPECTATIONS stated in the materials (consensus, prior guidance,
  the research note), not in absolute terms. When deciding between "clearly" and "slightly",
  judge against what is normal for THIS company if the materials indicate it.
- Part B: copy numbers exactly as stated in the materials. Do not compute, convert or
  estimate. Both numbers of a pair must use the same units and basis.
- Use null when the materials do not address an item or state a number. Do not guess.
- The company is referred to as "the company"; do not try to identify it.

Respond with JSON only, with exactly the keys listed, e.g. {"eps_vs_expectation": 1, ...}.
"""


def questionnaire_prompt(materials: str) -> str:
    items = "\n".join(f'{i}. "{q.key}" (integer {q.lo} to {q.hi}, or null): {q.text}'
                      for i, q in enumerate(QUESTIONS, start=1))
    numbers = "\n".join(f'- "{k}" (number or null): {desc}' for k, desc in NUMBER_FIELDS)
    numbers += f'\n- "guidance_metric" (one of {", ".join(GUIDANCE_METRICS)}, or null): the metric of the guidance above'
    return (f"Event materials:\n{materials}\n\nPart A, questionnaire:\n{items}\n\n"
            f"Part B, numbers copied from the materials:\n{numbers}\n\nReturn the JSON object now.")


_NAME_SUFFIX = re.compile(
    r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|"
    r"n\.?v|s\.?a|ag|se|lp|llc|/de|/md)\.?$", re.IGNORECASE)


def company_core_name(name: str | None) -> str | None:
    """'Autodesk, Inc.' -> 'Autodesk'; 'NVIDIA CORP' -> 'NVIDIA'; None when too short to mask."""
    if not name:
        return None
    core = name.split(",")[0].strip()
    for _ in range(3):
        core = _NAME_SUFFIX.sub("", core).strip()
    core = core.strip(" ./")
    return core if len(core) >= 3 else None


def anonymize(text: str, ticker: str, company_name: str | None = None) -> str:
    """Mask the ticker and the company name, so the model reads the news, not the name."""
    t = re.escape(ticker.upper())
    if len(ticker) <= 2:
        # Short tickers ("A", "ON") are ordinary words: only mask their unmistakable forms.
        out = re.sub(rf"\({t}\)|(?:NYSE|NASDAQ|Nasdaq|NYSEARCA)\s*:\s*{t}\b", "(the company)", text)
    else:
        out = re.sub(rf"(?<![A-Za-z0-9]){t}(?![A-Za-z0-9])", "the company", text)
    core = company_core_name(company_name)
    if core:
        out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(core)}(?![A-Za-z0-9])", "the company", out,
                     flags=re.IGNORECASE)
    return out


def _finite(v: object) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def parse_questionnaire(text: str) -> dict:
    """The model's JSON reply -> {item: value clipped to its scale, or None} plus the
    copied numbers and the guidance metric.

    Raises when there is no JSON object or not a single usable answer, so the
    caller can tell a failed call from an event the materials say nothing about.
    """
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise ValueError("no JSON object in the questionnaire reply")
    raw = json.loads(m.group(0))
    if not isinstance(raw, dict):
        raise ValueError("questionnaire reply is not a JSON object")
    out: dict[str, float | str | None] = {}
    for q in QUESTIONS:
        v = _finite(raw.get(q.key))
        out[q.key] = None if v is None else float(min(q.hi, max(q.lo, v)))
    if all(out[k] is None for k in QUESTION_KEYS):
        raise ValueError("questionnaire reply has no usable answers")
    for key in NUMBER_KEYS:
        v = _finite(raw.get(key))
        out[key] = v if v is not None and abs(v) < 1e15 else None
    metric = raw.get("guidance_metric")
    out["guidance_metric"] = metric.strip().lower() if isinstance(metric, str) and \
        metric.strip().lower() in GUIDANCE_METRICS else None
    return out


def average_answers(answers: list[dict]) -> dict:
    """Several questionnaire replies -> one: the mean of each scored item, the median of
    each copied number (robust to one reply in the wrong units), the most common metric."""
    out: dict[str, float | str | None] = {}
    for key in QUESTION_KEYS:
        vals = [a[key] for a in answers if a.get(key) is not None]
        out[key] = statistics.fmean(vals) if vals else None
    for key in NUMBER_KEYS:
        vals = [a[key] for a in answers if a.get(key) is not None]
        out[key] = statistics.median(vals) if vals else None
    metrics = [a["guidance_metric"] for a in answers if a.get("guidance_metric")]
    out["guidance_metric"] = max(set(metrics), key=metrics.count) if metrics else None
    return out


_Q_CLIENTS: dict[tuple, Any] = {}
_Q_LOCK = threading.Lock()


def _questionnaire_client(settings: AgentSettings):
    """openai SDK client pointed at DeepSeek (in the official image; no LangChain needed)."""
    key = (settings.deepseek_base_url, settings.deepseek_api_key, settings.questionnaire_timeout_s)
    with _Q_LOCK:
        if key not in _Q_CLIENTS:
            from openai import OpenAI

            _Q_CLIENTS[key] = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
                                     timeout=settings.questionnaire_timeout_s, max_retries=1)
        return _Q_CLIENTS[key]


def _ask_questionnaire(client, settings: AgentSettings, materials: str) -> dict[str, float | None]:
    kwargs = {}
    if settings.deepseek_temperature is not None:
        kwargs["temperature"] = settings.deepseek_temperature
    resp = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "system", "content": QUESTIONNAIRE_SYSTEM},
                  {"role": "user", "content": questionnaire_prompt(materials)}],
        response_format={"type": "json_object"},
        **kwargs,
    )
    return parse_questionnaire(resp.choices[0].message.content or "")


def run_questionnaire(materials: str, ticker: str, settings: AgentSettings, *,
                      company_name: str | None = None, client=None) -> dict[str, float | None]:
    """Fill in the questionnaire (averaged over AGENT_Q_SAMPLES calls). Raises if every call fails."""
    text = anonymize(materials, ticker, company_name)
    client = client or _questionnaire_client(settings)
    n = max(1, settings.questionnaire_samples)
    if n == 1:
        return _ask_questionnaire(client, settings, text)
    answers, last_error = [], None
    with ThreadPoolExecutor(max_workers=n) as pool:
        for fut in [pool.submit(_ask_questionnaire, client, settings, text) for _ in range(n)]:
            try:
                answers.append(fut.result())
            except Exception as exc:
                last_error = exc
    if not answers:
        raise last_error or RuntimeError("questionnaire failed")
    return average_answers(answers)


# ============================================================================
# 13. combiner
# ----------------------------------------------------------------------------
# Turns the questionnaire answers and the numeric context into one percentile,
# with weights learned from the archive by scripts/train_combiner.py:
#
#     feature_row()      answers + context numbers -> named features (training AND live)
#     Combiner.predict() standardise -> weighted sum -> centre/scale
#                        -> magnitude scaling -> clip to [0, 1]
#
# The weights come from a ridge regression of the realised percentile y on the
# features with the surprise percentile partialled out per quarter, which is the
# quantity the competition scores. Magnitude (implied move, past reaction size)
# is not a linear feature: it carries no direction. It stretches a prediction
# away from the centre instead, by (m / m_ref) ** power (the official
# option-statistics notebook uses power 0.5). Training can fit the weights jointly
# with that scaling (features multiplied by the same factor) or stretch afterwards;
# cross-validation picks which, and the exponent (0 = no scaling).
# ============================================================================

COMBINER_FORMAT = 1

# Signed context features the combiner may weigh (section 11 keys, plus option skew).
CONTEXT_FEATURES: tuple[str, ...] = (
    "abn_ret_5d", "abn_ret_20d", "abn_ret_60d", "pos_52w",
    "peer_mean_car", "sector_mean_car",
    "rev_yoy", "op_margin", "accruals", "earnings_yield", "log_market_cap",
    "skew_25d", "has_options",
)
# Surprises computed from the numbers the model copied, raw and on the company's own scale.
SURPRISE_FEATURES: tuple[str, ...] = (
    "revenue_surprise_pct", "revenue_surprise_z", "guidance_change_pct", "guidance_change_z", "eps_surprise_px",
)
FEATURE_NAMES: tuple[str, ...] = tuple(f"q_{k}" for k in QUESTION_KEYS) + SURPRISE_FEATURES + CONTEXT_FEATURES

MAX_REL_CHANGE = 0.5      # |actual / expected - 1| beyond this is almost surely a units mix-up
MAX_PRICE_SURPRISE = 0.2  # |EPS surprise| beyond 20% of the share price: same
MIN_GROWTH_VOL = 0.01     # floor of the company scale, so a very steady company cannot explode z
MAX_Z = 10.0
# Unsigned size-of-reaction measures, in order of preference.
MAGNITUDE_KEYS: tuple[str, ...] = ("implied_move", "own_mean_abs_car", "peer_mean_abs_car", "vol_60d")


def option_stats_values(bundle: object) -> dict[str, float | None]:
    """The option-implied statistics of the event's bundle as numbers (None when absent)."""
    out: dict[str, float | None] = {"implied_move": None, "implied_vol": None, "skew_25d": None}
    items = bundle.get("items") if isinstance(bundle, dict) else None
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("id") != OPTION_STATS_ID:
            continue
        content = item.get("content")
        if not isinstance(content, dict):
            continue
        for key, name in (("implied_move", "implied_absolute_earnings_move"),
                          ("implied_vol", "implied_earnings_volatility"), ("skew_25d", "skew_25_delta")):
            block = content.get(name)
            if isinstance(block, dict) and block.get("status") == "ok":
                out[key] = _finite(block.get("value"))
    return out


def _rel_change(new: float | None, ref: float | None) -> float | None:
    if new is None or ref is None or ref == 0:
        return None
    r = (new - ref) / abs(ref)
    return r if abs(r) <= MAX_REL_CHANGE else None


def surprise_features(answers: dict | None, ctx_data: dict | None) -> dict[str, float | None]:
    """Surprises from the copied numbers, raw and scaled by the company's own variability.

    revenue_surprise_pct   revenue actual / expected - 1
    revenue_surprise_z     ... divided by the std of the company's y/y revenue growth
                           (fundamentals.rev_growth_vol): "how unusual for THIS company"
    guidance_change_pct    new / reference guidance midpoint - 1 (revenue or unspecified metric)
    guidance_change_z      ... on the same company scale (revenue guidance only)
    eps_surprise_px        (EPS actual - expected) / last close, the textbook price scaling,
                           which stays sane when expected EPS is near zero
    """
    a, d = answers or {}, ctx_data or {}
    rev = _rel_change(_finite(a.get("revenue_actual")), _finite(a.get("revenue_expected")))
    metric = a.get("guidance_metric")
    guide = _rel_change(_finite(a.get("guidance_new_mid")), _finite(a.get("guidance_ref_mid"))) \
        if metric in ("revenue", None) else None
    vol = _finite(d.get("rev_growth_vol"))
    scale = max(vol, MIN_GROWTH_VOL) if vol is not None else None

    def z(x: float | None) -> float | None:
        return None if x is None or scale is None else max(-MAX_Z, min(MAX_Z, x / scale))

    eps_a, eps_e, px = _finite(a.get("eps_actual")), _finite(a.get("eps_expected")), _finite(d.get("last_close"))
    eps_px = None
    if eps_a is not None and eps_e is not None and px and px > 0:
        eps_px = (eps_a - eps_e) / px
        eps_px = eps_px if abs(eps_px) <= MAX_PRICE_SURPRISE else None
    return {"revenue_surprise_pct": rev, "revenue_surprise_z": z(rev), "guidance_change_pct": guide,
            "guidance_change_z": z(guide) if metric == "revenue" else None, "eps_surprise_px": eps_px}


def feature_row(answers: dict | None, ctx_data: dict | None, options: dict | None) -> dict[str, float | None]:
    """Named features for the combiner. The ONLY place features are defined, so the
    training script and the live pipeline cannot drift apart."""
    answers, d, options = answers or {}, ctx_data or {}, options or {}
    row: dict[str, float | None] = {f"q_{k}": _finite(answers.get(k)) for k in QUESTION_KEYS}
    row.update(surprise_features(answers, d))
    for key in ("abn_ret_5d", "abn_ret_20d", "abn_ret_60d", "pos_52w", "peer_mean_car", "sector_mean_car",
                "rev_yoy", "op_margin", "accruals", "earnings_yield"):
        row[key] = _finite(d.get(key))
    mcap = _finite(d.get("market_cap"))
    row["log_market_cap"] = math.log(mcap) if mcap and mcap > 0 else None
    row["skew_25d"] = _finite(options.get("skew_25d"))
    row["has_options"] = 1.0 if _finite(options.get("implied_move")) is not None else 0.0
    return row


def magnitude_inputs(ctx_data: dict | None, options: dict | None) -> dict[str, float | None]:
    d, options = ctx_data or {}, options or {}
    return {"implied_move": _finite(options.get("implied_move")),
            "own_mean_abs_car": _finite(d.get("own_mean_abs_car")),
            "peer_mean_abs_car": _finite(d.get("peer_mean_abs_car")),
            "vol_60d": _finite(d.get("vol_60d"))}


@dataclass(frozen=True)
class Combiner:
    features: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]            # per standardised feature
    scale: float = 1.0                    # linear score -> percentile units
    center: float = 0.5
    magnitude: dict | None = None         # {"base", "ratios", "ref", "power"} or None
    questionnaire_version: str = QUESTIONNAIRE_VERSION
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Combiner:
        if d.get("format") != COMBINER_FORMAT:
            raise ValueError(f"combiner format {d.get('format')!r}, expected {COMBINER_FORMAT}")
        feats = d["features"]
        names = tuple(f["name"] for f in feats)
        means = tuple(float(f["mean"]) for f in feats)
        stds = tuple(float(f["std"]) for f in feats)
        weights = tuple(float(f["weight"]) for f in feats)
        if any(not math.isfinite(x) for x in (*means, *stds, *weights)) or any(s <= 0 for s in stds):
            raise ValueError("combiner has non-finite values or non-positive standard deviations")
        unknown = set(names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"combiner uses unknown features {sorted(unknown)}")
        mag = d.get("magnitude")
        if mag is not None and not (mag.get("base") in MAGNITUDE_KEYS and float(mag.get("ref", 0)) > 0):
            raise ValueError("combiner magnitude block is invalid")
        return cls(names, means, stds, weights, float(d.get("scale", 1.0)), float(d.get("center", 0.5)),
                   mag, d.get("questionnaire_version", ""), d.get("meta") or {})

    def to_dict(self) -> dict:
        return {
            "format": COMBINER_FORMAT,
            "questionnaire_version": self.questionnaire_version,
            "features": [{"name": n, "mean": m, "std": s, "weight": w}
                         for n, m, s, w in zip(self.features, self.means, self.stds, self.weights)],
            "scale": self.scale, "center": self.center, "magnitude": self.magnitude, "meta": self.meta,
        }

    def linear(self, row: dict[str, float | None]) -> tuple[float, int]:
        """Weighted sum of standardised features; a missing feature sits at its mean (adds 0)."""
        z, present = 0.0, 0
        for name, mean, std, w in zip(self.features, self.means, self.stds, self.weights):
            v = _finite(row.get(name))
            if v is not None:
                z += w * (v - mean) / std
                present += 1
        return z, present

    def magnitude_of(self, mags: dict[str, float | None]) -> float | None:
        """Expected size of the reaction, in units of the base measure (None if unknown)."""
        if not self.magnitude:
            return None
        base = self.magnitude["base"]
        ratios = self.magnitude.get("ratios") or {}
        for key in (base, *[k for k in MAGNITUDE_KEYS if k != base and k in ratios]):
            v = _finite(mags.get(key))
            r = 1.0 if key == base else _finite(ratios.get(key))
            if v is not None and v > 0 and r:
                return v * r
        return None

    def predict(self, row: dict[str, float | None], mags: dict[str, float | None] | None = None) -> tuple[float, dict]:
        z, present = self.linear(row)
        p = self.center + self.scale * z
        m = self.magnitude_of(mags or {})
        if m is not None:
            p = self.center + (p - self.center) * (m / float(self.magnitude["ref"])) ** float(
                self.magnitude.get("power", 0.5))
        p = min(0.999, max(0.001, p))
        return p, {"z": round(z, 6), "features_present": present, "magnitude": m}


_COMBINERS: dict[str, tuple[float, Combiner | None]] = {}


def load_combiner(path: Path) -> Combiner | None:
    """The trained combiner at `path`, or None (missing, invalid or for another questionnaire)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _COMBINERS.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    combiner: Combiner | None
    try:
        combiner = Combiner.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if combiner.questionnaire_version != QUESTIONNAIRE_VERSION:
            print(f"[WARN] combiner {path} was trained on questionnaire {combiner.questionnaire_version!r}, "
                  f"this code asks {QUESTIONNAIRE_VERSION!r}; retrain it. Using the single-score chain.")
            combiner = None
    except Exception as exc:
        print(f"[WARN] combiner {path} is unusable ({type(exc).__name__}: {exc}); using the single-score chain.")
        combiner = None
    _COMBINERS[str(path)] = (mtime, combiner)
    return combiner


# ============================================================================
# 14. pipeline
# ----------------------------------------------------------------------------
# materials -> questionnaire + context -> combiner -> percentile when a trained
# combiner is present (sections 12-13), else materials -> context -> chain; with a
# 0.5 fallback at every step so an event is never left without a submission.
# ============================================================================

_warned = False  # one-shot warning when no key is configured
_chains: dict[tuple, object] = {}
_chains_lock = threading.Lock()


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    `event` is the verified webhook payload. Useful fields:
      event["event_type"]          e.g. "EARNINGS_RELEASE"
      event["focal_assets"]        list of {"identifier_type", "identifier_value"}
      event["information_url"]     short-lived signed URL to the event's materials
                                   (a JSON bundle of items; see `format_materials`)
      event["prediction_deadline"] ISO timestamp; submit before this fires
      event["knowledge_cutoff"]    not in the webhook payload; looked up in the
                                   calendar (or derived) by the context stage

    Required return: a list of dicts, one per focal asset:
      [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]

    `predicted_percentile` is a float in [0, 1] — where you predict the asset's
    next-day abnormal (market-adjusted) return will rank across all of the
    quarter's event outcomes: 0 = the quarter's most negative reaction,
    0.50 = median, 1 = its most positive. It's a cross-sectional rank across the
    quarter's events, not a percentile within the asset's own history.
    """
    predictions, _ = predict_with_trace(event)
    return predictions


def predict_with_trace(event: dict, settings: AgentSettings | None = None) -> tuple[list[dict], list[dict]]:
    """`predict`, plus one trace dict per asset (context, timings, reply) for debugging."""
    settings = settings or AgentSettings.from_env()
    combiner = _active_combiner(settings)
    if settings.deepseek_api_key and combiner is None and "ok" not in _LC_STATE:
        start_langchain_install()  # overlaps the pip install with materials + context
    bundle, materials_error = _fetch_bundle(event)
    materials = _materials_text(bundle)

    # One pipeline run per focal asset, in series — today every event carries a
    # single asset; if that changes, run them concurrently rather than raising
    # the time budget.
    predictions, traces = [], []
    for asset in event["focal_assets"]:
        ticker = asset["identifier_value"]
        trace: dict = {"ticker": ticker, "settings": settings.describe(),
                       "mode": "combiner" if combiner is not None else "chain"}
        if materials_error:
            trace["materials_error"] = materials_error
        predictions.append({
            "identifier_value": ticker,
            "predicted_percentile": _predict_asset(event, ticker, materials, settings, trace,
                                                   bundle=bundle, combiner=combiner),
        })
        traces.append(trace)
    return predictions, traces


def _active_combiner(settings: AgentSettings) -> Combiner | None:
    """The trained combiner, if enabled and present (deployed: after the Volume sync)."""
    if not (settings.deepseek_api_key and settings.use_combiner):
        return None
    try:
        get_data(settings).ensure_bundle()  # copies model/combiner.json from the Volume on Modal
        return load_combiner(settings.combiner_file)
    except Exception as exc:
        print(f"[WARN] could not load the combiner ({type(exc).__name__}: {exc})")
        return None


def _fetch_bundle(event: dict) -> tuple[object | None, str | None]:
    """The event's materials bundle (None on failure, with the error). Never raises."""
    try:
        resp = httpx.get(event["information_url"], timeout=SUMMARY_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:  # predict on the context alone rather than miss the event
        return None, f"{type(exc).__name__}: {exc}"


def _materials_text(bundle: object | None) -> str:
    if bundle is None:
        return "(the event materials could not be retrieved)"
    text = format_materials(bundle)
    if not text:
        # Not a bundle this code recognises. Show the model what arrived rather
        # than nothing.
        text = json.dumps(bundle)[:PREVIEW_MAX_CHARS]
    return text


def _load_materials(event: dict) -> tuple[str, str | None]:
    """The event's own materials as prompt text. Never raises."""
    bundle, error = _fetch_bundle(event)
    return _materials_text(bundle), error


def _predict_asset(event: dict, ticker: str, materials: str, settings: AgentSettings, trace: dict,
                   *, bundle: object | None = None, combiner: Combiner | None = None) -> float:
    """Context -> combiner (or chain) -> percentile for one asset (0.5 on any failure)."""
    global _warned
    if not settings.deepseek_api_key:
        if not _warned:
            print(
                "[WARN] DEEPSEEK_API_KEY not set — submitting 0.5 placeholder. "
                "Set the key (or edit predict.py) for real predictions."
            )
            _warned = True
        trace["fallback"] = "no DEEPSEEK_API_KEY"
        return 0.5

    if combiner is not None:
        try:
            return _predict_with_combiner(event, ticker, materials, bundle, settings, combiner, trace)
        except Exception as exc:  # never lose the event: fall back to the single-score chain
            trace["combiner_error"] = f"{type(exc).__name__}: {exc}"[:300]
            trace["mode"] = "chain (combiner failed)"

    # Market and peer context (never raises; failed blocks are skipped).
    context = trace.pop("_context", None)
    if context is None:
        try:
            context = build_context(event, ticker, settings)
        except Exception as exc:
            context = EventContext(ticker=ticker, cutoff=event.get("knowledge_cutoff", ""))
            context.errors["context"] = f"{type(exc).__name__}: {exc}"
    trace.update(cutoff=context.cutoff, context=context.to_prompt(), context_data=context.data,
                 context_errors=context.errors, timings=context.timings)

    t0 = time.monotonic()
    try:
        chain = _chain(settings) if combiner is None else _chain_without_install(settings)
        trace["chain"] = type(chain).__name__
        prediction: Prediction = chain.invoke({
            "event_type": event.get("event_type", "EARNINGS_RELEASE"),
            "ticker": ticker,
            "materials": materials,
            "context": context.to_prompt(),
        })
    except Exception as exc:
        # A missed event is imputed with your mean anyway; submit 0.5 instead.
        trace["fallback"] = f"{type(exc).__name__}: {exc}"[:300]
        return 0.5
    finally:
        trace["llm_seconds"] = round(time.monotonic() - t0, 2)
    trace["reason"] = prediction.reason
    return prediction.predicted_percentile


def _chain(settings: AgentSettings):
    """The LCEL chain (installing LangChain if needed), else the openai-SDK twin."""
    key = (settings.deepseek_model, settings.deepseek_base_url, settings.deepseek_api_key,
           settings.deepseek_temperature, settings.llm_timeout_s, settings.llm_max_retries)
    with _chains_lock:
        if key not in _chains:
            _chains[key] = build_chain(settings) if ensure_langchain() else OpenAIFallbackChain(settings)
        return _chains[key]


def _chain_without_install(settings: AgentSettings):
    """The chain for a fallback inside the combiner path: never blocks on a pip install."""
    if _LC_STATE.get("ok") or _langchain_importable():
        return _chain(settings)
    return OpenAIFallbackChain(settings)


def _predict_with_combiner(event: dict, ticker: str, materials: str, bundle: object | None,
                           settings: AgentSettings, combiner: Combiner, trace: dict) -> float:
    """Questionnaire (LLM, on the text) and context (numbers) in parallel -> combiner."""
    data = get_data(settings)
    name = (data.sec.profiles().get(ticker.upper()) or {}).get("name")
    pool = ThreadPoolExecutor(max_workers=1)
    t0 = time.monotonic()
    try:
        q_future = pool.submit(run_questionnaire, materials, ticker, settings, company_name=name) \
            if bundle is not None else None
        try:
            context = build_context(event, ticker, settings, data)
        except Exception as exc:
            context = EventContext(ticker=ticker, cutoff=event.get("knowledge_cutoff", ""))
            context.errors["context"] = f"{type(exc).__name__}: {exc}"
        trace["_context"] = context  # reused by the chain if the combiner step fails
        answers = None
        if q_future is None:
            trace["questionnaire_error"] = "no materials"
        else:
            try:
                answers = q_future.result(timeout=2 * settings.questionnaire_timeout_s + 10)
            except Exception as exc:  # the numeric features still carry signal
                trace["questionnaire_error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        pool.shutdown(wait=False)
    trace["llm_seconds"] = round(time.monotonic() - t0, 2)

    options = option_stats_values(bundle)
    row = feature_row(answers, context.data, options)
    p, info = combiner.predict(row, magnitude_inputs(context.data, options))
    trace.pop("_context", None)
    trace.update(cutoff=context.cutoff, context=context.to_prompt(), context_data=context.data,
                 context_errors=context.errors, timings=context.timings,
                 questionnaire=answers, features=row, combiner=info)
    return p