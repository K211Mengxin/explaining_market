"""Pre-fetch everything the agent's context blocks read, into the data store.

Run this locally (weekly at least; daily keeps prices fresh), then upload the
bundle to the Modal Volume that predict.py reads at runtime:

    uv run --with akshare python scripts/build_store.py               # archive + SEC + prices
    uv run --with akshare python scripts/build_store.py --skip-prices # SEC only (fast)
    uv run --with akshare python scripts/build_store.py --limit 200   # quick trial
    uv run modal volume put em-agent-store scripts/data/bundle /bundle --force

(`--with akshare` keeps pyproject.toml / uv.lock identical to the official repo.)

Steps (each one cached and resumable; re-running skips what is fresh):
  1. archive   every quarter file from GET /archive           -> data/archive/
  2. universe  every ticker that appears in the archive (+ SPY)
  3. SEC       ticker -> CIK, SIC (submissions) and compact XBRL facts
               -> data/sec/profiles.json.gz, data/sec/facts_all.json.gz
  4. prices    full daily history per ticker from AKShare      -> data/prices/
  5. history   the compact index of past events + reactions    -> data/history/
  6. bundle    the files the deployed agent needs, consolidated -> data/bundle/
               (profiles, facts_all, history index, last ~420 trading days of prices)

Needs EM_API_KEY (archive) and SEC_USER_AGENT ("Your Name your@email") in .env.
SEC allows ~10 requests/second (two per company); Sina takes ~2-4 s per ticker,
so a full first run takes roughly 10 minutes for SEC and 30-60 minutes for prices.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from predict import (  # noqa: E402
    BUNDLE_FILES, MARKET, AgentSettings, EmApi, EventHistory, MarketData, SecData, Store, akshare_fetch,
    yahoo_fetch,
)

BUNDLE_PRICE_DAYS = 420  # enough for 252-day features plus a season of peer windows



def parallel(label: str, fn, items: list[str], workers: int) -> tuple[int, list[str]]:
    ok, failed, t0 = 0, [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                if fut.result():
                    ok += 1
                else:
                    failed.append(futs[fut])
            except Exception as exc:
                failed.append(futs[fut])
                if len(failed) <= 5:
                    print(f"  {label} {futs[fut]}: {type(exc).__name__}: {str(exc)[:100]}")
            if i % 100 == 0 or i == len(items):
                print(f"  {label}: {i}/{len(items)} done, {ok} ok, {len(failed)} missing, {time.time() - t0:.0f}s")
    return ok, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-archive", action="store_true")
    ap.add_argument("--skip-sec", action="store_true")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only a random N tickers (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --limit")
    ap.add_argument("--tickers", nargs="*", default=[], help="always include these tickers")
    ap.add_argument("--price-workers", type=int, default=4)
    ap.add_argument("--sec-workers", type=int, default=6)
    args = ap.parse_args()

    settings = AgentSettings.from_env()
    store = Store(settings.store_dir)
    print(f"store: {store.root}")

    # 1. archive
    api = EmApi(settings.em_api_key, settings.em_api_base_url)
    if not args.skip_archive:
        if not api.enabled:
            print("[!] EM_API_KEY not set: skipping the archive download")
        else:
            for f in api.archive_files():
                path = api.download_archive(f["event_type"], f["quarter"], store.path("archive"))
                print(f"  archive {path.name}: {path.stat().st_size:,} bytes")

    # 2. universe
    history = EventHistory.from_store(store)
    tickers = sorted({ev.ticker for ev in history.events()})
    if args.limit:
        import random

        tickers = sorted(set(random.Random(args.seed).sample(tickers, min(args.limit, len(tickers))))
                         | {t.upper() for t in args.tickers})
    print(f"universe: {len(tickers)} tickers from {len(history)} archived events")

    # 3. SEC
    if not args.skip_sec:
        if not settings.sec_user_agent:
            sys.exit('SEC_USER_AGENT is not set; add e.g. SEC_USER_AGENT="Jane Doe jane@example.com" to .env')
        sec = SecData(store, settings.sec_user_agent, online=True)
        n_map = len(sec.ticker_map())
        print(f"SEC ticker map: {n_map} tickers")
        if not n_map:
            sys.exit("SEC refused the ticker map: SEC_USER_AGENT must contain a contact e-mail")
        parallel("SEC", lambda t: sec.profile(t) is not None and sec.facts(t) is not None,
                 tickers, args.sec_workers)
        profiles = {t: p for t, p in sec.profiles().items() if p}
        store.write_json(profiles, "sec", "profiles.json.gz")
        facts_all = {}
        for p in store.path("sec", "facts").glob("*.json.gz"):
            data = store.read_json("sec", "facts", p.name)
            if data:
                facts_all[str(int(p.name.split(".")[0]))] = data
        store.write_json(facts_all, "sec", "facts_all.json.gz")
        print(f"  wrote profiles for {len(profiles)} tickers, facts for {len(facts_all)} companies")

    # 4. prices
    if not args.skip_prices:
        sources = {"sina": 0, "yahoo": 0}

        def fetch(ticker: str):
            """Sina via AKShare first; Yahoo when Sina does not know the ticker
            (AKShare then fails to decode Sina's reply, e.g. a SyntaxError)."""
            try:
                series = akshare_fetch(ticker)
            except Exception:
                series = None
            if series is not None:
                sources["sina"] += 1
                return series
            series = yahoo_fetch(ticker)
            if series is not None:
                sources["yahoo"] += 1
            return series

        market = MarketData(store, online=True, fetch=fetch)
        need = (date.today() - timedelta(days=1)).isoformat()
        parallel("prices", lambda t: market.series(t, need_through=need) is not None,
                 [MARKET, *tickers], args.price_workers)
        print(f"  price sources this run: {sources['sina']} Sina, {sources['yahoo']} Yahoo")

    # 5. history index (already rebuilt above if the archive changed)
    print(f"history index: {len(EventHistory.from_store(store))} events")

    # 6. bundle for the deployed agent
    write_bundle(store, [MARKET, *tickers])
    print("\nUpload it to the Modal Volume the deployed predict.py reads:\n"
          "  uv run modal volume put em-agent-store scripts/data/bundle /bundle --force")


def write_bundle(store: Store, tickers: list[str]) -> None:
    """Consolidate what the deployed agent reads into scripts/data/bundle/."""
    import shutil
    from datetime import datetime, timezone

    out = store.path("bundle")
    out.mkdir(parents=True, exist_ok=True)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for name in ("profiles.json.gz", "facts_all.json.gz", "history.json.gz"):
        src = store.path(*BUNDLE_FILES[name])
        if src.exists():
            shutil.copyfile(src, out / name)
    series = {}
    for t in tickers:
        raw = store.read_json("prices", f"{t}.json.gz")
        if raw and raw.get("dates"):
            k = -BUNDLE_PRICE_DAYS
            series[t] = {"dates": raw["dates"][k:], "close": raw["close"][k:], "volume": raw["volume"][k:]}
    store.write_json({"built": built[:10], "series": series}, "bundle", "prices.json.gz")
    files = sorted(p.name for p in out.glob("*.gz"))
    store.write_json({"built_at": built, "files": files, "tickers_with_prices": len(series)},
                     "bundle", "manifest.json")
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f"bundle: {', '.join(files)} ({size / 1e6:.1f} MB, prices for {len(series)} tickers)")


if __name__ == "__main__":
    main()
