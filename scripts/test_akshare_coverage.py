"""Can AKShare supply daily prices for the competition's stocks?

Samples events from a cached archive quarter and, for each ticker, pulls the
daily history from AKShare (Sina source, `ak.stock_us_daily`). It checks:

  1. found       AKShare returns any data for the ticker at all
  2. window      both days of the event's return window are present
  3. r_i match   close(window_end) / close(window_start) - 1 agrees with the
                 archive's r_i (the competition's own stock return)
  4. history     how many trading days exist in the lookback before the
                 knowledge cutoff (what you would build features from)

Usage (from the repo root; Python >= 3.11). `--with akshare` installs it into a
throwaway environment, so pyproject.toml / uv.lock / the Modal image are untouched:
    uv run --with akshare python scripts/test_akshare_coverage.py            # 100 events
    uv run --with akshare python scripts/test_akshare_coverage.py -n 300 --seed 1
    uv run --with akshare python scripts/test_akshare_coverage.py --file scripts/data/archive/EARNINGS_RELEASE_2026Q3.jsonl.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import statistics
import time
from datetime import date, timedelta
from pathlib import Path

import akshare as ak

HERE = Path(__file__).resolve().parent
ARCHIVE_DIR = HERE / "data" / "archive"
OUT_DIR = HERE / "results"


def latest_archive() -> Path:
    files = sorted(ARCHIVE_DIR.glob("*.jsonl.gz"))
    if not files:
        raise SystemExit(
            f"No archive file in {ARCHIVE_DIR}. Run scripts/backtest.py once to download one, "
            "or pass --file."
        )
    return files[-1]


def sample_events(path: Path, n: int, seed: int) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            for asset in rec.get("focal_assets") or []:
                t = asset.get("identifier_value")
                leg = (rec.get("event_returns") or {}).get(t) or {}
                if leg.get("r_i") is None or not leg.get("window_start_date"):
                    continue
                rows.append({
                    "ticker": t,
                    "event_datetime": rec.get("event_datetime", "")[:16],
                    "cutoff_date": rec["knowledge_cutoff"][:10],
                    "window_start": leg["window_start_date"],
                    "window_end": leg["window_end_date"],
                    "r_i": float(leg["r_i"]),
                })
    if 0 < n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


_cache: dict[str, dict[str, float] | None] = {}


def fetch_closes(ticker: str, pause: float) -> tuple[dict[str, float] | None, str, str]:
    """Return ({'YYYY-MM-DD': close}, symbol used, error). Tries a few symbol spellings."""
    if ticker in _cache:
        return _cache[ticker], ticker, ""
    error = ""
    for symbol in dict.fromkeys([ticker, ticker.replace(".", "-"), ticker.replace(".", "")]):
        for attempt in range(4):  # Sina resets connections now and then; back off and retry
            try:
                df = ak.stock_us_daily(symbol=symbol, adjust="")
                time.sleep(pause)
                if df is None or df.empty:
                    error = "empty"
                    break
                closes = {str(d)[:10]: float(c) for d, c in zip(df["date"], df["close"])}
                _cache[ticker] = closes
                return closes, symbol, ""
            except Exception as exc:  # network hiccup, unknown symbol, parse error
                error = f"{type(exc).__name__}: {str(exc)[:80]}"
                if "Connection" not in type(exc).__name__ and "Timeout" not in type(exc).__name__:
                    break  # not a network blip: try the next spelling instead
                time.sleep(2 * 2**attempt)
    _cache[ticker] = None
    return None, ticker, error


def check(row: dict, lookback: int, tol: float, pause: float) -> dict:
    t0 = time.time()
    closes, symbol, error = fetch_closes(row["ticker"], pause)
    row.update(symbol=symbol, error=error, found=closes is not None)
    row["seconds"] = round(time.time() - t0, 2)
    if not closes:
        return row

    c0, c1 = closes.get(row["window_start"]), closes.get(row["window_end"])
    row["window"] = c0 is not None and c1 is not None
    if row["window"]:
        row["r_ak"] = c1 / c0 - 1
        row["abs_err"] = abs(row["r_ak"] - row["r_i"])
        row["r_match"] = row["abs_err"] <= tol

    # Features may only use days strictly before the cutoff date.
    cutoff = date.fromisoformat(row["cutoff_date"])
    start = (cutoff - timedelta(days=lookback)).isoformat()
    pre = sorted(d for d in closes if start <= d < row["cutoff_date"])
    row["pre_days"] = len(pre)
    if len(pre) > 1:
        gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(pre, pre[1:])]
        row["max_gap_days"] = max(gaps)
    row["first_date"] = min(closes)
    return row


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "0/0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="archive .jsonl.gz (default: latest cached)")
    ap.add_argument("-n", type=int, default=100, help="events to test (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lookback", type=int, default=60, help="calendar days of history to check")
    ap.add_argument("--tol", type=float, default=0.005, help="r_i match tolerance (0.005 = 0.5pp)")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    args = ap.parse_args()

    path = args.file or latest_archive()
    rows = sample_events(path, args.n, args.seed)
    print(f"{path.name}: testing {len(rows)} events with akshare {ak.__version__} (Sina)\n")

    for i, row in enumerate(rows, 1):
        check(row, args.lookback, args.tol, args.pause)
        if not row["found"]:
            status = f"NOT FOUND  {row['error']}"
        elif not row.get("window"):
            status = "found, event window missing"
        else:
            status = (f"r_i={row['r_i']:+.4f} ak={row['r_ak']:+.4f} "
                      f"{'OK ' if row['r_match'] else 'BAD'}  pre_days={row['pre_days']}")
        print(f"[{i:>3}/{len(rows)}] {row['ticker']:<6} {row['window_start']}  {status}")

    n = len(rows)
    found = [r for r in rows if r["found"]]
    window = [r for r in found if r.get("window")]
    match = [r for r in window if r.get("r_match")]
    full_hist = [r for r in found if r.get("pre_days", 0) >= 0.6 * args.lookback * 5 / 7]
    print("\n==== AKShare coverage ====")
    print(f"ticker found          {pct(len(found), n)}")
    print(f"event window present  {pct(len(window), n)}")
    print(f"r_i matches (±{args.tol:.1%})  {pct(len(match), len(window))} of those with a window")
    if window:
        print(f"median |r_ak - r_i|   {statistics.median(r['abs_err'] for r in window):.5f}")
    if found:
        print(f"median pre-cutoff trading days in {args.lookback}d: "
              f"{statistics.median(r.get('pre_days', 0) for r in found)}")
        print(f"tickers with >=60% of expected history: {pct(len(full_hist), len(found))}")
    missing = [r["ticker"] for r in rows if not r["found"]]
    bad = [f"{r['ticker']}({r['r_i']:+.3f} vs {r['r_ak']:+.3f})" for r in window if not r["r_match"]]
    if missing:
        print(f"\nnot found: {', '.join(missing)}")
    if bad:
        print(f"r_i mismatch: {', '.join(bad)}")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"akshare_coverage_{path.name.removesuffix('.jsonl.gz')}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = ["ticker", "symbol", "event_datetime", "cutoff_date", "window_start", "window_end",
            "r_i", "r_ak", "abs_err", "r_match", "found", "window", "pre_days", "max_gap_days",
            "first_date", "seconds", "error"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\ndetails -> {out}")


if __name__ == "__main__":
    main()
