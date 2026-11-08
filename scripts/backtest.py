"""Backtest predict() on real, already-scored events from the competition archive.

Same flow as the deployed webhook handler, minus the webhook and the submission:

    live:      webhook -> GET information_url -> predict(event) -> POST /predictions
    backtest:  GET /archive -> past event + its disclosure -> predict(event) -> compare
                                                                with the realized return

Every event is fed to the *unchanged* predict.py: the event dict has the same
shape as a webhook payload and `httpx.get(information_url)` returns that event's
real disclosure bundle (facts / preview / option stats). The realized return is
never shown to the model; it is only used afterwards for scoring. Nothing is
submitted.

Usage (from the repo root, on your own machine):
    uv run python scripts/backtest.py --list                 # quarters in the archive
    uv run python scripts/backtest.py                        # latest quarter, 20 events
    uv run python scripts/backtest.py --quarter 2026Q3 -n 100 --workers 8
    uv run python scripts/backtest.py --file some.jsonl.gz   # an archive file on disk

Ablations (same --seed -> same events, so runs are directly comparable):
    uv run python scripts/backtest.py -n 300 --blocks none            # materials only
    uv run python scripts/backtest.py -n 300 --blocks prices,history
    uv run python scripts/backtest.py -n 300 --peer-method sic        # vs knn / hybrid

Needs EM_API_KEY (archive download) and DEEPSEEK_API_KEY (the model) in .env;
the context blocks also want SEC_USER_AGENT and a built store
(scripts/build_store.py). Full per-event traces, including the context text the
model saw, are written next to the CSV as *_traces.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import predict as P  # noqa: E402  (after .env so the key is visible)
from predict import AgentSettings  # noqa: E402

API = os.environ.get("EM_API_BASE_URL", "https://api.explainingmarkets.ai/v1").rstrip("/")
CACHE = Path(__file__).with_name("data") / "archive"
OUT_DIR = Path(__file__).with_name("results")
BASELINES = {  # reference LLM baselines carried in archive lines (when present)
    "openai/ea-explain-contemp-summary": "gpt5nano_baseline",
    "gemini/ea-explain-contemp-summary": "gemini_baseline",
}


# ---------------------------------------------------------------- archive ---


def _api_get(path: str) -> dict:
    key = os.environ.get("EM_API_KEY")
    if not key:
        sys.exit("EM_API_KEY is not set in .env")
    r = httpx.get(f"{API}{path}", headers={"X-API-Key": key}, timeout=30)
    r.raise_for_status()
    return r.json()


def list_quarters() -> list[dict]:
    files = _api_get("/archive").get("files", [])
    return sorted(files, key=lambda f: (f["event_type"], f["quarter"]))


def download(event_type: str, quarter: str) -> Path:
    """Fetch one quarter file (cached; a sealed quarter never changes)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{event_type}_{quarter}.jsonl.gz"
    meta = _api_get(f"/archive/{event_type}/{quarter}")
    if target.exists() and meta.get("bytes") and target.stat().st_size == meta["bytes"]:
        print(f"cached   {target.name}")
        return target
    print(f"download {target.name} ({meta.get('bytes') or 0:,} bytes, sealed={meta.get('sealed')})")
    with httpx.stream("GET", meta["url"], follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        tmp = target.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 16):
                fh.write(chunk)
    tmp.replace(target)
    return target


def read_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- scoring ---
# Same transform as the competition scorer (explaining-markets/examples,
# scoring.py): the target y is the percentile rank of car1 across the quarter,
# ties share the average rank.


def percentile_ranks(values: list[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 / (n - 1)
        i = j + 1
    return ranks


def outcome_rows(records: list[dict]) -> list[dict]:
    """One row per scorable (event, asset), with y and surprise ranked over the quarter."""
    rows = []
    for rec in records:
        es = (rec.get("metrics") or {}).get("earnings_surprise") or {}
        surprise = es.get("surprise") if es.get("surprise_status") == "ok" else None
        for asset in rec.get("focal_assets") or []:
            t = asset.get("identifier_value")
            leg = (rec.get("event_returns") or {}).get(t) or {}
            if leg.get("car1") is None or not rec.get("disclosure"):
                continue
            row = {
                "event_id": rec["event_id"],
                "ticker": t,
                "event_datetime": rec.get("event_datetime"),
                "car1": float(leg["car1"]),
                "surprise": surprise,
                "_record": rec,
            }
            for key, col in BASELINES.items():
                preds = (rec.get("baseline_predictions") or {}).get(key) or {}
                if t in preds:
                    row[col] = float(preds[t])
            rows.append(row)
    for row, y in zip(rows, percentile_ranks([r["car1"] for r in rows])):
        row["y"] = y
    have = [r for r in rows if r["surprise"] is not None]
    for row, s in zip(have, percentile_ranks([r["surprise"] for r in have])):
        row["surprise_pct"] = s
    return rows


def ols_r2(y: list[float], xs: list[list[float]]) -> float:
    """R^2 of y on an intercept plus the given regressors (normal equations)."""
    n, k = len(y), len(xs) + 1
    X = [[1.0] + [x[i] for x in xs] for i in range(n)]
    A = [[sum(X[r][a] * X[r][b] for r in range(n)) for b in range(k)] for a in range(k)]
    v = [sum(X[r][a] * y[r] for r in range(n)) for a in range(k)]
    for c in range(k):  # Gaussian elimination with partial pivoting
        p = max(range(c, k), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-12:
            return float("nan")  # a regressor with no spread
        A[c], A[p], v[c], v[p] = A[p], A[c], v[p], v[c]
        for r in range(k):
            if r != c:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
                v[r] -= f * v[c]
    beta = [v[i] / A[i][i] for i in range(k)]
    ybar = sum(y) / n
    ss_tot = sum((t - ybar) ** 2 for t in y)
    ss_res = sum((y[i] - sum(b * x for b, x in zip(beta, X[i]))) ** 2 for i in range(n))
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sab / (sa * sb) if sa and sb else float("nan")


def report(rows: list[dict], col: str, label: str) -> None:
    use = [r for r in rows if r.get(col) is not None and r.get("surprise_pct") is not None]
    if len(use) < 5:
        print(f"{label:<22} too few rows with a surprise ({len(use)}) to fit")
        return
    y = [r["y"] for r in use]
    p = [r[col] for r in use]
    s = [r["surprise_pct"] for r in use]
    r2_s, r2_f = ols_r2(y, [s]), ols_r2(y, [p, s])
    print(f"{label:<22} n={len(use):<4} corr(pred,y)={corr(p, y):+.3f}  "
          f"R2 surprise={r2_s:.4f}  R2 full={r2_f:.4f}  dR2={r2_f - r2_s:+.4f}")


# ---------------------------------------------------------------- predict ---


def run_one(row: dict) -> dict:
    """Push one past event through predict() exactly as the webhook path would."""
    rec = row["_record"]
    event = {  # same shape as the webhook payload (WebhookPayload)
        "id": f"backtest_{rec['event_id']}",
        "event_id": rec["event_id"],
        "event_type": rec.get("event_type", "EARNINGS_RELEASE"),
        "timing_category": rec.get("timing_category"),
        "event_datetime": rec.get("event_datetime"),
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": row["ticker"]}],
        "information_url": f"archive://{rec['event_id']}",
        "prediction_deadline": None,
        # Not in a live payload (predict.py looks it up); passing it here keeps the
        # backtest from depending on the calendar API.
        "knowledge_cutoff": rec.get("knowledge_cutoff"),
    }
    t0 = time.time()
    try:
        preds, traces = P.predict_with_trace(event)
        row["pred"] = float(preds[0]["predicted_percentile"])
        trace = traces[0]
        row["trace"] = trace
        row["fallback"] = trace.get("fallback")
        row["reason"] = trace.get("reason")
        row["mode"] = trace.get("mode")
        row["questionnaire_error"] = trace.get("questionnaire_error") or trace.get("combiner_error")
        row["q_overall"] = (trace.get("questionnaire") or {}).get("overall")
        row["context_errors"] = "; ".join(f"{k}: {v}" for k, v in (trace.get("context_errors") or {}).items())
        row["context_seconds"] = (trace.get("timings") or {}).get("total")
        for k in CONTEXT_COLS:
            row[k] = (trace.get("context_data") or {}).get(k)
    except Exception as exc:  # live, this would be a missed event
        row["pred"], row["error"] = None, f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.time() - t0, 1)
    return row


CONTEXT_COLS = ["peers", "peers_reported", "peer_mean_car", "sector_mean_car", "own_mean_abs_car",
                "peer_mean_abs_car", "abn_ret_20d", "vol_60d", "rev_yoy", "op_margin", "rel_rev_yoy"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list archive quarters and exit")
    ap.add_argument("--quarter", help="e.g. 2026Q3 (default: latest in the archive)")
    ap.add_argument("--event-type", default="EARNINGS_RELEASE")
    ap.add_argument("--file", type=Path, help="use a local .jsonl.gz instead of downloading")
    ap.add_argument("-n", "--limit", type=int, default=20, help="events to predict (0 = all)")
    ap.add_argument("--workers", type=int, default=4, help="parallel model calls")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random event sample")
    ap.add_argument("--blocks", help='context blocks, e.g. "all", "none", "prices,history"')
    ap.add_argument("--peer-method", choices=["sic", "knn", "hybrid"])
    ap.add_argument("--calendar", action="store_true",
                    help="also query GET /events (off by default: the archive already has every past event)")
    ap.add_argument("--offline", action="store_true", help="use only the store, never fetch prices/SEC")
    args = ap.parse_args()

    # The agent reads its configuration from the environment on every call.
    if args.blocks is not None:
        os.environ["AGENT_BLOCKS"] = args.blocks
    if args.peer_method:
        os.environ["AGENT_PEER_METHOD"] = args.peer_method
    os.environ["AGENT_USE_CALENDAR"] = "1" if args.calendar else "0"
    if args.offline:
        os.environ["AGENT_ONLINE"] = "0"

    if args.list:
        for f in list_quarters():
            print(f"{f['event_type']:<18} {f['quarter']}  events={f.get('events')}  "
                  f"sealed={f.get('sealed')}  {f.get('event_datetime_min', '')[:10]}"
                  f" -> {f.get('event_datetime_max', '')[:10]}")
        return

    if args.file:
        path = args.file
    else:
        quarter = args.quarter
        if not quarter:
            files = [f for f in list_quarters() if f["event_type"] == args.event_type]
            quarter = files[-1]["quarter"]
        path = download(args.event_type, quarter)

    rows = outcome_rows(read_records(path))
    print(f"{path.name}: {len(rows)} scored (event, asset) outcomes")
    sample = rows if args.limit <= 0 or args.limit >= len(rows) else \
        random.Random(args.seed).sample(rows, args.limit)

    # Serve each event's real disclosure where predict() expects information_url.
    bundles = {f"archive://{r['event_id']}": r["_record"]["disclosure"] for r in sample}

    class _Resp:
        def __init__(self, url: str) -> None: self._b = bundles[url]
        def raise_for_status(self) -> None: ...
        def json(self) -> dict: return self._b

    P.httpx.get = lambda url, *a, **k: _Resp(url)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[!] DEEPSEEK_API_KEY not set: every prediction will be the 0.5 baseline")
    settings = AgentSettings.from_env()
    combiner = P.load_combiner(settings.combiner_file) if settings.use_combiner else None
    if combiner is not None:
        trained_on = combiner.meta.get("quarters") or []
        print(f"mode: questionnaire + combiner ({settings.combiner_file}, trained on {trained_on})")
        if path.name and any(q in path.name for q in trained_on):
            print("[!] this quarter was in the combiner's training data: the score below is IN-SAMPLE "
                  "(optimistic). Use the cross-validated numbers from train_combiner.py instead, "
                  "or set AGENT_USE_COMBINER=0 to backtest the single-score chain.")
    else:
        print("mode: single-score chain (no combiner file, or AGENT_USE_COMBINER=0)")
    print(f"predicting {len(sample)} events: {settings.describe()}, {args.workers} workers ...\n")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(run_one, r) for r in sample]):
            r = fut.result()
            done += 1
            shown = "ERROR" if r["pred"] is None else f"{r['pred']:.3f}"
            print(f"[{done:>3}/{len(sample)}] {r['ticker']:<6} {str(r['event_datetime'])[:10]}  "
                  f"pred={shown:<6} actual y={r['y']:.3f}  car1={r['car1']:+.2%}  {r['seconds']}s"
                  + (f"  peers reported={r['peers_reported']}" if r.get("peers_reported") is not None else "")
                  + (f"  FALLBACK {r['fallback']}" if r.get("fallback") else "")
                  + (f"  {r['error']}" if r.get("error") else ""))

    ok = [r for r in sample if r["pred"] is not None]
    print(f"\n==== results ({len(ok)}/{len(sample)} predicted; "
          f"y is ranked across all {len(rows)} outcomes of the quarter) ====")
    report(ok, "pred", "DeepSeek (predict.py)")
    for col in BASELINES.values():
        if any(col in r for r in ok):
            report(ok, col, col)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"backtest_{path.name.removesuffix('.jsonl.gz')}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = ["event_id", "ticker", "event_datetime", "pred", "y", "car1", "surprise",
            "surprise_pct", *BASELINES.values(), *CONTEXT_COLS, "mode", "q_overall", "reason", "fallback",
            "questionnaire_error", "context_errors", "context_seconds", "seconds", "error"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(sample, key=lambda r: str(r["event_datetime"])))
    traces = out.with_name(out.stem + "_traces.jsonl")
    with traces.open("w") as fh:
        for r in sample:
            if r.get("trace"):
                fh.write(json.dumps({"event_id": r["event_id"], "y": r["y"], "car1": r["car1"],
                                     "pred": r["pred"], **r["trace"]}, default=str) + "\n")
    print(f"\nper-event results -> {out.relative_to(ROOT)}\ntraces -> {traces.relative_to(ROOT)}")


if __name__ == "__main__":
    main()