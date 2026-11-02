"""Run the full pipeline on one event and print every stage.

Usage (from the repo root):
    uv run python scripts/debug_predict.py                    # built-in sample bundle (ACME)
    uv run python scripts/debug_predict.py --ticker ADSK      # a real event from the archive
    uv run python scripts/debug_predict.py --ticker ADSK --blocks prices,history
    uv run python scripts/debug_predict.py my_bundle.json AAPL

Stages printed: materials -> context (with timings/errors) -> chain reply -> prediction.
Nothing is submitted; the network is used for DeepSeek and for missing data.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import predict as P  # noqa: E402
from predict import AgentSettings  # noqa: E402

SAMPLE_BUNDLE = Path(__file__).with_name("sample_bundle.json")


def find_archive_event(ticker: str, store_dir: Path) -> dict | None:
    """The latest archived event for `ticker` (with its disclosure and outcome)."""
    best = None
    for path in sorted((store_dir / "archive").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if f'"{ticker}"' not in line:
                    continue
                rec = json.loads(line)
                if any(a.get("identifier_value") == ticker for a in rec.get("focal_assets") or []):
                    if best is None or rec.get("event_datetime", "") > best.get("event_datetime", ""):
                        best = rec
    return best


def rule(title: str) -> None:
    print("\n" + "=" * 25 + f" {title} " + "=" * 25)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?", type=Path, help="a disclosure bundle JSON file")
    ap.add_argument("bundle_ticker", nargs="?", default="ACME")
    ap.add_argument("--ticker", help="use this ticker's latest archived event instead")
    ap.add_argument("--blocks", help='override AGENT_BLOCKS, e.g. "none" or "prices,peers"')
    args = ap.parse_args()
    if args.blocks is not None:
        os.environ["AGENT_BLOCKS"] = args.blocks
    settings = AgentSettings.from_env()

    outcome = None
    if args.ticker:
        rec = find_archive_event(args.ticker.upper(), settings.store_dir)
        if rec is None:
            sys.exit(f"no archived event for {args.ticker} under {settings.store_dir / 'archive'}")
        ticker, bundle = args.ticker.upper(), rec["disclosure"]
        event = {k: rec.get(k) for k in ("event_id", "event_type", "timing_category",
                                         "event_datetime", "knowledge_cutoff")}
        outcome = (rec.get("event_returns") or {}).get(ticker)
    else:
        ticker = args.bundle_ticker
        bundle = json.loads((args.bundle or SAMPLE_BUNDLE).read_text())
        event = {"event_id": "local-debug", "event_type": "EARNINGS_RELEASE",
                 "event_datetime": "2026-08-27T21:00:00Z", "knowledge_cutoff": "2026-08-27T20:00:00Z"}
    event.update(id="debug", information_url="local://bundle", prediction_deadline=None,
                 focal_assets=[{"identifier_type": "TICKER", "identifier_value": ticker}])

    class _Resp:
        def raise_for_status(self) -> None: ...
        def json(self) -> dict: return bundle

    P.httpx.get = lambda *a, **k: _Resp()  # serve the bundle where information_url would be

    rule("SETTINGS")
    print(settings.describe(), f"| store={settings.store_dir}")
    rule("MATERIALS")
    print(P.format_materials(bundle) or "(empty: raw JSON will be sent)")

    preds, traces = P.predict_with_trace(event, settings)
    trace = traces[0]
    rule("CONTEXT")
    print(trace.get("context", "(not built: no DEEPSEEK_API_KEY)"))
    if trace.get("timings"):
        print("\ntimings:", trace["timings"])
    if trace.get("context_errors"):
        print("errors:", json.dumps(trace["context_errors"], indent=2))
    rule("CHAIN")
    print(f"reason: {trace.get('reason')!r}  llm={trace.get('llm_seconds')}s"
          + (f"\nFALLBACK: {trace['fallback']}" if trace.get("fallback") else ""))
    rule("PREDICTION")
    print(json.dumps(preds, indent=2))
    if outcome:
        print(f"actual: car1={outcome.get('car1'):+.2%} (r_i={outcome.get('r_i'):+.2%})")


if __name__ == "__main__":
    main()
