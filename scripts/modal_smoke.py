"""Run predict() on a real archived event INSIDE Modal, with the official image.

The portal's "Send test event" only checks the plumbing (TEST events skip
predict() and submit 0.5). This runs the real pipeline in a container built
exactly like modal_app.py's: same packages, same .env secret, the bundle read
from the em-agent-store Volume, LangChain installed on the cold start. Nothing
is submitted to the competition.

    uv run modal run scripts/modal_smoke.py                     # ADSK, cold + warm call
    uv run modal run scripts/modal_smoke.py --ticker NVDA --calls 1

Keep IMAGE in sync with modal_app.py (it is copied, not imported, because the
container only receives predict.py and explaining_markets).
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so add_local_python_source finds predict.py locally

IMAGE = (  # == modal_app.image
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "httpx", "openai", "pydantic")
    .add_local_python_source("explaining_markets", "predict")
)
SECRETS = [modal.Secret.from_dotenv(ROOT)]  # == modal_app.secrets (your .env)

app = modal.App("em-agent-smoke")


@app.function(image=IMAGE, secrets=SECRETS, timeout=600)
def smoke(event: dict, bundle: dict) -> dict:
    import predict

    class _Resp:  # serve the archived materials where information_url would be
        def raise_for_status(self) -> None: ...
        def json(self) -> dict: return bundle

    predict.httpx.get = lambda *a, **k: _Resp()
    t0 = time.time()
    preds, traces = predict.predict_with_trace(event)
    return {"seconds": round(time.time() - t0, 1), "predictions": preds, "trace": traces[0],
            "langchain": dict(predict._LC_STATE)}


def _archive_event(ticker: str) -> dict:
    best = None
    for path in sorted((ROOT / "scripts" / "data" / "archive").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if f'"{ticker}"' in line:
                    rec = json.loads(line)
                    if any(a.get("identifier_value") == ticker for a in rec.get("focal_assets") or []):
                        if best is None or rec["event_datetime"] > best["event_datetime"]:
                            best = rec
    if best is None:
        raise SystemExit(f"{ticker} not found in scripts/data/archive (run scripts/build_store.py first)")
    return best


@app.local_entrypoint()
def main(ticker: str = "ADSK", calls: int = 2) -> None:
    rec = _archive_event(ticker.upper())
    event = {k: rec.get(k) for k in ("event_id", "event_type", "timing_category", "event_datetime")}
    # knowledge_cutoff is deliberately NOT passed: a live payload lacks it, so this
    # also exercises the calendar lookup.
    event.update(id="smoke", information_url="smoke://bundle", prediction_deadline=None,
                 focal_assets=[{"identifier_type": "TICKER", "identifier_value": ticker.upper()}])
    outcome = (rec.get("event_returns") or {}).get(ticker.upper()) or {}
    for i in range(calls):
        out = smoke.remote(event, rec["disclosure"])
        t = out["trace"]
        print(f"\n===== call {i + 1} ({'cold' if i == 0 else 'warm'}): {out['seconds']}s total =====")
        print("prediction:", out["predictions"], "| actual car1:", outcome.get("car1"))
        print("chain:", t.get("chain"), "| langchain:", out["langchain"], "| llm:", t.get("llm_seconds"), "s")
        print("store:", (t.get("context_data") or {}).get("store"), "| cutoff:", t.get("cutoff"))
        print("timings:", t.get("timings"))
        if t.get("context_errors"):
            print("context errors:", json.dumps(t["context_errors"], indent=2))
        if t.get("fallback"):
            print("FALLBACK:", t["fallback"])
        print("reason:", t.get("reason"))
        print("--- context ---\n" + (t.get("context") or ""))
