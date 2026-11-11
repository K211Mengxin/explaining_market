"""What the questionnaire had to read, and what it answered. Free: no API calls.

    uv run python scripts/inspect_questionnaire.py

1. Per archive quarter already downloaded (scripts/data/archive): how many events carry a
   research note (preview) and option statistics, and how long the facts and the note are.
2. Per questionnaire answer cache (scripts/data/combiner/questionnaire_*.jsonl): failed
   calls, and how often each item came back null, split by whether the event had a preview.
3. A few events whose `overall` came back null, with their facts, to see why.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from predict import FACTS_ID, NUMBER_KEYS, OPTION_STATS_ID, PREVIEW_ID, QUESTION_KEYS  # noqa: E402

ARCHIVE = HERE / "data" / "archive"
CACHE_DIR = HERE / "data" / "combiner"


def _items(rec: dict) -> dict:
    items = (rec.get("disclosure") or {}).get("items") or []
    return {i.get("id"): i for i in items if isinstance(i, dict)}


def main() -> None:
    events: dict[str, dict] = {}   # "event_id|ticker" -> {"quarter", "preview", "facts"}
    print("1. materials per quarter")
    print(f"   {'quarter':<8} {'events':>6} {'preview':>8} {'options':>8} {'facts chars':>12} {'preview chars':>14}  item ids")
    for path in sorted(ARCHIVE.glob("*.jsonl.gz")):
        m = re.search(r"(\d{4}Q[1-4])", path.name)
        quarter = m.group(1) if m else path.stem
        n = prev = opt = 0
        facts_len, prev_len, ids = [], [], defaultdict(int)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                by_id = _items(rec)
                if not by_id:
                    continue
                n += 1
                for k in by_id:
                    ids[k] += 1
                preview = (by_id.get(PREVIEW_ID) or {}).get("content")
                has_prev = isinstance(preview, str) and bool(preview.strip())
                prev += has_prev
                opt += OPTION_STATS_ID in by_id
                if has_prev:
                    prev_len.append(len(preview))
                facts = (by_id.get(FACTS_ID) or {}).get("content")
                facts = facts if isinstance(facts, list) else []
                facts_len.append(len("\n".join(map(str, facts))))
                for asset in rec.get("focal_assets") or []:
                    events[f"{rec['event_id']}|{asset.get('identifier_value')}"] = {
                        "quarter": quarter, "preview": has_prev, "facts": facts}
        if n:
            top = ", ".join(f"{k} {v / n:.0%}" for k, v in sorted(ids.items(), key=lambda kv: -kv[1])[:5])
            print(f"   {quarter:<8} {n:>6} {prev / n:>8.0%} {opt / n:>8.0%} {sum(facts_len) / n:>12.0f} "
                  f"{(sum(prev_len) / len(prev_len)) if prev_len else 0:>14.0f}  {top}")
    if not events:
        print("   no archive files under", ARCHIVE)

    for cache in sorted(CACHE_DIR.glob("questionnaire_*.jsonl")):
        rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        latest = {r["key"]: r for r in rows}
        print(f"\n2. {cache.name}: {len(latest)} events")
        groups: dict[str, list[dict]] = {"with preview": [], "no preview": [], "unknown": []}
        errors: dict[str, int] = defaultdict(int)
        for key, r in latest.items():
            ev = events.get(key)
            label = "unknown" if ev is None else "with preview" if ev["preview"] else "no preview"
            if r.get("answers") is None:
                errors[str(r.get("error"))[:90]] += 1
            else:
                groups[label].append(r["answers"])
        for err, k in errors.items():
            print(f"   FAILED x{k}: {err}")
        cols = [g for g in groups if groups[g]]
        print("   share of answers that are NULL:")
        print(f"   {'item':<26}" + "".join(f"{g + ' (n=' + str(len(groups[g])) + ')':>24}" for g in cols))
        for item in (*QUESTION_KEYS, *NUMBER_KEYS):
            print(f"   {item:<26}" + "".join(
                f"{sum(a.get(item) is None for a in groups[g]) / len(groups[g]):>24.0%}" for g in cols))

        shown = 0
        for key, r in latest.items():
            a, ev = r.get("answers"), events.get(key)
            if a and ev and a.get("overall") is None and shown < 3:
                shown += 1
                print(f"\n3. overall is null: {key} ({ev['quarter']}, preview: {ev['preview']})")
                for fact in ev["facts"][:10]:
                    print("     -", str(fact)[:200])
                print("     answers:", {k: v for k, v in a.items() if v is not None})


if __name__ == "__main__":
    main()
