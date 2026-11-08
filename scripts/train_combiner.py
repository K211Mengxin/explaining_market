"""Train the combiner: questionnaire answers + context numbers -> one percentile.

The deployed agent (predict.py, sections 12-14) asks DeepSeek a FIXED questionnaire
about the event's text and computes the numeric context, then adds them up with the
weights this script learns from already-scored events in the competition archive.

Usage (repo root, on your own machine; needs EM_API_KEY, DEEPSEEK_API_KEY and a built
store, see scripts/build_store.py):

    uv run python scripts/train_combiner.py --list                       # quarters in the archive
    uv run python scripts/train_combiner.py --quarters 2026Q3 --limit 200   # a cheap first look
    uv run python scripts/train_combiner.py                              # every quarter, every event
    uv run python scripts/train_combiner.py --fit-only                   # refit from the cache only
    uv run modal volume put em-agent-store scripts/data/bundle /bundle --force   # ship it

Steps
  1. archive   download the chosen quarter files (cached; a sealed quarter never changes)
  2. targets   y = percentile of car1 and s = percentile of the surprise, ranked within
               each quarter over ALL of its outcomes, exactly like the scorer
  3. extract   per sampled event: the questionnaire (DeepSeek, cached per questionnaire
               version / model / samples), the context (build_context, point-in-time,
               cached) and the option statistics of its bundle
  4. fit       standardise -> (optionally) multiply by the magnitude factor
               (m / m_ref) ** p, so the weights are fitted jointly with the scaling the
               live code applies -> partial the surprise out within each quarter ->
               ridge. The penalty, the exponent p (0 = no scaling) and joint fitting vs
               stretching afterwards are chosen together by cross-validation (leave one quarter out when there are several quarters),
               scored with the official delta R^2. Held-out predictions go through
               predict.Combiner, the live code path.
  5. save      scripts/data/model/combiner.json and a copy in scripts/data/bundle/

Why partial the surprise out: the competition fits y = a + b1*prediction + b2*s and
scores the R^2 the prediction adds over s alone. By Frisch-Waugh-Lovell, regressing
y on the features with s partialled out estimates exactly the weights that maximise
that gain, and the part of a prediction that only repeats s earns nothing anyway.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import predict as P  # noqa: E402
from predict import (  # noqa: E402
    FEATURE_NAMES, MAGNITUDE_KEYS, QUESTIONNAIRE_VERSION, AgentSettings, Combiner, feature_row, magnitude_inputs,
    option_stats_values,
)

DATA = HERE / "data"
CACHE_DIR = DATA / "combiner"
MODEL_PATH = DATA / "model" / "combiner.json"
BUNDLE_COPY = DATA / "bundle" / "combiner.json"
BASELINES = {
    "openai/ea-explain-contemp-summary": "gpt5nano_baseline",
    "gemini/ea-explain-contemp-summary": "gemini_baseline",
}
DEFAULT_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
# Exponents of the magnitude scaling tried in CV; 0 = no scaling. 0.5 is the official
# option-statistics notebook's choice.
DEFAULT_POWERS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_ROWS_FOR_CV = 40


# ---------------------------------------------------------- official scoring ---
# Ports of explaining-markets/examples scoring.py (percentile_ranks, ols_fit, ols_fit2).

DEGENERATE_SD = 1e-6


def percentile_ranks(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
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


def _r2_surprise(s: np.ndarray, y: np.ndarray) -> float | None:
    s_c, y_c = s - s.mean(), y - y.mean()
    sxx, syy = float(s_c @ s_c), float(y_c @ y_c)
    if sxx <= len(s) * DEGENERATE_SD ** 2:
        return None
    beta = float(s_c @ y_c) / sxx
    ss_res = float(((y_c - beta * s_c) ** 2).sum())
    return 0.0 if syy == 0 else 1 - ss_res / syy


def delta_r2(pred, s, y) -> float | None:
    """R^2(y ~ pred + s) - R^2(y ~ s): the competition's ranking quantity."""
    p, s, y = (np.asarray(v, dtype=float) for v in (pred, s, y))
    n = len(y)
    if n < 3:
        return None
    base = _r2_surprise(s, y)
    if base is None:
        return None
    p_c, s_c, y_c = p - p.mean(), s - s.mean(), y - y.mean()
    s11, s22, s12 = float(p_c @ p_c), float(s_c @ s_c), float(p_c @ s_c)
    s1y, s2y, syy = float(p_c @ y_c), float(s_c @ y_c), float(y_c @ y_c)
    if s11 <= n * DEGENERATE_SD ** 2:
        return 0.0  # constant predictions explain nothing
    det = s11 * s22 - s12 * s12
    if det == 0:
        return None
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    ss_res = float(((y_c - b1 * p_c - b2 * s_c) ** 2).sum())
    full = 0.0 if syy == 0 else 1 - ss_res / syy
    return full - base


# ------------------------------------------------------------------- targets ---


def quarter_of(path: Path) -> str:
    m = re.search(r"(\d{4}Q[1-4])", path.name)
    return m.group(1) if m else path.stem


def outcome_rows(records: list[dict], quarter: str) -> list[dict]:
    """Every (event, asset) with a car1, with y and s ranked over the whole quarter
    (official scorer); only rows with a disclosure and a surprise are trainable."""
    rows = []
    for rec in records:
        es = (rec.get("metrics") or {}).get("earnings_surprise") or {}
        surprise = es.get("surprise") if es.get("surprise_status") == "ok" else None
        for asset in rec.get("focal_assets") or []:
            t = asset.get("identifier_value")
            leg = (rec.get("event_returns") or {}).get(t) or {}
            if leg.get("car1") is None:
                continue
            row = {"quarter": quarter, "event_id": rec["event_id"], "ticker": t, "car1": float(leg["car1"]),
                   "surprise": None if surprise is None else float(surprise), "_record": rec}
            for key, col in BASELINES.items():
                preds = (rec.get("baseline_predictions") or {}).get(key) or {}
                if t in preds:
                    row[col] = float(preds[t])
            rows.append(row)
    for row, y in zip(rows, percentile_ranks([r["car1"] for r in rows])):
        row["y"] = y
    have = [r for r in rows if r["surprise"] is not None]
    for row, s in zip(have, percentile_ranks([r["surprise"] for r in have])):
        row["s"] = s
    return rows


def trainable(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("s") is not None and r["_record"].get("disclosure")]


# -------------------------------------------------------------------- caches ---


class JsonlCache:
    """Append-only {key: value} cache, safe across threads and resumable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        rec = {"key": key, **value}
        with self._lock:
            self.data[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def questionnaire_cache(settings: AgentSettings) -> JsonlCache:
    name = f"questionnaire_{QUESTIONNAIRE_VERSION}_{_slug(settings.deepseek_model)}_n{settings.questionnaire_samples}.jsonl"
    return JsonlCache(CACHE_DIR / name)


# Bump when build_context starts putting new keys in ctx.data that feature_row reads
# (v2: last_close and rev_growth_vol for the company-scaled surprises).
CONTEXT_CACHE_VERSION = "v2"


def context_cache() -> JsonlCache:
    return JsonlCache(CACHE_DIR / f"context_{CONTEXT_CACHE_VERSION}.jsonl")


# ------------------------------------------------------------------- extract ---


def _event(rec: dict, ticker: str) -> dict:
    return {"event_id": rec["event_id"], "event_type": rec.get("event_type", "EARNINGS_RELEASE"),
            "event_datetime": rec.get("event_datetime"), "knowledge_cutoff": rec.get("knowledge_cutoff"),
            "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}]}


def extract_one(row: dict, settings: AgentSettings, qcache: JsonlCache | None, ccache: JsonlCache,
                *, ask: bool, refresh_context: bool, questionnaire_fn=None) -> dict:
    """Questionnaire answers + context numbers for one archive row (cached)."""
    rec, ticker = row["_record"], row["ticker"]
    key = f"{rec['event_id']}|{ticker}"
    out: dict = {"key": key}

    c = None if refresh_context else ccache.get(key)
    if c is None and ask:
        data = P.get_data(settings)
        ctx = P.build_context(_event(rec, ticker), ticker, settings, data)
        c = {"ctx_data": ctx.data, "ctx_errors": ctx.errors}
        ccache.put(key, c)
    out["ctx_data"] = (c or {}).get("ctx_data") or {}

    if qcache is not None:
        q = qcache.get(key)
        if q is None and ask:
            materials = P._materials_text(rec["disclosure"])
            name = (P.get_data(settings).sec.profiles().get(ticker.upper()) or {}).get("name")
            fn = questionnaire_fn or P.run_questionnaire
            try:
                q = {"answers": fn(materials, ticker, settings, company_name=name), "error": None}
            except Exception as exc:
                q = {"answers": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
            qcache.put(key, q)
        out["answers"] = (q or {}).get("answers")
        out["q_error"] = (q or {}).get("error")
    return out


def extract_all(rows: list[dict], settings: AgentSettings, *, workers: int, ask: bool, use_questionnaire: bool,
                refresh_context: bool, questionnaire_fn=None) -> list[dict]:
    qcache = questionnaire_cache(settings) if use_questionnaire else None
    ccache = context_cache()
    feats: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(extract_one, r, settings, qcache, ccache, ask=ask, refresh_context=refresh_context,
                            questionnaire_fn=questionnaire_fn): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                feats[f"{r['event_id']}|{r['ticker']}"] = fut.result()
            except Exception as exc:
                print(f"  [skip] {r['ticker']} {r['event_id']}: {type(exc).__name__}: {exc}")
            if i % 25 == 0 or i == len(rows):
                print(f"  extracted {i}/{len(rows)} ({time.time() - t0:.0f}s)")
    out = []
    for r in rows:
        f = feats.get(f"{r['event_id']}|{r['ticker']}")
        if f is None:
            continue
        options = option_stats_values(r["_record"].get("disclosure"))
        out.append({**{k: v for k, v in r.items() if k != "_record"},
                    "features": feature_row(f.get("answers"), f.get("ctx_data"), options),
                    "mags": magnitude_inputs(f.get("ctx_data"), options),
                    "q_error": f.get("q_error")})
    return out


# ----------------------------------------------------------------------- fit ---


def _matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[np.nan if r["features"].get(n) is None else float(r["features"][n]) for n in names]
                     for r in rows], dtype=float).reshape(len(rows), len(names))


def choose_features(rows: list[dict], min_coverage: float) -> tuple[list[str], dict[str, float]]:
    X = _matrix(rows, list(FEATURE_NAMES))
    coverage = {n: float(np.mean(~np.isnan(X[:, j]))) for j, n in enumerate(FEATURE_NAMES)}
    keep = []
    for j, n in enumerate(FEATURE_NAMES):
        col = X[:, j][~np.isnan(X[:, j])]
        if coverage[n] >= min_coverage and len(col) >= 3 and float(np.std(col)) > 1e-9:
            keep.append(n)
    return keep, coverage


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing columns in a CV fold
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where(np.isfinite(stds) & (stds > 1e-12), stds, 1.0)
    Z = (X - means) / stds
    return np.nan_to_num(Z, nan=0.0), means, stds  # a missing value sits at the mean


def _partial_out(Z: np.ndarray, y: np.ndarray, s: np.ndarray, quarters: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Residualise y and every feature on [1, s] within each quarter (the scorer's period)."""
    Zr, yr = Z.copy(), y.copy()
    groups: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(quarters):
        groups[q].append(i)
    for idx in groups.values():
        idx = np.array(idx)
        A = np.column_stack([np.ones(len(idx)), s[idx]]) if len(idx) >= 3 else np.ones((len(idx), 1))
        coef, *_ = np.linalg.lstsq(A, np.column_stack([y[idx], Z[idx]]), rcond=None)
        resid = np.column_stack([y[idx], Z[idx]]) - A @ coef
        yr[idx], Zr[idx] = resid[:, 0], resid[:, 1:]
    return Zr, yr


def magnitude_config(rows: list[dict], power: float = 0.5, min_rows: int = 20) -> dict | None:
    """Pick the base magnitude measure and the ratios that put the others on its scale."""
    vals = {k: np.array([np.nan if (v := r["mags"].get(k)) is None or v <= 0 else v for r in rows], dtype=float)
            for k in MAGNITUDE_KEYS}
    base = next((k for k in MAGNITUDE_KEYS if np.sum(~np.isnan(vals[k])) >= min_rows), None)
    if base is None:
        return None
    ratios = {}
    for k in MAGNITUDE_KEYS:
        if k == base:
            continue
        both = ~np.isnan(vals[base]) & ~np.isnan(vals[k])
        if np.sum(both) >= 10:
            ratios[k] = float(np.median(vals[base][both] / vals[k][both]))
    cfg = {"base": base, "ratios": ratios, "ref": 1.0, "power": power}
    probe = Combiner((), (), (), (), magnitude=cfg)
    ms = [m for r in rows if (m := probe.magnitude_of(r["mags"])) is not None]
    if len(ms) < min_rows:
        return None
    cfg["ref"] = float(np.mean(ms))
    return cfg


def magnitude_factors(rows: list[dict], cfg: dict | None) -> np.ndarray:
    """(m / m_ref) ** power per row, exactly as Combiner.predict applies it; 1 when m is unknown."""
    if not cfg:
        return np.ones(len(rows))
    probe = Combiner((), (), (), (), magnitude=cfg)
    power, ref = float(cfg.get("power", 0.5)), float(cfg["ref"])
    return np.array([1.0 if (m := probe.magnitude_of(r["mags"])) is None else (m / ref) ** power for r in rows])


def fit(rows: list[dict], names: list[str], lam: float, *, power: float = 0.0, joint: bool = True,
        scale_sd: float = 0.10, meta: dict | None = None) -> Combiner:
    """Ridge weights for the live formula  0.5 + scale * g * (z . w),  g = (m / m_ref) ** power.

    joint=True  fits the weights on the features multiplied by g, the quantity the live
                code computes: the best weights given the scaling that will be applied.
    joint=False fits them without g and only stretches afterwards (the official
                notebook's recipe). Joint fitting also gives high-magnitude events more
                pull on the weights, which helps or hurts depending on the noise, so
                cross-validation chooses. power = 0 means no scaling (g = 1).
    """
    X = _matrix(rows, names)
    Z, means, stds = _standardise(X)
    cfg = magnitude_config(rows, power) if power > 0 else None
    g = magnitude_factors(rows, cfg)
    y = np.array([r["y"] for r in rows], dtype=float)
    s = np.array([r["s"] for r in rows], dtype=float)
    Gr, yr = _partial_out(Z * g[:, None] if joint else Z, y, s, [r["quarter"] for r in rows])
    n, k = Gr.shape
    beta = np.linalg.solve(Gr.T @ Gr + lam * n * np.eye(k), Gr.T @ yr) if k else np.zeros(0)
    z_sd = float(np.std(g * (Z @ beta))) if k else 0.0
    scale = scale_sd / z_sd if z_sd > 1e-12 else 1.0
    return Combiner(tuple(names), tuple(map(float, means)), tuple(map(float, stds)), tuple(map(float, beta)),
                    scale=scale, center=0.5, magnitude=cfg,
                    questionnaire_version=QUESTIONNAIRE_VERSION, meta=meta or {})


def predict_rows(c: Combiner, rows: list[dict]) -> np.ndarray:
    return np.array([c.predict(r["features"], r["mags"])[0] for r in rows], dtype=float)


def score(pred: np.ndarray, rows: list[dict]) -> float | None:
    return delta_r2(pred, [r["s"] for r in rows], [r["y"] for r in rows])


def folds_for(rows: list[dict], seed: int, k_random: int = 5) -> list[tuple[str, list[int], list[int]]]:
    quarters = sorted({r["quarter"] for r in rows})
    if len(quarters) >= 2:
        return [(q, [i for i, r in enumerate(rows) if r["quarter"] != q],
                 [i for i, r in enumerate(rows) if r["quarter"] == q]) for q in quarters]
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    parts = [idx[i::k_random] for i in range(k_random)]
    return [(f"fold{i + 1}", sorted(j for p in parts[:i] + parts[i + 1:] for j in p), sorted(part))
            for i, part in enumerate(parts)]


def _cv_label(lam: float, power: float, joint: bool) -> str:
    return f"{lam}|p={power}|{'none' if not power else 'joint' if joint else 'after'}"


def cross_validate(rows: list[dict], names: list[str], lambdas, seed: int, powers=DEFAULT_POWERS) -> dict:
    """Held-out delta R^2 per fold for every (lambda, exponent, joint-or-after) setting."""
    folds = folds_for(rows, seed)
    table: dict[tuple[float, float, bool], list[float]] = defaultdict(list)
    for label, train_i, test_i in folds:
        train, test = [rows[i] for i in train_i], [rows[i] for i in test_i]
        for power in powers:
            if power > 0 and magnitude_config(train, power) is None:
                continue  # too few rows with any magnitude measure
            for joint in ((True,) if not power else (True, False)):
                for lam in lambdas:
                    d = score(predict_rows(fit(train, names, lam, power=power, joint=joint), test), test)
                    if d is not None:
                        table[(lam, power, joint)].append(d)
    n_folds = len(folds)
    full = {key: v for key, v in table.items() if len(v) == n_folds}  # compare on the same folds only
    summary = {_cv_label(*key): {"mean": float(np.mean(v)), "folds": [round(x, 5) for x in v]}
               for key, v in sorted(full.items(), key=lambda kv: (kv[0][1], not kv[0][2], kv[0][0]))}
    base = {"folds": [f[0] for f in folds], "lambda": 1.0, "power": 0.0, "joint": True, "table": summary}
    if not full:
        return base
    # ties go to the larger (safer) penalty, the smaller (simpler) exponent, then the joint fit
    best = max(full, key=lambda k: (round(float(np.mean(full[k])), 6), k[0], -k[1], k[2]))
    return base | {"lambda": best[0], "power": best[1], "joint": best[2]}


# ---------------------------------------------------------------------- main ---


def load_quarters(args) -> list[tuple[str, list[dict]]]:
    from backtest import download, list_quarters, read_records  # noqa: E402  (needs EM_API_KEY)

    if args.files:
        return [(quarter_of(p), read_records(p)) for p in args.files]
    files = [f for f in list_quarters() if f["event_type"] == args.event_type]
    wanted = set(args.quarters.split(",")) if args.quarters else {f["quarter"] for f in files}
    return [(f["quarter"], read_records(download(args.event_type, f["quarter"])))
            for f in files if f["quarter"] in wanted]


def report_baselines(rows: list[dict]) -> None:
    for col in ("q_overall_single", *BASELINES.values()):
        if col == "q_overall_single":
            use = [r for r in rows if r["features"].get("q_overall") is not None]
            pred = [r["features"]["q_overall"] for r in use]
            label = "questionnaire 'overall'"
        else:
            use = [r for r in rows if r.get(col) is not None]
            pred = [r[col] for r in use]
            label = col
        if len(use) >= 10:
            by_q: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(use):
                by_q[r["quarter"]].append(i)
            vals = [d for idx in by_q.values()
                    if (d := delta_r2([pred[i] for i in idx], [use[i]["s"] for i in idx], [use[i]["y"] for i in idx]))
                    is not None]
            if vals:
                print(f"  {label:<28} n={len(use):<5} mean per-quarter dR2={np.mean(vals):+.4f} (in-sample, untrained)")


def main(argv: list[str] | None = None, *, questionnaire_fn=None) -> Combiner | None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list archive quarters and exit")
    ap.add_argument("--quarters", help="comma list, e.g. 2026Q2,2026Q3 (default: all)")
    ap.add_argument("--files", type=Path, nargs="*", help="local archive .jsonl.gz files instead of the API")
    ap.add_argument("--event-type", default="EARNINGS_RELEASE")
    ap.add_argument("--limit", type=int, default=0, help="events sampled per quarter (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lambdas", default=",".join(map(str, DEFAULT_LAMBDAS)))
    ap.add_argument("--powers", default=",".join(map(str, DEFAULT_POWERS)),
                    help="magnitude exponents to try (0 = no scaling)")
    ap.add_argument("--min-coverage", type=float, default=0.05, help="drop features present on fewer rows")
    ap.add_argument("--fit-only", action="store_true", help="never call the LLM or build context; use caches")
    ap.add_argument("--no-questionnaire", action="store_true", help="context features only")
    ap.add_argument("--refresh-context", action="store_true", help="recompute cached context features")
    ap.add_argument("--offline", action="store_true", help="context from the store only (no price/SEC fetches)")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    args = ap.parse_args(argv)

    if args.list:
        from backtest import list_quarters
        for f in list_quarters():
            print(f"{f['event_type']:<18} {f['quarter']}  events={f.get('events')}  sealed={f.get('sealed')}")
        return None

    os.environ.setdefault("AGENT_USE_CALENDAR", "0")  # the archive already holds every past event
    if args.offline:
        os.environ["AGENT_ONLINE"] = "0"
    settings = AgentSettings.from_env()
    use_q = not args.no_questionnaire
    if use_q and not args.fit_only and not settings.deepseek_api_key and questionnaire_fn is None:
        sys.exit("DEEPSEEK_API_KEY is not set (or pass --no-questionnaire / --fit-only)")

    # 1-2. archive and targets
    rows: list[dict] = []
    rng = random.Random(args.seed)
    for quarter, records in load_quarters(args):
        q_rows = trainable(outcome_rows(records, quarter))
        if args.limit and len(q_rows) > args.limit:
            q_rows = rng.sample(q_rows, args.limit)
        print(f"{quarter}: {len(q_rows)} trainable outcomes")
        rows.extend(q_rows)
    if not rows:
        sys.exit("no trainable outcomes (need a car1, a surprise and a disclosure)")

    # 3. extract
    print(f"extracting features for {len(rows)} outcomes "
          f"(questionnaire {QUESTIONNAIRE_VERSION if use_q else 'off'}, model {settings.deepseek_model}, "
          f"samples {settings.questionnaire_samples}) ...")
    data = extract_all(rows, settings, workers=args.workers, ask=not args.fit_only, use_questionnaire=use_q,
                       refresh_context=args.refresh_context, questionnaire_fn=questionnaire_fn)
    failed = sum(1 for r in data if use_q and r["features"].get("q_overall") is None)
    print(f"{len(data)} rows ready; questionnaire missing on {failed}")

    # 4. fit
    names, coverage = choose_features(data, args.min_coverage)
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    powers = [float(x) for x in args.powers.split(",") if x.strip()]
    print("\nreference points:")
    report_baselines(data)
    if len(data) >= MIN_ROWS_FOR_CV:
        cv = cross_validate(data, names, lambdas, args.seed, powers)
        print(f"\ncross-validation over {cv['folds']} (held-out delta R^2, mean over folds; "
              "p = magnitude exponent, 0 = no scaling):")
        for key, v in cv["table"].items():
            lam, p, how = key.split("|")
            print(f"  lambda={lam:<6} {p:<7} {how:<6} {v['mean']:+.4f}   folds {v['folds']}")
        print(f"chosen: lambda={cv['lambda']}, magnitude exponent {cv['power']}, "
              + ("no scaling" if not cv["power"] else "weights fitted jointly" if cv["joint"]
                 else "stretched after fitting"))
    else:
        cv = {"folds": [], "lambda": 1.0, "power": 0.0, "joint": True, "table": {}}
        print(f"\n[!] only {len(data)} rows: skipping cross-validation, lambda=1.0, no magnitude scaling")

    meta = {"trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "quarters": sorted({r["quarter"] for r in data}), "n_rows": len(data),
            "model": settings.deepseek_model, "questionnaire_samples": settings.questionnaire_samples,
            "lambda": cv["lambda"], "cv": cv, "coverage": {n: round(coverage[n], 3) for n in names},
            "dropped": sorted(set(FEATURE_NAMES) - set(names))}
    meta["magnitude_power"], meta["magnitude_joint"] = cv["power"], cv["joint"]
    combiner = fit(data, names, cv["lambda"], power=cv["power"], joint=cv["joint"], meta=meta)

    print("\nweights (per standard deviation; sorted by size):")
    for n, w in sorted(zip(combiner.features, combiner.weights), key=lambda t: -abs(t[1])):
        print(f"  {n:<26} {w:+.4f}   coverage {coverage[n]:.0%}")
    if combiner.magnitude:
        print(f"magnitude: base={combiner.magnitude['base']} ratios={combiner.magnitude['ratios']} "
              f"ref={combiner.magnitude['ref']:.4f} power={combiner.magnitude['power']}")
    print(f"in-sample delta R^2 (optimistic): {score(predict_rows(combiner, data), data):+.4f}")

    # 5. save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combiner.to_dict(), indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")
    if args.out.resolve() == MODEL_PATH.resolve():
        BUNDLE_COPY.parent.mkdir(parents=True, exist_ok=True)
        BUNDLE_COPY.write_text(args.out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {BUNDLE_COPY}; ship it with:\n"
              "  uv run modal volume put em-agent-store scripts/data/bundle /bundle --force")
    return combiner


if __name__ == "__main__":
    main()