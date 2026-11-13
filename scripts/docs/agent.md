# The prediction agent (all in predict.py)

Only `predict.py` differs from the official starter. `modal_app.py`, `pyproject.toml`,
`uv.lock` and the tests are untouched; everything else lives in `scripts/`.

## Two modes

| Mode | When | What produces the number |
|---|---|---|
| **questionnaire + combiner** (default) | a trained `combiner.json` is present | DeepSeek fills in a fixed questionnaire about the text; the combiner weighs the answers and the numeric context with weights learned from the archive |
| single-score chain | no combiner file, or `AGENT_USE_COMBINER=0` | DeepSeek reads the text and the context and returns one percentile |

Why the combiner: the competition scores the R² a prediction adds over the earnings surprise
(`y = a + b1·prediction + b2·surprise`). A single LLM score mostly repeats the surprise and
weighs the context numbers by guesswork, differently on every event. The combiner splits
the work: the LLM only reads text (what it is good at), and the weights come from a ridge
regression of the realised percentile on the features **with the surprise partialled out
per quarter**, which is exactly the quantity the scorer rewards (Frisch–Waugh–Lovell).

## predict.py sections

| # | Section | What it does |
|---|---|---|
| 0 | materials | the event's facts, preview and option stats as prompt text (official code) |
| 1 | settings | configuration from the environment / `.env` |
| 2 | store | on-disk cache; deployed, filled from the Modal Volume `em-agent-store` via the Modal API |
| 3 | em_api | competition archive + events calendar |
| 4 | market_data | daily prices (store built with AKShare; fresh days from Yahoo), pre-event features, event CARs |
| 5 | sec_data | SEC EDGAR SIC industry and XBRL statements |
| 6 | fundamentals | point-in-time ratios: only filings with `filed` < cutoff |
| 7 | event_history | every past event and its realized reaction |
| 8 | peers | `sic` / `knn` / `hybrid` peer groups |
| 9 | prompts | system + user prompt (single-score chain) |
| 10 | chain | LangChain LCEL `prompt \| ChatOpenAI(DeepSeek, JSON) \| parser`; installed on cold start, openai-SDK twin as fallback |
| 11 | features | the six context blocks, each optional and time-bounded |
| 12 | questionnaire | fixed items on anchored scales; materials anonymised (no ticker, no company name, no date) |
| 13 | combiner | `feature_row()` (shared with training) → standardise → weights → magnitude scaling → clip |
| 14 | pipeline | `predict()`: materials → combiner (or chain) → 0.5 fallback |

## The questionnaire (`QUESTIONNAIRE_VERSION = "q4"`, facts only)

Thinking: DeepSeek's thinking mode is ON by default at high effort, and the reasoning is
billed as output tokens. The questionnaire turns it off (`AGENT_Q_THINKING=off`, the
default; `low` / `high` / `max` to compare). Train and deploy with the same value: the
answer cache is kept per value, and `combiner.json` records it in `meta`. The training
script prints the tokens per call (input, cached, output, reasoning) after extraction.

What it reads: ONLY the earnings-call facts. The research note (preview) exists only from
2026Q3 on and the option statistics only in 2026Q3, so a model trained on earlier quarters
must not read them live either, or training and live inputs differ; the option statistics
reach the combiner as numbers anyway. A SECOND questionnaire (`p1`, "model B") runs only for
events that carry a note: it reads the note AND the facts and grades the release against
the note's own expectations (revenue / guidance / the note's key metrics vs the note, which
scenario hit, what the note missed, a mandatory `overall_vs_note`), and copies the note's
consensus numbers, which pair with q4's actuals into revenue and EPS surprises on the
company's own scale. Its features are simply missing on events without a note (plus a
`has_preview` flag), so one combiner serves both; `combiner.json` records the preview
version and is rejected on a mismatch. The prompt puts the fixed questionnaire FIRST and the materials LAST, so every call
shares one prefix for DeepSeek's context cache. Answers are cached per (version, model,
samples, thinking), and `--limit` samples are nested, so a run can be extended without
paying twice.

Part A, scored items:

Every item is answerable from the facts alone — the EPS surprise itself is NOT asked for in
detail, because the scorer already controls for it and repeating it earns nothing:

| Item | Scale | Asks |
|---|---|---|
| `guidance_change` | −2..2 | guidance vs the company's OWN previous guidance (null only if none mentioned) |
| `growth_trend` | −2..2 | revenue growth accelerating or decelerating vs the company's own recent quarters |
| `earnings_quality` | −1..1 | operating improvement vs one-offs (tax, asset sales, milestones, buybacks) |
| `margin_trend` | −1..1 | margins vs a year earlier |
| `demand_outlook` | −2..2 | forward commentary on orders, backlog, pipeline |
| `vs_stated_expectation` | −2..2 | results vs an expectation the materials EXPLICITLY state (else null) |
| `new_risks` | 0..3 | material new negatives |
| `overall` | −5..5 | holistic read vs the company's own trajectory; MUST always be answered |

Part B, numbers the model **copies** from the text (it never computes): `revenue_actual`,
`revenue_yoy_pct` (this quarter's y/y growth as stated), `eps_actual`, `guidance_new_mid` /
`guidance_ref_mid` (the company's previous guidance midpoint) and `guidance_metric`.

### Changes on each company's own scale

Growth speeding up by 2 points means more at a steady company than at one whose growth
swings by 10. The code turns the copied numbers into changes and divides them by the
company's own variability:

| Feature | Definition |
|---|---|
| `growth_accel_pp` | copied y/y growth − the last filed quarter's y/y growth (SEC, point-in-time) |
| `growth_accel_z` | ÷ `rev_growth_vol`: std of the company's y/y quarterly revenue growth over the last 8 quarters filed before the cutoff (floor 1 %, cap ±10) |
| `guidance_change_pct` | new / previous guidance midpoint − 1 (revenue or unspecified metric) |
| `guidance_change_z` | same company scale (revenue guidance only) |

Guards: a guidance change beyond ±50 %, an acceleration beyond 75 points, or a copied
"growth percent" beyond 400 is treated as a units mix-up and dropped; with several samples
(`AGENT_Q_SAMPLES`) numbers are combined by the median. Raw and scaled versions both go to
the regression, which decides which helps.

Changing an item or the prompt: bump `QUESTIONNAIRE_VERSION`. Cached answers and combiners
are tied to it; a combiner trained on another version is ignored (falls back to the chain).

## Features and magnitude

* Linear features: the nine answers, the five surprise features above, plus signed context numbers (`abn_ret_*`, `pos_52w`,
  `peer_mean_car`, `sector_mean_car`, `rev_yoy`, `op_margin`, `accruals`, `earnings_yield`,
  `log_market_cap`, `skew_25d`, `has_options`). Features present on < 5 % of training rows are
  dropped. A missing value sits at the training mean (contributes 0).
* Magnitude (`implied_move`, else `own_mean_abs_car`, `peer_mean_abs_car`, `vol_60d`, each
  converted to the base measure's scale by a median ratio) has no direction, so it is not a
  linear feature. It stretches the prediction away from the centre by `(m / m_ref) ** p`
  (the official option-statistics notebook uses p = 0.5).
* Cross-validation chooses three things together: the ridge penalty λ, the exponent p
  (0, 0.25, 0.5, 0.75, 1; 0 = no scaling) and how the weights meet the scaling —
  **joint** (features multiplied by the same factor when training, so the weights are
  the best ones given the scaling) or **after** (weights fitted unscaled, stretched
  afterwards, the notebook's recipe). Joint fitting also gives high-magnitude events
  more pull on the weights, which can help or hurt, so the data decides. The live
  formula is the same either way: `0.5 + scale · g · z`, `g = (m / m_ref) ** p`.

## Commands

```bash
# 1. build the store and the deploy bundle (weekly; daily keeps prices fresher)
uv run --with akshare python scripts/build_store.py
# 2. train the combiner (cheap first look, then everything); prints CV dR2 per quarter,
#    the chosen penalty and the weights; writes scripts/data/model/combiner.json + bundle copy
uv run python scripts/train_combiner.py --quarters 2026Q3 --limit 200
uv run python scripts/train_combiner.py
uv run python scripts/train_combiner.py --fit-only          # refit from the cache, no API calls
# train and tune on three quarters, score once on a fourth that is never fitted or tuned on;
# prints the combiner's dR2 with a bootstrap interval and the official baselines on the same rows
uv run python scripts/train_combiner.py --quarters 2025Q4,2026Q1,2026Q2 --limit 200 \
    --holdout 2026Q3 --holdout-limit 600
# model B: Q3's research notes train the preview features; Q4 (unsealed, so provisional) tests.
# --cv random: quarter folds can never train AND test on the one quarter that has notes.
# Add --no-preview for model A on the same rows (the note's increment = the difference).
uv run python scripts/train_combiner.py --quarters 2025Q4,2026Q1,2026Q2,2026Q3 --limit 600 \
    --holdout 2026Q4 --holdout-limit 600 --cv random
# every --holdout run also prints POST-PROCESSING: the same weights with no scaling, our
# scaling, no scaling + the option-only square-root calibration (scripts/calibration.py if
# present, else a copy), and both stacked, each with a paired interval against ours. Free with
# --fit-only.
# Q4 is unsealed and keeps growing, and a grown quarter re-draws its sample, so the
# --holdout-limit 600 rows are no longer the 600 already paid for. Re-score those, free:
uv run python scripts/train_combiner.py --quarters 2025Q4,2026Q1,2026Q2,2026Q3 --limit 600 \
    --holdout 2026Q4 --cv random --fit-only --holdout-from-cache
uv run python scripts/train_combiner.py --no-questionnaire  # ablation: context numbers only
# 3. upload the bundle (includes combiner.json)
uv run modal volume put em-agent-store scripts/data/bundle /bundle --force
# 4. one event, every stage printed
uv run python scripts/debug_predict.py --ticker ADSK
# 5. backtest / ablations (same --seed = same events). A quarter the combiner was trained
#    on gives an IN-SAMPLE score; the script warns. AGENT_USE_COMBINER=0 backtests the chain.
uv run python scripts/backtest.py -n 300
AGENT_USE_COMBINER=0 uv run python scripts/backtest.py -n 300 --blocks none
# 6. tests
uv run pytest                                                         # official suite
uv run --with langchain-core --with langchain-openai pytest scripts/tests
# 7. deploy, exactly as the starter README says
uv run modal deploy modal_app.py
```

`.env` needs `DEEPSEEK_API_KEY`, `EM_API_KEY`, `SEC_USER_AGENT="Your Name your@email"`.
Modal ships the whole `.env` to the container, so the same keys work there.
`AGENT_Q_SAMPLES` (default 1) averages several questionnaire calls; train and deploy with
the same value, or the weights no longer match the features' noise level.

## Reading the training output

* **CV ΔR² per fold** is the honest number: each quarter is predicted by a model that never
  saw it. In-sample ΔR² is printed too, and is optimistic.
* With 300 events a fold's ΔR² is noisy (±0.003 or more at zero signal). Compare variants on
  the same events and look at whether the sign is consistent across folds.
* LLM recall: answers for events before DeepSeek's training cutoff may be inflated by memory
  of the outcome, even with the text anonymised. Prefer quarters after the cutoff for the
  final judgement.

## Point-in-time rules (pinned by scripts/tests/test_agent.py)

* Fundamentals: facts filed before the cutoff date only.
* Peer and history reactions: return window closed before the cutoff date.
* Prices: trading days strictly before the cutoff date.

## Time budget

Combiner mode: 15 s materials, then the questionnaire (60 s timeout, one retry) runs in
parallel with the context (60 s budget). No LangChain install happens in this mode.
Chain mode: 15 s materials + 60 s context + 2 × 90 s LLM = 255 s worst case, plus the
LangChain install on a cold container.