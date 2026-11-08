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

## The questionnaire (`QUESTIONNAIRE_VERSION = "q2"`)

Part A, scored items:

| Item | Scale | Asks |
|---|---|---|
| `eps_vs_expectation` | −2..2 | EPS vs the expectation stated in the materials |
| `revenue_vs_expectation` | −2..2 | revenue vs expectation |
| `guidance_change` | −2..2 | guidance raised / maintained / cut (null if none) |
| `earnings_quality` | −1..1 | operating improvement vs one-offs (tax, asset sales, buybacks) |
| `margin_trend` | −1..1 | margins vs a year earlier |
| `demand_outlook` | −2..2 | forward commentary on orders, backlog, pipeline |
| `vs_preview` | −2..2 | results vs what the pre-release research note expected (null if no note) |
| `new_risks` | 0..3 | material new negatives |
| `overall` | −5..5 | holistic view; kept as one more feature, so nothing is lost vs the old score |

Part B, numbers the model **copies** from the text (it never computes): `revenue_actual` /
`revenue_expected`, `eps_actual` / `eps_expected` (same basis), `guidance_new_mid` /
`guidance_ref_mid` (previous guidance, else consensus) and `guidance_metric`.

### Surprises on each company's own scale

The same 2% revenue beat is large for a company whose growth barely varies and small for
one whose growth swings by 10 points. So the code turns the copied pairs into surprises and
divides them by the company's own variability:

| Feature | Definition |
|---|---|
| `revenue_surprise_pct` | revenue actual / expected − 1 |
| `revenue_surprise_z` | ÷ `rev_growth_vol`: std of the company's y/y quarterly revenue growth over the last 8 quarters filed before the cutoff (floor 1 %, cap ±10) |
| `guidance_change_pct` | new / reference guidance midpoint − 1 (revenue or unspecified metric) |
| `guidance_change_z` | same company scale (revenue guidance only) |
| `eps_surprise_px` | (EPS actual − expected) / last close: textbook price scaling, sane when expected EPS ≈ 0 |

Guards: a relative change beyond ±50 % or an EPS surprise beyond 20 % of the price is treated
as a units mix-up and dropped; with several samples (`AGENT_Q_SAMPLES`) numbers are combined
by the median. Raw and scaled versions both go to the regression, which decides which helps.

Changing an item or the prompt: bump `QUESTIONNAIRE_VERSION`. Cached answers and combiners
are tied to it; a combiner trained on another version is ignored (falls back to the chain).

## Features and magnitude

* Linear features: the nine answers, the five surprise features above, plus signed context numbers (`abn_ret_*`, `pos_52w`,
  `peer_mean_car`, `sector_mean_car`, `rev_yoy`, `op_margin`, `accruals`, `earnings_yield`,
  `log_market_cap`, `skew_25d`, `has_options`). Features present on < 5 % of training rows are
  dropped. A missing value sits at the training mean (contributes 0).
* Magnitude (`implied_move`, else `own_mean_abs_car`, `peer_mean_abs_car`, `vol_60d`, each
  converted to the base measure's scale by a median ratio) has no direction, so it is not a
  linear feature. It stretches the prediction away from the centre by `sqrt(m / m_ref)`
  (official option-statistics notebook), and only if that improved cross-validated ΔR².

## Commands

```bash
# 1. build the store and the deploy bundle (weekly; daily keeps prices fresher)
uv run --with akshare python scripts/build_store.py
# 2. train the combiner (cheap first look, then everything); prints CV dR2 per quarter,
#    the chosen penalty and the weights; writes scripts/data/model/combiner.json + bundle copy
uv run python scripts/train_combiner.py --quarters 2026Q3 --limit 200
uv run python scripts/train_combiner.py
uv run python scripts/train_combiner.py --fit-only          # refit from the cache, no API calls
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