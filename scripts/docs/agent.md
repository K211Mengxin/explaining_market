# The prediction agent (all in predict.py)

Only `predict.py` differs from the official starter. `modal_app.py`, `pyproject.toml`,
`uv.lock` and the tests are untouched; everything else lives in the git-ignored `scripts/`.

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
| 9 | prompts | system + user prompt |
| 10 | chain | LangChain LCEL `prompt \| ChatOpenAI(DeepSeek, JSON) \| parser`; installed on cold start, openai-SDK twin as fallback |
| 11 | features | the six context blocks, each optional and time-bounded |
| 12 | pipeline | `predict()`: materials → context → chain → 0.5 fallback |

## Why it works inside the official modal_app.py

* The official image has only fastapi/httpx/openai/pydantic. A container's first call
  pip-installs `langchain-core` + `langchain-openai` into a temp dir (~7 s measured), in
  parallel with the context stage. If that fails, the identical prompt runs through the
  openai SDK, so a prediction is never lost.
* The official app mounts no volume, so `predict.py` reads the prebuilt bundle through the
  Modal API (`modal.Volume.from_name("em-agent-store").read_file(...)`) once per container.
* AKShare is not needed online: the bundle carries ~420 trading days of prices; newer days
  come from Yahoo's chart API over plain httpx (same raw closes as Sina).
* SEC facts for the event's own company are refreshed live when the bundle is > 7 days old.

## Commands

```bash
# 1. build the store and the deploy bundle (weekly; daily keeps prices fresher)
uv run --with akshare python scripts/build_store.py
# 2. upload the bundle (no file changes needed)
uv run modal volume put em-agent-store scripts/data/bundle /bundle --force
# 3. one event, every stage printed
uv run python scripts/debug_predict.py --ticker ADSK
# 4. backtest / ablations (same --seed = same events)
uv run python scripts/backtest.py -n 300 --blocks none
uv run python scripts/backtest.py -n 300
uv run python scripts/backtest.py -n 300 --peer-method sic
# 5. tests
uv run pytest                                                         # official suite
uv run --with langchain-core --with langchain-openai pytest scripts/tests
# 6. deploy, exactly as the starter README says
uv run modal deploy modal_app.py
```

`.env` needs `DEEPSEEK_API_KEY`, `EM_API_KEY`, `SEC_USER_AGENT="Your Name your@email"`.
Modal ships the whole `.env` to the container, so the same keys work there.

## Point-in-time rules (pinned by scripts/tests/test_agent.py)

* Fundamentals: facts filed before the cutoff date only.
* Peer and history reactions: return window closed before the cutoff date.
* Prices: trading days strictly before the cutoff date.

## Time budget

15 s materials + 60 s context (`AGENT_FEATURE_BUDGET`) + 2 × 90 s LLM = 255 s worst case.
A cold container also downloads the bundle and installs LangChain; both overlap the budget
and are skipped if they fail.
