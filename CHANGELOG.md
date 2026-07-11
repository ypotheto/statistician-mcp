# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versioning is semver.

## Unreleased — Phase 7a (DO Spaces + Postgres key store)

- **`SpacesBackend`** (`storage.py`): a DigitalOcean Spaces (S3-compatible, via
  `boto3`) implementation of `StorageBackend`, selected by setting
  `STATMCP_SPACES_BUCKET` (plus endpoint/key/secret/region) instead of the default
  `LocalDirBackend`. Unlocks ephemeral-disk compute (e.g. DO App Platform) for
  dataset/artifact storage. Namespaced under `STATMCP_SPACES_PREFIX` (default
  `statistician-mcp`) so the bucket can be safely shared with other services
  (each their own prefix, each their own bucket-scoped Spaces key) without key
  collisions — verified live against a real shared bucket, including that two
  prefixes round-trip independently.
- Storage-backend contract tests (`tests/test_storage.py`) run against both
  backends via a shared parametrized fixture, `SpacesBackend`'s half mocked with
  `moto` — no real Spaces bucket needed to verify the two stay behaviorally
  identical (write/read roundtrip, missing-key `FileNotFoundError`, idempotent
  delete, prefix-scoped `list`), plus a dedicated cross-service isolation test.
- **`apikeys.py` restructured into a `KeyStore` ABC**: `SqliteKeyStore` (the
  original behavior, now a class) and a new `PostgresKeyStore` (`psycopg` +
  `psycopg_pool`), selected by `STATMCP_DATABASE_URL`. `AuthMiddleware`'s
  `TokenVerifier` is now async and runs the lookup via
  `anyio.to_thread.run_sync`, so a Postgres-backed verification's network round
  trip never blocks the event loop. Verified live end-to-end against a real DO
  Postgres cluster (issue a key via `scripts/issue_key.py`, use it against a
  running container, confirm 200).
- Fixed a test-isolation bug this uncovered: `Settings`' new `env_file=".env"`
  meant test fixtures that didn't explicitly pin `database_url`/`spaces_bucket`
  would silently inherit real credentials from a developer's `.env` and hit
  live infrastructure. `tests/conftest.py` now pins both to `None` explicitly.
- `tests/test_apikeys.py`: contract tests run against both key stores via a
  shared parametrized fixture, `PostgresKeyStore`'s half against a real
  `postgres:16-alpine` Docker container spun up for the test session (not
  mocked — SQL-dialect differences between SQLite and Postgres are exactly what
  a mock would paper over).

## v0.2.1 — Hardening (Phase 7, minus deployment)

Deployment (DO Spaces backend, actual hosting, ypotheto-core round-trip against a
hosted instance) is split out to Phase 7a and not part of this release; the plan's
`v0.3.0` marker is reserved for when that's done and verified.

- **Dockerfile**: slim non-root image, uvicorn entrypoint, `/healthz` healthcheck.
  Not build-tested in this environment (no Docker daemon available) — do a
  `docker build` sanity check before relying on it.
- **API-key auth mode** (`STATMCP_AUTH_MODE=keys`): a real per-tenant SQLite key
  table (hashed keys only) alongside the existing single-static-token mode, behind
  one shared `AuthMiddleware` built around a pluggable token-verifier so both
  modes share one code path. `scripts/issue_key.py` admin CLI (issue/disable/list).
  Default stays `token` mode, so local Claude Desktop/MCP Inspector testing is
  unaffected.
- **Per-request timeout** for tool calls (`STATMCP_REQUEST_TIMEOUT_SECONDS`,
  default 120s), scoped to POST only so it never cuts off a health check, artifact
  download, or a long-lived streamable-HTTP server-push stream.
- **Concurrency fixes found via a threaded stress test, not assumed away**:
  - A lock around `DatasetStore`'s LRU cache (the compound check/evict sequence
    wasn't atomic).
  - A closed-over-in-`envelope.tool` guard that closes any matplotlib figure a
    plotting tool leaked by raising between figure creation and `render_png`
    (matplotlib keeps figures registered globally until closed).
  - **`LocalDirBackend`'s path-traversal check** used `Path.resolve()` (a real
    filesystem lookup) and was found to spuriously reject valid paths under
    concurrent directory creation on Windows; replaced with a purely lexical
    containment check, which is both race-free and more robust against TOCTOU/
    symlink tricks than resolve-and-compare.
  - **A TOCTOU gap in `DatasetStore.list()`**: enumerating datasets then reading
    each one is two steps, and a concurrent `delete()` between them raised instead
    of just omitting the vanished entry. Same fix applied to `get_dataframe`/
    `get_info`, and to `LocalDirBackend.list()`'s own directory walk.
  - **Windows-specific file-locking retries**: unlike Linux (the actual production
    target), Windows can transiently raise `PermissionError` when one thread's
    read races another's delete of the same file; added a short, shared retry
    around reads/writes/deletes.
- **Closed a cross-cutting resource-limit gap**: the 200k-row analysis cap was
  only ever enforced by `eda.py`'s own private helper — every other analysis
  module (inference, DOE, SPC, MSA, regression, advisor) called
  `store.get_dataframe()` directly, bypassing it entirely. Moved to a shared
  `get_dataframe_for_analysis` helper used by every analysis module.

## v0.2.0 — DOE, SPC, MSA, regression, advisor (Phases 4-6)

- **DOE** (`design_experiment`, `evaluate_design`, `analyze_factorial`,
  `analyze_response_surface`, `optimize_response`): full/fractional factorial,
  Plackett-Burman, CCD, Box-Behnken, and LHS design generation via pyDOE3; alias
  structure/resolution for fractional designs; effects fitting with half-normal/
  Pareto-of-effects (Lenth's method), lack-of-fit testing, response-surface
  stationary-point analysis; Derringer-Suich desirability optimization.
- **SPC** (`create_control_chart`, `assess_capability`, `run_stability_check`):
  Xbar-R/Xbar-S/I-MR/p/np/c/u/EWMA/CUSUM charts, the 8 Nelson/Western-Electric
  rules, and Cp/Cpk/Pp/Ppk capability with DPMO/sigma level. Control-chart
  constants (c4, d2, d3, A2, A3, B3, B4, D3, D4) are computed from first
  principles (closed-form c4, numerically-integrated d2/d3) rather than
  transcribed from a published table.
- **MSA** (`analyze_gauge_rr`, `analyze_attribute_agreement`): crossed Gauge R&R
  via the ANOVA method with variance components, %Contribution, %StudyVar,
  %Tolerance, and ndc; Fleiss' kappa for attribute agreement.
- **Regression** (`fit_linear_model`, `fit_logistic_model`, `compare_models`,
  `predict_from_model`, `fit_distribution`): OLS with VIF/Cook's-distance
  diagnostics, logistic regression with ROC/AUC, nested-F/AIC/BIC model
  comparison, stateless prediction with CIs/PIs, and distribution fitting
  (normal/lognormal/weibull/exponential/gamma) ranked by a generic
  Anderson-Darling statistic.
- **Advisor** (`recommend_analysis`, `explain_concept`) and 3 MCP prompts
  (`plan_an_experiment`, `analyze_my_experiment`, `set_up_spc`).
- Added a restricted patsy-style model-formula validator (column names / `~ + -
  : *` only) that runs before any formula reaches patsy — confirmed against the
  installed patsy version that unfiltered formula terms execute arbitrary
  Python. All formula-accepting tools now use it.

## v0.1.0 — EDA, inference, power (Phases 0-3)

First genuinely useful release; the trigger point for wiring up the
ypotheto-core client.

- Server scaffold: FastMCP over stdio and streamable HTTP, health check, CI.
- Workspace/dataset/artifact chassis: bearer-token auth, per-workspace dataset
  storage (parquet), artifact serving, the `{ok, results, assumptions,
  interpretation, artifacts, meta}` result envelope, and usage-event logging.
- Dataset tools: load from CSV/URL, describe, sample, a restricted
  AST-validated transform expression grammar (filter/select/rename/derive/
  stack/unstack), delete.
- EDA tools: summarize_columns, plot_distribution, test_normality,
  detect_outliers, compute_correlations, plot_scatter, plot_time_series,
  crosstab, backed by a reusable normality/variance assumptions engine.
- Inference tools: compare_means (Welch by default, automatic nonparametric
  fallback), compare_multiple_groups (ANOVA with automatic Tukey/Welch+Games-
  Howell/Kruskal-Wallis+Dunn), compare_proportions, test_equivalence (TOST),
  compute_confidence_interval, test_variance.
- Power tools: compute_power_or_sample_size, plot_power_curve.

## v0.0.0 — Scaffold (Phase 0)

Package skeleton, `ping` health-check tool, dual transport (stdio + streamable
HTTP), CI workflow.
