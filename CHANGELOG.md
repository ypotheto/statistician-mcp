# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versioning is semver.

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
