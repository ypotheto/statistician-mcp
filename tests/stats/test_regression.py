from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats as sp_stats

from statistician_mcp.stats import regression as reg
from statistician_mcp.utils.formulas import FormulaError


def test_fit_linear_model_matches_statsmodels_directly() -> None:
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 5 + 2 * x1 - 1 * x2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    result = reg.fit_linear_model(df, "y ~ x1 + x2")
    ref = smf.ols("y ~ x1 + x2", data=df).fit()

    by_term = {c["term"]: c for c in result["coefficients"]}
    assert by_term["x1"]["estimate"] == pytest.approx(ref.params["x1"])
    assert by_term["x1"]["p_value"] == pytest.approx(ref.pvalues["x1"])
    assert result["r_squared"] == pytest.approx(ref.rsquared)
    assert result["aic"] == pytest.approx(ref.aic)
    # near-orthogonal independent predictors -> VIF close to 1 (no collinearity)
    assert result["vif"]["x1"] == pytest.approx(1.0, abs=0.2)


def test_fit_linear_model_rejects_unsafe_formula() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(FormulaError):
        reg.fit_linear_model(df, "y ~ __import__('os').system('dir')")


def test_fit_logistic_model_matches_statsmodels_directly() -> None:
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.8 * x1)))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"x1": x1, "y": y})

    result = reg.fit_logistic_model(df, "y ~ x1")
    ref = smf.logit("y ~ x1", data=df).fit(disp=0)

    by_term = {c["term"]: c for c in result["coefficients"]}
    assert by_term["x1"]["estimate"] == pytest.approx(ref.params["x1"])
    assert by_term["x1"]["odds_ratio"] == pytest.approx(np.exp(ref.params["x1"]))
    assert result["aic"] == pytest.approx(ref.aic)
    assert 0.5 <= result["roc"]["auc"] <= 1.0


def test_roc_curve_auc_is_near_one_for_a_perfectly_separable_case() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    fpr, tpr, _ = reg._roc_curve(y_true, y_score)
    auc = float(np.trapezoid(tpr, fpr))
    assert auc == pytest.approx(1.0)


def test_compare_models_nested_f_test_matches_hand_computation() -> None:
    rng = np.random.default_rng(2)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 3 + 2 * x1 + 1.5 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    result = reg.compare_models(df, ["y ~ x1", "y ~ x1 + x2"], family="linear")

    small = smf.ols("y ~ x1", data=df).fit()
    large = smf.ols("y ~ x1 + x2", data=df).fit()
    df1 = small.df_resid - large.df_resid
    df2 = large.df_resid
    f_stat = ((small.ssr - large.ssr) / df1) / (large.ssr / df2)
    p_value = float(1 - sp_stats.f.cdf(f_stat, df1, df2))

    assert result["nested_f_test"]["f_statistic"] == pytest.approx(f_stat)
    assert result["nested_f_test"]["p_value"] == pytest.approx(p_value)
    assert result["best_by_aic"] == "y ~ x1 + x2"


def test_predict_from_model_matches_statsmodels_get_prediction() -> None:
    rng = np.random.default_rng(3)
    n = 100
    x1 = rng.normal(0, 1, n)
    y = 10 + 3 * x1 + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"x1": x1, "y": y})
    new_data = pd.DataFrame({"x1": [0.0, 1.0]})

    result = reg.predict_from_model(df, "y ~ x1", new_data)
    ref_model = smf.ols("y ~ x1", data=df).fit()
    ref_pred = ref_model.get_prediction(new_data).summary_frame(alpha=0.05)

    assert result["predicted"] == pytest.approx(ref_pred["mean"].tolist())
    ref_lower = ref_pred["mean_ci_lower"].tolist()
    assert result["confidence_interval"]["lower"] == pytest.approx(ref_lower)


def test_fit_distribution_identifies_the_true_generating_family() -> None:
    rng = np.random.default_rng(4)
    normal_data = rng.normal(50, 5, 500)
    result = reg.fit_distribution(normal_data, ["normal", "lognormal", "gamma"])
    assert result["best_fit"] == "normal"

    weibull_data = rng.weibull(2.0, 500) * 10 + 0.01
    result2 = reg.fit_distribution(weibull_data, ["normal", "weibull", "exponential"])
    assert result2["best_fit"] == "weibull"


def test_fit_distribution_skips_positive_only_families_for_negative_data() -> None:
    data = np.array([-5.0, -1.0, 0.0, 1.0, 5.0, -2.0, 3.0, -4.0])
    result = reg.fit_distribution(data, ["normal", "weibull", "exponential"])
    fitted_names = {f["distribution"] for f in result["fits"]}
    assert fitted_names == {"normal"}
