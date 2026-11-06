# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
Unit tests for complete Composite orchestration, composition consistency,
additive forecasts, diagnostics, repeated calls, disabled exogenous inputs, and
short- and long-memory residual dynamics.

Usage examples
--------------
.. code-block:: bash

    python -m pytest seawave/tests/test_composite.py -v
'''

import numpy as np
import pytest

from seawave import SeaWaveConfig
from seawave.composite import SeaWaveComposite
from seawave.diagnostics import ForecastDiagnostics
from seawave.trend import (
    GLinTrendComponent,
    PolyTrendComponent,
    RWTrendComponent,
    TrendCandidate,
)


def composite_config(**overrides):
    '''Return a fast full-workflow configuration with injected simple trends.'''
    parameters = {
        "inner_window": 30,
        "n_win": 2,
        "topk": 2,
        "g_win": 160,
        "g_min_train": 50,
        "g_min_valid": 50,
        "lgb_num_boost_round": 150,
        "lgb_early_stopping": 20,
        "lgb_min_child_samples": 5,
        "lgb_num_leaves": 15,
    }
    parameters.update(overrides)
    return SeaWaveConfig(**parameters)


def trend_candidates():
    '''Inject non-HP candidates to keep orchestration tests focused and fast.'''
    return [
        TrendCandidate("poly2", PolyTrendComponent, 0.8),
        TrendCandidate("rw", RWTrendComponent, 0.7),
        TrendCandidate("glin", GLinTrendComponent, 0.3),
    ]


def make_composite(**config_overrides):
    '''Create a Composite with deterministic lightweight dependencies.'''
    return SeaWaveComposite(
        config=composite_config(**config_overrides),
        trend_candidates=trend_candidates(),
        min_train=50,
    )


@pytest.fixture(scope="module")
def composite_result():
    '''Run one full forecast on fixed promotion and autocorrelated residual data.'''
    n, h = 250, 14
    rng = np.random.default_rng(31)
    t = np.arange(n + h)
    promo = (rng.random(n + h) < 0.08).astype(float)
    memory = np.zeros(n + h)
    noise = rng.normal(0, 0.08, n + h)
    for i in range(1, n + h):
        memory[i] = 0.65 * memory[i - 1] + noise[i]
    y = 2.0 + 0.01 * t + np.sin(2 * np.pi * t / 7) + 0.8 * promo + memory
    composite = make_composite()
    forecast, info = composite.forecast(y[:n], h=h, promo=promo)
    return composite, y, promo, forecast, info


class TestSeaWaveComposite:
    '''Verify full workflow outputs and internal residual consistency.'''

    def test_component_order_and_finite_output(self, composite_result):
        '''All four algorithm stages execute in the designed order.'''
        composite, _, _, forecast, _ = composite_result
        assert composite.workflow == ("trend", "season", "residual", "exog")
        assert forecast.shape == (14,)
        assert np.isfinite(forecast).all()
        np.testing.assert_array_equal(composite.forecast_, forecast)

    def test_additive_decomposition(self, composite_result):
        '''The final forecast is exactly structure plus G plus exogenous contribution.'''
        composite, _, _, forecast, _ = composite_result
        expected = (
            composite.structural_forecast_
            + composite.residual_model_.predict(14)
            + composite.exog_forecast_
        )
        np.testing.assert_allclose(forecast, expected, rtol=0, atol=1e-12)

    def test_composition_consistency(self, composite_result):
        '''G is trained on the residual of the same blended Top-K structure.'''
        composite, y, _, _, _ = composite_result
        expected_residual = y[:250] - composite.structural_fitted_
        np.testing.assert_allclose(composite.residual_, expected_residual)
        np.testing.assert_allclose(composite.residual_model_.residual_, expected_residual)

    def test_diagnostic_fields(self, composite_result):
        '''The public info dictionary exposes scores, weights, and gate evidence.'''
        composite, _, _, _, info = composite_result
        assert isinstance(composite.diagnostics_, ForecastDiagnostics)
        assert info["top"] == composite.trend_.selected_names_
        assert info["weights"].sum() == pytest.approx(1.0)
        assert set(info["scores"]) == {"poly2", "rw", "glin"}
        assert all(len(values) == 2 for values in info["window_scores"].values())
        assert info["use_g"] == composite.residual_model_.use_g_
        assert info["g_scale"] == composite.residual_model_.g_scale_
        assert info["g_gate"]["n_train"] >= 50
        assert info["g_gate"]["n_gate"] >= 50

    def test_repeated_calls_are_deterministic(self, composite_result):
        '''A Composite can be called repeatedly without leaking mutable fit state.'''
        composite, y, promo, forecast, _ = composite_result
        repeated, _ = composite.forecast(y[:250], h=14, promo=promo)
        np.testing.assert_allclose(repeated, forecast)

    def test_disabled_exogenous_channel(self, composite_result):
        '''promo=None follows the same structural workflow with zero exogenous output.'''
        _, y, _, _, _ = composite_result
        composite = make_composite()
        forecast, _ = composite.forecast(y[:250], h=14, promo=None)
        assert composite.exog_model_.enabled_ is False
        np.testing.assert_array_equal(composite.exog_forecast_, np.zeros(14))
        assert forecast.shape == (14,)

    def test_long_memory_residual(self):
        '''A slow AR(1) wave also produces a finite direct multi-step forecast.'''
        n, h = 250, 14
        rng = np.random.default_rng(37)
        t = np.arange(n + h)
        memory = np.zeros(n + h)
        noise = rng.normal(0, 0.1, n + h)
        for i in range(1, n + h):
            memory[i] = 0.9 * memory[i - 1] + noise[i]
        y = 2.0 + 0.005 * t + np.sin(2 * np.pi * t / 7) + memory
        forecast, info = make_composite().forecast(y[:n], h=h)
        assert forecast.shape == (14,)
        assert np.isfinite(forecast).all()
        assert isinstance(info["use_g"], bool)

    def test_invalid_inputs(self, composite_result):
        '''The Composite rejects invalid y, h, and future promotion coverage early.'''
        _, y, promo, _, _ = composite_result
        with pytest.raises(ValueError, match="finite"):
            make_composite().forecast(np.array([1.0, np.nan, 2.0]), h=14)
        with pytest.raises(ValueError, match="positive integer"):
            make_composite().forecast(y[:250], h=0)
        with pytest.raises(ValueError, match="g_hmax"):
            make_composite().forecast(y[:250], h=15)
        with pytest.raises(ValueError, match="n\\+h"):
            make_composite().forecast(y[:250], h=14, promo=promo[:-1])

    def test_empty_trend_backtest_is_propagated(self):
        '''Insufficient history produces the clear trend-window error.'''
        with pytest.raises(ValueError, match="No valid inner backtest windows"):
            make_composite().forecast(np.arange(70, dtype=float), h=14)
