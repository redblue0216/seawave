# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
Unit tests for HP numerical correctness, trend extrapolation, candidate
penalties, staggered backtests, stable Softmax selection, and full ensembles.

Usage examples
--------------
.. code-block:: bash

    python -m pytest seawave/tests/test_trend.py -v
'''

import numpy as np
import pytest

from seawave import SeaWaveConfig
from seawave.mixin import AlgorithmMixin
from seawave.trend import (
    GLinTrendComponent,
    HPTrendComponent,
    PolyTrendComponent,
    RWTrendComponent,
    TrendCandidate,
    TrendEnsembleComponent,
    TrendScore,
    build_trend_candidates,
    score_trend_candidates,
    select_trend_candidates,
)


class TestHPTrendComponent:
    '''Verify the JAX HP solve and damping behavior.'''

    def test_hp_matches_dense_numpy_reference(self):
        '''The JAX system agrees with an independent NumPy K'K reference.'''
        rng = np.random.default_rng(7)
        y = 2.0 + 0.03 * np.arange(60) + rng.normal(0, 0.05, 60)
        lamb = 1600.0
        n = len(y)
        K = np.zeros((n - 2, n))
        for i in range(n - 2):
            K[i, i : i + 3] = [1.0, -2.0, 1.0]
        expected = np.linalg.solve(np.eye(n) + lamb * K.T @ K, y)
        actual = AlgorithmMixin.hp_filter(y, lamb=lamb)
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)

    def test_constant_series_remains_constant(self):
        '''A constant signal has zero curvature and a flat forecast.'''
        component = HPTrendComponent(lamb=129600.0, damp=1.0).fit(np.full(50, 3.5))
        np.testing.assert_allclose(component.trend_, 3.5, atol=1e-4)
        np.testing.assert_allclose(component.predict(14), 3.5, atol=1e-4)

    def test_damping_reduces_slope_commitment(self):
        '''A lower damping value produces a less aggressive extrapolation.'''
        y = 1.0 + 0.02 * np.arange(80)
        conservative = HPTrendComponent(lamb=1600.0, damp=0.3).fit(y)
        aggressive = HPTrendComponent(lamb=1600.0, damp=1.0).fit(y)
        assert conservative.predict(14)[-1] < aggressive.predict(14)[-1]


class TestParametricTrends:
    '''Verify exact recovery on known polynomial and linear signals.'''

    def test_quadratic_fit_and_forecast(self):
        '''The degree-two candidate recovers a quadratic without noise.'''
        t = np.arange(50)
        y = 1.5 + 0.2 * t + 0.003 * t**2
        component = PolyTrendComponent().fit(y)
        np.testing.assert_allclose(component.trend_, y, atol=1e-10)
        expected = 1.5 + 0.2 * np.arange(50, 64) + 0.003 * np.arange(50, 64) ** 2
        np.testing.assert_allclose(component.predict(14), expected, atol=1e-10)

    def test_global_linear_fit_and_forecast(self):
        '''The global candidate recovers a straight line.'''
        t = np.arange(40)
        y = 2.0 - 0.05 * t
        component = GLinTrendComponent().fit(y)
        np.testing.assert_allclose(
            component.predict(5), 2.0 - 0.05 * np.arange(40, 45), atol=1e-12
        )

    def test_local_window_never_exceeds_short_history(self):
        '''Short sequences use all available observations rather than slicing empty data.'''
        component = RWTrendComponent(window=90).fit(np.arange(10, dtype=float))
        assert component.effective_window_ == 10
        assert component.predict(3).shape == (3,)

    @pytest.mark.parametrize(
        "component",
        [HPTrendComponent(), PolyTrendComponent(), RWTrendComponent(), GLinTrendComponent()],
    )
    def test_predict_before_fit(self, component):
        '''Every trend component rejects prediction before fitting.'''
        with pytest.raises(RuntimeError, match="has not been fitted"):
            component.predict(14)

    def test_invalid_series_and_horizon(self):
        '''Non-finite values and invalid horizons fail early.'''
        with pytest.raises(ValueError, match="finite"):
            GLinTrendComponent().fit(np.array([1.0, np.nan]))
        component = GLinTrendComponent().fit(np.arange(10.0))
        with pytest.raises(ValueError, match="positive integer"):
            component.predict(0)


class TestTrendSelection:
    '''Verify candidate construction, worst-window scores, and Softmax weights.'''

    def test_default_candidate_pool(self):
        '''The reference configuration creates eight candidates with design penalties.'''
        candidates = build_trend_candidates(SeaWaveConfig())
        assert len(candidates) == 8
        assert [candidate.name for candidate in candidates[:4]] == [
            "hp_l129600_d3",
            "hp_l129600_d6",
            "hp_l129600_d9",
            "hp_l129600_d10",
        ]
        high_frequency = next(c for c in candidates if c.name == "hp_l1600_d10")
        assert high_frequency.penalty == 1.0
        assert {c.name: c.penalty for c in candidates}["glin"] == 0.3

    def test_quadratic_candidate_wins_backtest(self):
        '''A quadratic candidate outscores a line on a curved validation history.'''
        t = np.arange(240)
        y = 2.0 + 0.01 * t + 0.0002 * t**2
        candidates = [
            TrendCandidate("poly2", PolyTrendComponent, 0.8),
            TrendCandidate("glin", GLinTrendComponent, 0.3),
        ]
        scores = score_trend_candidates(
            y, candidates, inner_window=30, n_win=2, gamma=0.1, min_train=50
        )
        assert len(scores["poly2"].window_scores) == 2
        assert scores["poly2"].score < scores["glin"].score
        expected = max(scores["glin"].window_scores) * (1 + 0.1 * 0.3)
        assert scores["glin"].score == pytest.approx(expected)

    def test_empty_backtest_windows_are_rejected(self):
        '''An empty origin list raises a clear error instead of max() failing.'''
        candidates = [TrendCandidate("glin", GLinTrendComponent, 0.3)]
        with pytest.raises(ValueError, match="No valid inner backtest windows"):
            score_trend_candidates(np.arange(100.0), candidates)

    def test_softmax_is_stable_for_large_scores(self):
        '''Subtracting the minimum score prevents overflow and underflow loss.'''
        scores = {
            "a": TrendScore("a", 0.5, (10000.0,), 10000.0),
            "b": TrendScore("b", 0.5, (10001.0,), 10001.0),
        }
        selected, weights = select_trend_candidates(scores, topk=2, tau=0.1)
        assert [item.name for item in selected] == ["a", "b"]
        assert np.isfinite(weights).all()
        assert weights.sum() == pytest.approx(1.0)
        assert weights[0] > weights[1]

    def test_full_trend_ensemble(self):
        '''The ensemble fits selected candidates on full history and blends paths.'''
        t = np.arange(240)
        y = 1.0 + 0.02 * t + 0.0001 * t**2
        candidates = [
            TrendCandidate("poly2", PolyTrendComponent, 0.8),
            TrendCandidate("rw", RWTrendComponent, 0.7),
            TrendCandidate("glin", GLinTrendComponent, 0.3),
        ]
        config = SeaWaveConfig(inner_window=30, n_win=2, topk=2)
        component = TrendEnsembleComponent(
            config=config, candidates=candidates, min_train=50
        ).fit(y)
        forecast = component.predict(14)
        assert component.selected_names_[0] == "poly2"
        assert len(component.models_) == 2
        assert component.trend_.shape == (240,)
        assert forecast.shape == (14,)
        assert np.isfinite(forecast).all()
        assert component.weights_.sum() == pytest.approx(1.0)
        assert "TrendEnsembleComponent" in component._info()
