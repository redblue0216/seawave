# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
Unit tests for promotion lag alignment, second-stage LightGBM fitting, future
input requirements, disabled exogenous behavior, and deterministic prediction.

Usage examples
--------------
.. code-block:: bash

    python -m pytest seawave/tests/test_exog.py -v
'''

import numpy as np
import pytest

from seawave import SeaWaveConfig
from seawave.mixin import AlgorithmMixin
from seawave.exog import PromoLagComponent


def exog_config():
    '''Return a small deterministic LightGBM configuration.'''
    return SeaWaveConfig(
        promo_lag=3,
        lgb_num_boost_round=150,
        lgb_early_stopping=20,
        lgb_min_child_samples=5,
        lgb_num_leaves=15,
    )


@pytest.fixture(scope="module")
def promotion_data():
    '''Generate known lagged promotion effects and a future promotion path.'''
    rng = np.random.default_rng(23)
    n, h = 240, 14
    promo = (rng.random(n + h) < 0.12).astype(float)
    features = AlgorithmMixin.promo_lag_cols(promo, 3)
    residual = 1.4 * features[:n, 0] + 0.5 * features[:n, 1]
    residual += rng.normal(0, 0.02, n)
    return residual, promo


class TestPromotionLags:
    '''Verify exact lag alignment and zero padding.'''

    def test_lag_columns(self):
        '''Columns contain current, lag-1, and lag-2 values in reference order.'''
        actual = AlgorithmMixin.promo_lag_cols(np.array([1.0, 2.0, 3.0, 4.0]), 3)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0],
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_lag(self):
        '''A zero lag is not a valid promotion feature design.'''
        with pytest.raises(ValueError, match="lag"):
            AlgorithmMixin.promo_lag_cols(np.array([1.0]), 0)


class TestPromoLagComponent:
    '''Verify second-stage fitting and future contribution behavior.'''

    def test_fit_and_future_prediction(self, promotion_data):
        '''The component learns a known lagged effect and returns finite contributions.'''
        residual, promo = promotion_data
        component = PromoLagComponent(exog_config()).fit(residual, promo)
        predicted = component.predict(promo, n=len(residual), h=14)
        assert component.enabled_ is True
        assert predicted.shape == (14,)
        assert np.isfinite(predicted).all()
        assert np.any(np.abs(predicted) > 1e-8)
        assert "3 lags" in component._info()

    def test_disabled_component_returns_zeros(self):
        '''promo=None explicitly disables both training and future contribution.'''
        component = PromoLagComponent(exog_config()).fit(np.arange(50.0), promo=None)
        predicted = component.predict(None, n=50, h=14)
        assert component.enabled_ is False
        assert component.booster_ is None
        np.testing.assert_array_equal(predicted, np.zeros(14))

    def test_future_promotion_length_is_checked(self, promotion_data):
        '''An enabled component requires known promotion inputs through n+h.'''
        residual, promo = promotion_data
        component = PromoLagComponent(exog_config()).fit(residual, promo)
        with pytest.raises(ValueError, match="n\\+h"):
            component.predict(promo[:-1], n=len(residual), h=14)

    def test_training_length_mismatch_is_checked(self, promotion_data):
        '''The prediction origin must match the fitted residual length.'''
        residual, promo = promotion_data
        component = PromoLagComponent(exog_config()).fit(residual, promo)
        with pytest.raises(ValueError, match="residual length"):
            component.predict(promo, n=len(residual) - 1, h=14)

    def test_short_training_promotion_is_rejected(self, promotion_data):
        '''Training cannot begin with fewer promotion rows than residuals.'''
        residual, promo = promotion_data
        with pytest.raises(ValueError, match="at least as many"):
            PromoLagComponent(exog_config()).fit(residual, promo[:100])

    def test_predict_before_fit(self):
        '''Future prediction requires a fitted or explicitly disabled component.'''
        with pytest.raises(RuntimeError, match="has not been fitted"):
            PromoLagComponent().predict(None, n=20, h=14)

    def test_deterministic_reproduction(self, promotion_data):
        '''The same data and seed produce identical second-stage contributions.'''
        residual, promo = promotion_data
        first = PromoLagComponent(exog_config()).fit(residual, promo)
        second = PromoLagComponent(exog_config()).fit(residual, promo)
        np.testing.assert_allclose(
            first.predict(promo, len(residual), 14),
            second.predict(promo, len(residual), 14),
        )
