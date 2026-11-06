# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
End-to-end tests for the installed-style public SeaWave API using the default
HP candidate pool, Fourier seasonality, direct residual learning, and promotion
inputs. No internal Composite method is called by these tests.

Usage examples
--------------
.. code-block:: bash

    python -m pytest seawave/tests/test_interface.py -v
'''

import numpy as np
import pytest

from seawave import SeaWaveConfig, SeaWaveForecastInterface


@pytest.fixture(scope="module")
def public_result():
    '''Execute the public API with default trend candidates and fixed data.'''
    n, h = 320, 14
    rng = np.random.default_rng(41)
    t = np.arange(n + h)
    promo = (rng.random(n + h) < 0.08).astype(float)
    memory = np.zeros(n + h)
    noise = rng.normal(0, 0.08, n + h)
    for i in range(2, n + h):
        memory[i] = 0.5 * memory[i - 1] + 0.18 * memory[i - 2] + noise[i]
    y = 2.5 + 0.004 * t + np.sin(2 * np.pi * t / 7) + 0.7 * promo + memory
    config = SeaWaveConfig(
        inner_window=40,
        n_win=2,
        topk=2,
        g_win=160,
        g_min_train=50,
        g_min_valid=50,
        lgb_num_boost_round=150,
        lgb_early_stopping=20,
        lgb_min_child_samples=5,
        lgb_num_leaves=15,
    )
    model = SeaWaveForecastInterface(config=config)
    forecast, info = model(y=y[:n], h=h, promo=promo)
    return model, y, promo, forecast, info


class TestSeaWaveForecastInterface:
    '''Verify real forecasting through the public callable object.'''

    def test_real_public_forecast(self, public_result):
        '''The default candidate pool returns 14 finite predictions.'''
        _, _, _, forecast, info = public_result
        assert forecast.shape == (14,)
        assert np.isfinite(forecast).all()
        assert len(info["top"]) == 2
        assert len(info["scores"]) == 8
        assert info["weights"].shape == (2,)
        assert np.isfinite(info["weights"]).all()

    def test_public_configuration_is_applied(self, public_result):
        '''The custom Top-K and LightGBM settings reach the complete workflow.'''
        model, _, _, _, _ = public_result
        assert model.config.topk == 2
        assert model.composite.config.lgb_num_boost_round == 150
        assert model.composite.residual_model_.config is model.config

    def test_public_input_validation(self, public_result):
        '''The public API preserves clear Composite validation errors.'''
        model, y, promo, _, _ = public_result
        with pytest.raises(ValueError, match="g_hmax"):
            model(y=y[:320], h=15, promo=promo)
        with pytest.raises(ValueError, match="n\\+h"):
            model(y=y[:320], h=14, promo=promo[:-1])
        invalid = y[:320].copy()
        invalid[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            model(y=invalid, h=14, promo=promo)

    def test_public_metadata(self, public_result):
        '''Interface and Composite metadata describe the integrated algorithm.'''
        model, _, _, _, _ = public_result
        assert "SeaWaveForecastInterface" in model._info()
        assert "gated residual" in model.composite._info()
