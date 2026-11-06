# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
Unit tests for direct multi-step dataset construction, temporal separation,
margin gate opening and closing, zero baselines, direct forecasts, fitted paths,
and deterministic LightGBM behavior.

Usage examples
--------------
.. code-block:: bash

    python -m pytest seawave/tests/test_residual.py -v
'''

import numpy as np
import pytest

from seawave import SeaWaveConfig
from seawave.mixin import AlgorithmMixin
from seawave.residual import DirectGBDTComponent


def residual_config(**overrides):
    '''Return a fast deterministic configuration for residual unit tests.'''
    parameters = {
        "inner_window": 60,
        "g_step": 2,
        "g_win": 200,
        "g_hmax": 14,
        "g_min_train": 50,
        "g_min_valid": 50,
        "lgb_num_boost_round": 200,
        "lgb_early_stopping": 20,
        "lgb_min_child_samples": 5,
        "lgb_num_leaves": 15,
    }
    parameters.update(overrides)
    return SeaWaveConfig(**parameters)


@pytest.fixture(scope="module")
def gated_component():
    '''Train one shared component on a deterministic learnable periodic residual.'''
    t = np.arange(360)
    residual = np.sin(2 * np.pi * t / 7) + 0.2 * np.cos(4 * np.pi * t / 7)
    component = DirectGBDTComponent(residual_config(margin=0.02)).fit(residual, h=14)
    return component, residual


class TestDirectDataset:
    '''Verify feature and target indices against a known integer sequence.'''

    def test_feature_and_target_construction(self):
        '''Each origin emits hmax rows with horizon as the eleventh feature.'''
        residual = np.arange(60, dtype=float)
        X, y = AlgorithmMixin.direct_multistep_dataset(residual, [28], 14)
        assert X.shape == (14, 11)
        assert y.shape == (14,)
        np.testing.assert_array_equal(y, np.arange(28, 42))
        np.testing.assert_array_equal(X[:, -1], np.arange(1, 15))
        assert X[0, 0] == 27
        assert X[0, 6] == 21
        assert X[0, 7] == pytest.approx(np.mean(residual[21:28]))
        assert X[0, 8] == pytest.approx(np.mean(residual[14:28]))
        assert X[0, 9] == pytest.approx(np.std(residual[14:28]))

    def test_invalid_origin_is_rejected(self):
        '''An origin cannot address targets beyond the residual history.'''
        with pytest.raises(ValueError, match="exceed"):
            AlgorithmMixin.direct_multistep_dataset(np.arange(40.0), [35], 14)


class TestDirectGBDTGate:
    '''Verify the held-out margin gate and its diagnostic evidence.'''

    def test_gate_opens_on_learnable_residual(self, gated_component):
        '''A periodic residual yields positive held-out improvement and finite output.'''
        component, _ = gated_component
        forecast = component.predict(14)
        assert component.use_g_ is True
        assert component.relative_improvement_ > 0.02
        assert 0 < component.g_scale_ <= 1
        assert forecast.shape == (14,)
        assert np.isfinite(forecast).all()
        assert component.diagnostics_["reason"] == "gate_open"

    def test_training_targets_precede_gate(self, gated_component):
        '''No target used for training or early stopping enters the gate interval.'''
        component, _ = gated_component
        last_training_target = max(component.train_origins_) + component.config.g_hmax
        first_gate_origin = min(component.gate_origins_)
        assert last_training_target <= first_gate_origin

    def test_fitted_path_matches_training_length(self, gated_component):
        '''The exogenous staging path has one value per historical residual.'''
        component, residual = gated_component
        path = component.predict_path()
        assert path.shape == residual.shape
        assert np.isfinite(path).all()
        np.testing.assert_array_equal(path[:14], np.zeros(14))

    def test_deterministic_reproduction(self, gated_component):
        '''The same residual and seed reproduce the same direct predictions.'''
        component, residual = gated_component
        reproduced = DirectGBDTComponent(component.config).fit(residual, h=14)
        np.testing.assert_allclose(reproduced.predict(14), component.predict(14))
        assert reproduced.use_g_ == component.use_g_

    def test_margin_closes_gate_on_noise(self):
        '''An impossible-to-exceed margin forces a closed gate and zero contribution.'''
        rng = np.random.default_rng(19)
        residual = rng.normal(size=300)
        component = DirectGBDTComponent(residual_config(margin=1.0)).fit(residual, h=14)
        assert component.use_g_ is False
        assert component.g_scale_ == 0.0
        assert component.diagnostics_["reason"] == "margin_not_exceeded"
        np.testing.assert_array_equal(component.predict(14), np.zeros(14))

    def test_zero_baseline_closes_gate(self):
        '''A zero-error baseline cannot produce a relative-improvement division.'''
        component = DirectGBDTComponent(residual_config()).fit(np.zeros(300), h=14)
        assert component.use_g_ is False
        assert component.relative_improvement_ == 0.0
        assert component.diagnostics_["reason"] == "zero_baseline_error"

    def test_insufficient_training_samples_close_gate(self):
        '''Short history returns a diagnostic zero path instead of training an invalid model.'''
        component = DirectGBDTComponent().fit(np.arange(100, dtype=float), h=14)
        assert component.use_g_ is False
        assert component.booster_ is None
        assert component.diagnostics_["reason"] == "insufficient_training_samples"
        np.testing.assert_array_equal(component.predict(14), np.zeros(14))

    def test_predict_before_fit(self):
        '''Residual prediction requires fitted historical features.'''
        with pytest.raises(RuntimeError, match="has not been fitted"):
            DirectGBDTComponent().predict(14)

    def test_horizon_cannot_exceed_training_coverage(self):
        '''The component rejects horizons unsupported by direct training.'''
        with pytest.raises(ValueError, match="g_hmax"):
            DirectGBDTComponent().fit(np.arange(100, dtype=float), h=15)
