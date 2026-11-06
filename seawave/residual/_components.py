# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------

DirectGBDTComponent learns the residual left by the selected trend and Fourier
structure. It uses direct multi-step training rather than recursive prediction,
so an h-step forecast does not feed earlier residual predictions back as inputs.

Mathematical principle
----------------------
Each origin contributes seven lag values, rolling 7/14 means, rolling 14
standard deviation, and horizon as features. The target is R[origin+horizon-1].
This avoids the recursive error accumulation sometimes called the recursive tax.

The margin gate evaluates strictly later, held-out origins:

    $rel = (MAE_base - MAE_base_g) / MAE_base$
    $use_g = rel > margin$
    $g_scale = min(1, rel / g_rel_scale)$

All training targets precede the first gate origin. Consequently, the gate does
not evaluate observations already used to train or early-stop the booster.

Usage examples
--------------
.. code-block:: python

    from seawave import SeaWaveConfig
    from seawave.residual import DirectGBDTComponent

    component = DirectGBDTComponent(SeaWaveConfig())
    component.fit(residual=structural_residual, h=14)
    residual_forecast = component.predict(14)
    print(component.use_g_, component.g_scale_, component.diagnostics_)
'''

import numpy as np

from seawave.config import SeaWaveConfig
from seawave.mixin import AlgorithmMixin, UtilMixin
from seawave.residual._base import BaseResidualComponent


class DirectGBDTComponent(BaseResidualComponent, AlgorithmMixin, UtilMixin):
    '''Class Introduction:

    DirectGBDTComponent trains one horizon-aware LightGBM residual model and
    activates its contribution only when held-out relative improvement exceeds
    the configured margin. A closed gate produces an exact zero vector.

    :parameters:
        - config (SeaWaveConfig) - Residual windows, sample thresholds, gate, and GBDT settings.
    '''

    def __init__(self, config: SeaWaveConfig | None = None):
        '''Initialize an unfitted residual component.'''
        self.config = SeaWaveConfig() if config is None else config
        if not isinstance(self.config, SeaWaveConfig):
            raise TypeError("config must be a SeaWaveConfig instance")
        self.residual_ = None
        self.booster_ = None
        self.use_g_ = False
        self.g_scale_ = 0.0
        self.relative_improvement_ = None
        self.train_origins_ = []
        self.gate_origins_ = []
        self.diagnostics_ = {}

    def _require_fitted(self):
        '''Reject prediction before residual history is available.'''
        if self.residual_ is None:
            raise RuntimeError("DirectGBDTComponent has not been fitted; call fit first")

    def fit(self, residual: np.ndarray, h: int):
        '''Train on pre-gate origins and evaluate the margin on later origins.'''
        series = self.as_series(residual, min_length=28)
        h = self.validate_horizon(h)
        if h > self.config.g_hmax:
            raise ValueError("h must not exceed config.g_hmax for direct residual learning")

        self.residual_ = series
        self.booster_ = None
        self.use_g_ = False
        self.g_scale_ = 0.0
        self.relative_improvement_ = None
        n = len(series)
        gate_start = max(28, n - self.config.inner_window)
        train_start = max(28, gate_start - self.config.g_win)
        self.train_origins_ = list(
            range(
                train_start,
                gate_start - self.config.g_hmax + 1,
                self.config.g_step,
            )
        )
        self.gate_origins_ = list(
            range(gate_start, n - self.config.g_hmax + 1, self.config.g_step)
        )
        self.diagnostics_ = {
            "reason": "insufficient_training_samples",
            "n_train": 0,
            "n_gate": 0,
            "baseline_mae": None,
            "gated_mae": None,
        }

        X_train, y_train = self.direct_multistep_dataset(
            series, self.train_origins_, self.config.g_hmax
        )
        X_gate, y_gate = self.direct_multistep_dataset(
            series, self.gate_origins_, self.config.g_hmax
        )
        self.diagnostics_["n_train"] = len(y_train)
        self.diagnostics_["n_gate"] = len(y_gate)
        if len(y_train) < self.config.g_min_train:
            return self
        if len(y_gate) < self.config.g_min_valid:
            self.diagnostics_["reason"] = "insufficient_gate_samples"
            return self

        self.booster_ = self.fit_gbm(
            X_train,
            y_train,
            self.config,
            n_valid=min(84, len(y_train) // 4),
        )
        gate_prediction = self.booster_.predict(X_gate)
        baseline_mae = float(np.mean(np.abs(y_gate)))
        gated_mae = float(np.mean(np.abs(y_gate - gate_prediction)))
        self.diagnostics_.update(
            {"baseline_mae": baseline_mae, "gated_mae": gated_mae}
        )
        if baseline_mae <= 1e-12:
            self.diagnostics_["reason"] = "zero_baseline_error"
            self.relative_improvement_ = 0.0
            return self

        relative = (baseline_mae - gated_mae) / baseline_mae
        self.relative_improvement_ = float(relative)
        if relative > self.config.margin:
            self.use_g_ = True
            self.g_scale_ = min(1.0, relative / self.config.g_rel_scale)
            self.diagnostics_["reason"] = "gate_open"
        else:
            self.diagnostics_["reason"] = "margin_not_exceeded"
        return self

    def predict(self, h: int) -> np.ndarray:
        '''Return scaled direct residual predictions or zeros when gated off.'''
        self._require_fitted()
        h = self.validate_horizon(h)
        if h > self.config.g_hmax:
            raise ValueError("h must not exceed config.g_hmax")
        if not self.use_g_ or self.booster_ is None:
            return np.zeros(h)
        base = self.residual_base_features(self.residual_, len(self.residual_))
        rows = np.asarray([base + [horizon] for horizon in range(1, h + 1)])
        return self.g_scale_ * self.booster_.predict(rows)

    def predict_path(self) -> np.ndarray:
        '''Return fitted one-step residual contributions for exogenous staging.'''
        self._require_fitted()
        n = len(self.residual_)
        path = np.zeros(n)
        if not self.use_g_ or self.booster_ is None:
            return path
        rows = [
            self.residual_base_features(self.residual_, origin) + [1]
            for origin in range(14, n)
        ]
        path[14:] = self.g_scale_ * self.booster_.predict(np.asarray(rows))
        return path

    def _info(self) -> str:
        '''Return residual component metadata.'''
        return "DirectGBDTComponent: direct multi-step LightGBM with a held-out margin gate"
