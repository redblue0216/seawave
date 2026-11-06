# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------

PromoLagComponent implements SeaWave's two-stage exogenous model. It is fitted
only after the structural and residual main model has produced its residual:

    $R_2 = y - T_star - S_beta - g$
    $R_2 ~ LightGBM(promo_0, promo_lag_1, ..., promo_lag_(p-1))$

The future promotion contribution is added directly to the main forecast. This
prevents the trend channel from absorbing and distorting known promotion signals.

Usage examples
--------------
.. code-block:: python

    from seawave import SeaWaveConfig
    from seawave.exog import PromoLagComponent

    component = PromoLagComponent(SeaWaveConfig())
    component.fit(residual=main_residual, promo=promo)
    contribution = component.predict(promo=promo, n=len(main_residual), h=14)

Passing promo=None disables the component and produces an exact zero vector.
'''

import numpy as np

from seawave.config import SeaWaveConfig
from seawave.mixin import AlgorithmMixin, UtilMixin
from seawave.exog._base import BaseExogComponent


class PromoLagComponent(BaseExogComponent, AlgorithmMixin, UtilMixin):
    '''Class Introduction:

    PromoLagComponent fits lagged promotion indicators to residuals left by the
    main model. Future inputs must contain at least n+h known promotion values.

    :parameters:
        - config (SeaWaveConfig) - Promotion lag and LightGBM settings.
    '''

    def __init__(self, config: SeaWaveConfig | None = None):
        '''Initialize an unfitted exogenous component.'''
        self.config = SeaWaveConfig() if config is None else config
        if not isinstance(self.config, SeaWaveConfig):
            raise TypeError("config must be a SeaWaveConfig instance")
        self.booster_ = None
        self.enabled_ = False
        self.n_ = None

    def fit(self, residual: np.ndarray, promo: np.ndarray | None = None):
        '''Fit the second-stage model or explicitly disable it for promo=None.'''
        series = self.as_series(residual, min_length=2)
        self.n_ = len(series)
        self.booster_ = None
        self.enabled_ = promo is not None
        if promo is None:
            return self
        promotion = self.as_series(promo)
        if len(promotion) < self.n_:
            raise ValueError("promo must contain at least as many values as residual")
        features = self.promo_lag_cols(promotion, self.config.promo_lag)
        self.booster_ = self.fit_gbm(
            features[: self.n_],
            series,
            self.config,
            n_valid=min(84, self.n_ // 4),
        )
        return self

    def predict(
        self, promo: np.ndarray | None, n: int, h: int
    ) -> np.ndarray:
        '''Return future promotion contributions or zeros when disabled.'''
        if self.n_ is None:
            raise RuntimeError("PromoLagComponent has not been fitted; call fit first")
        h = self.validate_horizon(h)
        if n != self.n_:
            raise ValueError("n must equal the residual length used during fit")
        if not self.enabled_:
            return np.zeros(h)
        if promo is None:
            raise ValueError("promo is required because the component was fitted with it")
        promotion = self.as_series(promo)
        if len(promotion) < n + h:
            raise ValueError("promo must contain at least n+h values")
        features = self.promo_lag_cols(promotion, self.config.promo_lag)
        return self.booster_.predict(features[n : n + h])

    def _info(self) -> str:
        '''Return exogenous component metadata.'''
        return f"PromoLagComponent: two-stage model with {self.config.promo_lag} lags"
