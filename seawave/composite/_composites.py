# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------

SeaWaveComposite assembles the complete structural and residual forecast. Its
product hypothesis space is H = T x S x G x X: trend, Fourier seasonality,
direct GBDT residual dynamics, and exogenous promotion effects. The conceptual
mixed block-coordinate objective is:

    $min ||y - T - S*beta - g - X*w||^2 + Omega(g) + lambda*||Delta^2 T||^2$

The implementation solves this through ordered structural selection, seasonal
projection, residual gating, and exogenous submodel fitting rather than through
one end-to-end gradient optimizer.

Engineering flow closure
------------------------
.. code-block:: text

    Interface
        -> Composite
            -> TrendComponent: multi-window Top-K selection
            -> SeasonComponent: projection for each selected trend
            -> ResidualComponent: direct multi-step margin gate
            -> ExogComponent: second-stage promotion residual model
        -> T_star + S_star + g_scale*G + X_sub

Composition consistency
-----------------------
The selected Top-K structures are fully fitted before R = y - T_star - S_star
is constructed. G is trained and predicted on this same blended residual, never
on a residual belonging to another candidate. The exogenous target is R-G_path.

Usage examples
--------------
.. code-block:: python

    from seawave import SeaWaveConfig
    from seawave.composite import SeaWaveComposite

    composite = SeaWaveComposite(SeaWaveConfig())
    forecast, info = composite.forecast(y=y_train, h=14, promo=promo)
'''

import numpy as np

from seawave.composite._base import BaseSeaWaveComposite
from seawave.config import SeaWaveConfig
from seawave.diagnostics import ForecastDiagnostics, GateDiagnostics
from seawave.exog import PromoLagComponent
from seawave.mixin import UtilMixin
from seawave.residual import DirectGBDTComponent
from seawave.season import FourierSeasonComponent
from seawave.trend import TrendCandidate, TrendEnsembleComponent


class SeaWaveComposite(BaseSeaWaveComposite, UtilMixin):
    '''Class Introduction:

    SeaWaveComposite executes trend, season, residual, and exogenous components
    in a fixed order and returns their additive forecast with diagnostics.

    :parameters:
        - config (SeaWaveConfig) - Complete algorithm configuration.
        - trend_candidates (list[TrendCandidate] | None) - Optional injected candidates for testing or research.
        - min_train (int) - Minimum history accepted at a trend backtest origin.
    '''

    workflow = ("trend", "season", "residual", "exog")

    def __init__(
        self,
        config: SeaWaveConfig | None = None,
        trend_candidates: list[TrendCandidate] | None = None,
        min_train: int = 150,
    ):
        '''Store configuration and optional trend dependency injection.'''
        self.config = SeaWaveConfig() if config is None else config
        if not isinstance(self.config, SeaWaveConfig):
            raise TypeError("config must be a SeaWaveConfig instance")
        self.trend_candidates = trend_candidates
        self.min_train = min_train
        self.diagnostics_ = None
        self.forecast_ = None

    def forecast(self, y: np.ndarray, h: int, promo: np.ndarray | None = None):
        '''Execute the full SeaWave workflow and return ``(forecast, info)``.'''
        series = self.as_series(y, min_length=3)
        h = self.validate_horizon(h)
        if h > self.config.g_hmax:
            raise ValueError("h must not exceed config.g_hmax")
        n = len(series)
        if promo is not None:
            promotion = self.as_series(promo)
            if len(promotion) < n + h:
                raise ValueError("promo must contain at least n+h values")
        else:
            promotion = None

        trend = TrendEnsembleComponent(
            config=self.config,
            candidates=self.trend_candidates,
            min_train=self.min_train,
        ).fit(series)
        t = np.arange(n)
        future_t = np.arange(n, n + h)
        seasons = []
        fitted_structures = []
        future_structures = []
        for model in trend.models_:
            season = FourierSeasonComponent(
                period=self.config.period, K=self.config.fourier_K
            ).fit(series - model.trend_, t)
            seasons.append(season)
            fitted_structures.append(model.trend_ + season.seasonal_)
            future_structures.append(model.predict(h) + season.predict(future_t))

        structural_fitted = np.sum(
            np.asarray(fitted_structures) * trend.weights_[:, None], axis=0
        )
        structural_forecast = np.sum(
            np.asarray(future_structures) * trend.weights_[:, None], axis=0
        )
        residual = series - structural_fitted

        residual_model = DirectGBDTComponent(self.config).fit(residual, h=h)
        residual_forecast = residual_model.predict(h)
        residual_path = residual_model.predict_path()

        exog_model = PromoLagComponent(self.config).fit(
            residual - residual_path, promo=promotion
        )
        exog_forecast = exog_model.predict(promotion, n=n, h=h)
        forecast = structural_forecast + residual_forecast + exog_forecast

        gate = GateDiagnostics.from_component(residual_model)
        self.diagnostics_ = ForecastDiagnostics(
            top=tuple(trend.selected_names_),
            weights=trend.weights_.copy(),
            scores={name: value.score for name, value in trend.scores_.items()},
            window_scores={
                name: value.window_scores for name, value in trend.scores_.items()
            },
            gate=gate,
        )
        self.trend_ = trend
        self.seasons_ = seasons
        self.structural_fitted_ = structural_fitted
        self.structural_forecast_ = structural_forecast
        self.residual_ = residual
        self.residual_model_ = residual_model
        self.residual_path_ = residual_path
        self.exog_model_ = exog_model
        self.exog_forecast_ = exog_forecast
        self.forecast_ = forecast
        return forecast, self.diagnostics_.to_dict()

    def _info(self) -> str:
        '''Return composite metadata.'''
        return "SeaWaveComposite: trend + Fourier + gated residual + promotion workflow"
