# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------

This module implements SeaWave trend candidates, worst-window backtest scoring,
and the Top-K Softmax ensemble. The structural selector uses forecast risk
rather than residual ACF because a trend that absorbs a slow wave can destroy
the diagnostic distinction between candidates.

Mathematical principle
----------------------
For candidate j and staggered window k:

    $s_j = max_k(MAE_j^(k) / sigma^(k)) * (1 + gamma * pen_j)$

The worst window is selected because extrapolation failures are tail-driven.
For the K lowest-risk candidates:

    $w_j = exp(-s_j / tau) / sum_l exp(-s_l / tau)$
    $T_star = sum_j w_j T_j$

The implementation subtracts the minimum selected score before exponentiation,
which is algebraically equivalent and numerically stable.

Algorithm flow
--------------
.. code-block:: text

    y -> candidate pool -> staggered backtests -> penalized worst scores
      -> Top-K stable Softmax -> full-history fits -> blended trend

Class Description
-----------------
(1) HPTrendComponent: damped HP trend with JAX dense-system fitting.
(2) PolyTrendComponent: global polynomial trend.
(3) RWTrendComponent: local-window linear extrapolation.
(4) GLinTrendComponent: conservative global linear extrapolation.
(5) TrendEnsembleComponent: multi-window selection and soft trend ensemble.
'''

from dataclasses import dataclass
from typing import Callable

import numpy as np

from seawave.config import SeaWaveConfig
from seawave.mixin import AlgorithmMixin, UtilMixin
from seawave.trend._base import BaseTrendComponent


class HPTrendComponent(BaseTrendComponent, AlgorithmMixin, UtilMixin):
    '''Class Introduction:

    HPTrendComponent minimizes ||y-T||^2 + lambda*||Delta^2 T||^2. The last
    20 fitted values estimate the terminal slope. Forecasts use
    T[n+h] = T[n] + slope*damp*h, so smaller damping values represent a stronger
    mean-reversion prior.

    :parameters:
        - lamb (float) - Positive HP smoothing penalty.
        - damp (float) - Terminal slope commitment in the interval (0, 1].
        - slope_window (int) - Number of terminal values used for slope fitting.
    '''

    def __init__(self, lamb: float = 1600.0, damp: float = 1.0, slope_window: int = 20):
        '''Initialize an unfitted HP trend candidate.'''
        if lamb <= 0:
            raise ValueError("lamb must be positive")
        if not 0 < damp <= 1:
            raise ValueError("damp must be in the interval (0, 1]")
        if slope_window < 2:
            raise ValueError("slope_window must be at least 2")
        self.lamb = float(lamb)
        self.damp = float(damp)
        self.slope_window = slope_window
        self.trend_ = None
        self.slope_ = None

    def fit(self, y: np.ndarray):
        '''Fit the HP path and estimate its terminal linear slope.'''
        series = self.as_series(y, min_length=3)
        trend = self.hp_filter(series, lamb=self.lamb)
        width = min(self.slope_window, len(trend))
        coefficients = np.polyfit(np.arange(width), trend[-width:], 1)
        self.slope_ = float(coefficients[0])
        self._set_fitted(trend)
        return self

    def predict(self, h: int) -> np.ndarray:
        '''Extrapolate the terminal HP level with the damped slope.'''
        h = self._require_fitted(h)
        steps = np.arange(1, h + 1)
        return self.trend_[-1] + self.slope_ * self.damp * steps

    def _info(self) -> str:
        '''Return candidate metadata.'''
        return f"HPTrendComponent: lambda={self.lamb}, damp={self.damp}"


class PolyTrendComponent(BaseTrendComponent, UtilMixin):
    '''Class Introduction:

    PolyTrendComponent fits a global polynomial to all observations. The
    default degree two candidate captures global curvature and receives a higher
    flexibility penalty than global linear extrapolation.
    '''

    def __init__(self, degree: int = 2):
        '''Initialize an unfitted polynomial trend.'''
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
            raise ValueError("degree must be a positive integer")
        self.degree = degree
        self.trend_ = None
        self.coefficients_ = None

    def fit(self, y: np.ndarray):
        '''Fit polynomial coefficients on time indices 0 through n-1.'''
        series = self.as_series(y, min_length=self.degree + 1)
        t = np.arange(len(series))
        self.coefficients_ = np.polyfit(t, series, self.degree)
        self._set_fitted(np.polyval(self.coefficients_, t))
        return self

    def predict(self, h: int) -> np.ndarray:
        '''Evaluate the fitted polynomial at n through n+h-1.'''
        h = self._require_fitted(h)
        return np.polyval(self.coefficients_, np.arange(self.n_, self.n_ + h))

    def _info(self) -> str:
        '''Return candidate metadata.'''
        return f"PolyTrendComponent: degree={self.degree}"


class RWTrendComponent(BaseTrendComponent, UtilMixin):
    '''Class Introduction:

    RWTrendComponent estimates a line from a recent window. Despite the
    historical candidate name ``rw``, this is local linear extrapolation rather
    than a random-walk point forecast.
    '''

    def __init__(self, window: int = 90):
        '''Initialize the maximum local lookback length.'''
        if not isinstance(window, int) or isinstance(window, bool) or window < 2:
            raise ValueError("window must be an integer of at least 2")
        self.window = window
        self.trend_ = None
        self.coefficients_ = None
        self.effective_window_ = None

    def fit(self, y: np.ndarray):
        '''Fit a line to the recent window without exceeding available history.'''
        series = self.as_series(y, min_length=2)
        n = len(series)
        width = min(n, self.window, max(30, n // 3))
        self.effective_window_ = width
        self.coefficients_ = np.polyfit(
            np.arange(width), series[-width:], 1
        )
        self._set_fitted(np.polyval(self.coefficients_, np.arange(n)))
        return self

    def predict(self, h: int) -> np.ndarray:
        '''Continue the local line at future global time indices.'''
        h = self._require_fitted(h)
        return np.polyval(self.coefficients_, np.arange(self.n_, self.n_ + h))

    def _info(self) -> str:
        '''Return candidate metadata.'''
        return f"RWTrendComponent: window={self.window}"


class GLinTrendComponent(BaseTrendComponent, UtilMixin):
    '''Class Introduction:

    GLinTrendComponent fits one global line. Its low flexibility penalty makes
    it a conservative endpoint when flexible trends show unstable backtest risk.
    '''

    def __init__(self):
        '''Initialize an unfitted global linear trend.'''
        self.trend_ = None
        self.coefficients_ = None

    def fit(self, y: np.ndarray):
        '''Fit one line to all observations.'''
        series = self.as_series(y, min_length=2)
        t = np.arange(len(series))
        self.coefficients_ = np.polyfit(t, series, 1)
        self._set_fitted(np.polyval(self.coefficients_, t))
        return self

    def predict(self, h: int) -> np.ndarray:
        '''Continue the global line for h steps.'''
        h = self._require_fitted(h)
        return np.polyval(self.coefficients_, np.arange(self.n_, self.n_ + h))

    def _info(self) -> str:
        '''Return candidate metadata.'''
        return "GLinTrendComponent: global linear trend"


@dataclass(frozen=True)
class TrendCandidate:
    '''Define a named component factory and its selection-layer penalty.'''

    name: str
    factory: Callable[[], BaseTrendComponent]
    penalty: float


@dataclass(frozen=True)
class TrendScore:
    '''Store per-window risk and the final penalized worst-window score.'''

    name: str
    penalty: float
    window_scores: tuple[float, ...]
    score: float


def build_trend_candidates(config: SeaWaveConfig) -> list[TrendCandidate]:
    '''Build the eight default candidates from the active configuration.'''
    candidates = []
    for lamb in config.hp_lambda_grid:
        for damp in config.damp_grid:
            if lamb == 1600 and damp != 1.0:
                continue
            penalty = 1.0 if lamb == 1600 else 0.5
            name = f"hp_l{int(lamb)}_d{int(damp * 10)}"
            candidates.append(
                TrendCandidate(
                    name=name,
                    factory=lambda lamb=lamb, damp=damp: HPTrendComponent(
                        lamb=lamb, damp=damp
                    ),
                    penalty=penalty,
                )
            )
    candidates.extend(
        [
            TrendCandidate("poly2", PolyTrendComponent, 0.8),
            TrendCandidate(
                "rw", lambda: RWTrendComponent(window=config.rw_window), 0.7
            ),
            TrendCandidate("glin", GLinTrendComponent, 0.3),
        ]
    )
    return candidates


def score_trend_candidates(
    y: np.ndarray,
    candidates: list[TrendCandidate],
    inner_window: int = 84,
    n_win: int = 3,
    gamma: float = 0.1,
    min_train: int = 150,
) -> dict[str, TrendScore]:
    '''Score every candidate on identical staggered historical windows.'''
    series = UtilMixin.as_series(y, min_length=3)
    if not candidates:
        raise ValueError("At least one trend candidate is required")
    origins = UtilMixin.backtest_origins(
        len(series), inner_window, n_win, min_train=min_train
    )
    scores = {}
    for candidate in candidates:
        window_scores = []
        for origin in origins:
            model = candidate.factory()
            model.fit(series[:origin])
            predicted = model.predict(inner_window)
            actual = series[origin : origin + inner_window]
            window_scores.append(UtilMixin.normalized_mae(actual, predicted))
        worst = max(window_scores)
        scores[candidate.name] = TrendScore(
            name=candidate.name,
            penalty=candidate.penalty,
            window_scores=tuple(window_scores),
            score=worst * (1.0 + gamma * candidate.penalty),
        )
    return scores


def select_trend_candidates(
    scores: dict[str, TrendScore], topk: int, tau: float
) -> tuple[list[TrendScore], np.ndarray]:
    '''Select the lowest scores and compute stable Softmax ensemble weights.'''
    if not scores:
        raise ValueError("scores must not be empty")
    if topk < 1 or tau <= 0:
        raise ValueError("topk and tau must be positive")
    ranked = sorted(scores.values(), key=lambda item: item.score)
    selected = ranked[: min(topk, len(ranked))]
    raw_scores = np.asarray([item.score for item in selected])
    logits = -(raw_scores - raw_scores.min()) / tau
    weights = np.exp(logits)
    weights /= weights.sum()
    return selected, weights


class TrendEnsembleComponent(BaseTrendComponent, UtilMixin):
    '''Class Introduction:

    TrendEnsembleComponent executes the complete P0 trend gate. It backtests all
    candidates, selects Top-K, fits only the selected candidates on full history,
    and blends both fitted and future trend paths.

    :example:
        >>> import numpy as np
        >>> from seawave import SeaWaveConfig
        >>> from seawave.trend import TrendEnsembleComponent
        >>> y = np.linspace(1.0, 5.0, 400)
        >>> component = TrendEnsembleComponent(SeaWaveConfig())
        >>> component.fit(y)
        >>> forecast = component.predict(14)
    '''

    def __init__(
        self,
        config: SeaWaveConfig | None = None,
        candidates: list[TrendCandidate] | None = None,
        min_train: int = 150,
    ):
        '''Initialize candidate selection and ensemble configuration.'''
        self.config = SeaWaveConfig() if config is None else config
        if not isinstance(self.config, SeaWaveConfig):
            raise TypeError("config must be a SeaWaveConfig instance")
        self.candidates = (
            build_trend_candidates(self.config) if candidates is None else list(candidates)
        )
        if not self.candidates:
            raise ValueError("At least one trend candidate is required")
        if min_train < 1:
            raise ValueError("min_train must be positive")
        self.min_train = min_train
        self.trend_ = None
        self.scores_ = None
        self.selected_names_ = None
        self.weights_ = None
        self.models_ = None

    def fit(self, y: np.ndarray):
        '''Backtest, select, fully fit, and blend trend candidates.'''
        series = self.as_series(y, min_length=3)
        self.scores_ = score_trend_candidates(
            series,
            self.candidates,
            inner_window=self.config.inner_window,
            n_win=self.config.n_win,
            gamma=self.config.gamma,
            min_train=self.min_train,
        )
        selected, self.weights_ = select_trend_candidates(
            self.scores_, self.config.topk, self.config.tau
        )
        candidate_by_name = {candidate.name: candidate for candidate in self.candidates}
        self.selected_names_ = [item.name for item in selected]
        self.models_ = []
        fitted_paths = []
        for name in self.selected_names_:
            model = candidate_by_name[name].factory()
            model.fit(series)
            self.models_.append(model)
            fitted_paths.append(model.trend_)
        self._set_fitted(
            np.sum(np.asarray(fitted_paths) * self.weights_[:, None], axis=0)
        )
        return self

    def predict(self, h: int) -> np.ndarray:
        '''Return the weighted h-step extrapolation of selected candidates.'''
        h = self._require_fitted(h)
        predictions = [model.predict(h) for model in self.models_]
        return np.sum(np.asarray(predictions) * self.weights_[:, None], axis=0)

    def _info(self) -> str:
        '''Return ensemble metadata.'''
        return "TrendEnsembleComponent: worst-window Top-K Softmax trend ensemble"
