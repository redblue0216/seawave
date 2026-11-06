# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------

This module centralizes every effective SeaWave hyperparameter in a validated
dataclass. Defaults reproduce the flat reference implementation while making
structural selection, residual gating, LightGBM training, and promotion lags
configurable without editing algorithm internals.

Usage examples
--------------
.. code-block:: python

    from seawave import SeaWaveConfig

    config = SeaWaveConfig(margin=0.05, tau=0.05, topk=2)
    print(config.lgb_params)

Class Description
-----------------
(1) SeaWaveConfig: validated configuration and LightGBM parameter factory.
'''

from dataclasses import dataclass, field


@dataclass
class SeaWaveConfig:
    '''Class Introduction:

    SeaWaveConfig stores structural, gating, GBDT, and exogenous parameters.
    Validation is performed at construction and whenever dataclasses.replace
    creates a modified configuration.

    Structural parameters
    ---------------------
    period=7, fourier_K=3, hp_lambda_grid=[129600.0, 1600.0],
    damp_grid=[0.3, 0.6, 0.9, 1.0], and rw_window=90 define the candidate pool.
    The 1600 HP candidate retains only damp=1.0 and receives penalty 1.0.

    Selection and gate parameters
    -----------------------------
    inner_window=84 and n_win=3 define staggered historical validation.
    gamma=0.10 controls flexibility penalties; topk=3 and tau=0.1 control the
    Softmax ensemble. margin=0.02 and g_rel_scale=0.10 control G activation.

    LightGBM parameters
    -------------------
    learning_rate=0.05, num_leaves=31, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    num_boost_round=3000, early_stopping=50, and random_seed=42 are converted to
    the LightGBM parameter dictionary by the lgb_params property.

    Direct residual and exogenous parameters
    ----------------------------------------
    g_step=3, g_win=200, g_hmax=14, g_min_train=150, and g_min_valid=50 define
    direct sample coverage. promo_lag=3 creates current, lag-1, and lag-2
    promotion indicators. h greater than g_hmax is rejected explicitly.

    :parameters:
        - period (int) - Seasonal period; seven represents weekly daily data.
        - fourier_K (int) - Number of Fourier harmonics.
        - hp_lambda_grid (list[float]) - HP smoothing candidates.
        - damp_grid (list[float]) - Trend extrapolation damping candidates.
        - rw_window (int) - Maximum local linear lookback window.
        - inner_window (int) - Length of each inner backtest window.
        - n_win (int) - Number of staggered backtest windows.
        - gamma (float) - Flexibility penalty coefficient.
        - topk (int) - Number of candidates in the structural ensemble.
        - tau (float) - Positive Softmax temperature.
        - margin (float) - Minimum relative improvement required for G.
        - g_rel_scale (float) - Positive residual shrinkage denominator.
        - promo_lag (int) - Current plus lagged promotion feature count.
    '''

    period: int = 7
    fourier_K: int = 3
    hp_lambda_grid: list[float] = field(
        default_factory=lambda: [129600.0, 1600.0]
    )
    damp_grid: list[float] = field(
        default_factory=lambda: [0.3, 0.6, 0.9, 1.0]
    )
    rw_window: int = 90

    inner_window: int = 84
    n_win: int = 3
    gamma: float = 0.10
    topk: int = 3
    tau: float = 0.1
    margin: float = 0.02
    g_rel_scale: float = 0.10

    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31
    lgb_min_child_samples: int = 20
    lgb_feature_fraction: float = 0.8
    lgb_bagging_fraction: float = 0.8
    lgb_bagging_freq: int = 1
    lgb_num_boost_round: int = 3000
    lgb_early_stopping: int = 50
    random_seed: int = 42

    g_step: int = 3
    g_win: int = 200
    g_hmax: int = 14
    g_min_train: int = 150
    g_min_valid: int = 50

    promo_lag: int = 3

    def __post_init__(self):
        '''Validate values before any algorithm component consumes them.'''
        positive_ints = {
            "period": self.period,
            "fourier_K": self.fourier_K,
            "rw_window": self.rw_window,
            "inner_window": self.inner_window,
            "n_win": self.n_win,
            "topk": self.topk,
            "lgb_num_leaves": self.lgb_num_leaves,
            "lgb_min_child_samples": self.lgb_min_child_samples,
            "lgb_num_boost_round": self.lgb_num_boost_round,
            "lgb_early_stopping": self.lgb_early_stopping,
            "g_step": self.g_step,
            "g_win": self.g_win,
            "g_hmax": self.g_hmax,
            "g_min_train": self.g_min_train,
            "g_min_valid": self.g_min_valid,
            "promo_lag": self.promo_lag,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if self.rw_window < 2:
            raise ValueError("rw_window must be at least 2")
        if self.lgb_num_leaves < 2:
            raise ValueError("lgb_num_leaves must be at least 2")
        if not self.hp_lambda_grid or any(
            value <= 0 for value in self.hp_lambda_grid
        ):
            raise ValueError("hp_lambda_grid must contain positive values")
        if not self.damp_grid or any(
            value <= 0 or value > 1 for value in self.damp_grid
        ):
            raise ValueError("damp_grid values must be in the interval (0, 1]")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.g_rel_scale <= 0:
            raise ValueError("g_rel_scale must be positive")
        if self.lgb_learning_rate <= 0:
            raise ValueError("lgb_learning_rate must be positive")
        if not 0 < self.lgb_feature_fraction <= 1:
            raise ValueError("lgb_feature_fraction must be in (0, 1]")
        if not 0 < self.lgb_bagging_fraction <= 1:
            raise ValueError("lgb_bagging_fraction must be in (0, 1]")
        if self.lgb_bagging_freq < 0:
            raise ValueError("lgb_bagging_freq must be non-negative")
        if self.g_win < 14:
            raise ValueError("g_win must be at least 14")

    @property
    def lgb_params(self):
        '''Return deterministic LightGBM regression parameters.'''
        return {
            "objective": "regression",
            "learning_rate": self.lgb_learning_rate,
            "num_leaves": self.lgb_num_leaves,
            "min_child_samples": self.lgb_min_child_samples,
            "feature_fraction": self.lgb_feature_fraction,
            "bagging_fraction": self.lgb_bagging_fraction,
            "bagging_freq": self.lgb_bagging_freq,
            "verbose": -1,
            "force_row_wise": True,
            "seed": self.random_seed,
        }

    def _info(self) -> str:
        '''Return a concise configuration description.'''
        return "SeaWaveConfig: validated structural, residual, and exogenous settings"
