# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------

This module defines reusable SeaWave Mixin capabilities. JAX is the primary
engine for dense HP linear systems, Fourier bases, and seasonal projections;
LightGBM is the residual and promotion learning engine. NumPy is the validation,
feature-construction, and array-bridge layer. Float64 is enabled before any JAX
array creation because float32 introduces visible errors in high-lambda HP
systems and periodic trigonometric evaluations.

Mathematical principle
----------------------
The HP trend solves $(I + lambda K^T K)T = y$, where K is the second-order
difference matrix. The Fourier basis contains sin(2*pi*k*t/period) and
cos(2*pi*k*t/period) for k=1,...,K. Direct residual samples map seven lags,
rolling statistics, and horizon to R[origin+horizon-1]. Promotion columns follow
the reference order current, lag-1, ..., lag-(p-1).

Class Description
-----------------
(1) CoreMixin: common component metadata capability.
(2) AlgorithmMixin: reusable JAX and chronological LightGBM operations.
(3) UtilMixin: validation, metrics, and backtest window utilities.
'''

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
import lightgbm as lgb
import numpy as np

jax.config.update("jax_enable_x64", True)


def _hp_filter_jax(y: jnp.ndarray, lamb: float) -> jnp.ndarray:
    '''Solve the dense HP system using a JAX linear solver.'''
    n = y.shape[0]
    rows = jnp.arange(n - 2)
    second_difference = jnp.zeros((n - 2, n), dtype=y.dtype)
    second_difference = second_difference.at[rows, rows].set(1.0)
    second_difference = second_difference.at[rows, rows + 1].set(-2.0)
    second_difference = second_difference.at[rows, rows + 2].set(1.0)
    system = jnp.eye(n, dtype=y.dtype) + lamb * (
        second_difference.T @ second_difference
    )
    return jnp.linalg.solve(system, y)


@partial(jit, static_argnames=("period", "K"))
def _fourier_terms_jax(t: jnp.ndarray, period: int, K: int) -> jnp.ndarray:
    '''Construct a JIT-compiled Fourier design matrix.'''
    columns = []
    for k in range(1, K + 1):
        columns.append(jnp.sin(2.0 * jnp.pi * k * t / period))
        columns.append(jnp.cos(2.0 * jnp.pi * k * t / period))
    return jnp.column_stack(columns)


@partial(jit, static_argnames=("ridge",))
def _fourier_projection_jax(
    design: jnp.ndarray, residual: jnp.ndarray, ridge: float
) -> jnp.ndarray:
    '''Solve the ridge-stabilized normal equations for Fourier coefficients.'''
    n_features = design.shape[1]
    regularizer = ridge * jnp.eye(n_features, dtype=design.dtype)
    return jnp.linalg.solve(design.T @ design + regularizer, design.T @ residual)


class CoreMixin:
    '''Provide common metadata behavior for algorithm components.'''

    def _info(self) -> str:
        '''Return the component class name as default metadata.'''
        return self.__class__.__name__


class AlgorithmMixin:
    '''Provide reusable JAX HP and Fourier numerical operations.'''

    @staticmethod
    def hp_filter(y: np.ndarray, lamb: float = 1600.0) -> np.ndarray:
        '''Return the dense HP trend for a one-dimensional series.'''
        series = UtilMixin.as_series(y, min_length=3)
        if lamb <= 0:
            raise ValueError("lamb must be positive")
        return np.asarray(_hp_filter_jax(jnp.asarray(series), float(lamb)))

    @staticmethod
    def fourier_terms(t: np.ndarray, period: int, K: int) -> np.ndarray:
        '''Bridge time indices to the JIT Fourier basis and return NumPy.'''
        indices = UtilMixin.as_time_indices(t)
        if period < 1 or K < 1:
            raise ValueError("period and K must be positive integers")
        terms = _fourier_terms_jax(jnp.asarray(indices), int(period), int(K))
        return np.asarray(terms)

    @staticmethod
    def fourier_projection(
        design: np.ndarray, residual: np.ndarray, ridge: float
    ) -> np.ndarray:
        '''Return JAX-solved Fourier coefficients as a NumPy vector.'''
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        coefficients = _fourier_projection_jax(
            jnp.asarray(design), jnp.asarray(residual), float(ridge)
        )
        return np.asarray(coefficients)

    @staticmethod
    def fit_gbm(X: np.ndarray, y: np.ndarray, config, n_valid: int = 84):
        '''Train LightGBM with a chronological train/validation row split.'''
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=float)
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError("X must be a matrix and y must be a vector")
        if len(features) != len(labels) or len(features) < 2:
            raise ValueError("X and y must contain the same number of at least two rows")
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise ValueError("LightGBM inputs must contain only finite values")
        n_valid = min(max(1, int(n_valid)), len(features) - 1)
        train = lgb.Dataset(features[:-n_valid], label=labels[:-n_valid])
        valid = lgb.Dataset(features[-n_valid:], label=labels[-n_valid:])
        return lgb.train(
            config.lgb_params,
            train,
            num_boost_round=config.lgb_num_boost_round,
            valid_sets=[valid],
            callbacks=[
                lgb.early_stopping(config.lgb_early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

    @staticmethod
    def residual_base_features(residual: np.ndarray, origin: int) -> list[float]:
        '''Construct seven lags, two rolling means, and one rolling standard deviation.'''
        series = UtilMixin.as_series(residual)
        if origin < 14 or origin > len(series):
            raise ValueError("origin must leave at least 14 historical residuals")
        window = series[origin - 14 : origin]
        return [
            series[origin - 1],
            series[origin - 2],
            series[origin - 3],
            series[origin - 4],
            series[origin - 5],
            series[origin - 6],
            series[origin - 7],
            window[-7:].mean(),
            window.mean(),
            window.std(),
        ]

    @staticmethod
    def direct_multistep_dataset(
        residual: np.ndarray, origins, hmax: int
    ) -> tuple[np.ndarray, np.ndarray]:
        '''Construct direct rows with horizon as a feature and future residual as target.'''
        series = UtilMixin.as_series(residual)
        if hmax < 1:
            raise ValueError("hmax must be positive")
        rows = []
        targets = []
        for origin in origins:
            if origin < 14 or origin + hmax > len(series):
                raise ValueError("origin and hmax exceed the available residual history")
            base = AlgorithmMixin.residual_base_features(series, origin)
            for horizon in range(1, hmax + 1):
                rows.append(base + [horizon])
                targets.append(series[origin + horizon - 1])
        return (
            np.asarray(rows, dtype=float).reshape(-1, 11),
            np.asarray(targets, dtype=float),
        )

    @staticmethod
    def promo_lag_cols(promo: np.ndarray, lag: int) -> np.ndarray:
        '''Construct current through lag-(lag-1) features with zero history padding.'''
        series = UtilMixin.as_series(promo)
        if lag < 1:
            raise ValueError("lag must be positive")
        padded = np.concatenate([np.zeros(lag), series])
        return np.column_stack(
            [padded[i : len(series) + i] for i in range(lag, 0, -1)]
        )


class UtilMixin:
    '''Provide reusable validation, metrics, and historical windows.'''

    @staticmethod
    def as_series(y: np.ndarray, min_length: int = 1) -> np.ndarray:
        '''Return a finite float vector or raise a clear input error.'''
        series = np.asarray(y, dtype=float)
        if series.ndim != 1:
            raise ValueError("y must be a one-dimensional array")
        if len(series) < min_length:
            raise ValueError(f"y must contain at least {min_length} observations")
        if not np.isfinite(series).all():
            raise ValueError("y must contain only finite values")
        return series

    @staticmethod
    def as_time_indices(t: np.ndarray) -> np.ndarray:
        '''Return a finite one-dimensional time-index vector.'''
        indices = np.asarray(t, dtype=float)
        if indices.ndim != 1 or not np.isfinite(indices).all():
            raise ValueError("t must be a finite one-dimensional array")
        return indices

    @staticmethod
    def validate_horizon(h: int) -> int:
        '''Require a positive integer forecast horizon.'''
        if not isinstance(h, int) or isinstance(h, bool) or h < 1:
            raise ValueError("h must be a positive integer")
        return h

    @staticmethod
    def backtest_origins(
        n: int, inner_window: int, n_win: int, min_train: int = 150
    ) -> list[int]:
        '''Return staggered origins and reject an empty backtest design.'''
        origins = [
            n - inner_window * k
            for k in range(1, n_win + 1)
            if n - inner_window * k >= min_train
        ]
        if not origins:
            raise ValueError(
                "No valid inner backtest windows; provide more history or reduce "
                "inner_window, n_win, or min_train"
            )
        return origins

    @staticmethod
    def normalized_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        '''Compute MAE normalized by the validation segment standard deviation.'''
        return float(np.mean(np.abs(actual - predicted)) / (np.std(actual) + 1e-9))
