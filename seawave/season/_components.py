# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------

FourierSeasonComponent performs closed-form seasonal projection after a trend
candidate has been fitted. Fitting seasonality on y-T rather than on y prevents
the trend channel and periodic channel from competing for the same signal.

Mathematical principle
----------------------
For harmonics k=1,...,K, the design matrix F contains:

    $F_k(t) = [sin(2*pi*k*t/period), cos(2*pi*k*t/period)]$

The coefficients are estimated with a small ridge stabilizer:

    $beta_star = (F^T F + ridge I)^(-1) F^T (y - T_star)$

JAX computes both the trigonometric design and the linear solve. NumPy serves
only as the public array bridge.

Usage examples
--------------
.. code-block:: python

    import numpy as np
    from seawave.season import FourierSeasonComponent

    t = np.arange(56)
    residual = np.sin(2 * np.pi * t / 7)
    component = FourierSeasonComponent(period=7, K=3)
    component.fit(residual=residual, t=t)
    future = component.predict(np.arange(56, 70))
'''

import numpy as np

from seawave.mixin import AlgorithmMixin, UtilMixin
from seawave.season._base import BaseSeasonComponent


class FourierSeasonComponent(BaseSeasonComponent, AlgorithmMixin, UtilMixin):
    '''Class Introduction:

    FourierSeasonComponent learns periodic sine and cosine coefficients from a
    trend residual and projects them at arbitrary integer or fractional times.

    :parameters:
        - period (int) - Positive seasonal period.
        - K (int) - Positive number of Fourier harmonics.
        - ridge (float) - Positive diagonal stabilizer for the projection.
    '''

    def __init__(self, period: int = 7, K: int = 3, ridge: float = 1e-6):
        '''Initialize an unfitted Fourier projection.'''
        if not isinstance(period, int) or isinstance(period, bool) or period < 1:
            raise ValueError("period must be a positive integer")
        if not isinstance(K, int) or isinstance(K, bool) or K < 1:
            raise ValueError("K must be a positive integer")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.period = period
        self.K = K
        self.ridge = float(ridge)
        self.beta_ = None
        self.seasonal_ = None

    def fit(self, residual: np.ndarray, t: np.ndarray):
        '''Fit Fourier coefficients to one trend-residual vector.'''
        values = self.as_series(residual)
        indices = self.as_time_indices(t)
        if len(values) != len(indices):
            raise ValueError("residual and t must have the same length")
        if len(values) < 2 * self.K:
            raise ValueError("residual must contain at least 2*K observations")
        design = self.fourier_terms(indices, self.period, self.K)
        self.beta_ = self.fourier_projection(design, values, self.ridge)
        self.seasonal_ = design @ self.beta_
        return self

    def predict(self, t: np.ndarray) -> np.ndarray:
        '''Project seasonal values at the supplied future or historical times.'''
        self._require_fitted()
        indices = self.as_time_indices(t)
        design = self.fourier_terms(indices, self.period, self.K)
        return design @ self.beta_

    def _info(self) -> str:
        '''Return seasonal component metadata.'''
        return f"FourierSeasonComponent: period={self.period}, K={self.K}"
