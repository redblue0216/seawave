# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''Module Introduction
-------------------
Abstract contract for trend components. A fitted trend retains its in-sample
path and produces an h-step extrapolation through ``predict``.
'''

from abc import ABC, abstractmethod
import numpy as np
from seawave.base import MetaRequestMethod
from seawave.mixin import CoreMixin, UtilMixin


class BaseTrendComponent(
    ABC, CoreMixin, metaclass=MetaRequestMethod, enforce_abstract=False
):
    '''Standardize trend fitting, extrapolation, state, and metadata.'''

    def _set_fitted(self, trend: np.ndarray):
        '''Store a fitted in-sample path and its training length.'''
        self.trend_ = np.asarray(trend, dtype=float)
        self.n_ = len(self.trend_)

    def _require_fitted(self, h: int) -> int:
        '''Validate forecast state and horizon before extrapolation.'''
        if getattr(self, "trend_", None) is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted; call fit first"
            )
        return UtilMixin.validate_horizon(h)

    @abstractmethod
    def fit(self, y: np.ndarray):
        '''Fit a trend to one observed series.'''

    @abstractmethod
    def predict(self, h: int) -> np.ndarray:
        '''Return h out-of-sample trend values.'''
