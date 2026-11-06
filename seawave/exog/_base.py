# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''Module Introduction
-------------------
Abstract contract for two-stage exogenous residual components. ``promo=None``
represents an explicitly disabled exogenous channel.
'''

from abc import ABC, abstractmethod
import numpy as np
from seawave.base import MetaRequestMethod
from seawave.mixin import CoreMixin


class BaseExogComponent(
    ABC, CoreMixin, metaclass=MetaRequestMethod, enforce_abstract=False
):
    '''Standardize exogenous fitting, future contribution, and metadata.'''

    @abstractmethod
    def fit(self, residual: np.ndarray, promo: np.ndarray | None = None):
        '''Fit an exogenous model on residuals left by the main model.'''

    @abstractmethod
    def predict(self, promo: np.ndarray | None, n: int, h: int) -> np.ndarray:
        '''Return h future exogenous contributions.'''
