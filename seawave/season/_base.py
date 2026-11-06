# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''Module Introduction
-------------------
Abstract contract for seasonal components fitted on trend residuals.
'''

from abc import ABC, abstractmethod
import numpy as np
from seawave.base import MetaRequestMethod
from seawave.mixin import CoreMixin


class BaseSeasonComponent(
    ABC, CoreMixin, metaclass=MetaRequestMethod, enforce_abstract=False
):
    '''Standardize seasonal fitting, projection, state, and metadata.'''

    def _require_fitted(self):
        '''Reject seasonal projection before coefficients are available.'''
        if getattr(self, "beta_", None) is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted; call fit first"
            )

    @abstractmethod
    def fit(self, residual: np.ndarray, t: np.ndarray):
        '''Fit seasonal coefficients on trend residuals.'''

    @abstractmethod
    def predict(self, t: np.ndarray) -> np.ndarray:
        '''Project seasonal values at the supplied time indices.'''
