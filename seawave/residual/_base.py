# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''Module Introduction
-------------------
Abstract contract for direct multi-step residual learning components.
'''

from abc import ABC, abstractmethod
import numpy as np
from seawave.base import MetaRequestMethod
from seawave.mixin import CoreMixin


class BaseResidualComponent(ABC, CoreMixin, metaclass=MetaRequestMethod, enforce_abstract=False):
    '''Standardize residual fitting, direct forecasting, and metadata.'''

    @abstractmethod
    def fit(self, residual: np.ndarray, h: int):
        '''Fit the residual model for a required forecast horizon.'''

    @abstractmethod
    def predict(self, h: int) -> np.ndarray:
        '''Return h direct residual predictions.'''
