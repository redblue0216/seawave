# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''Module Introduction
-------------------
BaseSeaWaveInterface defines the public function-like forecasting contract.
'''

from abc import abstractmethod
import numpy as np
from seawave.base import BaseInterface


class BaseSeaWaveInterface(BaseInterface, enforce_abstract=False):
    '''Define callable forecasting and metadata behavior.'''

    @abstractmethod
    def __call__(self, y: np.ndarray, h: int, promo: np.ndarray | None = None):
        '''Return an h-step forecast and diagnostic information.'''

    @abstractmethod
    def _info(self) -> str:
        '''Return interface metadata.'''
