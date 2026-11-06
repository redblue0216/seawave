# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''Module Introduction
-------------------
BaseSeaWaveComposite combines lifecycle management with the forecast contract
implemented by concrete algorithm composites.
'''

from abc import ABC, abstractmethod
import numpy as np
from seawave.base import BaseComposite


class BaseSeaWaveComposite(BaseComposite, ABC, enforce_abstract=False):
    '''Define the common SeaWave orchestration interface.'''

    @abstractmethod
    def forecast(self, y: np.ndarray, h: int, promo: np.ndarray | None = None):
        '''Return an h-step forecast and its diagnostic dictionary.'''

    @abstractmethod
    def _info(self) -> str:
        '''Return composite metadata.'''
