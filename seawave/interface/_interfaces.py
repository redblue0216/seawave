# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# Email: 15021408795@163.com
# License: Apache-2.0
# Copyright (c) 2026 shihua. All rights reserved.

'''
Module Introduction
-------------------
SeaWaveForecastInterface is the function-like public entry point. It delegates
to one SeaWaveComposite and returns ``(forecast, info)``.

- Main functions:

    (1) Accept one finite one-dimensional training series

    (2) Accept known current/future promotion indicators or promo=None

    (3) Return h additive forecasts and serializable diagnostics

Usage examples
--------------
.. code-block:: python

    from seawave import SeaWaveConfig, SeaWaveForecastInterface

    config = SeaWaveConfig(margin=0.05, tau=0.05, topk=2)
    model = SeaWaveForecastInterface(config=config)
    forecast, info = model(y=y_train, h=14, promo=promo)
    print(info["top"], info["weights"])
    print(info["g_gate"])

Return information
------------------
The info dictionary contains top, weights, scores, window_scores, use_g,
g_scale, relative_improvement, and g_gate. The g_gate dictionary additionally
contains sample counts, baseline_mae, gated_mae, and the gate reason.

Precautions
-----------
The promotion path must contain at least n+h values when enabled. The forecast
horizon must not exceed config.g_hmax. Both restrictions prevent silent future
input truncation or unsupported LightGBM horizon extrapolation.

References
----------
SeaWave Design SH V001 and SeaWave Coding Design SH V002.
'''

import numpy as np
from seawave.composite import SeaWaveComposite
from seawave.config import SeaWaveConfig
from seawave.interface._base import BaseSeaWaveInterface


class SeaWaveForecastInterface(BaseSeaWaveInterface):
    '''Provide the stable public forecasting signature.'''

    def __init__(self, config: SeaWaveConfig | None = None):
        '''Create the composite owned by this interface.'''
        self.config = SeaWaveConfig() if config is None else config
        self.composite = SeaWaveComposite(config=self.config)

    def __call__(self, y: np.ndarray, h: int, promo: np.ndarray | None = None):
        '''Delegate a forecasting request to the composite.

        :parameters:
            - y (np.ndarray) - (n,) Finite observed training series.
            - h (int) - Positive horizon no greater than config.g_hmax.
            - promo (np.ndarray | None) - Known path with at least n+h values.

        :return:
            - forecast (np.ndarray) - (h,) Additive SeaWave predictions.
            - info (dict) - Structural selection and residual-gate diagnostics.
        '''
        return self.composite.forecast(y=y, h=h, promo=promo)

    def _info(self) -> str:
        '''Return public interface metadata.'''
        return "SeaWaveForecastInterface: callable SeaWave forecasting entry point"
