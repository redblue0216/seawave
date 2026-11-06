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

SeaWave is a daily/weekly business time-series forecasting package built from
explicit structural decomposition and gated GBDT residual learning. It is
designed to make trend commitment, seasonal projection, residual dynamics, and
promotion effects separately inspectable in one additive forecast.

- Design mode:

    (1) Composite pattern: Interface -> Composite -> Component workflow

    (2) Mixin pattern: reusable numerical and feature capabilities

    (3) Metaprogramming: abstract contracts and controlled object creation

- Key points:

    (1) JAX is the numerical engine for HP systems and Fourier projection

    (2) LightGBM is the learning engine for residual and promotion models

    (3) NumPy is the array, validation, metric, and LightGBM bridge

    (4) __all__ and underscore-prefixed modules protect public boundaries

Architecture
------------
.. code-block:: text

    Interface layer
        SeaWaveForecastInterface
            |
    Composite layer
        SeaWaveComposite
            |
    Component layer                 Mixin implementation layer
        TrendEnsembleComponent ----> AlgorithmMixin / UtilMixin
        FourierSeasonComponent ----> AlgorithmMixin
        DirectGBDTComponent ------> AlgorithmMixin / UtilMixin
        PromoLagComponent --------> AlgorithmMixin / UtilMixin

Algorithm flow
--------------
.. code-block:: text

    y[1:n], known promotion path
        -> P0: HP damping / polynomial / local / global trend pool
        -> P0: staggered worst-window risk, penalties, Top-K Softmax
        -> P1: Fourier projection on each selected trend residual
        -> P2: direct multi-step G with held-out margin gate and shrinkage
        -> P3: promotion submodel fitted on the remaining residual
        -> y_hat[n+1:n+h] = T_star + S_star + g_scale*G + X_sub

Mathematical principle
----------------------
SeaWave combines the product hypothesis space H = T x S x G x X with structured
block-coordinate fitting. Its conceptual objective is:

    $min ||y - T - S*beta - g - X*w||^2 + Omega(g) + lambda*||Delta^2 T||^2$

Trend selection uses predictive risk rather than residual ACF because trend
absorption of a slow wave can make residual diagnostics indistinguishable.

Usage examples
--------------
.. code-block:: python

    from seawave import SeaWaveConfig, SeaWaveForecastInterface

    config = SeaWaveConfig(margin=0.05, tau=0.05, topk=2)
    model = SeaWaveForecastInterface(config=config)
    forecast, info = model(y=y_train, h=14, promo=promo)

Class Description
-----------------
(1) SeaWaveConfig: validated algorithm hyperparameters.
(2) SeaWaveForecastInterface: public function-like forecasting entry point.

References
----------
SeaWave Design SH V001 and SeaWave Coding Design SH V002.
'''



####### Load Packages ##############################################################################
####################################################################################################

import importlib.metadata as _metadata
from pathlib import Path
import tomllib

from seawave.config import SeaWaveConfig
from seawave.interface import SeaWaveForecastInterface



####### Version and Public Interface ################################################################
####################################################################################################


def _read_version():
    '''Read the single version declared in pyproject.toml or package metadata.'''
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as stream:
            return tomllib.load(stream)["project"]["version"]
    return _metadata.version("seawave")


__version__ = _read_version()

__all__ = ["SeaWaveConfig", "SeaWaveForecastInterface", "__version__"]



### End of file
