# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
Unit tests for JAX Fourier basis generation, closed-form coefficient recovery,
periodic projection, input validation, and component state.

Usage examples
--------------
.. code-block:: bash

    python -m pytest seawave/tests/test_season.py -v
'''

import numpy as np
import pytest

from seawave.mixin import AlgorithmMixin
from seawave.season import FourierSeasonComponent


class TestFourierBasis:
    '''Verify trigonometric values against an independent NumPy reference.'''

    def test_terms_match_numpy(self):
        '''The JAX basis uses alternating sine and cosine columns.'''
        t = np.arange(14)
        actual = AlgorithmMixin.fourier_terms(t, period=7, K=3)
        columns = []
        for k in range(1, 4):
            columns.append(np.sin(2 * np.pi * k * t / 7))
            columns.append(np.cos(2 * np.pi * k * t / 7))
        expected = np.column_stack(columns)
        assert actual.shape == (14, 6)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_basis_is_periodic(self):
        '''Advancing time by one period leaves every Fourier term unchanged.'''
        t = np.arange(21)
        first = AlgorithmMixin.fourier_terms(t, period=7, K=3)
        shifted = AlgorithmMixin.fourier_terms(t + 7, period=7, K=3)
        np.testing.assert_allclose(first, shifted, atol=1e-6)


class TestFourierSeasonComponent:
    '''Verify closed-form fitting and future seasonal projection.'''

    def test_recovers_known_seasonal_signal(self):
        '''A known two-harmonic signal is reconstructed within JAX precision.'''
        t = np.arange(84)
        residual = (
            0.8 * np.sin(2 * np.pi * t / 7)
            + 0.3 * np.cos(2 * np.pi * t / 7)
            - 0.2 * np.sin(4 * np.pi * t / 7)
        )
        component = FourierSeasonComponent(period=7, K=3).fit(residual, t)
        np.testing.assert_allclose(component.seasonal_, residual, atol=2e-5)
        future_t = np.arange(84, 98)
        expected = (
            0.8 * np.sin(2 * np.pi * future_t / 7)
            + 0.3 * np.cos(2 * np.pi * future_t / 7)
            - 0.2 * np.sin(4 * np.pi * future_t / 7)
        )
        np.testing.assert_allclose(component.predict(future_t), expected, atol=2e-5)
        assert component.beta_.shape == (6,)

    def test_predict_before_fit(self):
        '''Projection requires fitted coefficients.'''
        with pytest.raises(RuntimeError, match="has not been fitted"):
            FourierSeasonComponent().predict(np.arange(14))

    def test_mismatched_residual_and_time(self):
        '''Residual and time vectors must have identical lengths.'''
        with pytest.raises(ValueError, match="same length"):
            FourierSeasonComponent().fit(np.arange(10.0), np.arange(9))

    def test_non_finite_time_is_rejected(self):
        '''Non-finite time indices cannot enter a trigonometric projection.'''
        component = FourierSeasonComponent().fit(np.arange(10.0), np.arange(10))
        with pytest.raises(ValueError, match="finite"):
            component.predict(np.array([1.0, np.inf]))

    def test_invalid_configuration(self):
        '''Invalid period, harmonic count, and ridge values fail at construction.'''
        with pytest.raises(ValueError, match="period"):
            FourierSeasonComponent(period=0)
        with pytest.raises(ValueError, match="K"):
            FourierSeasonComponent(K=0)
        with pytest.raises(ValueError, match="ridge"):
            FourierSeasonComponent(ridge=0)

    def test_info(self):
        '''Seasonal metadata identifies the configured period and harmonics.'''
        info = FourierSeasonComponent(period=7, K=3)._info()
        assert "period=7" in info
        assert "K=3" in info
