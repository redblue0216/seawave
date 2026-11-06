# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
Unit tests for SeaWave abstract interface enforcement, five-stage Composite
construction, public Interface delegation, and explicit framework boundaries.

Usage examples
--------------
.. code-block:: bash

    python -m pytest seawave/tests/test_frame.py -v
'''

from abc import abstractmethod
import numpy as np
import pytest

from seawave import SeaWaveConfig, SeaWaveForecastInterface
from seawave.base import BaseComposite, MetaRequestMethod


class TestMetaRequestMethod:
    '''Verify that abstract contracts are enforced at class creation.'''

    def test_missing_abstract_method_is_rejected(self):
        '''A concrete class with an unimplemented contract must not be created.'''
        class Contract(metaclass=MetaRequestMethod, enforce_abstract=False):
            @abstractmethod
            def fit(self):
                '''Define a required fitting operation.'''

        with pytest.raises(NotImplementedError, match="fit"):
            class MissingImplementation(Contract):
                pass

    def test_complete_implementation_is_accepted(self):
        '''A class implementing the complete contract can be instantiated.'''
        class Contract(metaclass=MetaRequestMethod, enforce_abstract=False):
            @abstractmethod
            def fit(self):
                '''Define a required fitting operation.'''

        class CompleteImplementation(Contract):
            def fit(self):
                return "fitted"

        assert CompleteImplementation().fit() == "fitted"


class TestMetaBuildInstance:
    '''Verify the exact five-stage instance creation order.'''

    def test_lifecycle_order(self):
        '''Every lifecycle hook runs once and in the agreed order.'''
        class LifecycleComposite(BaseComposite):
            events = []

            @classmethod
            def doprenew(cls, *args, **kwargs):
                cls.events.append("preNew")
                return super().doprenew(*args, **kwargs)

            @classmethod
            def donew(cls, *args, **kwargs):
                cls.events.append("new")
                return super().donew(*args, **kwargs)

            @classmethod
            def dopreinit(cls, obj, *args, **kwargs):
                cls.events.append("preInit")
                return super().dopreinit(obj, *args, **kwargs)

            def __init__(self, value):
                self.events.append("init")
                self.value = value

            @classmethod
            def dopostinit(cls, obj, *args, **kwargs):
                cls.events.append("postInit")
                return super().dopostinit(obj, *args, **kwargs)

        instance = LifecycleComposite(value=7)

        assert instance.value == 7
        assert instance.events == ["preNew", "new", "preInit", "init", "postInit"]


class TestSeaWaveForecastInterface:
    '''Verify public Interface construction and Composite delegation.'''

    def test_default_configuration_and_info(self):
        '''The Interface owns a default configuration and readable metadata.'''
        interface = SeaWaveForecastInterface()
        assert isinstance(interface.config, SeaWaveConfig)
        assert interface.composite.config is interface.config
        assert "SeaWaveForecastInterface" in interface._info()

    def test_invalid_configuration_type(self):
        '''A non-SeaWaveConfig object is rejected before forecasting.'''
        with pytest.raises(TypeError, match="SeaWaveConfig"):
            SeaWaveForecastInterface(config=object())

    def test_interface_uses_composite_workflow(self):
        '''The public entry point owns the integrated SeaWaveComposite.'''
        interface = SeaWaveForecastInterface()
        assert interface.composite.workflow == ("trend", "season", "residual", "exog")
        assert "trend + Fourier" in interface.composite._info()

    def test_call_delegates_with_keywords(self):
        '''The fixed public signature delegates y, h, and promo unchanged.'''
        interface = SeaWaveForecastInterface()
        y = np.arange(30, dtype=float)
        promo = np.zeros(44)
        calls = []

        def forecast(*, y, h, promo):
            calls.append((y, h, promo))
            return np.zeros(h), {"top": []}

        interface.composite.forecast = forecast
        forecast_values, info = interface(y=y, h=14, promo=promo)

        assert calls == [(y, 14, promo)]
        assert forecast_values.shape == (14,)
        assert info == {"top": []}
