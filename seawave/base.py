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

This module defines the metaclasses that standardize SeaWave component creation
and abstract interface enforcement.

- Design mode:

    (1) Abstract base class pattern

    (2) Template method pattern

- Key points:

    (1) Abstract methods are checked when a concrete class is created

    (2) Composite instances are created through five explicit lifecycle stages

Main functions
--------------
(1) MetaRequestMethod enforces agreed abstract interfaces through ABCMeta's
class-creation process and the collected __abstractmethods__ set.

(2) MetaBuildInstance separates preNew, new, preInit, init, and postInit.

UML
---
.. code-block:: text

    classDiagram
        ABCMeta <|-- MetaRequestMethod
        MetaRequestMethod <|-- MetaBuildInstance
        MetaBuildInstance <|.. BaseComposite
        MetaRequestMethod <|.. BaseInterface

        MetaRequestMethod: +__new__(enforce_abstract=True)
        MetaBuildInstance: +__call__()
        BaseComposite: +doprenew()
        BaseComposite: +donew()
        BaseComposite: +dopreinit()
        BaseComposite: +doinit()
        BaseComposite: +dopostinit()
        BaseInterface: +_info()

Usage examples
--------------
.. code-block:: python

    from abc import abstractmethod
    from seawave.base import MetaRequestMethod

    class BaseAlgorithm(metaclass=MetaRequestMethod, enforce_abstract=False):
        @abstractmethod
        def fit(self):
            pass

    class ConcreteAlgorithm(BaseAlgorithm):
        def fit(self):
            return self

Precautions
-----------
Abstract framework bases use enforce_abstract=False so they may declare an
incomplete contract. Concrete subclasses use the default True and fail early if
any abstract method remains unresolved. The five lifecycle hooks isolate side
effects and permit dependency checks or post-initialization validation.

References
----------
Python data model: https://docs.python.org/3/reference/datamodel.html
SeaWave Coding Design SH V002.
'''



####### Load Packages ##############################################################################
####################################################################################################

from abc import ABCMeta



####### Metaclasses and Base Classes ################################################################
####################################################################################################


class MetaRequestMethod(ABCMeta):
    '''Class Introduction:

    This metaclass rejects concrete subclasses that leave any agreed abstract
    method unimplemented. Abstract framework bases can opt out with the
    ``enforce_abstract=False`` class keyword.
    '''

    def __new__(mcls, name, bases, namespace, enforce_abstract=True, **kwargs):
        '''Create a class and validate its inherited abstract methods.'''
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        missing = getattr(cls, "__abstractmethods__", frozenset())
        if enforce_abstract and missing:
            methods = ", ".join(sorted(missing))
            raise NotImplementedError(
                f"Concrete class {name} must implement abstract methods: {methods}"
            )
        return cls


class MetaBuildInstance(MetaRequestMethod):
    '''Class Introduction:

    This metaclass controls Composite construction through five overridable
    stages while retaining abstract interface validation.
    '''

    def __call__(cls, *args, **kwargs):
        '''Execute preNew, new, preInit, init, and postInit in order.'''
        cls.doprenew(*args, **kwargs)
        obj = cls.donew(*args, **kwargs)
        cls.dopreinit(obj, *args, **kwargs)
        cls.doinit(obj, *args, **kwargs)
        cls.dopostinit(obj, *args, **kwargs)
        return obj


class BaseComposite(metaclass=MetaBuildInstance, enforce_abstract=False):
    '''Class Introduction:

    BaseComposite supplies the default implementation of the five lifecycle
    hooks used by SeaWave algorithm composites.
    '''

    @classmethod
    def doprenew(cls, *args, **kwargs):
        '''Run optional processing before memory allocation.'''
        return None

    @classmethod
    def donew(cls, *args, **kwargs):
        '''Allocate an instance without invoking its initializer.'''
        return cls.__new__(cls)

    @classmethod
    def dopreinit(cls, obj, *args, **kwargs):
        '''Run optional processing immediately before initialization.'''
        return None

    @classmethod
    def doinit(cls, obj, *args, **kwargs):
        '''Invoke the instance initializer with the original arguments.'''
        obj.__init__(*args, **kwargs)

    @classmethod
    def dopostinit(cls, obj, *args, **kwargs):
        '''Run optional validation after initialization.'''
        return None


class BaseInterface(metaclass=MetaRequestMethod, enforce_abstract=False):
    '''Class Introduction:

    BaseInterface marks the common type of function-like SeaWave interfaces.
    Concrete interfaces implement ``__call__`` and ``_info``.
    '''

    def _info(self) -> str:
        '''Return a textual description of the interface.'''
        raise NotImplementedError



### End of file
