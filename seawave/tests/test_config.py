# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
Unit tests for SeaWaveConfig defaults, validation, independent mutable fields,
LightGBM parameters, public exports, and the single package version source.

Usage examples
--------------
.. code-block:: bash

    python -m pytest seawave/tests/test_config.py -v
'''

from dataclasses import replace
from pathlib import Path
import tomllib

import pytest

import seawave
from seawave import SeaWaveConfig


class TestSeaWaveConfig:
    '''Verify configuration values and early validation.'''

    def test_reference_defaults(self):
        '''Defaults match the flat reference implementation.'''
        config = SeaWaveConfig()
        assert config.period == 7
        assert config.fourier_K == 3
        assert config.hp_lambda_grid == [129600.0, 1600.0]
        assert config.damp_grid == [0.3, 0.6, 0.9, 1.0]
        assert config.inner_window == 84
        assert config.n_win == 3
        assert config.topk == 3
        assert config.g_hmax == 14
        assert config.promo_lag == 3

    def test_mutable_grids_are_independent(self):
        '''One configuration cannot mutate another configuration's grids.'''
        first = SeaWaveConfig()
        second = SeaWaveConfig()
        first.hp_lambda_grid.append(100.0)
        assert second.hp_lambda_grid == [129600.0, 1600.0]

    def test_custom_conservative_configuration(self):
        '''The conservative example values are accepted without mutation.'''
        config = SeaWaveConfig(margin=0.05, tau=0.05, topk=2)
        assert config.margin == 0.05
        assert config.tau == 0.05
        assert config.topk == 2

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("period", 0),
            ("fourier_K", 0),
            ("hp_lambda_grid", []),
            ("hp_lambda_grid", [1600.0, -1.0]),
            ("damp_grid", [0.0]),
            ("rw_window", 1),
            ("inner_window", 0),
            ("n_win", 0),
            ("topk", 0),
            ("tau", 0.0),
            ("gamma", -0.1),
            ("margin", -0.01),
            ("g_rel_scale", 0.0),
            ("lgb_learning_rate", 0.0),
            ("lgb_num_leaves", 1),
            ("lgb_feature_fraction", 1.1),
            ("lgb_bagging_fraction", 0.0),
            ("lgb_bagging_freq", -1),
            ("g_step", 0),
            ("g_win", 13),
            ("g_hmax", 0),
            ("promo_lag", 0),
        ],
    )
    def test_invalid_values(self, field, value):
        '''Invalid values fail at construction instead of deep in a workflow.'''
        with pytest.raises(ValueError, match=field):
            replace(SeaWaveConfig(), **{field: value})

    def test_lightgbm_parameter_bridge(self):
        '''LightGBM parameters are derived from the active configuration.'''
        config = SeaWaveConfig(random_seed=19, lgb_num_leaves=15)
        params = config.lgb_params
        assert params["objective"] == "regression"
        assert params["num_leaves"] == 15
        assert params["seed"] == 19
        assert params["verbose"] == -1
        assert params["force_row_wise"] is True

    def test_info(self):
        '''Configuration metadata is a non-empty string.'''
        assert "SeaWaveConfig" in SeaWaveConfig()._info()


class TestPublicPackage:
    '''Verify public names and pyproject version consistency.'''

    def test_public_exports(self):
        '''Only agreed top-level symbols are explicitly exported.'''
        assert seawave.__all__ == [
            "SeaWaveConfig",
            "SeaWaveForecastInterface",
            "__version__",
        ]

    def test_version_matches_pyproject(self):
        '''The source package reads its version from pyproject.toml.'''
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject.open("rb") as stream:
            expected = tomllib.load(stream)["project"]["version"]
        assert seawave.__version__ == expected
