# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spread calibration of ensemble models: losses and spread–skill metrics.

Reference the classes by full path in hydra configs, e.g.
``_target_: anemoi.training.spread.SpreadScaledCRPS``.
"""

from anemoi.training.spread.crps import SpreadScaledCRPS
from anemoi.training.spread.metrics import EnsembleMeanSquaredError
from anemoi.training.spread.metrics import EnsembleVariance
from anemoi.training.spread.metrics import spread_skill_ratio
from anemoi.training.spread.multiscale import MultiscaleSpreadLossWrapper

__all__ = [
    "EnsembleMeanSquaredError",
    "EnsembleVariance",
    "MultiscaleSpreadLossWrapper",
    "SpreadScaledCRPS",
    "spread_skill_ratio",
]
