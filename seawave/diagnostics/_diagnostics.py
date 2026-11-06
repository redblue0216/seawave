# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''
Module Introduction
-------------------
Typed diagnostics preserve trend risk, ensemble weights, and the residual gate
decision. ``to_dict`` provides the dictionary returned by the public Interface
without exposing mutable component internals.
'''

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GateDiagnostics:
    '''Describe residual training coverage, validation errors, and gate output.'''

    reason: str
    n_train: int
    n_gate: int
    baseline_mae: float | None
    gated_mae: float | None
    relative_improvement: float | None
    use_g: bool
    g_scale: float

    @classmethod
    def from_component(cls, component):
        '''Create an immutable diagnostic snapshot from DirectGBDTComponent.'''
        values = component.diagnostics_
        return cls(
            reason=values["reason"],
            n_train=values["n_train"],
            n_gate=values["n_gate"],
            baseline_mae=values["baseline_mae"],
            gated_mae=values["gated_mae"],
            relative_improvement=component.relative_improvement_,
            use_g=component.use_g_,
            g_scale=component.g_scale_,
        )

    def to_dict(self) -> dict:
        '''Return a plain dictionary suitable for serialization.'''
        return {
            "reason": self.reason,
            "n_train": self.n_train,
            "n_gate": self.n_gate,
            "baseline_mae": self.baseline_mae,
            "gated_mae": self.gated_mae,
            "relative_improvement": self.relative_improvement,
            "use_g": self.use_g,
            "g_scale": self.g_scale,
        }


@dataclass(frozen=True)
class ForecastDiagnostics:
    '''Store structural selection and residual-gate diagnostics for one forecast.'''

    top: tuple[str, ...]
    weights: np.ndarray
    scores: dict[str, float]
    window_scores: dict[str, tuple[float, ...]]
    gate: GateDiagnostics

    def to_dict(self) -> dict:
        '''Return the stable public info dictionary with defensive array copies.'''
        return {
            "top": list(self.top),
            "weights": self.weights.copy(),
            "scores": dict(self.scores),
            "window_scores": dict(self.window_scores),
            "use_g": self.gate.use_g,
            "g_scale": self.gate.g_scale,
            "relative_improvement": self.gate.relative_improvement,
            "g_gate": self.gate.to_dict(),
        }
