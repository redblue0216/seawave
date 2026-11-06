# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# License: Apache-2.0

'''Public trend boundary with explicit component and selection exports.'''

from seawave.trend._components import (
    GLinTrendComponent,
    HPTrendComponent,
    PolyTrendComponent,
    RWTrendComponent,
    TrendCandidate,
    TrendEnsembleComponent,
    TrendScore,
    build_trend_candidates,
    score_trend_candidates,
    select_trend_candidates,
)

__all__ = [
    "GLinTrendComponent",
    "HPTrendComponent",
    "PolyTrendComponent",
    "RWTrendComponent",
    "TrendCandidate",
    "TrendEnsembleComponent",
    "TrendScore",
    "build_trend_candidates",
    "score_trend_candidates",
    "select_trend_candidates",
]
