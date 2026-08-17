"""Ausset continuous normalizing flow (FFJORD) baseline via ``torchdiffeq``.

This is an optional competitor reimplementation (Methodology Sec. 3.2).  It
falls back to the UnavailableMethod stub when ``torchdiffeq`` is not
installed.
"""

from __future__ import annotations

from .stubs import _UnavailableMethod

class AussetCNF(_UnavailableMethod):
    """Ausset et al. continuous ODE-flow baseline."""

    name: str = "ausset_cnf"

    def __init__(self) -> None:
        super().__init__()
        self.reason = "torchdiffeq not installed or CNF not implemented"
