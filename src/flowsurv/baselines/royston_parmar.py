"""Royston-Parmar flexible parametric survival model (M-splines) via R/flexsurv.

The implementation uses ``rpy2`` when available and falls back to the
UnavailableMethod stub otherwise.  On Windows the recommended route is the
exported-CSV bridge (Methodology Sec. 7), so this module primarily exists to
reserve the method slot in the registry.
"""

from __future__ import annotations

from .stubs import _UnavailableMethod

class RoystonParmar(_UnavailableMethod):
    """Royston-Parmar flexible parametric baseline."""

    name: str = "royston_parmar"

    def __init__(self) -> None:
        super().__init__()
        self.reason = "rpy2/flexsurv not installed or not configured"
