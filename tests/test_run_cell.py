"""`_fit_config` device wiring for `_FlowSurvWrapper` baselines.

Regression test: `_fit_config` used to gate the `device` injection on
`method_name.startswith("flowsurv")`, which matched flowsurv_aft/gauss but
missed ausset_cnf (also a `_FlowSurvWrapper`) -- it silently fell back to
the wrapper's CPU default regardless of the runner's requested device.
"""

from __future__ import annotations

from flowsurv.baselines import METHODS
from flowsurv.eval.run_cell import _fit_config


def test_device_injected_for_all_flowsurv_wrapper_methods():
    for name in ("flowsurv_aft", "flowsurv_gauss", "ausset_cnf"):
        method = METHODS[name]()
        config = _fit_config(name, method, tuner=None, cell=None, rep=1, device="cuda")
        assert config.get("device") == "cuda", name


def test_device_not_injected_for_non_wrapper_methods():
    method = METHODS["cox_ph"]()
    config = _fit_config("cox_ph", method, tuner=None, cell=None, rep=1, device="cuda")
    assert "device" not in config
