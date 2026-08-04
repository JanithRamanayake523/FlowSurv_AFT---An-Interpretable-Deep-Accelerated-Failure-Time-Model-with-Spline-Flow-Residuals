"""Smoke tests for the Phase 0 skeleton.

Replaced/extended by the six pre-experiment gate tests in Phase 1
(Methodology §6); these exist so CI has a green baseline from day one.
"""

import importlib


def test_package_imports():
    import flowsurv

    assert flowsurv.__version__ == "0.1.0"


def test_subpackages_import():
    for name in ["models", "data", "baselines", "metrics", "eval"]:
        importlib.import_module(f"flowsurv.{name}")


def test_torch_and_zuko_available():
    torch = importlib.import_module("torch")
    zuko = importlib.import_module("zuko")

    assert torch.__version__ and zuko.__version__
