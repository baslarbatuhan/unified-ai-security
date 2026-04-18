"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load environment variables (HF_TOKEN, etc.) for all pytest runs
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from output_agency_defense.behavior_monitor import BehaviorMonitor


@pytest.fixture
def m() -> BehaviorMonitor:
    """BehaviorMonitor instance for tests/test_behavior_monitor.py scenarios."""
    return BehaviorMonitor()


@pytest.fixture
def _m(m: BehaviorMonitor) -> BehaviorMonitor:
    """Alias for scenarios that name the fixture `_m`."""
    return m
