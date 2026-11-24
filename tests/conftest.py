"""Shared pytest configuration."""

import matplotlib
import pytest

# Headless plotting: tests must not open windows or need a display.
matplotlib.use("Agg")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat an empty test run as success while the skeleton has no tests yet."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
