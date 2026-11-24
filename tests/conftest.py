"""Shared pytest configuration."""

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat an empty test run as success while the skeleton has no tests yet."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
