"""Test fixtures for the honey package.

This module contains pytest fixtures that are used across multiple test files.
These fixtures provide common test data and resources to ensure consistent testing.

Security Notes:
- S101 (assert usage): Asserts are the standard way to validate test conditions in pytest.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def root() -> Path:
    """Provide the repository root path (directory containing pyproject.toml)."""
    return Path(__file__).parent.parent


@pytest.fixture
def logger(request) -> logging.Logger:
    """Provide a logger named after the calling test."""
    return logging.getLogger(request.node.nodeid)
