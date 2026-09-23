"""Scaffold smoke tests — no model load."""

from deliberation_judge import __version__


def test_version():
    assert __version__ == "0.1.0"
