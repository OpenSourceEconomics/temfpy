"""
Temfpy package.

temfpy is an open-source package providing test models and functions
for standard numerical components in computational economic models
"""
__version__ = "1.1.4"

import pytest

from temfpy.config import ROOT_DIR


def test(*args, **kwargs):
    """Run basic tests of the package."""
    pytest.main([str(ROOT_DIR), *args], **kwargs)
