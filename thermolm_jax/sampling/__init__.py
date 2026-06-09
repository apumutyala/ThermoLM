"""
Sampling utilities for DTM.

This module contains sampling algorithms for DTM, including
chromatic Gibbs sampling and THRML integration.
"""

from .chromatic_gibbs import chromatic_gibbs_sample

__all__ = ["chromatic_gibbs_sample"]
