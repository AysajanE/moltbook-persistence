#!/usr/bin/env python3
"""Deprecated wrapper for results-presentation artifact generation.

Canonical generator:
- analysis/10_results_presentation_category4.py

This wrapper forwards execution to the canonical script to avoid duplicate
implementations producing divergent values.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("10_results_presentation_category4.py")
    runpy.run_path(str(target), run_name="__main__")
