"""
Canonical feature list -- imported from features.py (single source of truth).
Import FEATURE_COLUMNS and TARGET_COLUMN from here everywhere in the codebase.
"""

from src.data.features import FEATURE_COLUMNS, TARGET_COLUMN

__all__ = ["FEATURE_COLUMNS", "TARGET_COLUMN"]
