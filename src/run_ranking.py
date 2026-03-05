"""
Thin wrapper: all ranking logic lives in main.py (single source of truth).
Re-export run_ranking_for_snapshot for replay_snapshot and other callers.
"""
from main import run_ranking_for_snapshot

__all__ = ["run_ranking_for_snapshot"]
