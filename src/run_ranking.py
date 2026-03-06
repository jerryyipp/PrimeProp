"""
Thin wrapper: ranking logic lives in src.main (single source of truth).
Re-export run_ranking_for_snapshot for replay_snapshot and other callers.
"""
from src.main import run_ranking_for_snapshot

__all__ = ["run_ranking_for_snapshot"]
