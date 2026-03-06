"""
Injury/availability filtering: build a set of unavailable player IDs from an optional API
and/or injuries_override.json (list of names). Used in main to exclude props before ranking.

Env: INJURY_FILTER_MODE=on (default) to enable; INJURY_SKIP_STATUSES=OUT,DOUBTFUL,QUESTIONABLE.
If injury API cannot be fetched, logs "injury feed unavailable" and proceeds (override file still applied).
"""
import json
import os
from pathlib import Path
from typing import Dict, Set

from .models import Player

# Project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OVERRIDE_PATH = _PROJECT_ROOT / "injuries_override.json"


def _env_injury_filter_mode() -> bool:
    """True if INJURY_FILTER_MODE is on/1/true."""
    return os.getenv("INJURY_FILTER_MODE", "on").strip().lower() in ("on", "1", "true", "yes")


def _env_skip_statuses() -> Set[str]:
    """INJURY_SKIP_STATUSES comma-separated, default OUT,DOUBTFUL,QUESTIONABLE."""
    raw = os.getenv("INJURY_SKIP_STATUSES", "OUT,DOUBTFUL,QUESTIONABLE").strip()
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def _load_override_names(path: Path) -> list:
    """Load list of player names from JSON file. Returns [] if missing/invalid."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x).strip() for x in data if x]
        return []
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def _names_to_player_ids(names: list, players_by_id: Dict[str, Player]) -> Set[str]:
    """Match override names to player IDs (canonical_name or provider_name, case-insensitive)."""
    out: Set[str] = set()
    name_set = {n.lower() for n in names if n}
    for pid, p in players_by_id.items():
        if not p:
            continue
        c = (p.canonical_name or "").strip().lower()
        pr = (p.provider_name or "").strip().lower()
        if c in name_set or pr in name_set:
            out.add(pid)
        for al in getattr(p, "aliases", []) or []:
            if (al or "").strip().lower() in name_set:
                out.add(pid)
                break
    return out


def get_unavailable_player_ids(
    players_by_id: Dict[str, Player],
    api_key: str | None = None,
    override_path: Path | None = None,
) -> Set[str]:
    """
    Build set of player_ids to exclude (injured/unavailable).
    - Optionally fetch from injury API; on failure log "injury feed unavailable" and do not crash.
    - Load injuries_override.json (list of names) and match to player_ids.
    """
    out: Set[str] = set()
    skip_statuses = _env_skip_statuses()

    # Optional: try injury API (e.g. The Odds API if they add one)
    if api_key:
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://api.the-odds-api.com/v4/sports/basketball_nba/injuries?apiKey=" + api_key
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            # Expected shape: list of { "player": "Name", "status": "OUT" | ... } or similar
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    status = (item.get("status") or item.get("injury_status") or "").strip().upper()
                    if status in skip_statuses:
                        name = (item.get("player") or item.get("name") or "").strip()
                        if name:
                            out |= _names_to_player_ids([name], players_by_id)
        except Exception:
            print("injury feed unavailable")

    # Override file: list of names to treat as unavailable
    path = override_path or DEFAULT_OVERRIDE_PATH
    names = _load_override_names(path)
    out |= _names_to_player_ids(names, players_by_id)

    return out


def is_injury_filter_enabled() -> bool:
    """Whether injury filtering is turned on via env."""
    return _env_injury_filter_mode()
