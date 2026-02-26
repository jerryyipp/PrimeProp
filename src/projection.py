"""
Historical stat projection engine.

Calculates a player's expected performance for a given stat type using
their last N games (default 10) via weighted average or simple average.
"""

from typing import List, Literal

# Canonical stat types aligned with PropLine; used for validation and projection keys.
STAT_TYPES = ("Points", "Rebounds", "Assists", "PRA", "Threes")
StatType = Literal["Points", "Rebounds", "Assists", "PRA", "Threes"]


def _linear_weights(n: int) -> List[float]:
    """
    Weights for last n games: index 0 = oldest (lowest), index n-1 = newest (highest).

    This matches the assumption that historical_values is ordered [oldest, ..., newest],
    so the most recent game (last element) gets the largest multiplier.
    """
    if n <= 0:
        return []
    total = n * (n + 1) / 2
    # Ascending weights so the final (newest) element has the heaviest weight.
    return [float(i) / total for i in range(1, n + 1)]


def compute_weighted_average(values: List[float]) -> float:
    """
    Weighted average of values; most recent (last element) has highest weight.
    Uses linear weights so recent games matter more than older ones.
    """
    if not values:
        return 0.0
    weights = _linear_weights(len(values))
    return sum(v * w for v, w in zip(values, weights))


def compute_simple_average(values: List[float]) -> float:
    """Simple mean of values (equal weight for each game)."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def get_projection(
    player_id: str,
    stat_type: StatType,
    historical_values: List[float],
    *,
    n_games: int = 10,
    method: Literal["weighted_average", "simple_average"] = "weighted_average",
) -> float:
    """
    Returns a projected stat value for a player based on their recent history.

    Uses the last `n_games` values from `historical_values` (or all if fewer
    are available), then applies the chosen baseline method.

    Args:
        player_id: Unique player identifier (used by callers for lookup; not used in math).
        stat_type: Which stat is being projected (must match PropLine stat_type).
        historical_values: Ordered list of stat values, most recent last (or first; see note).
        n_games: Number of most recent games to use (default 10).
        method: "weighted_average" (recent games weighted higher) or "simple_average".

    Returns:
        Projected value (float). Returns 0.0 if no historical data.

    Note:
        Assumes `historical_values` is ordered most-recent last (e.g. [oldest, ..., newest]).
        If your data is newest-first, slice and reverse before calling, e.g.:
        get_projection(pid, "Points", list(reversed(recent_values[:n_games]))).
    """
    if stat_type not in STAT_TYPES:
        raise ValueError(f"stat_type must be one of {STAT_TYPES}, got {stat_type!r}")

    # Use the last n_games (most recent) from the list.
    recent = historical_values[-n_games:] if len(historical_values) > n_games else historical_values
    if not recent:
        return 0.0

    if method == "weighted_average":
        return compute_weighted_average(recent)
    if method == "simple_average":
        return compute_simple_average(recent)
    raise ValueError(f"method must be 'weighted_average' or 'simple_average', got {method!r}")
