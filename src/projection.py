"""
Historical stat projection engine.

Calculates a player's expected performance for a given stat type using
their last N games (default 10) via weighted average, simple average, or exponential.
Supports variance (sample stdev) and confidence labeling for projections.
Optional winsorization (call winsorize(values, pct) with pct e.g. 0.05) clamps to [p5, p95]; enable via WINSORIZE_PCT in main.
"""

import os
from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Literal, Tuple

# Standard confidence labels for ProjectionResult
ConfidenceType = Literal["insufficient", "low", "high_variance", "ok"]
CONFIDENCE_CV_THRESHOLD = 0.45

# Default: only PTS/REB/AST. Set ENABLE_EXTRA_MARKETS=true to allow PRA and Threes.
CORE_STAT_TYPES: Tuple[str, ...] = ("Points", "Rebounds", "Assists")
EXTRA_STAT_TYPES: Tuple[str, ...] = ("PRA", "Threes")
_extra_markets = os.getenv("ENABLE_EXTRA_MARKETS", "").strip().lower() in ("1", "true", "yes")
STAT_TYPES: Tuple[str, ...] = CORE_STAT_TYPES + (EXTRA_STAT_TYPES if _extra_markets else ())
StatType = Literal["Points", "Rebounds", "Assists", "PRA", "Threes"]

# Supported projection methods; invalid names fall back to weighted_average.
VALID_METHODS = ("weighted_average", "simple_average", "exponential", "blend_short_long")
DEFAULT_METHOD = "weighted_average"

# Coefficient of variation above this => high_variance confidence label.
DEFAULT_CV_THRESHOLD = 0.35

# Minimum stdev for ensemble output (conservative floor).
ENSEMBLE_STDEV_FLOOR = 0.5

# Matchup adjustment clamp (max ±6% from baseline; keep safe).
MATCHUP_DELTA_CLAMP = 0.06


def winsorize(values: List[float], pct: float) -> Tuple[List[float], int]:
    """
    Clamp values to [percentile(pct), percentile(1-pct)]. pct in (0, 0.5), e.g. 0.05 -> [p5, p95].
    Returns (winsorized list, number of values that were clamped).
    """
    if not values or pct <= 0 or pct >= 0.5:
        return (list(values), 0)
    n = len(values)
    sorted_vals = sorted(values)
    lo_idx = min(n - 1, max(0, int((n - 1) * pct)))
    hi_idx = max(0, min(n - 1, int((n - 1) * (1 - pct))))
    p_lo = sorted_vals[lo_idx]
    p_hi = sorted_vals[hi_idx]
    if p_lo >= p_hi:
        return (list(values), 0)
    winsorized: List[float] = []
    count = 0
    for v in values:
        if v < p_lo:
            winsorized.append(p_lo)
            count += 1
        elif v > p_hi:
            winsorized.append(p_hi)
            count += 1
        else:
            winsorized.append(v)
    return (winsorized, count)


# Metric keys for matchup adjustment (must exist in opponent/league metrics).
MATCHUP_METRIC_PTS = "pts_allowed_per_game"
MATCHUP_METRIC_REB = "reb_allowed_per_game"
MATCHUP_METRIC_AST = "ast_allowed_per_game"


def adjust_for_matchup(
    baseline_mean: float,
    opponent_metrics: Dict[str, float],
    league_avg_metrics: Dict[str, float],
    matchup_strength: float = 0.5,
    metric_key: str = MATCHUP_METRIC_PTS,
) -> float:
    """
    Adjust baseline projection for opponent using the given defensive metric.
    factor = league_avg / opponent_allowance;
    adjusted = baseline_mean * (1 + clamp((factor - 1) * matchup_strength, -clamp, +clamp)).
    metric_key: pts_allowed_per_game, reb_allowed_per_game, or ast_allowed_per_game.
    """
    league_avg = league_avg_metrics.get(metric_key) or 1.0
    opponent_allowance = opponent_metrics.get(metric_key) or league_avg
    if opponent_allowance <= 0:
        return baseline_mean
    factor = league_avg / opponent_allowance
    delta_raw = (factor - 1.0) * matchup_strength
    delta = max(-MATCHUP_DELTA_CLAMP, min(MATCHUP_DELTA_CLAMP, delta_raw))
    return baseline_mean * (1.0 + delta)


def compute_confidence(
    mean: float,
    stdev: float,
    n: int,
    min_games: int,
) -> ConfidenceType:
    """
    Standard confidence label for a projection.
    - n < min_games => "insufficient"
    - stdev == 0 => "low"
    - else cv = stdev / max(abs(mean), 1e-6); cv > 0.45 => "high_variance" else "ok"
    """
    if n < min_games:
        return "insufficient"
    if stdev == 0:
        return "low"
    cv = stdev / max(abs(mean), 1e-6)
    return "high_variance" if cv > CONFIDENCE_CV_THRESHOLD else "ok"


@dataclass(frozen=True)
class ProjectionResult:
    """Projection with mean, sample stdev, sample size, and standardized confidence."""
    mean: float
    stdev: float
    n: int
    confidence: ConfidenceType = "ok"


def compute_stdev(values: List[float]) -> float:
    """Sample standard deviation of values. Returns 0 if n < 2."""
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return sqrt(variance)


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


def compute_exponential_average(values: List[float], decay: float = 0.85) -> float:
    """
    Exponential moving average; most recent (last element) has highest weight.
    Weights are decay^0, decay^1, ... from newest to oldest, then normalized.
    """
    if not values:
        return 0.0
    n = len(values)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_w = sum(weights)
    if total_w <= 0:
        return compute_simple_average(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def get_confidence_label(
    mean: float,
    stdev: float,
    n: int,
    min_games: int = 5,
) -> ConfidenceType:
    """Return standardized confidence label. Delegates to compute_confidence."""
    return compute_confidence(mean, stdev, n, min_games)


def _normalize_method(method: str) -> str:
    """Return a valid method; fallback to DEFAULT_METHOD if unknown. Logs when falling back."""
    m = (method or "").strip().lower()
    if m in VALID_METHODS:
        return m
    print(
        f"Projection: unknown method {method!r}; falling back to {DEFAULT_METHOD}. "
        f"Valid methods: {', '.join(VALID_METHODS)}."
    )
    return DEFAULT_METHOD


def _compute_mean_and_stdev(
    values: List[float],
    method: str,
) -> tuple[float, float]:
    """Compute mean and stdev for values using the given method."""
    if not values:
        return (0.0, 0.0)
    if method == "weighted_average":
        mean = compute_weighted_average(values)
    elif method == "simple_average":
        mean = compute_simple_average(values)
    elif method == "exponential":
        mean = compute_exponential_average(values)
    else:
        mean = compute_weighted_average(values)
    stdev = compute_stdev(values)
    return (mean, stdev)


def ensemble_projection(
    strategies_with_weights: List[Tuple[str, float]],
    values_short: List[float],
    values_long: List[float],
    blend_alpha: float = 0.65,
    *,
    min_games: int = 5,
) -> Tuple[ProjectionResult | None, Dict[str, float]]:
    """
    Combine multiple strategies by weight. Each strategy produces mean and stdev;
    means are combined by weights; stdev is weighted avg with a floor.

    strategies_with_weights: list of (strategy_name, weight), e.g.
        [("weighted_average", 0.5), ("simple_average", 0.2), ("blend_short_long", 0.3)]
    values_short / values_long: used for short/long windows (blend uses both; others use long).

    Returns (ProjectionResult, strategy_means_dict for logging). Returns (None, {}) if no data.
    """
    if not values_long:
        return (None, {})

    means: Dict[str, float] = {}
    stdevs: Dict[str, float] = {}

    for name, weight in strategies_with_weights:
        method = (name or "").strip().lower()
        if method == "blend_short_long":
            res, _ = blend_short_long_result(
                values_short, values_long, blend_alpha, DEFAULT_METHOD
            )
            if res is not None:
                means[name] = res.mean
                stdevs[name] = res.stdev
        elif method in ("weighted_average", "simple_average", "exponential"):
            m, s = _compute_mean_and_stdev(values_long, method)
            means[name] = m
            stdevs[name] = s

    if not means:
        return (None, {})

    total_weight = sum(w for n, w in strategies_with_weights if n in means)
    if total_weight <= 0:
        return (None, {})

    mean_ens = sum(
        (weight / total_weight) * means[name]
        for name, weight in strategies_with_weights
        if name in means
    )
    stdev_weighted = sum(
        (weight / total_weight) * stdevs.get(name, 0.0)
        for name, weight in strategies_with_weights
        if name in stdevs
    )
    stdev_ens = max(stdev_weighted, ENSEMBLE_STDEV_FLOOR)
    n = len(values_long)
    confidence = compute_confidence(mean_ens, stdev_ens, n, min_games)
    return (
        ProjectionResult(mean=mean_ens, stdev=stdev_ens, n=n, confidence=confidence),
        dict(means),
    )


def blend_short_long_result(
    short_values: List[float],
    long_values: List[float],
    alpha: float,
    base_method: str = "weighted_average",
    *,
    min_games: int = 5,
) -> tuple[ProjectionResult | None, str]:
    """
    Regression-to-mean blend: mean = alpha*mean_short + (1-alpha)*mean_long.
    stdev = alpha*stdev_short + (1-alpha)*stdev_long (weighted).
    Returns (ProjectionResult, strategy_label). Returns (None, "...") if insufficient data.
    """
    if not long_values:
        return (None, "blend_short_long")
    mean_short, stdev_short = _compute_mean_and_stdev(short_values, base_method)
    mean_long, stdev_long = _compute_mean_and_stdev(long_values, base_method)
    mean = alpha * mean_short + (1.0 - alpha) * mean_long
    stdev = alpha * stdev_short + (1.0 - alpha) * stdev_long
    n = len(long_values)
    confidence = compute_confidence(mean, stdev, n, min_games)
    return (ProjectionResult(mean=mean, stdev=stdev, n=n, confidence=confidence), "blend_short_long")


def get_projection_result(
    player_id: str,
    stat_type: StatType,
    historical_values: List[float],
    *,
    n_games: int = 10,
    method: Literal["weighted_average", "simple_average", "exponential", "blend_short_long"] = "weighted_average",
    min_games: int = 5,
) -> ProjectionResult | None:
    """
    Returns a ProjectionResult (mean, stdev, n, confidence) for the player's recent history.
    Returns None if no historical data. Uses same method logic as get_projection.
    For blend_short_long, historical_values should be long window; short is derived by caller.
    """
    if stat_type not in STAT_TYPES:
        raise ValueError(f"stat_type must be one of {STAT_TYPES}, got {stat_type!r}")

    method = _normalize_method(method)
    recent = historical_values[-n_games:] if len(historical_values) > n_games else historical_values
    if not recent:
        return None

    if method == "blend_short_long":
        # Caller must use blend_short_long_result directly with short/long values
        return None

    if method == "weighted_average":
        mean = compute_weighted_average(recent)
    elif method == "simple_average":
        mean = compute_simple_average(recent)
    elif method == "exponential":
        mean = compute_exponential_average(recent)
    else:
        mean = compute_weighted_average(recent)

    stdev = compute_stdev(recent)
    n = len(recent)
    confidence = compute_confidence(mean, stdev, n, min_games)
    return ProjectionResult(mean=mean, stdev=stdev, n=n, confidence=confidence)


def get_projection(
    player_id: str,
    stat_type: StatType,
    historical_values: List[float],
    *,
    n_games: int = 10,
    method: Literal["weighted_average", "simple_average", "exponential"] = "weighted_average",
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
        method: "weighted_average", "simple_average", or "exponential". Invalid values fall back to weighted_average.

    Returns:
        Projected value (float). Returns 0.0 if no historical data.

    Note:
        Assumes `historical_values` is ordered most-recent last (e.g. [oldest, ..., newest]).
        If your data is newest-first, slice and reverse before calling, e.g.:
        get_projection(pid, "Points", list(reversed(recent_values[:n_games]))).
    """
    if stat_type not in STAT_TYPES:
        raise ValueError(f"stat_type must be one of {STAT_TYPES}, got {stat_type!r}")

    method = _normalize_method(method)

    # Use the last n_games (most recent) from the list.
    recent = historical_values[-n_games:] if len(historical_values) > n_games else historical_values
    if not recent:
        return 0.0

    if method == "weighted_average":
        return compute_weighted_average(recent)
    if method == "simple_average":
        return compute_simple_average(recent)
    if method == "exponential":
        return compute_exponential_average(recent)
    return compute_weighted_average(recent)
