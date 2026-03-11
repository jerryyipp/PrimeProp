import json
import math
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Literal, Tuple, Union

from pydantic import BaseModel, Field

from .models import MarketSnapshot
from .projection import StatType, ProjectionResult


# Projections are supplied by the caller, typically backed by historical stats.
ProjectionProvider = Callable[[str, StatType], Optional[float]]
ProjectionResultProvider = Callable[[str, StatType], Optional[ProjectionResult]]
LastNValuesProvider = Callable[[str, StatType], Optional[List[float]]]

# Default path for calibration params (project root)
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "calibration_params.json"


def normal_cdf(z: float) -> float:
    """Standard normal CDF: P(Z <= z). Uses math.erf (stdlib only)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def model_prob_over(line: float, mean: float, stdev: float) -> float:
    """
    Model probability that the stat is strictly above the line, assuming normal(mean, stdev).
    - If stdev < 1e-6: return 1.0 if mean > line, 0.5 if mean == line, 0.0 if mean < line.
    - Else: z = (line - mean) / stdev, P(Over) = 1 - Phi(z).
    """
    if stdev < 1e-6:
        if mean > line:
            return 1.0
        if mean < line:
            return 0.0
        return 0.5
    z = (line - mean) / stdev
    return 1.0 - normal_cdf(z)


def p_over_model(line: float, mean: float, stdev: float) -> float:
    """P(stat > line) from Normal(mean, stdev). 1/0/0.5 when stdev < 1e-6. Alias for model_prob_over."""
    return model_prob_over(line, mean, stdev)


def _prob_method_env() -> str:
    """PROB_METHOD: normal (default) or empirical."""
    return (os.getenv("PROB_METHOD", "normal") or "normal").strip().lower()


def _empirical_smoothing_env() -> float:
    """EMPIRICAL_SMOOTHING for Laplace-style smoothing (default 1.0)."""
    try:
        return float(os.getenv("EMPIRICAL_SMOOTHING", "1.0").strip())
    except (ValueError, TypeError):
        return 1.0


def empirical_prob_over(line: float, values: List[float], smoothing: float = 1.0) -> float:
    """
    P(stat > line) from empirical CDF with Laplace smoothing.
    p_over = (count(values > line) + smoothing) / (n + 2*smoothing).
    Clamped to [0, 1].
    """
    if not values:
        return 0.5
    n = len(values)
    count_over = sum(1 for v in values if v > line)
    p = (count_over + smoothing) / (n + 2.0 * smoothing)
    return max(0.0, min(1.0, p))


def profit_per_unit(odds: Optional[float]) -> Optional[float]:
    """
    Profit per $1 stake if the bet wins (American odds).
    - If odds < 0: profit = 100 / abs(odds)  (e.g. -110 -> 100/110)
    - If odds > 0: profit = odds / 100        (e.g. +150 -> 1.5)
    Returns None if odds is None.
    """
    if odds is None:
        return None
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def american_odds_to_profit_per_dollar(odds: Optional[float]) -> Optional[float]:
    """American odds -> profit per $1 stake if win. Alias for profit_per_unit."""
    return profit_per_unit(odds)


def american_to_decimal(odds: Optional[float]) -> Optional[float]:
    """Convert American odds to decimal (payout per $1 stake). Returns None if odds is None."""
    p = profit_per_unit(odds)
    return (1.0 + p) if p is not None else None


def calculate_implied_probability(odds: Optional[float]) -> Optional[float]:
    """
    Convert American odds (e.g. -110 or +125) to implied probability (0 to 1).

    Returns None if odds is None.
    """
    if odds is None:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def ev_per_unit(p_win: float, profit_if_win: float) -> float:
    """EV per $1 stake: p_win * profit - (1 - p_win) * 1."""
    return p_win * profit_if_win - (1.0 - p_win) * 1.0


def _clamp_calibrated(p: float, a: float, b: float) -> float:
    """Apply linear calibration and clamp to [0, 1]: p' = clamp(a*p + b, 0, 1)."""
    return max(0.0, min(1.0, a * p + b))


def _prob_bounds_from_env() -> Tuple[float, float]:
    """
    Read P_MIN and P_MAX from env (defaults 0.05 and 0.80) and return as a sorted (low, high) tuple.
    """
    def _read(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)).strip())
        except (ValueError, TypeError):
            return default

    p_min = _read("P_MIN", 0.05)
    p_max = _read("P_MAX", 0.80)
    # Ensure sensible ordering and clamp to [0,1]
    low = max(0.0, min(1.0, min(p_min, p_max)))
    high = max(0.0, min(1.0, max(p_min, p_max)))
    if high < low:
        return (0.05, 0.80)
    return (low, high)


def load_calibration(path: Path) -> Optional[Tuple[float, float]]:
    """
    Load calibration params (a, b) from JSON. Expects {"a": float, "b": float}.
    Returns None if file missing or invalid.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        a = float(data.get("a", 1.0))
        b = float(data.get("b", 0.0))
        return (a, b)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def load_calibration_params(path: Union[str, Path, None] = None) -> Optional[Tuple[float, float]]:
    """
    Load calibration (a, b) from JSON for use in EV. Default path: calibration_params.json in project root.
    Returns None if file missing or invalid.
    """
    p = DEFAULT_CALIBRATION_PATH if path is None else (Path(path) if isinstance(path, str) else path)
    return load_calibration(p)


def save_calibration(path: Path, a: float, b: float) -> None:
    """Write calibration params to JSON with updated_at (ISO)."""
    from datetime import datetime, timezone
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "a": a,
                "b": b,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
    except OSError:
        pass


def fit_calibration_from_picks(rows: List[Any]) -> Tuple[float, float]:
    """
    Fit linear calibration p' = clamp(a*p + b, 0, 1) from historical graded picks.
    Each row must have p_over_model, recommended_side, won (0/1). P(win) = p_over if Over else 1-p_over.
    Fits realized hit rate vs model P(win); returns (a, b). If < 10 usable rows, returns (1.0, 0.0).
    Save/load via save_calibration(path, a, b) and load_calibration(path); use calibration_params.json in project root.
    """
    def _get(r: Any, key: str, default: Any = None) -> Any:
        try:
            return r[key]
        except (KeyError, TypeError):
            return getattr(r, key, default)

    points: List[Tuple[float, float]] = []
    for r in rows:
        p_over = _get(r, "p_over_model")
        if p_over is None:
            continue
        side = (_get(r, "recommended_side") or "").strip()
        won = _get(r, "won")
        if won is None or won == -1:
            continue  # skip ungraded and pushes (void)
        if side == "Over":
            p_win = float(p_over)
        elif side == "Under":
            p_win = 1.0 - float(p_over)
        else:
            continue
        points.append((p_win, 1.0 if won == 1 else 0.0))

    if len(points) < 10:
        return (1.0, 0.0)

    # Linear regression: realized = a * p_model + b
    n = len(points)
    mean_x = sum(p for p, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    cov = sum((p - mean_x) * (y - mean_y) for p, y in points)
    var_x = sum((p - mean_x) ** 2 for p, _ in points)
    if var_x < 1e-10:
        return (1.0, 0.0)
    a = cov / var_x
    b = mean_y - a * mean_x
    return (a, b)


class PropEdge(BaseModel):
    """Represents the value of a single prop; ranked by best_ev when available."""

    player_id: str = Field(..., description="Canonical Player.id for this prop")
    stat_type: StatType = Field(..., description="Stat category (points, rebounds, etc.)")
    provider: str = Field(..., description="Book/provider offering this prop")
    market_line: float = Field(..., gt=0, description="Posted line (threshold) from the book")
    projected: float = Field(..., description="Model's projected stat value (mean)")
    edge: float = Field(..., description="(projected - market_line) / market_line (secondary)")
    recommended_side: Literal['Over', 'Under', 'Pass'] = Field(..., description="Recommended side by EV, or Pass")
    p_over_model: Optional[float] = Field(None, description="Model P(stat > line)")
    p_under_model: Optional[float] = Field(None, description="Model P(stat < line)")
    ev_over: Optional[float] = Field(None, description="EV per $1 for Over")
    ev_under: Optional[float] = Field(None, description="EV per $1 for Under")
    best_ev: Optional[float] = Field(None, description="max(ev_over, ev_under) when > 0")
    projected_stdev: Optional[float] = Field(None, description="Projection stdev for display")
    recommended_odds: Optional[float] = Field(None, description="American odds for recommended side")
    over_odds: Optional[float] = Field(None, description="Best over odds (for display)")
    under_odds: Optional[float] = Field(None, description="Best under odds (for display)")
    over_provider: Optional[str] = Field(None, description="Book with best over odds")
    under_provider: Optional[str] = Field(None, description="Book with best under odds")
    recommended_provider: Optional[str] = Field(None, description="Book for recommended side")
    open_line: Optional[float] = Field(None, description="Earliest line of the day (for movement)")
    current_line: Optional[float] = Field(None, description="Current market line (same as market_line when set)")
    delta_line: Optional[float] = Field(None, description="current_line - open_line")


def compute_edge(projected: float, market_line: float) -> float:
    """
    Computes Edge = (projected - market_line) / market_line.

    Raises if market_line is not strictly positive.
    """
    if market_line <= 0.0:
        raise ValueError("market_line must be > 0 to compute edge.")
    return (projected - market_line) / market_line


def rank_props_by_edge(
    snapshot: MarketSnapshot,
    get_projection: ProjectionProvider,
    get_projection_result: Optional[ProjectionResultProvider] = None,
    get_last_n_values: Optional[LastNValuesProvider] = None,
    calibration_params: Optional[Tuple[float, float]] = None,
    ev_threshold: Optional[float] = None,
) -> List[PropEdge]:
    """
    Returns a ranked list of props with EV-based ranking (best_ev desc, fallback abs(edge)).

    Uses rec_threshold = ev_threshold if ev_threshold is not None else 0.02 for recommendation
    checks (never compare to None). When EV is computed, recommends Over/Under only if
    best_ev >= rec_threshold; else Pass and recommended_odds/recommended_provider None.
    When best_ev is positive but below rec_threshold, best_ev is set to None so Pass plays
    don't rank above real bets. Fallback when EV cannot be computed: edge-based recommendation
    unchanged. If ev_threshold is not None, filter results at end; otherwise no filter.

    Args:
        snapshot: MarketSnapshot containing the available prop lines.
        get_projection: Returns projected mean for (player_id, stat_type); None skips prop.
        get_projection_result: Optional. Returns ProjectionResult(mean, stdev, n, confidence) for
            P(Over)/EV computation. If None, EV fields are null; fallback to edge-based recommendation.
        calibration_params: Optional (a, b). If set, EV uses p' = clamp(a*p + b, 0, 1)
            for P(Over) before computing EV; PropEdge still stores raw p_over_model.
        ev_threshold: Optional. If set, only recommend when best_ev >= ev_threshold and filter at end.
            If None, rec_threshold=0.02 and no filtering.
    """
    ranked: List[PropEdge] = []
    rec_threshold = ev_threshold if ev_threshold is not None else 0.02
    prob_method = _prob_method_env()
    empirical_smoothing = _empirical_smoothing_env()

    # Calibration (a, b) applied to p_over before EV when supplied by caller; main.py loads and logs.
    p_min, p_max = _prob_bounds_from_env()

    for line in snapshot.lines:
        projected = get_projection(line.player_id, line.stat_type)
        if projected is None:
            continue

        try:
            edge_value = compute_edge(projected, line.threshold)
        except ValueError:
            continue

        if abs(edge_value) > 0.40:
            continue

        p_over_model_val: Optional[float] = None
        p_under_model_val: Optional[float] = None
        projected_stdev_val: Optional[float] = None
        result = get_projection_result(line.player_id, line.stat_type) if get_projection_result else None
        if result is not None:
            projected_stdev_val = result.stdev
            if prob_method == "empirical" and get_last_n_values is not None:
                last_n = get_last_n_values(line.player_id, line.stat_type)
                if last_n:
                    p_over_model_val = empirical_prob_over(line.threshold, last_n, empirical_smoothing)
                    p_under_model_val = 1.0 - p_over_model_val
                else:
                    p_over_model_val = model_prob_over(line.threshold, result.mean, result.stdev)
                    p_under_model_val = 1.0 - p_over_model_val
            else:
                p_over_model_val = model_prob_over(line.threshold, result.mean, result.stdev)
                p_under_model_val = 1.0 - p_over_model_val

        ev_over_val: Optional[float] = None
        ev_under_val: Optional[float] = None
        recommended_side = 'Pass'
        recommended_odds_val: Optional[float] = None
        over_provider_val = getattr(line, "over_provider", None)
        under_provider_val = getattr(line, "under_provider", None)
        recommended_provider_val: Optional[str] = None

        if (
            p_over_model_val is not None
            and p_under_model_val is not None
            and line.over_odds is not None
            and line.under_odds is not None
        ):
            # Optionally apply calibration for EV only (PropEdge keeps raw p_over/p_under),
            # then clamp EV probabilities to [P_MIN, P_MAX] from env.
            p_over_ev = p_over_model_val
            p_under_ev = p_under_model_val
            if calibration_params is not None:
                a, b = calibration_params
                p_over_ev = _clamp_calibrated(p_over_model_val, a, b)
                p_under_ev = 1.0 - p_over_ev
            # Clamp to configured probability bounds (for EV only)
            p_over_ev = max(p_min, min(p_max, p_over_ev))
            p_under_ev = 1.0 - p_over_ev
            profit_over = american_odds_to_profit_per_dollar(line.over_odds)
            profit_under = american_odds_to_profit_per_dollar(line.under_odds)
            if profit_over is not None and profit_under is not None:
                ev_over_val = ev_per_unit(p_over_ev, profit_over)
                ev_under_val = ev_per_unit(p_under_ev, profit_under)
                best_ev_here = max(ev_over_val, ev_under_val)
                if best_ev_here >= rec_threshold:
                    if ev_over_val >= ev_under_val:
                        recommended_side = 'Over'
                        recommended_odds_val = line.over_odds
                        recommended_provider_val = over_provider_val or line.provider
                    else:
                        recommended_side = 'Under'
                        recommended_odds_val = line.under_odds
                        recommended_provider_val = under_provider_val or line.provider
                else:
                    recommended_side = 'Pass'
                    recommended_odds_val = None
                    recommended_provider_val = None
        if recommended_side == 'Pass' and (ev_over_val is None or ev_under_val is None):
            # Fallback to edge-based recommendation
            if edge_value > 0.05:
                recommended_side = 'Over'
                recommended_odds_val = line.over_odds
                recommended_provider_val = over_provider_val or line.provider
            elif edge_value < -0.05:
                recommended_side = 'Under'
                recommended_odds_val = line.under_odds
                recommended_provider_val = under_provider_val or line.provider

        best_ev_val: Optional[float] = None
        if ev_over_val is not None and ev_under_val is not None:
            best_ev_val = max(ev_over_val, ev_under_val)
            if best_ev_val <= 0:
                best_ev_val = None
            elif best_ev_val < rec_threshold:
                best_ev_val = None

        ranked.append(
            PropEdge(
                player_id=line.player_id,
                stat_type=line.stat_type,
                provider=line.provider,
                market_line=line.threshold,
                projected=projected,
                edge=edge_value,
                recommended_side=recommended_side,
                p_over_model=p_over_model_val,
                p_under_model=p_under_model_val,
                ev_over=ev_over_val,
                ev_under=ev_under_val,
                best_ev=best_ev_val,
                projected_stdev=projected_stdev_val,
                recommended_odds=recommended_odds_val,
                over_odds=line.over_odds,
                under_odds=line.under_odds,
                over_provider=over_provider_val,
                under_provider=under_provider_val,
                recommended_provider=recommended_provider_val,
                open_line=None,
                current_line=None,
                delta_line=None,
            )
        )

    if ev_threshold is not None:
        def _passes_filter(item: PropEdge) -> bool:
            if item.best_ev is not None:
                return item.best_ev >= ev_threshold
            return item.edge >= ev_threshold
        ranked = [e for e in ranked if _passes_filter(e)]

    # Sort: EV bets first (higher best_ev first), then non-Pass by |edge|, then Pass last
    def _sort_key(item: PropEdge) -> tuple:
        if item.recommended_side == "Pass":
            return (2, 0)  # Pass always below any EV or edge bet
        ev = item.best_ev
        if ev is not None:
            return (0, -ev)
        return (1, -abs(item.edge))

    ranked.sort(key=_sort_key)
    return ranked

