"""
Backtest graded picks: hit rate, Brier score, log loss, calibration bin report (avg predicted vs actual hit rate),
EV vs realized profit. Use --fit-calibration to fit and save calibration (a, b) and print before/after Brier/logloss.

Run from project root: python -m src.backtest
Or: python -m src.backtest --fit-calibration
"""

import sys
import math
from pathlib import Path
from typing import Optional

# Allow running as script from src/ (e.g. python backtest.py)
_root = Path(__file__).resolve().parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

from src.database import DatabaseManager
from src.optimizer import (
    profit_per_unit,
    fit_calibration_from_picks,
    save_calibration,
    DEFAULT_CALIBRATION_PATH,
    _clamp_calibrated,
)


# Bins for model probability (Over): (low, high) in [0, 1]
P_OVER_BINS = [
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 1.00),
]


def _p_win_for_pick(row) -> Optional[float]:
    """Model P(win) for this pick: p_over if Over, 1-p_over if Under."""
    try:
        p_over = row["p_over_model"] if "p_over_model" in row.keys() else None
    except (KeyError, TypeError):
        p_over = None
    if p_over is None:
        return None
    side = (row["recommended_side"] if "recommended_side" in row.keys() else "").strip()
    if side == "Over":
        return float(p_over)
    if side == "Under":
        return 1.0 - float(p_over)
    return None


def run_backtest(db_path: Optional[Path] = None) -> None:
    """
    Load graded picks from SQLite and print:
    - overall hit rate
    - hit rate by p_over_model bins (0.50-0.55, ... 0.75-0.80)
    - EV vs realized profit (per unit)
    """
    db = DatabaseManager(db_path=db_path)
    rows = db.get_graded_picks()
    db.close()

    if not rows:
        print("No graded picks (won IS NOT NULL). Grade picks to run backtest.")
        return

    # Exclude pushes (won == -1) from wins/losses and hit rate
    settled = [r for r in rows if r["won"] in (0, 1)]
    n = len(settled)
    if n == 0:
        print("No settled picks (all pushes). Grade picks to run backtest.")
        return
    wins = sum(1 for r in settled if r["won"] == 1)
    hit_rate_pct = round(wins / n * 100.0, 2)
    print("--- Backtest (graded picks only, pushes excluded) ---")
    print(f"Overall: n={n}, wins={wins}, hit rate={hit_rate_pct}%")

    # Brier score and log loss (picks with p_win; exclude pushes)
    p_win_outcomes: list[tuple[float, int]] = []
    for r in settled:
        p_win = _p_win_for_pick(r)
        if p_win is None:
            continue
        if r["won"] == -1:
            continue
        outcome = 1 if r["won"] == 1 else 0
        p_win_outcomes.append((p_win, outcome))
    if p_win_outcomes:
        n_prob = len(p_win_outcomes)
        brier = sum((p - y) ** 2 for p, y in p_win_outcomes) / n_prob
        eps = 1e-15
        log_loss = -sum(
            y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
            for p, y in p_win_outcomes
        ) / n_prob
        print("\nBrier score (lower better): {:.4f}".format(brier))
        print("Log loss (lower better): {:.4f}".format(log_loss))

    # Calibration bin report: avg predicted vs actual hit rate
    print("\nCalibration bins (avg predicted vs actual hit rate):")
    for low, high in P_OVER_BINS:
        in_bin = []
        for r in settled:
            p_win = _p_win_for_pick(r)
            if p_win is None:
                continue
            if r["won"] == -1:
                continue
            if low <= p_win < high:
                in_bin.append(r)
        if not in_bin:
            print(f"  {low:.2f}-{high:.2f}: (no picks)")
            continue
        avg_pred = sum(_p_win_for_pick(r) for r in in_bin) / len(in_bin)
        b_wins = sum(1 for r in in_bin if r["won"] == 1)
        actual_hr = b_wins / len(in_bin)
        print(f"  {low:.2f}-{high:.2f}: n={len(in_bin)}, avg_predicted={avg_pred:.3f}, actual_hit_rate={actual_hr:.3f}")

    # Expected EV (model) vs realized profit (actual), per $1 stake
    total_ev = 0.0
    total_realized = 0.0
    used = 0
    for r in settled:
        if r["won"] == -1:
            continue
        p_win = _p_win_for_pick(r)
        try:
            odds = r["odds"] if "odds" in r.keys() else None
        except (KeyError, TypeError):
            odds = None
        if p_win is None or odds is None:
            continue
        profit = profit_per_unit(odds)
        if profit is None:
            continue
        total_ev += p_win * profit - (1.0 - p_win) * 1.0
        won = 1 if r["won"] == 1 else 0
        total_realized += won * profit - (1 - won) * 1.0
        used += 1

    print("\nExpected EV (model) vs realized profit (actual), per $1 stake (pushes excluded):")
    print(f"  Picks used: {used} / {n}")
    print(f"  Sum expected EV:   {total_ev:.4f}")
    print(f"  Sum realized:      {total_realized:.4f}")
    print(f"  Difference (EV − realized): {total_ev - total_realized:.4f}")


def _brier_and_logloss(p_win_outcomes: list[tuple[float, int]]) -> tuple[float, float]:
    """Brier score and log loss from list of (p_win, outcome). Returns (brier, logloss)."""
    if not p_win_outcomes:
        return (0.0, 0.0)
    n = len(p_win_outcomes)
    eps = 1e-15
    brier = sum((p - y) ** 2 for p, y in p_win_outcomes) / n
    logloss = -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in p_win_outcomes
    ) / n
    return (brier, logloss)


if __name__ == "__main__":
    if "--fit-calibration" in sys.argv:
        db = DatabaseManager()
        rows = db.get_graded_picks()
        db.close()
        if len(rows) < 10:
            print("Need at least 10 graded picks to fit calibration. Run backtest without --fit-calibration first.")
        else:
            # Build (p_win, outcome) for each pick (exclude pushes)
            p_win_outcomes: list[tuple[float, int]] = []
            for r in rows:
                if r["won"] == -1:
                    continue
                p_win = _p_win_for_pick(r)
                if p_win is None:
                    continue
                outcome = 1 if r["won"] == 1 else 0
                p_win_outcomes.append((float(p_win), outcome))
            brier_before, logloss_before = _brier_and_logloss(p_win_outcomes)
            print("Before calibration: Brier = {:.4f}, Log loss = {:.4f}".format(brier_before, logloss_before))

            a, b = fit_calibration_from_picks(rows)
            save_calibration(DEFAULT_CALIBRATION_PATH, a, b)
            print(f"Fitted calibration a={a:.4f}, b={b:.4f} -> saved to {DEFAULT_CALIBRATION_PATH}")

            # After: p' = clamp(a*p + b, 0, 1)
            p_cal_outcomes = [(_clamp_calibrated(p, a, b), y) for p, y in p_win_outcomes]
            brier_after, logloss_after = _brier_and_logloss(p_cal_outcomes)
            print("After calibration:  Brier = {:.4f}, Log loss = {:.4f}".format(brier_after, logloss_after))
    run_backtest()
