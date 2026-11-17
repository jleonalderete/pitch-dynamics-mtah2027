"""Decode per-frame class log-probabilities into labels with a three-state Viterbi."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pitch_dynamics.matches import Interval

CLASSES = ("H", "A", "D")
_LABELS = np.array(CLASSES)


def change_costs(labels: NDArray[Any], mask: NDArray[np.bool_]) -> tuple[float, float]:
    """Switch costs log(P(stay) / P(change)) from consecutive labelled frames, add-one smoothed.

    Returns the cost of a change between teams and the cost of entering or leaving a dead ball.
    """
    pairs = mask[:-1] & mask[1:]
    before, after = labels[:-1][pairs], labels[1:][pairs]
    live = np.isin(before, ("H", "A")) & np.isin(after, ("H", "A"))
    stay = np.sum(before == after) + 1
    team = np.sum(live & (before != after)) + 1
    dead = np.sum((before != after) & ((before == "D") | (after == "D"))) + 1
    return float(np.log(stay / team)), float(np.log(stay / dead))


def viterbi(scores: NDArray[Any], switch_cost: float, dead_switch_cost: float) -> NDArray[np.int8]:
    """Best state path over (frames, 3) log-scores; a state change costs the given penalty."""
    n_frames = len(scores)
    if not n_frames:
        return np.empty(0, dtype=np.int8)
    previous = scores[0].copy()
    back = np.zeros((n_frames, 3), dtype=np.int8)
    costs = np.array(
        [
            [0, switch_cost, dead_switch_cost],
            [switch_cost, 0, dead_switch_cost],
            [dead_switch_cost, dead_switch_cost, 0],
        ],
        dtype=np.float64,
    )
    states_index = np.arange(3)
    for frame in range(1, n_frames):
        candidates = previous[:, None] - costs
        parents = np.argmax(candidates, axis=0)
        back[frame] = parents
        previous = candidates[parents, states_index] + scores[frame]
        previous -= np.max(previous)  # stable even for a full-match sequence
    states = np.empty(n_frames, dtype=np.int8)
    states[-1] = np.argmax(previous)
    for frame in range(n_frames - 1, 0, -1):
        states[frame - 1] = back[frame, states[frame]]
    return states


def decode(
    scores: NDArray[Any],
    intervals: Sequence[Interval],
    forced_dead: Sequence[tuple[int, int]],
    switch_cost: float,
    dead_switch_cost: float,
) -> NDArray[np.str_]:
    """Labels for every frame: each half decoded alone, `forced_dead` frames dead, X elsewhere."""
    labels = np.full(len(scores), "X", dtype="<U1")
    for _, start, stop in intervals:
        local = scores[start:stop].copy()
        for first, last in forced_dead:
            low, high = max(first, start) - start, min(last, stop) - start
            if low < high:
                local[low:high, :2] = -np.inf
                local[low:high, 2] = 0
        labels[start:stop] = _LABELS[viterbi(local, switch_cost, dead_switch_cost)]
    return labels
