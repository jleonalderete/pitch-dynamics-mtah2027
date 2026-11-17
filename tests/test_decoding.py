"""Viterbi decoding, forced dead ball, half isolation and switch costs."""

import itertools

import numpy as np
import pytest

from pitch_dynamics.decoding import CLASSES, change_costs, decode, viterbi


def energy(scores: np.ndarray, costs: np.ndarray, path: tuple[int, ...]) -> float:
    gain = sum(scores[i, state] for i, state in enumerate(path))
    return gain - sum(costs[path[i - 1], path[i]] for i in range(1, len(path)))


def cost_matrix(switch: float, dead: float) -> np.ndarray:
    return np.array([[0, switch, dead], [switch, 0, dead], [dead, dead, 0]], dtype=float)


def test_classes_are_home_away_dead():
    assert CLASSES == ("H", "A", "D")


def test_viterbi_matches_exhaustive_optimum():
    scores = np.array([[2.0, 0.0, -2.0], [0.0, 3.0, -1.0], [0.0, -1.0, 4.0], [1.0, 0.0, 0.0]])
    costs = cost_matrix(2, 3)
    best = max(energy(scores, costs, path) for path in itertools.product(range(3), repeat=4))
    assert energy(scores, costs, tuple(viterbi(scores, 2, 3))) == best


@pytest.mark.parametrize("seed", range(5))
def test_viterbi_matches_exhaustive_optimum_on_random_scores(seed: int):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 2, (6, 3)).astype(np.float32)
    switch, dead = rng.uniform(0, 4, 2)
    costs = cost_matrix(switch, dead)
    best = max(energy(scores, costs, path) for path in itertools.product(range(3), repeat=6))
    found = energy(scores, costs, tuple(viterbi(scores, switch, dead)))
    assert found == pytest.approx(best, abs=1e-4)


def test_a_brief_deflection_is_ignored_but_clear_evidence_switches():
    scores = np.tile([2.0, 0.0, -5.0], (30, 1))
    scores[10] = [0, 2, -5]
    assert (viterbi(scores, 8, 6) == 0).all()
    scores[10:13] = [0, 10, -5]
    result = viterbi(scores, 8, 6)
    assert (result[10:13] == 1).all()
    assert (result[:10] == 0).all() and (result[13:] == 0).all()


def test_viterbi_handles_empty_and_single_frame_sequences():
    assert viterbi(np.empty((0, 3)), 1, 1).shape == (0,)
    assert viterbi(np.array([[0.0, 1.0, 0.5]]), 1, 1).tolist() == [1]


def test_ties_pick_the_lowest_state():
    assert viterbi(np.zeros((4, 3)), 1, 1).tolist() == [0, 0, 0, 0]


def test_decode_forces_dead_inside_halves_and_x_outside():
    scores = np.tile([2.0, 0.0, -2.0], (40, 1))
    intervals = [("1", 5, 15), ("2", 25, 35)]
    labels = decode(scores, intervals, [(25, 29)], 8, 6)
    assert "".join(labels[25:30]) == "DDDDH"
    assert (labels[:5] == "X").all() and (labels[15:25] == "X").all() and (labels[35:] == "X").all()
    assert (labels[5:15] == "H").all() and (labels[30:35] == "H").all()
    assert labels.dtype == np.dtype("<U1")


def test_forced_dead_outside_every_half_changes_nothing():
    scores = np.tile([2.0, 0.0, -2.0], (40, 1))
    intervals = [("1", 10, 20)]
    free = decode(scores, intervals, [], 8, 6)
    np.testing.assert_array_equal(decode(scores, intervals, [(0, 10), (20, 40)], 8, 6), free)


def test_forced_dead_stretch_spanning_a_half_start_is_clipped():
    scores = np.tile([2.0, 0.0, -2.0], (30, 1))
    labels = decode(scores, [("2", 10, 30)], [(5, 14)], 8, 6)
    assert "".join(labels[10:15]) == "DDDDH"
    assert (labels[:10] == "X").all()


def test_halves_are_decoded_independently():
    first = np.tile([5.0, 0.0, -9.0], (12, 1))
    second = np.tile([0.0, 5.0, -9.0], (10, 1))
    scores = np.vstack([first, second])
    labels = decode(scores, [("1", 0, 12), ("2", 12, 22)], [], 1000, 1000)
    assert "".join(labels) == "H" * 12 + "A" * 10
    # One sequence would pay the huge switch cost and stay home throughout.
    assert (viterbi(scores, 1000, 1000) == 0).all()


def test_decode_does_not_modify_the_scores():
    scores = np.tile([2.0, 0.0, -2.0], (20, 1))
    before = scores.copy()
    decode(scores, [("1", 0, 20)], [(5, 10)], 8, 6)
    np.testing.assert_array_equal(scores, before)


def test_change_costs_count_stays_team_switches_and_dead_switches():
    labels = np.array(list("HHHAADDH"))
    team, dead = change_costs(labels, np.ones(len(labels), dtype=bool))
    # pairs: HH HH HA AA AD DD DH -> stay 4, team change 1, dead change 2 (each +1)
    assert team == pytest.approx(np.log(5 / 2))
    assert dead == pytest.approx(np.log(5 / 3))


def test_change_costs_ignore_unlabelled_frames():
    labels = np.array(list("HHXAA"))
    mask = labels != "X"
    team, dead = change_costs(labels, mask)
    assert team == pytest.approx(np.log(3 / 1)) and dead == pytest.approx(np.log(3 / 1))


def test_team_switches_are_dearer_than_staying():
    rng = np.random.default_rng(0)
    labels = np.repeat(rng.choice(list("HAD"), 200), 25)
    team, dead = change_costs(labels, labels != "X")
    assert team > 0 and dead > 0
