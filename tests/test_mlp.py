"""The model contract on synthetic matches: halves, forced dead ball, seeds and threads."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from synthetic import install_matches, use_cache
from threadpoolctl import threadpool_info

from pitch_dynamics import mlp
from pitch_dynamics.challenge import FORCED_DEAD
from pitch_dynamics.design import N_COLUMNS
from pitch_dynamics.matches import Match, Tracking, load_match

TRAIN_IDS = ["J03WMX", "J03WOH", "J03WOY"]
HELD_ID = "J03WPY"


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("cache")
    with pytest.MonkeyPatch.context() as patch:
        use_cache(patch, root)
        install_matches(root, [*TRAIN_IDS, HELD_ID])
        yield root


@pytest.fixture(scope="module")
def held(cache: Path) -> Match:
    """A held-out match: labels hidden, a gap between the halves and a forced-dead stretch."""
    match = load_match(HELD_ID)
    match.labels = np.full(len(match.labels), "X", dtype="<U1")
    return replace(
        match, intervals=[("1", 100, 1100), ("2", 1300, 2300)], forced_dead=[(1400, 1500)]
    )


@pytest.fixture(scope="module")
def trained(cache: Path) -> mlp.Model:
    return mlp.fit([load_match(m) for m in TRAIN_IDS])


class TestPredict:
    def test_labels_every_frame_and_respects_halves_and_forced_dead(
        self, trained: mlp.Model, held: Match
    ):
        model = trained
        labels = mlp.predict(model, held)
        assert labels.shape == (len(held.labels),) and labels.dtype == np.dtype("<U1")
        assert set(labels) <= {"H", "A", "D", "X"}
        assert (labels[:100] == "X").all() and (labels[1100:1300] == "X").all()
        assert (labels[2300:] == "X").all()
        assert (labels[1400:1500] == "D").all()
        assert (labels[100:1100] != "X").all() and (labels[1300:2300] != "X").all()
        assert len(set(labels[100:1100])) > 1  # it does not collapse to a single class

    def test_never_reads_the_labels(self, trained: mlp.Model, held: Match):
        class Poison:
            def __getattribute__(self, name: str) -> None:
                raise AssertionError("predict read match.labels")

        model = trained
        expected = mlp.predict(model, held)
        poisoned = replace(held, labels=Poison())
        np.testing.assert_array_equal(mlp.predict(model, poisoned), expected)

    def test_log_proba_are_log_probabilities_in_class_order(self, trained: mlp.Model, held: Match):
        model = trained
        scores = mlp.log_proba(model, held)
        assert scores.shape == (len(held.labels), 3)
        inside = np.zeros(len(scores), dtype=bool)
        inside[100:1100] = inside[1300:2300] = True
        np.testing.assert_allclose(np.exp(scores[inside]).sum(axis=1), 1.0, atol=1e-3)
        assert (scores[inside] <= 1e-6).all()

    def test_a_full_length_match_with_the_published_dead_stretch(
        self, trained: mlp.Model, held: Match
    ):
        model = trained
        n = 199319
        reps = -(-n // len(held.labels))
        tracking = Tracking(
            np.tile(held.tracking.players, (reps, 1, 1))[:n],
            np.tile(held.tracking.ball, (reps, 1))[:n],
            held.tracking.teams,
            held.tracking.team_names,
        )
        challenge_like = Match(
            "challenge",
            tracking,
            np.full(n, "X", dtype="<U1"),
            [("1", 11185, 90652), ("2", 110808, 190198)],
            held.pitch,
            held.fps,
            forced_dead=FORCED_DEAD,
            profile="challenge",
        )
        labels = mlp.predict(model, challenge_like)
        assert labels.shape == (n,)
        assert (labels[110808:113742] == "D").all()
        assert (labels[:11185] == "X").all() and (labels[90652:110808] == "X").all()

    def test_the_same_seed_gives_the_same_model_and_another_seed_a_different_one(
        self, trained: mlp.Model, held: Match
    ):
        model = trained
        again = mlp.fit([load_match(m) for m in TRAIN_IDS], seed=0)
        other = mlp.fit([load_match(m) for m in TRAIN_IDS], seed=1)
        base = mlp.log_proba(model, held)
        np.testing.assert_array_equal(mlp.log_proba(again, held), base)
        assert not np.array_equal(mlp.log_proba(other, held), base)


class TestMlp:
    def test_the_ensemble_has_two_seeded_members_with_the_design_width(self, cache: Path):
        model = mlp.fit([load_match(m) for m in TRAIN_IDS])
        assert len(model.members) == 2
        first, second = model.members
        assert first.w1.shape == (N_COLUMNS, 64) and first.w2.shape == (64, 3)
        assert not np.array_equal(first.w1, second.w1)
        assert model.switch_cost > 0 and model.dead_switch_cost > 0

    def test_fit_and_predict_run_on_a_single_blas_thread(
        self, cache: Path, held: Match, monkeypatch: pytest.MonkeyPatch
    ):
        seen: list[int] = []

        def blas_threads() -> None:
            seen.extend(i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas")

        real_fit, real_log_proba = mlp.fit_member, mlp.Member.log_proba

        def spy_fit(x: np.ndarray, y: np.ndarray, seed: int) -> mlp.Member:
            blas_threads()
            return real_fit(x, y, seed)

        def spy_log_proba(self: mlp.Member, x: np.ndarray) -> np.ndarray:
            blas_threads()
            return real_log_proba(self, x)

        monkeypatch.setattr(mlp, "fit_member", spy_fit)
        monkeypatch.setattr(mlp.Member, "log_proba", spy_log_proba)
        model = mlp.fit([load_match(m) for m in TRAIN_IDS])
        mlp.predict(model, held)
        assert len(seen) >= 4 and set(seen) == {1}

    def test_probabilities_are_averaged_not_log_probabilities(self, cache: Path, held: Match):
        model = mlp.fit([load_match(m) for m in TRAIN_IDS])
        x = mlp.design_matrix(held)
        mean = np.mean([np.exp(m.log_proba(x)) for m in model.members], axis=0)
        np.testing.assert_allclose(np.exp(mlp.log_proba(model, held)), np.maximum(mean, 1e-12))
