"""Leave-one-match-out and holdout scoring, with a fake model over synthetic matches."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from synthetic import install_matches, use_cache

from pitch_dynamics import cli, evaluation, matches
from pitch_dynamics.matches import DEV, HOLDOUT_DFL, HOLDOUT_LALIGA, TRAINING, Match

ALL_IDS = (*DEV, *HOLDOUT_DFL, *HOLDOUT_LALIGA)


class FakeModel(SimpleNamespace):
    """Predicts the true labels of every match it is shown; records what it was trained on."""

    def __init__(self, truth: dict[str, np.ndarray]) -> None:
        super().__init__(DESCRIPTION="fake", truth=truth, seen=[], fits=[])

    def fit(self, matches_: list[Match], seed: int = 0) -> SimpleNamespace:
        self.fits.append(([m.id for m in matches_], seed))
        return SimpleNamespace(seed=seed)

    def predict(self, model: SimpleNamespace, match: Match) -> np.ndarray:
        self.seen.append(set(np.unique(match.labels)))
        return self.truth[match.id].copy()


@pytest.fixture(autouse=True)
def synthetic_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDECODE", raising=False)
    use_cache(monkeypatch, tmp_path)
    install_matches(tmp_path, ALL_IDS, n_half=300)


@pytest.fixture
def truth() -> dict[str, np.ndarray]:
    return {match_id: matches.load_match(match_id).labels.copy() for match_id in ALL_IDS}


@pytest.fixture
def fake(truth: dict[str, np.ndarray], monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel(truth)
    monkeypatch.setattr(evaluation, "mlp", model)
    return model


class TestScore:
    def test_a_perfect_prediction_scores_one(self, truth: dict[str, np.ndarray]):
        labels = truth["J03WMX"]
        scores = evaluation.score(labels, labels)
        assert scores["spell_f1"] == scores["macro_f1"] == scores["d_f1"] == 1.0
        assert "spell_bootstrap_low" not in scores

    def test_bootstrap_fields_appear_on_request(self, truth: dict[str, np.ndarray]):
        labels = truth["J03WMX"]
        scores = evaluation.score(labels, labels, 20)
        assert {"spell_bootstrap_low", "spell_bootstrap_high", "team_acc_gt_alive"} <= set(scores)

    def test_a_constant_prediction_scores_lower(self, truth: dict[str, np.ndarray]):
        labels = truth["J03WMX"]
        scores = evaluation.score(labels, np.full(len(labels), "H", dtype="<U1"))
        assert scores["spell_f1"] < 1.0 and scores["d_f1"] == 0.0


class TestDev:
    def test_trains_on_the_other_dev_matches_for_each_seed_and_hides_labels(self, fake: FakeModel):
        shown: list[tuple[str, float]] = []
        report = evaluation.dev((0, 1), on_match=lambda m, s: shown.append((m, s["spell_f1"])))
        expected = [([m for m in DEV if m != held], seed) for held in DEV for seed in (0, 1)]
        assert fake.fits == expected
        assert all(labels == {"X"} for labels in fake.seen) and len(fake.seen) == 10
        assert list(report["matches"]) == list(DEV) and report["seeds"] == [0, 1]
        assert [m for m, _ in shown] == list(DEV)
        for fold in report["matches"].values():
            assert fold["spell_f1"] == 1.0 and fold["per_seed_spell_f1"] == [1.0, 1.0]
        assert report["mean"] == {"spell_f1": 1.0, "macro_f1": 1.0, "d_f1": 1.0}
        assert report["description"] == "fake"

    def test_folds_are_the_mean_over_seeds(
        self, fake: FakeModel, truth: dict[str, np.ndarray], monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            fake,
            "predict",
            lambda model, match: (
                truth[match.id].copy()
                if model.seed == 0
                else np.full(len(match.labels), "H", dtype="<U1")
            ),
        )
        report = evaluation.dev()
        for fold in report["matches"].values():
            first, second = fold["per_seed_spell_f1"]
            assert first == 1.0 and second < 1.0
            assert fold["spell_f1"] == pytest.approx((first + second) / 2)


class TestHoldout:
    def test_sets_are_fixed(self):
        assert evaluation.HOLDOUT_SETS == {
            "dfl": (DEV, HOLDOUT_DFL),
            "laliga": (TRAINING, HOLDOUT_LALIGA),
        }
        for trained_on, held_out in evaluation.HOLDOUT_SETS.values():
            assert set(trained_on).isdisjoint(held_out)

    @pytest.mark.parametrize(
        ("which", "trained_on", "held_out"),
        [("dfl", DEV, HOLDOUT_DFL), ("laliga", TRAINING, HOLDOUT_LALIGA)],
    )
    def test_trains_on_the_set_and_scores_the_reserved_matches(
        self, which: str, trained_on: tuple[str, ...], held_out: tuple[str, ...], fake: FakeModel
    ):
        report = evaluation.holdout(which)
        assert fake.fits == [(list(trained_on), 0)]
        assert all(labels == {"X"} for labels in fake.seen)
        assert list(report["matches"]) == list(held_out)
        assert report["trained_on"] == list(trained_on) and report["set"] == which
        assert report["mean_spell_f1"] == 1.0
        assert "spell_bootstrap_low" in report["matches"][held_out[0]]
        json.dumps(report)


class TestCommandLine:
    @pytest.fixture(autouse=True)
    def artifacts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "ARTIFACTS", tmp_path / "artifacts")

    def test_evaluate_prints_compact_metrics_and_writes_the_report(
        self, fake: FakeModel, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ):
        cli.main(["evaluate", "--seeds", "0"])
        lines = capsys.readouterr().out.splitlines()
        assert [line.split()[0] for line in lines[:6]] == [*DEV, "mean"]
        assert all("spellF1 1.0000" in line for line in lines[:6])
        path = tmp_path / "artifacts" / "evaluate" / "report.json"
        assert lines[6] == str(path)
        assert json.loads(path.read_text())["seeds"] == [0]

    def test_a_missing_cache_exits_with_the_data_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(matches, "CACHE", tmp_path / "empty")
        with pytest.raises(SystemExit, match="pitch-dynamics data"):
            cli.main(["evaluate"])

    def test_holdout_prints_the_mean_and_writes_one_report_per_set(
        self, fake: FakeModel, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ):
        cli.main(["holdout", "--set", "laliga"])
        lines = capsys.readouterr().out.splitlines()
        assert [line.split()[0] for line in lines[:2]] == list(HOLDOUT_LALIGA)
        assert lines[2] == "mean spellF1 1.0000"
        report = tmp_path / "artifacts" / "holdout" / "laliga.json"
        assert json.loads(report.read_text())["set"] == "laliga"

    def test_data_downloads_the_training_matches_and_optionally_the_laliga_holdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        fetched: list[str] = []
        monkeypatch.setattr(matches, "fetch", fetched.append)
        cli.main(["data"])
        assert fetched == list(TRAINING)
        out = capsys.readouterr().out.splitlines()
        assert [line.split(":")[0] for line in out] == list(TRAINING)
        assert all("600 frames" in line for line in out)
        fetched.clear()
        cli.main(["data", "--laliga"])
        assert fetched == list(ALL_IDS)

    def test_every_command_documents_itself(self, capsys: pytest.CaptureFixture[str]):
        for command in ([], ["data"], ["evaluate"], ["holdout"], ["align"], ["predict"]):
            with pytest.raises(SystemExit) as raised:
                cli.main([*command, "--help"])
            assert raised.value.code == 0
        assert "usage: pitch-dynamics" in capsys.readouterr().out
