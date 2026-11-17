"""The submitted pipeline end to end, with the model and the broadcast step replaced by fakes."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from synthetic import write_atd_bundle

from pitch_dynamics import official, submission
from pitch_dynamics.challenge import FORCED_DEAD
from pitch_dynamics.matches import TRAINING, Match

N_FRAMES = 199319


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_atd_bundle(tmp_path_factory.mktemp("bundle"))


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A model that labels the first half H and the second A, and a no-op broadcast step."""
    seen = SimpleNamespace(fitted=[], filled=[])

    def fill_match(match: Match, data_dir: Path | None) -> SimpleNamespace:
        seen.filled.append(match.id)
        return SimpleNamespace(match=match, summary={"offsets": []})

    def predict(model: str, match: Match) -> np.ndarray:
        labels = np.full(len(match.labels), "X", dtype="<U1")
        for (_, start, stop), label in zip(match.intervals, "HA", strict=True):
            labels[start:stop] = label
        labels[slice(*FORCED_DEAD[0])] = "D"
        return labels

    model = SimpleNamespace(
        DESCRIPTION="fake",
        fit=lambda matches: seen.fitted.append([m.id for m in matches]) or "model",
        predict=predict,
    )
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.setattr(submission, "fill_match", fill_match)
    monkeypatch.setattr(submission, "mlp", model)
    monkeypatch.setattr(submission, "load_match", lambda match_id: SimpleNamespace(id=match_id))
    return seen


def test_writes_a_validated_submission_and_its_provenance(
    bundle: Path, tmp_path: Path, fakes: SimpleNamespace
):
    path = submission.predict(tmp_path / "run", bundle)
    labels = path.read_text().splitlines()[1:]
    assert path == tmp_path / "run" / "submission.csv" and len(labels) == N_FRAMES
    assert labels[0].endswith(",X") and labels[110808].endswith(",D")
    assert labels[113742].endswith(",A")
    assert fakes.filled == ["challenge"] and fakes.fitted == [list(TRAINING)]
    record = json.loads((tmp_path / "run" / "provenance.json").read_text())
    assert record["trained_on"] == list(TRAINING)
    assert record["broadcast_alignment"] == {"offsets": []}
    assert {"commit", "python", "numpy", "pandas", "platform"} <= set(record)


def test_the_official_validator_rejects_a_bad_file(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,label\n0,H\n")
    with pytest.raises(subprocess.CalledProcessError):
        official.validate(bad)
