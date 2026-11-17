"""The submitted pipeline: fill the ball from the broadcast, train, label and validate."""

import json
from pathlib import Path

from pitch_dynamics import mlp, official
from pitch_dynamics.broadcast import fill_match
from pitch_dynamics.challenge import load_challenge, provenance, write_submission
from pitch_dynamics.guard import refuse_agent
from pitch_dynamics.matches import TRAINING, load_match


def predict(out_dir: Path, data_dir: Path | None = None) -> Path:
    """Write `submission.csv` and `provenance.json` to `out_dir`; returns the validated CSV path.

    The tactical ball gaps are filled from the aligned broadcast ATD, the model is fitted on every
    external DFL match, and the official validator checks the file.
    """
    refuse_agent()
    filled = fill_match(load_challenge(data_dir), data_dir)
    model = mlp.fit([load_match(m) for m in TRAINING])
    labels = mlp.predict(model, filled.match)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission = out_dir / "submission.csv"
    write_submission(submission, labels.tolist())
    record = {
        "description": mlp.DESCRIPTION,
        "trained_on": list(TRAINING),
        "broadcast_alignment": filled.summary,
        **provenance(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    official.validate(submission)
    return submission
