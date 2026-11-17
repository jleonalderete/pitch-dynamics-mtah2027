"""Reserved steps refuse to run inside an AI agent, before they touch any data."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pitch_dynamics import broadcast, challenge, cli, evaluation, official, submission
from pitch_dynamics.guard import refuse_agent


@pytest.fixture
def inside_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")


def forbidden(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("touched data or models after the guard should have refused")


def test_refuse_agent_exits_with_a_clear_message_inside_an_agent(inside_agent: None):
    with pytest.raises(SystemExit) as raised:
        refuse_agent()
    assert "own terminal" in str(raised.value)


def test_refuse_agent_lets_people_through(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLAUDECODE", raising=False)
    refuse_agent()
    monkeypatch.setenv("CLAUDECODE", "")
    refuse_agent()


def test_load_challenge_refuses_before_opening_anything(inside_agent: None, tmp_path: Path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(official, "games", forbidden)
        with pytest.raises(SystemExit):
            challenge.load_challenge(tmp_path)


def test_broadcast_alignment_refuses_before_opening_anything(inside_agent: None, tmp_path: Path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(official, "games", forbidden)
        patch.setattr(broadcast, "load_challenge", forbidden)
        with pytest.raises(SystemExit):
            broadcast.load_broadcast(("Barcelona", "Athletic Club"), tmp_path)
        with pytest.raises(SystemExit):
            broadcast.run(tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


def test_predict_refuses_before_loading_a_model_or_data(inside_agent: None, tmp_path: Path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(submission, "load_challenge", forbidden)
        patch.setattr(submission, "load_match", forbidden)
        with pytest.raises(SystemExit):
            submission.predict(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_holdout_refuses_before_loading_matches(inside_agent: None):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluation, "load_match", forbidden)
        with pytest.raises(SystemExit):
            evaluation.holdout("laliga")


def test_command_line_handlers_refuse_before_calling_anything(inside_agent: None):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli.evaluation, "holdout", forbidden)
        patch.setattr(cli.submission, "predict", forbidden)
        patch.setattr(cli.broadcast, "run", forbidden)
        for command in (["holdout", "--set", "dfl"], ["predict"], ["align"]):
            with pytest.raises(SystemExit) as raised:
                cli.main(command)
            assert "own terminal" in str(raised.value)


def test_data_and_evaluate_are_open_to_agents(inside_agent: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli.matches, "fetch", lambda match_id: None)
    monkeypatch.setattr(cli.matches, "summary", lambda match_id: {"frames": 1, "labels": {}})
    cli.main(["data"])


@pytest.mark.parametrize("command", [["holdout", "--set", "dfl"], ["predict"], ["align"]])
def test_the_installed_command_refuses_inside_an_agent(command: list[str], tmp_path: Path):
    env = {**os.environ, "CLAUDECODE": "1", "PITCH_DYNAMICS_CACHE": str(tmp_path / "cache")}
    done = subprocess.run(
        [sys.executable, "-m", "pitch_dynamics", *command],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert done.returncode == 1
    assert "own terminal" in done.stderr
    assert not (tmp_path / "cache").exists() and not (tmp_path / "artifacts").exists()
