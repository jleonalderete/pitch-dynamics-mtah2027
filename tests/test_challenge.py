"""Challenge ingestion, the forced-dead rule and submission writing."""

from pathlib import Path

import numpy as np
import pytest
from synthetic import write_atd_bundle

from pitch_dynamics import challenge
from pitch_dynamics.challenge import FORCED_DEAD, load_challenge, load_tracking, write_submission
from pitch_dynamics.matches import Match

N_FRAMES = 199319


class TestAtdIngestion:
    @pytest.fixture(autouse=True)
    def files(self, tmp_path: Path):
        self.meta = tmp_path / "meta.xml"
        self.meta.write_text(
            '<root xmlns="urn:epts"><Team id="h"><Name>Barcelona</Name></Team>'
            '<Team id="a"><Name>Athletic Club</Name></Team>'
            '<Player teamId="h"/><Player teamId="a"/></root>'
        )
        self.raw = tmp_path / "raw.txt"

    def test_alignment_namespace_missing_and_slot_termination(self):
        self.raw.write_text(
            "1:0.2,0.5,3;0.8,0.5,4;:NaN,NaN,NaN\n2:NaN,NaN,NaN;0.8,0.5,4:0.8,0.5,1\n"
        )
        result = load_tracking(self.meta, self.raw, 2)
        assert result.players.shape == (2, 2, 3)
        assert result.players.dtype == np.float32
        assert np.isnan(result.ball[0]).all()
        assert np.isnan(result.players[1, 0]).all()
        np.testing.assert_array_equal(result.teams, [0, 1])
        assert result.team_names == ("Barcelona", "Athletic Club")

    @pytest.mark.parametrize(
        "text",
        [
            "2:0,0,1;1,1,1:0,0,1\n",  # frame ids must run 1, 2, ...
            "1:0,0,1:0,0,1\n",  # one slot instead of two
            "1:0,0,1;1,1,1:inf,0,1\n",  # infinity
            "1:0,0,1;1,1,1:0,0,1\n",  # one record instead of two
            "1:0,0,1;1,1,1:0,0\n",  # an observation needs x, y and speed
        ],
    )
    def test_bad_ids_lengths_and_infinity_are_errors(self, text: str):
        self.raw.write_text(text)
        with pytest.raises(ValueError):
            load_tracking(self.meta, self.raw, 2)

    def test_more_records_than_frames_is_an_error(self):
        self.raw.write_text("1:0,0,1;1,1,1:0,0,1\n2:0,0,1;1,1,1:0,0,1\n")
        with pytest.raises(ValueError, match="more than 1"):
            load_tracking(self.meta, self.raw, 1)

    def test_unknown_team_is_not_silently_away(self):
        self.meta.write_text(self.meta.read_text().replace('teamId="a"', 'teamId="unknown"'))
        with pytest.raises(ValueError):
            challenge.read_metadata(self.meta)


class TestSubmission:
    def test_writes_one_row_per_frame_with_unix_newlines(self, tmp_path: Path):
        path = tmp_path / "out" / "submission.csv"
        write_submission(path, ["X", "H", "A", "D"])
        assert path.read_bytes() == b"frame,label\n0,X\n1,H\n2,A\n3,D\n"
        assert not list(path.parent.glob("*.tmp"))

    def test_accepts_numpy_labels(self, tmp_path: Path):
        path = tmp_path / "submission.csv"
        write_submission(path, np.array(["H", "X"]).tolist())
        assert path.read_text() == "frame,label\n0,H\n1,X\n"

    def test_refuses_unknown_labels_without_writing(self, tmp_path: Path):
        path = tmp_path / "submission.csv"
        with pytest.raises(ValueError, match="invalid"):
            write_submission(path, ["H", "Z"])
        assert not path.exists()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_atd_bundle(tmp_path_factory.mktemp("bundle"))


@pytest.fixture(autouse=True)
def outside_the_agent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLAUDECODE", raising=False)


@pytest.fixture(scope="module")
def challenge_match(bundle: Path) -> Match:
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("CLAUDECODE", raising=False)
        return load_challenge(bundle)


class TestLoadChallenge:
    def test_match_has_the_shape_of_an_external_one(self, challenge_match: Match):
        match = challenge_match
        assert match.id == "challenge" and match.profile == "challenge"
        assert match.fps == 25 and match.pitch == (105.3, 68.0)
        assert match.intervals == [("1", 11185, 90652), ("2", 110808, 190198)]
        assert match.forced_dead == FORCED_DEAD == [(110808, 113742)]
        assert len(match.labels) == N_FRAMES and set(match.labels) == {"X"}
        assert len(match.source) == 16
        tracking = match.tracking
        assert tracking.players.shape == (N_FRAMES, 2, 3)
        assert tracking.players.dtype == np.float32 and tracking.ball.dtype == np.float32
        assert np.isnan(tracking.ball[:, 2]).all()  # positions only
        np.testing.assert_array_equal(tracking.teams, [0, 1])

    def test_player_speeds_come_from_positions_inside_halves(self, challenge_match: Match):
        speeds = challenge_match.tracking.players[:, :, 2]
        assert np.isfinite(speeds[20000]).all() and speeds[20000, 0] == 0
        assert np.isnan(speeds[0]).all()  # outside the halves
        assert np.isnan(speeds[11185 : 11185 + 3]).all()  # no centred difference at a half edge

    def test_missing_assets_are_listed(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Missing local match assets"):
            load_challenge(tmp_path)

    def test_data_directory_defaults_to_the_configured_one(
        self, bundle: Path, challenge_match: Match, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(challenge.paths, "DATA", bundle)
        assert load_challenge().source == challenge_match.source
