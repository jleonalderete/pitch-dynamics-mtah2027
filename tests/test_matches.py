"""Match loading, dataset splits, tracking degradation and the external-data parsers."""

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from synthetic import DFL_META, dfl_positions_xml, install_matches, synthetic_raw, use_cache

from pitch_dynamics import matches, sources
from pitch_dynamics.matches import DEV, HOLDOUT_DFL, HOLDOUT_LALIGA, PROFILES, TRAINING


def test_splits_are_disjoint_and_cover_the_dfl_matches():
    assert set(DEV).isdisjoint(HOLDOUT_DFL)
    assert TRAINING == DEV + HOLDOUT_DFL
    assert set(TRAINING) == set(sources.DFL_MATCHES)
    assert HOLDOUT_LALIGA == ("skc_2841", "skc_3442")
    assert {f"skc_{match}" for match in sources.SKC_MATCHES} == set(HOLDOUT_LALIGA)


def test_default_profile_follows_the_source():
    assert matches.default_profile("J03WMX") == "atd_like_v1"
    assert matches.default_profile("skc_2841") == "broadcast_v1"
    assert set(PROFILES) == {"clean", "atd_like_v1", "broadcast_v1"}
    assert PROFILES["broadcast_v1"].ball_missing == 0.0


class TestDflParser:
    def test_meta_and_positions_follow_the_published_format(self, tmp_path: Path):
        meta, positions = tmp_path / "meta.xml", tmp_path / "positions.xml"
        meta.write_text(DFL_META)
        positions.write_text(dfl_positions_xml())
        pitch, team_of = sources.read_dfl_meta(meta)
        players, ball, teams, labels, intervals = sources.read_dfl_positions(
            positions, team_of, pitch
        )
        assert pitch == (105.0, 68.0)
        assert intervals == [(1, 0, 3), (2, 3, 5)]
        assert labels.tolist() == ["H", "A", "D", "X", "H"]
        np.testing.assert_allclose(ball[:3, :2], [[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]])
        assert teams.tolist() == [0, 1]  # referee skipped, home before away
        np.testing.assert_allclose(players[0, 0, :2], [0.4, 0.6])
        assert np.isnan(players[1, 0, :2]).all()  # frame the player did not report
        np.testing.assert_allclose(players[4, 1, :2], [0.5, 1.0])

    def test_meta_needs_one_home_and_one_guest_team(self, tmp_path: Path):
        meta = tmp_path / "meta.xml"
        meta.write_text(DFL_META.replace('Role="guest"', 'Role="home"'))
        with pytest.raises(ValueError, match="one home and one guest"):
            sources.read_dfl_meta(meta)

    def test_files_are_checked_against_the_pinned_digests(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        meta, positions = sources.dfl_paths("J03WMX", tmp_path)
        meta.write_text(DFL_META)
        positions.write_text(dfl_positions_xml())
        with pytest.raises(ValueError, match="pinned"):
            sources.parse_dfl("J03WMX", tmp_path)
        pinned = sources.DflMatch(
            "DFL-COM-000001", sources.git_blob_id(meta), sources.sha256(positions)
        )
        monkeypatch.setitem(sources.DFL_MATCHES, "J03WMX", pinned)
        raw = sources.parse_dfl("J03WMX", tmp_path)
        assert raw["labels"].tolist() == ["H", "A", "D", "X", "H"]
        assert raw["fps"] == 25
        provenance = json.loads(raw["provenance"])
        assert provenance["revision"] == sources.DFL_REVISION
        assert provenance["files"][positions.name] == pinned.positions_sha256

    def test_digest_helpers_match_known_values(self, tmp_path: Path):
        path = tmp_path / "hello.txt"
        path.write_bytes(b"hello\n")
        assert sources.git_blob_id(path) == "ce013625030ba8dba906f756967f9e9ca394464a"
        assert sources.sha256(path) == (
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        )

    def test_mirror_file_names(self):
        meta, positions = sources.dfl_paths("J03WOH", Path("x"))
        assert meta.name == "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml"
        assert positions.name == (
            "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml"
        )


class TestSkillCorner:
    META: ClassVar[dict[str, Any]] = {
        "home_team": {"id": 1},
        "away_team": {"id": 2},
        "pitch_length": 105,
        "pitch_width": 68,
        "ball": {"trackable_object": 55},
        "players": [
            {"id": 10, "trackable_object": 100, "team_id": 1},
            {"id": 20, "trackable_object": 200, "team_id": 2},
            {"id": 30, "trackable_object": 300, "team_id": 9},  # not on either team
        ],
    }

    def test_resample_is_linear_and_keeps_gaps(self):
        values = np.array([[0.0], [10.0], [np.nan], [30.0]])
        out = sources.resample(values, 2.5)
        assert len(out) == 8
        np.testing.assert_allclose(out[:3, 0], [0.0, 4.0, 8.0])
        assert np.isnan(out[3, 0])  # between 10 and the gap
        assert np.isnan(out[-1, 0])  # between the gap and 30
        np.testing.assert_allclose(sources.resample(np.array([0.0, 2.0]), 2.5), [0.0, 0.8, 1.6])

    def test_frames_give_team_labels_and_x_without_owner(self):
        frames = [
            {
                "frame": f,
                "period": 1,
                "possession": {"group": "home team" if f < 5 else None},
                "data": [
                    {"trackable_object": 55, "x": 0.0, "y": 0.0},
                    {"trackable_object": 100, "x": -52.5, "y": 34.0},
                    {"trackable_object": None, "group_name": "away team", "x": 1, "y": 1},
                ],
            }
            for f in range(10)
        ] + [{"frame": 99, "period": None, "data": []}]
        pitch, teams, sections = sources.read_skc_tracking(self.META, frames)
        players, ball, labels, intervals = sources.skc_assemble(sections, pitch)
        assert list(teams) == [0, 1]
        assert intervals == [(1, 0, len(labels))]
        assert players.shape[1:] == (2, 3)
        np.testing.assert_allclose(players[0, 0, :2], [0.0, 1.0])
        assert np.isnan(players[0, 1, 0])  # unidentified tracks are skipped
        np.testing.assert_allclose(ball[0, :2], [0.5, 0.5])
        assert set(labels[:10]) == {"H"}
        assert labels[-1] == "X"
        assert "D" not in set(labels)


class TestDegradation:
    def test_gap_mask_hits_the_target_share_in_runs(self):
        mask = matches.gap_mask(200000, 0.4, 50, np.random.default_rng(0))
        assert mask.mean() == pytest.approx(0.4, abs=0.03)
        starts = np.flatnonzero(np.diff(mask.astype(int)) == 1)
        assert mask.sum() / len(starts) == pytest.approx(50, abs=8)

    def test_gap_mask_without_missing_share_draws_nothing(self):
        rng = np.random.default_rng(0)
        assert not matches.gap_mask(100, 0.0, 1.0, rng).any()
        assert rng.random() == np.random.default_rng(0).random()

    def test_degradation_is_deterministic_and_label_independent(self):
        raw = synthetic_raw(0)
        profile = PROFILES["atd_like_v1"]
        first, second = matches.degrade(raw, profile, 7), matches.degrade(raw, profile, 7)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        missing = np.isnan(first[1][:, 0])
        dead = raw["labels"] == "D"
        assert abs(missing[dead].mean() - missing[~dead].mean()) < 0.15
        assert np.isnan(first[1][:, 2]).all()  # the ball has no speed column
        other = matches.degrade(raw, profile, 8)
        assert not np.array_equal(first[1], other[1], equal_nan=True)

    def test_clean_profile_keeps_positions_and_derives_speeds(self):
        raw = synthetic_raw(0)
        players, ball = matches.degrade(raw, PROFILES["clean"], 7)
        np.testing.assert_allclose(ball[:, :2], raw["ball"][:, :2])
        np.testing.assert_array_equal(players[:, :, :2], raw["players"][:, :, :2])
        assert np.isfinite(players[1000, :, 2]).all()
        assert np.isnan(players[0, :, 2]).all()  # no centred difference at the half boundary

    def test_broadcast_profile_adds_noise_but_no_gaps(self):
        raw = synthetic_raw(0)
        players, ball = matches.degrade(raw, PROFILES["broadcast_v1"], 3)
        assert np.isfinite(players[:, :, :2]).all()
        assert np.isfinite(ball[:, :2]).all()
        assert not np.array_equal(ball[:, :2], raw["ball"][:, :2])

    def test_player_speeds_stay_inside_halves(self):
        xy = np.zeros((20, 1, 2), dtype=np.float32)
        xy[:, 0, 0] = np.arange(20) * 0.001
        speed = matches.player_speeds(xy, (100.0, 50.0), 25, [("1", 0, 10), ("2", 10, 20)])
        assert np.isnan(speed[:3]).all() and np.isnan(speed[7:13]).all()
        np.testing.assert_allclose(speed[3], 0.001 * 100.0 * 25.0, rtol=1e-5)


class TestLoadMatch:
    @pytest.fixture(autouse=True)
    def cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        use_cache(monkeypatch, tmp_path)
        install_matches(tmp_path, ["J03WMX", "skc_2841"])

    def test_loads_a_degraded_match_with_the_default_profile(self):
        match = matches.load_match("J03WMX")
        assert match.profile == "atd_like_v1"
        assert match.intervals == [("1", 0, 1200), ("2", 1200, 2400)]
        assert match.fps == 25 and match.pitch == (105.0, 68.0)
        assert match.forced_dead == []
        assert match.tracking.team_names == ("Home", "Away")
        assert match.tracking.teams.dtype == np.int8
        assert match.tracking.players.dtype == np.float32
        assert set(match.labels) == {"H", "A", "D"}
        assert match.source.endswith("/0") and len(match.source) == 18
        assert np.isnan(match.tracking.ball[:, 2]).all()
        assert matches.load_match("skc_2841").profile == "broadcast_v1"

    def test_same_arguments_give_identical_inputs_and_offsets_give_fresh_draws(self):
        first, second = matches.load_match("J03WMX"), matches.load_match("J03WMX")
        np.testing.assert_array_equal(first.tracking.ball, second.tracking.ball)
        other = matches.load_match("J03WMX", seed_offset=1)
        assert other.source.endswith("/1")
        assert not np.array_equal(first.tracking.ball, other.tracking.ball, equal_nan=True)
        np.testing.assert_array_equal(first.labels, other.labels)

    def test_a_missing_match_points_to_the_data_command(self):
        with pytest.raises(FileNotFoundError, match="pitch-dynamics data"):
            matches.load_match("J03WN1")

    def test_summary_counts_labels(self):
        info = matches.summary("J03WMX")
        assert info["frames"] == 2400
        assert sum(info["labels"].values()) == 2400


def test_fetch_downloads_and_parses_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    use_cache(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(sources, "fetch_dfl", lambda match: calls.append(f"fetch {match}"))
    monkeypatch.setattr(sources, "parse_dfl", lambda match: synthetic_raw(0))
    path = matches.fetch("J03WMX")
    assert path == matches.raw_path("J03WMX") and path.is_file()
    assert matches.fetch("J03WMX") == path
    assert calls == ["fetch J03WMX"]
    assert not list(path.parent.glob("*.tmp.npz"))


def test_cached_arrays_are_computed_once_per_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    use_cache(monkeypatch, tmp_path)
    calls = []

    def compute() -> dict[str, np.ndarray]:
        calls.append(1)
        return {"a": np.arange(3)}

    first = matches.cached("demo", {"match": "m", "source": "s1"}, compute)
    again = matches.cached("demo", {"source": "s1", "match": "m"}, compute)
    changed = matches.cached("demo", {"match": "m", "source": "s2"}, compute)
    np.testing.assert_array_equal(first["a"], again["a"])
    np.testing.assert_array_equal(first["a"], changed["a"])
    assert len(calls) == 2


@pytest.mark.dfl
def test_cached_dfl_matches_have_provider_labels():
    for match_id in DEV:
        if not matches.raw_path(match_id).is_file():
            pytest.skip("external DFL matches are not in the local cache")
        match = matches.load_match(match_id)
        assert match.fps == 25 and len(match.intervals) == 2
        assert {"H", "A", "D"} <= set(match.labels)
