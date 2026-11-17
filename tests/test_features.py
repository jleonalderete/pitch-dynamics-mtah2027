"""Frame evidence and design matrix: football failure modes on synthetic tracking."""

from pathlib import Path

import numpy as np
import pytest
from synthetic import install_matches, tracking, use_cache

from pitch_dynamics import design
from pitch_dynamics.design import FEATURES, N_COLUMNS, N_EXTRA, WINDOWS
from pitch_dynamics.features import build_features, rolling_mean
from pitch_dynamics.matches import Match, Tracking, load_match

PITCH = (105.3, 68.0)


def evidence(track: Tracking, intervals: list | None = None) -> dict[str, np.ndarray]:
    return build_features(track, PITCH, 25, intervals or [("1", 0, len(track.ball))])


def match_of(track: Tracking, source: str = "") -> Match:
    n = len(track.ball)
    labels = np.full(n, "X", dtype="<U1")
    return Match("m", track, labels, [("1", 0, n)], PITCH, 25, source=source)


class TestEvidence:
    @pytest.mark.parametrize(("missing", "quiet"), [(True, False), (True, True), (False, False)])
    def test_missing_ball_and_quiet_players_alone_are_not_dead(self, missing: bool, quiet: bool):
        track = tracking()
        if missing:
            track.ball[:] = np.nan
        if quiet:
            track.players[:, :, 2] = 0
        if not missing:
            track.ball[:, 2] = 0
        assert evidence(track)["dead"].max() == 0

    def test_stationary_ball_and_quiet_players_support_dead(self):
        track = tracking()
        track.players[:, :, 2] = 0
        track.ball[:, 2] = 0
        assert evidence(track)["dead"][60] > 0.9

    def test_the_reported_ball_speed_column_is_used_when_present(self):
        track = tracking()
        track.players[:, :, 2] = 0
        track.ball[:, 2] = 20  # reported fast although the positions do not move
        assert evidence(track)["dead"][60] == 0

    def test_team_slot_permutation_does_not_change_features(self):
        original, changed = tracking(slots=4), tracking(slots=4)
        original.players[:, 2, 0] = changed.players[:, 2, 0] = 0.35
        changed.players[50:, [0, 2]] = changed.players[50:, [2, 0]]
        for key in ("geometry", "control", "dead", "nearest_home_m", "nearest_away_m"):
            np.testing.assert_array_equal(evidence(original)[key], evidence(changed)[key])

    def test_nearest_players_are_measured_per_team_in_metres(self):
        track = tracking(10)
        track.ball[:, :2] = [0.2, 0.5]  # on top of the home players
        result = evidence(track)
        assert result["nearest_home_m"][5] == 0
        assert result["nearest_away_m"][5] == pytest.approx(0.6 * 105.3)

    def test_features_do_not_modify_the_tracking(self):
        track = tracking()
        before = (track.players.copy(), track.ball.copy())
        evidence(track)
        np.testing.assert_array_equal(track.players, before[0])
        np.testing.assert_array_equal(track.ball, before[1])

    def test_smoothing_never_crosses_halftime(self):
        values = np.r_[np.ones(10), np.full(10, np.nan), -np.ones(10)]
        result = rolling_mean(values, 100, [("1", 0, 10), ("2", 20, 30)])
        np.testing.assert_array_equal(result[:10], 1)
        np.testing.assert_array_equal(result[20:], -1)
        assert np.isnan(result[10:20]).all()


class TestRestartEvidence:
    """Each restart cue needs its whole conjunction; one ingredient alone stays at zero."""

    def stopped(self, ball_xy: tuple[float, float], n: int = 60) -> dict[str, np.ndarray]:
        track = tracking(n)
        track.ball[:, :2] = ball_xy
        track.ball[:, 2] = 0
        track.players[:, :, 2] = 0  # quiet players
        return evidence(track)

    def test_stationary_ball_on_each_restart_spot(self):
        spots = {
            "restart_corner": (0.0, 0.0),
            "restart_touchline": (0.4, 1.0),
            "restart_goal_area": (0.03, 0.5),
            "restart_centre": (0.5, 0.5),
        }
        for name, xy in spots.items():
            result = self.stopped(xy)
            assert result[name][30] > 0.9, name
            skipped = {"restart_touchline"} if name == "restart_corner" else set()
            others = set(spots) - {name} - skipped
            assert all(result[other][30] == 0 for other in others), name

    def test_moving_or_missing_ball_is_no_restart(self):
        track = tracking(60)
        track.ball[:, 0] = np.linspace(0.3, 0.7, 60)
        track.ball[:, 1] = 1.0  # rolling along the touchline
        track.ball[:, 2] = 12
        assert evidence(track)["restart_touchline"][30] == 0
        missing = self.stopped((np.nan, np.nan))
        names = ("restart_touchline", "restart_corner", "restart_goal_area", "restart_centre")
        assert all(missing[name].max() == 0 for name in names)

    def test_kickoff_regroup_needs_separated_halves_and_quiet_players(self):
        separated = self.stopped((0.5, 0.5))  # fixture: home at x=0.2, away at x=0.8
        assert separated["kickoff_regroup"][30] > 0.9
        track = tracking(60)
        track.players[:, :, 2] = 0
        track.players[:, ::4, 0] = 0.6  # some home players across halfway: open play
        assert evidence(track)["kickoff_regroup"][30] < 0.5
        moving = tracking(60)  # separated but running players
        moving.players[:, :, 2] = 6
        assert evidence(moving)["kickoff_regroup"][30] == 0

    def test_ball_stillness_ignores_the_reported_speed_column(self):
        track = tracking(60)
        track.ball[:, 2] = 5  # the ATD column claims motion; positions do not move
        assert evidence(track)["ball_still"][30] > 0.9

    def test_kickoff_taker_on_the_halfway_line_still_counts(self):
        track = tracking(60)
        track.players[:, :, 2] = 0
        track.players[:, 0, 0] = 0.51  # a home taker just across halfway
        assert evidence(track)["kickoff_regroup"][30] > 0.9


class TestDesign:
    def test_column_layout(self):
        assert N_EXTRA == 43
        assert N_COLUMNS == len(FEATURES) * (1 + len(WINDOWS)) + 2 + N_EXTRA == 99

    def test_frame_design_keeps_the_four_constant_columns_at_zero(self):
        track = tracking(200)
        frame = design.frame_design(evidence(track), [("1", 0, 200)])
        assert frame.shape == (200, len(FEATURES) * 3 + 2)
        for name in ("tag_home", "tag_away", "tag_conflict", "restart"):
            first = FEATURES.index(name) * 3
            assert not frame[:, first : first + 3].any()
        geometry = FEATURES.index("geometry") * 3
        assert frame[:, geometry : geometry + 3].any()
        assert frame[:, -2:].max() <= 1.0  # distances are capped and scaled

    @pytest.mark.filterwarnings("ignore:Mean of empty slice")
    def test_design_matrix_is_float32_and_finite_with_a_missing_team_and_ball(self):
        track = tracking(300)
        track.ball[100:150] = np.nan
        track.players[:, 1::2] = np.nan  # the whole away team unobserved
        matrix = design.build_design(match_of(track), evidence(track))
        assert matrix.shape == (300, N_COLUMNS) and matrix.dtype == np.float32
        assert np.isfinite(matrix).all()

    def test_last_touch_carries_the_owner_and_the_time_since(self):
        feats = {
            "nearest_home_m": np.array([0.5, 5.0, 5.0, 5.0]),
            "nearest_away_m": np.array([5.0, 5.0, 0.5, 5.0]),
        }
        columns = design.last_touch(feats, [("1", 0, 4)], 25, 1.0)
        np.testing.assert_array_equal(columns[:, 0], [1, 1, 0, 0])  # home owns
        np.testing.assert_array_equal(columns[:, 1], [0, 0, 1, 1])  # away owns
        np.testing.assert_allclose(columns[:, 2], [0, 0.004, 0, 0.004])  # seconds since / cap
        np.testing.assert_array_equal(columns[:, 3], [1, 0, 0, 0])  # touching now
        np.testing.assert_array_equal(columns[:, 4], [0, 0, 1, 0])

    def test_before_the_first_touch_nobody_owns_and_time_is_capped(self):
        feats = {"nearest_home_m": np.full(3, 9.0), "nearest_away_m": np.full(3, 9.0)}
        columns = design.last_touch(feats, [("1", 0, 3)], 25, 1.0)
        assert not columns[:, :2].any() and (columns[:, 2] == 1).all()

    def test_extra_blocks_have_the_documented_widths(self):
        track = tracking(120)
        feats = evidence(track)
        match = match_of(track)
        assert design.kicks(match, feats).shape == (120, 8)
        assert design.pressure(match).shape == (120, 8)
        assert design.team_shape(match).shape == (120, 12)

    def test_swapping_teams_swaps_the_evidence(self):
        track = tracking(120)
        original = match_of(track)
        mirrored = design.swapped(original)
        assert mirrored.tracking.team_names == ("Athletic Club", "Barcelona")
        np.testing.assert_array_equal(mirrored.tracking.teams, 1 - track.teams)
        a, b = design.match_features(original), design.match_features(mirrored)
        np.testing.assert_array_equal(a["nearest_home_m"], b["nearest_away_m"])
        np.testing.assert_array_equal(a["nearest_away_m"], b["nearest_home_m"])
        np.testing.assert_array_equal(design.swapped(mirrored).tracking.teams, track.teams)

    def test_swapping_exchanges_home_and_away_labels_only(self):
        match = match_of(tracking(6))
        match.labels = np.array(list("HADXHA"))
        assert "".join(design.swapped(match).labels) == "AHDXAH"
        assert design.swapped(match).source == ""
        match.source = "abc"
        assert design.swapped(match).source == "abc/swap"


class TestCaches:
    @pytest.fixture
    def builds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        use_cache(monkeypatch, tmp_path)
        design.clear_memo()
        calls: list[int] = []
        real = design.build_features

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(design, "build_features", counting)
        return calls

    def test_features_are_cached_per_match_digest(self, builds: list[int]):
        track = tracking(120)
        first = design.match_features(match_of(track, "digest-a"))
        again = design.match_features(match_of(track, "digest-a"))
        assert len(builds) == 1
        for key in first:
            np.testing.assert_array_equal(first[key], again[key])
        design.match_features(match_of(track, "digest-b"))
        assert len(builds) == 2

    def test_a_match_without_a_digest_is_never_cached(self, builds: list[int]):
        match = match_of(tracking(120))
        design.match_features(match)
        design.match_features(match)
        assert len(builds) == 2

    def test_design_matrices_are_memoised_only_with_a_digest(self, builds: list[int]):
        track = tracking(120)
        keyed = match_of(track, "digest-a")
        assert design.design_matrix(keyed) is design.design_matrix(keyed)
        anonymous = match_of(track)
        assert design.design_matrix(anonymous) is not design.design_matrix(anonymous)
        design.clear_memo()
        first = design.design_matrix(keyed)
        design.clear_memo()
        assert design.design_matrix(keyed) is not first


def test_augmented_adds_two_draws_and_the_swap_of_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    use_cache(monkeypatch, tmp_path)
    install_matches(tmp_path, ["J03WMX", "J03WOH"])
    base = [load_match("J03WMX"), load_match("J03WOH")]
    copies = design.augmented(base)
    assert len(copies) == 2 * 3 * 2
    sources = [m.source for m in copies]
    assert len(set(sources)) == len(sources)
    assert sources[:2] == [base[0].source, base[1].source]
    ids = ["J03WMX", "J03WOH", "J03WMX", "J03WMX", "J03WOH", "J03WOH"]
    assert [m.id for m in copies[:6]] == ids
    swapped = copies[6:]
    for original, mirror in zip(copies[:6], swapped, strict=True):
        assert mirror.source == original.source + "/swap"
        assert mirror.id == original.id and mirror.profile == original.profile
        assert (original.labels == "H").sum() == (mirror.labels == "A").sum()
