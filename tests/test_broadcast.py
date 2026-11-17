"""Broadcast alignment: offsets, frame assignment, replays and the ball it fills."""

import numpy as np
import pytest

from pitch_dynamics import broadcast
from pitch_dynamics.broadcast import (
    align,
    complete_ball,
    fill_ball,
    recode_teams,
    supported_offsets,
)
from pitch_dynamics.matches import Match, Tracking, gap_mask

PITCH = (105.0, 68.0)
NAMES = ("Barcelona", "Athletic Club")
N_TACTICAL = 9000
# Two live pieces of the TV feed: (first broadcast frame, one past the last, offset).
PIECES = ((0, 3500, 1000), (3500, 7000, 1700))
REPLAY = (2000, 2300)  # shows the tactical moment REPLAY_DELAY_FRAMES earlier
REPLAY_DELAY_FRAMES = 500
CLOSE_UP = (2300, 2400)  # nothing tracked
CAMERA_HALF_WIDTH_M = 30.0
BROADCAST_SLOTS_PER_TEAM = 15


def world(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """True player (frames, 22, 2) and ball (frames, 2) positions in metres."""
    rng = np.random.default_rng(seed)
    size = np.asarray(PITCH)
    players = np.empty((N_TACTICAL, 22, 2))
    players[0] = rng.uniform(0, 1, (22, 2)) * size
    velocity = np.zeros((22, 2))
    for frame in range(1, N_TACTICAL):
        velocity = 0.98 * velocity + rng.normal(0, 0.024, (22, 2))
        players[frame] = np.clip(players[frame - 1] + velocity, 0, size)
    ball = np.empty((N_TACTICAL, 2))
    carrier = previous = switched = 0
    next_switch = 75
    for frame in range(N_TACTICAL):
        if frame == next_switch:
            previous, carrier, switched = carrier, int(rng.integers(22)), frame
            next_switch = frame + int(rng.integers(50, 100))
        share = min((frame - switched) / 12, 1.0)  # the pass takes half a second
        ball[frame] = (1 - share) * players[frame, previous] + share * players[frame, carrier]
    return players, ball + 0.5


def tactical_view(players: np.ndarray, ball: np.ndarray, seed: int) -> Tracking:
    """Every player with short gaps, the ball missing 40 % of the time, 0.2 m of noise."""
    rng = np.random.default_rng(seed)
    xy = players + rng.normal(0, 0.2, players.shape)
    for slot in range(22):
        xy[gap_mask(N_TACTICAL, 0.08, 50, rng), slot] = np.nan
    seen_ball = ball + rng.normal(0, 0.2, ball.shape)
    seen_ball[gap_mask(N_TACTICAL, 0.4, 50, rng)] = np.nan
    return _tracking(xy, seen_ball, np.r_[np.zeros(11), np.ones(11)], NAMES)


def broadcast_truth() -> np.ndarray:
    """Tactical frame each broadcast frame shows live, -1 in replays and close-ups."""
    truth = np.concatenate([np.arange(start, stop) + offset for start, stop, offset in PIECES])
    truth[slice(*REPLAY)] = -1
    truth[slice(*CLOSE_UP)] = -1
    return truth


def broadcast_view(players: np.ndarray, ball: np.ndarray, seed: int) -> Tracking:
    """The TV feed: players near the ball only, in shuffled slots, teams listed away first."""
    rng = np.random.default_rng(seed)
    n_frames = PIECES[-1][1]
    shown = np.concatenate([np.arange(start, stop) + offset for start, stop, offset in PIECES])
    shown[slice(*REPLAY)] -= REPLAY_DELAY_FRAMES
    teams = np.repeat([0, 1], BROADCAST_SLOTS_PER_TEAM)  # 0 is away in this recording
    xy = np.full((n_frames, len(teams), 2), np.nan)
    seen_ball = ball[shown] + rng.normal(0, 0.3, (n_frames, 2))
    for frame, moment in enumerate(shown):
        if CLOSE_UP[0] <= frame < CLOSE_UP[1]:
            continue
        near = np.abs(players[moment, :, 0] - ball[moment, 0]) < CAMERA_HALF_WIDTH_M
        for code, true_team in ((0, slice(11, 22)), (1, slice(0, 11))):
            visible = players[moment, true_team][near[true_team]]
            slots = rng.permutation(BROADCAST_SLOTS_PER_TEAM)[: len(visible)]
            xy[frame, code * BROADCAST_SLOTS_PER_TEAM + slots] = visible
    xy += rng.normal(0, 0.3, xy.shape)
    seen_ball[gap_mask(n_frames, 0.45, 40, rng)] = np.nan
    seen_ball[slice(*CLOSE_UP)] = np.nan
    return _tracking(xy, seen_ball, teams, NAMES[::-1])


def _tracking(xy: np.ndarray, ball: np.ndarray, teams: np.ndarray, names: tuple) -> Tracking:
    size = np.asarray(PITCH)
    players = np.concatenate([xy / size, np.full((*xy.shape[:2], 1), np.nan)], axis=2)
    ball3 = np.concatenate([ball / size, np.full((len(ball), 1), np.nan)], axis=1)
    return Tracking(
        players.astype(np.float32), ball3.astype(np.float32), teams.astype(np.int8), names
    )


@pytest.fixture(scope="module")
def scene() -> tuple[Tracking, Tracking]:
    players, ball = world(0)
    tactical = tactical_view(players, ball, 1)
    return tactical, recode_teams(broadcast_view(players, ball, 2), tactical.team_names)


@pytest.fixture(scope="module")
def alignment(scene: tuple[Tracking, Tracking]) -> broadcast.Alignment:
    return align(*scene, PITCH)


class TestAlign:
    def test_finds_the_offset_of_each_live_piece(self, alignment: broadcast.Alignment):
        assert [offset for offset, _ in alignment.offsets] == [1000, 1700]

    def test_matched_frames_land_on_their_own_instant(self, alignment: broadcast.Alignment):
        truth = broadcast_truth()
        matched = alignment.tactical_frame >= 0
        np.testing.assert_array_equal(alignment.tactical_frame[matched], truth[matched])
        assert matched[truth >= 0].mean() > 0.9

    def test_replays_and_close_ups_stay_unmatched(self, alignment: broadcast.Alignment):
        assert (alignment.tactical_frame[slice(*REPLAY)] == -1).all()
        assert (alignment.tactical_frame[slice(*CLOSE_UP)] == -1).all()

    def test_an_unrelated_recording_is_an_error(self, scene: tuple[Tracking, Tracking]):
        players, ball = world(7)
        other = recode_teams(broadcast_view(players, ball, 2), NAMES)
        with pytest.raises(ValueError, match="do not line up"):
            align(scene[0], other, PITCH)


def test_votes_merge_within_the_tolerance_and_need_support():
    votes = [1000, 1000, 1001, 999, 1700, 1700, 1700, 1700, 5000]
    assert supported_offsets(votes) == [(1000, 4), (1700, 4)]
    assert supported_offsets([]) == []


def test_fill_only_replaces_missing_tactical_ball():
    nan = np.nan
    tactical = np.array([[0.1, 0.1], [nan, nan], [nan, nan], [0.4, 0.4], [nan, nan]], np.float32)
    seen = np.array([[0.9, 0.9], [0.2, 0.2], [0.3, 0.3], [nan, nan], [0.5, 0.5]], np.float32)
    tactical_frame = np.array([0, 1, -1, 4, 1])
    ball, filled = fill_ball(tactical, seen, tactical_frame)
    expected = np.array([[0.1, 0.1], [0.2, 0.2], [nan, nan], [0.4, 0.4], [nan, nan]])
    np.testing.assert_array_equal(ball, expected.astype(np.float32))
    np.testing.assert_array_equal(filled, [False, True, False, False, False])
    assert np.isnan(tactical[1]).all()  # the input is not modified


def test_teams_are_renumbered_by_name_and_must_match():
    tracking = _tracking(np.zeros((1, 2, 2)), np.zeros((1, 2)), np.array([0, 1]), NAMES[::-1])
    np.testing.assert_array_equal(recode_teams(tracking, NAMES).teams, [1, 0])
    with pytest.raises(ValueError, match="differ"):
        recode_teams(tracking, ("Barcelona", "Atletic Club"))


def test_completed_match_keeps_observations_and_changes_its_cache_key(
    scene: tuple[Tracking, Tracking],
):
    tactical, seen = scene
    match = Match(
        "challenge",
        tactical,
        np.full(N_TACTICAL, "X"),
        [("1", 0, 4500), ("2", 5200, N_TACTICAL)],
        PITCH,
        25,
        source="tactical1",
    )
    result = complete_ball(match, seen, "broadcast1")
    before, after = tactical.ball[:, :2], result.match.tracking.ball[:, :2]
    observed = np.isfinite(before).all(axis=1)
    np.testing.assert_array_equal(after[observed], before[observed])
    assert result.filled.any() and not result.filled[observed].any()
    assert np.isfinite(after[result.filled]).all()
    assert result.match.tracking.players is tactical.players
    assert result.match.source == "tactical1+broadcast1"
    assert match.source == "tactical1" and np.isnan(match.tracking.ball[result.filled]).all()
