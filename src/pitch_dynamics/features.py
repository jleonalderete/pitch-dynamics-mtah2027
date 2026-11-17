"""Per-frame evidence from tracking, with explicit missingness and no player identities."""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pitch_dynamics.matches import Interval, Tracking

Array = NDArray[Any]

CONTROL_RADIUS_M = 3.0
DISTANCE_SCALE_M = 2.0
MAX_BALL_SPEED_MPS = 55.0
MAX_PLAYER_SPEED_MPS = 13.0
EVIDENCE_WINDOW_FRAMES = 12
STATIONARY_BALL_MPS = 0.7
QUIET_PLAYERS_MPS = 1.8
STILLNESS_LAG_FRAMES = 5

# Laws of the Game geometry in metres, widened by a tolerance for tracking noise.
LINE_TOLERANCE_M = 1.5
CORNER_RADIUS_M = 2.5
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M = 5.5, 9.16
CENTRE_RADIUS_M = 2.0
KICKOFF_MIN_PLAYERS = 8  # per team; fewer visible players cannot show a regrouped side
HALFWAY_TOLERANCE_M = 1.5


def rolling_mean(values: Array, radius: int, intervals: Sequence[Interval]) -> Array:
    """Centred, NaN-aware means, strictly isolated inside each half."""
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    for _, start, stop in intervals:
        window = values[start:stop]
        good = np.isfinite(window)
        sums = np.concatenate(([0.0], np.cumsum(np.where(good, window, 0))))
        counts = np.concatenate(([0], np.cumsum(good)))
        indices = np.arange(len(window))
        low = np.maximum(0, indices - radius)
        high = np.minimum(len(window), indices + radius + 1)
        denominator = counts[high] - counts[low]
        result[start:stop] = np.divide(
            sums[high] - sums[low],
            denominator,
            out=np.full(len(window), np.nan),
            where=denominator > 0,
        )
    return result


def ball_observations(
    tracking: Tracking, pitch: Array, fps: int, intervals: Sequence[Interval]
) -> tuple[Array, Array, Array]:
    """Ball positions, speed in m/s (NaN when unreliable) and a 0/1 quality per frame."""
    ball = tracking.ball.copy()
    observed = np.isfinite(ball[:, :2]).all(axis=1)
    quality = observed.astype(np.float32)
    speed = np.full(len(ball), np.nan, dtype=np.float32)
    reported = np.isfinite(ball[:, 2]) & (ball[:, 2] >= 0)
    speed[reported & observed] = ball[reported & observed, 2]
    derived = np.full(len(ball), np.nan)
    for _, start, stop in intervals:
        if stop - start < 2:
            continue
        good = observed[start : stop - 1] & observed[start + 1 : stop]
        delta = np.linalg.norm(np.diff(ball[start:stop, :2], axis=0) * pitch, axis=1) * fps
        derived[start + 1 : stop] = np.where(good, delta, np.nan)
    speed[~np.isfinite(speed)] = derived[~np.isfinite(speed)]
    speed = np.fmax(speed, derived).astype(np.float32)
    anomalous = observed & (
        (speed > MAX_BALL_SPEED_MPS)
        | (derived > MAX_BALL_SPEED_MPS)
        | (np.abs(ball[:, :2] - 0.5) > 0.8).any(axis=1)
    )
    quality[anomalous] = 0
    speed[quality == 0] = np.nan
    return ball, speed, quality


def ball_stillness(
    ball: Array,
    quality: Array,
    pitch: Array,
    fps: int,
    threshold_mps: float,
    window: int,
    intervals: Sequence[Interval],
) -> Array:
    """Stillness from displacement alone, smoothed inside halves.

    The ATD ball speed column does not behave as a speed and sits above the stationary threshold
    most of the time, so a still ball must be read from positions.
    """
    lag = STILLNESS_LAG_FRAMES
    still = np.zeros(len(ball))
    for _, start, stop in intervals:
        if stop - start <= lag:
            continue
        xy = ball[start:stop, :2]
        speed = np.linalg.norm((xy[lag:] - xy[:-lag]) * pitch, axis=1) * fps / lag
        with np.errstate(invalid="ignore"):
            still[start + lag : stop] = np.nan_to_num(np.clip(1 - speed / threshold_mps, 0, 1))
    return np.nan_to_num(rolling_mean(still * quality, window, intervals))


def restart_evidence(
    ball: Array,
    quality: Array,
    player_xy: Array,
    visible: Array,
    teams: Array,
    stationary: Array,
    quiet: Array,
    pitch: Array,
) -> dict[str, Array]:
    """Restart-shaped situations, each a conjunction so no single cue reads as a stoppage.

    A still ball on a line, in a corner, in a goal area or on the centre spot is where throw-ins,
    corners, goal kicks and kick-offs are taken. Both teams wholly in their own half while play
    is quiet is the regrouping before a kick-off. `stationary` is the displacement-based
    stillness of `ball_stillness`.
    """
    length, width = pitch
    x, y = ball[:, 0] * length, ball[:, 1] * width
    to_goal, to_touch = np.minimum(x, length - x), np.minimum(y, width - y)
    tolerance = LINE_TOLERANCE_M
    with np.errstate(invalid="ignore"):  # unobserved balls are NaN and match no spot
        spots = {
            "restart_touchline": np.abs(to_touch) <= tolerance,
            "restart_corner": np.hypot(to_goal, to_touch) <= CORNER_RADIUS_M,
            "restart_goal_area": (to_goal >= -tolerance)
            & (to_goal <= GOAL_AREA_DEPTH_M + tolerance)
            & (np.abs(y - width / 2) <= GOAL_AREA_HALF_WIDTH_M + tolerance),
            "restart_centre": np.hypot(x - length / 2, y - width / 2) <= CENTRE_RADIUS_M,
        }
    held = quality * stationary
    result = {name: mask.astype(np.float64) * held for name, mask in spots.items()}

    # Kick-off takers stand on the halfway line, so a player within the tolerance of it counts
    # as being in either half.
    margin = HALFWAY_TOLERANCE_M / length
    with np.errstate(invalid="ignore"):
        left = (player_xy[:, :, 0] < 0.5 + margin) & visible
        right = (player_xy[:, :, 0] > 0.5 - margin) & visible
    shares: dict[tuple[int, str], Array] = {}
    for side in (0, 1):
        mask = teams == side
        count = visible[:, mask].sum(axis=1)
        enough = count >= KICKOFF_MIN_PLAYERS
        for name, where in (("left", left), ("right", right)):
            share = where[:, mask].sum(axis=1) / np.maximum(count, 1)
            shares[side, name] = np.where(enough, share, np.nan)
    # Either orientation: teams swap ends at half-time.
    separation = np.nan_to_num(
        np.maximum(
            np.minimum(shares[0, "left"], shares[1, "right"]),
            np.minimum(shares[0, "right"], shares[1, "left"]),
        )
    )
    # Linear from 80 % to 100 % of both teams in their own half; open play rarely reaches it.
    result["kickoff_regroup"] = np.clip((separation - 0.8) / 0.2, 0, 1) * quiet
    return result


def build_features(
    tracking: Tracking,
    pitch: tuple[float, float],
    fps: int,
    intervals: Sequence[Interval],
) -> dict[str, Array]:
    """Frame evidence used by the models: ball control geometry, dead-ball cues, nearest players."""
    n_frames = len(tracking.ball)
    pitch_m = np.asarray(pitch)
    ball, ball_speed, quality = ball_observations(tracking, pitch_m, fps, intervals)

    player_xy = tracking.players[:, :, :2]
    visible = np.isfinite(player_xy).all(axis=2)
    distances = np.linalg.norm((player_xy - ball[:, None, :2]) * pitch_m, axis=2)
    distances[~visible | (quality[:, None] <= 0)] = np.inf
    nearest = np.full((n_frames, 2), np.inf)
    for side in (0, 1):
        mask = tracking.teams == side
        if mask.any():
            nearest[:, side] = distances[:, mask].min(axis=1)

    speeds = tracking.players[:, :, 2].copy()
    speeds[~visible | (speeds < 0) | (speeds > MAX_PLAYER_SPEED_MPS)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        collective_speed = np.nanmedian(speeds, axis=1)
    speed_counts = np.isfinite(speeds).sum(axis=1)
    sufficient_motion = np.minimum(1, speed_counts / 10.0)
    quiet = np.clip(1 - collective_speed / QUIET_PLAYERS_MPS, 0, 1)
    quiet = np.nan_to_num(quiet) * sufficient_motion
    stationary = np.nan_to_num(np.clip(1 - ball_speed / STATIONARY_BALL_MPS, 0, 1))
    stationary *= quality
    out = ((ball[:, :2] < 0) | (ball[:, :2] > 1)).any(axis=1) * quality

    both = np.isfinite(nearest).all(axis=1)
    advantage = np.zeros(n_frames)
    advantage[both] = np.tanh((nearest[both, 1] - nearest[both, 0]) / DISTANCE_SCALE_M)
    proximity = np.exp(-nearest.min(axis=1) / CONTROL_RADIUS_M)
    # A fast ball beside a player is weaker evidence of control than a slow one.
    control = quality * proximity / (1 + np.nan_to_num(ball_speed, nan=15.0) / 10.0)
    geometry = rolling_mean(advantage * control, EVIDENCE_WINDOW_FRAMES, intervals)
    geometry = np.nan_to_num(geometry)
    quiet = np.nan_to_num(rolling_mean(quiet, EVIDENCE_WINDOW_FRAMES, intervals))
    stationary = np.nan_to_num(rolling_mean(stationary, EVIDENCE_WINDOW_FRAMES, intervals))
    out = np.nan_to_num(rolling_mean(out, EVIDENCE_WINDOW_FRAMES, intervals))
    # Conjunctions are deliberate: missing ball, quiet players, or stationary ball alone
    # must never turn into positive dead-ball evidence.
    dead = np.maximum(stationary * quiet, out * quiet)
    still = ball_stillness(
        ball, quality, pitch_m, fps, STATIONARY_BALL_MPS, EVIDENCE_WINDOW_FRAMES, intervals
    )
    restarts = {
        "ball_still": still,
        **restart_evidence(
            ball, quality, player_xy, visible, tracking.teams, still, quiet, pitch_m
        ),
    }
    return {
        "nearest_home_m": nearest[:, 0],
        "nearest_away_m": nearest[:, 1],
        "ball_quality": quality,
        "ball_missing": ~np.isfinite(tracking.ball[:, :2]).all(axis=1),
        "geometry": geometry,
        "control": control,
        "quiet": quiet,
        "stationary": stationary,
        "outside_pitch": out,
        "dead": dead,
        **restarts,
    }
