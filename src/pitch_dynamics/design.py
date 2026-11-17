"""The per-frame design matrix of the model."""

import hashlib
import inspect
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pitch_dynamics import features as features_module
from pitch_dynamics.features import build_features, rolling_mean
from pitch_dynamics.matches import Interval, Match, Tracking, cached, load_match

Array = NDArray[Any]

# Evidence columns, each raw and smoothed over WINDOWS. The four tag columns are constant zero
# (no tagging input exists), but they keep the column layout, and with it the shape and random
# initialisation of every model weight, unchanged.
FEATURES = (
    "geometry",
    "control",
    "tag_home",
    "tag_away",
    "tag_conflict",
    "restart",
    "dead",
    "quiet",
    "stationary",
    "outside_pitch",
    "ball_quality",
    "ball_missing",
    "ball_still",
    "restart_touchline",
    "restart_corner",
    "restart_goal_area",
    "restart_centre",
    "kickoff_regroup",
)
ABSENT_FEATURES = frozenset({"tag_home", "tag_away", "tag_conflict", "restart"})
WINDOWS = (25, 100)  # centred 1 s and 4 s means
DISTANCE_CAP_M = 30.0

TOUCH_RADII_M = (1.0, 1.5, 2.5)  # a player this close, and closer than any opponent, touches it
SINCE_CAP_S = 10.0
KICK_LAG_FRAMES = 2
ACCEL_CAP_MPS2 = 60.0
BALL_SPEED_CAP_MPS = 40.0
KICK_RADIUS_M = 3.0
PRESSURE_RADII_M = (5.0, 10.0)
SHAPE_LAG_S = 1.0
SPREAD_CAP_M = 30.0

TRAIN_STRIDE = 3  # fit on every 3rd labelled frame; change costs still use every frame
AUGMENT_SEEDS = (1, 2)  # extra degradation draws per training match; evaluation stays at 0

N_EXTRA = 5 * len(TOUCH_RADII_M) + 8 + 4 * len(PRESSURE_RADII_M) + 12
N_COLUMNS = len(FEATURES) * (1 + len(WINDOWS)) + 2 + N_EXTRA

_FEATURES_CODE = hashlib.sha256(inspect.getsource(features_module).encode()).hexdigest()[:16]
_MEMO: dict[tuple[Any, ...], NDArray[np.float32]] = {}


def match_features(match: Match) -> dict[str, Array]:
    """Frame evidence of a match, cached on disk when the match has a data digest."""

    def compute() -> dict[str, Array]:
        return build_features(match.tracking, match.pitch, match.fps, match.intervals)

    if not match.source:
        return compute()
    key = {
        "match": match.id,
        "profile": match.profile,
        "source": match.source,
        "intervals": match.intervals,
        "pitch": match.pitch,
        "fps": match.fps,
        "code": _FEATURES_CODE,
    }
    return cached("features", key, compute)


def frame_design(features: Mapping[str, Array], intervals: Sequence[Interval]) -> Array:
    """Frame matrix: each feature raw and smoothed, plus capped nearest-player distances."""
    n_frames = len(features["geometry"])
    columns: list[Array] = []
    for name in FEATURES:
        if name in ABSENT_FEATURES:
            values = np.zeros(n_frames)
        else:
            values = np.nan_to_num(np.asarray(features[name], dtype=np.float64))
        columns.append(values)
        columns += [np.nan_to_num(rolling_mean(values, radius, intervals)) for radius in WINDOWS]
    for name in ("nearest_home_m", "nearest_away_m"):
        distance = np.asarray(features[name], dtype=np.float64)
        columns.append(
            np.nan_to_num(
                np.minimum(distance, DISTANCE_CAP_M), nan=DISTANCE_CAP_M, posinf=DISTANCE_CAP_M
            )
            / DISTANCE_CAP_M
        )
    return np.column_stack(columns)


def last_touch(
    features: Mapping[str, Array], intervals: Sequence[Interval], fps: int, radius: float
) -> Array:
    """Team of the latest touch, carried forward inside each half, and seconds since it."""
    home, away = (
        np.asarray(features[k], dtype=np.float64) for k in ("nearest_home_m", "nearest_away_m")
    )
    touch = np.where(
        (home <= radius) & (home < away), 0, np.where((away <= radius) & (away < home), 1, -1)
    )
    owner = np.full(len(touch), -1)
    since = np.full(len(touch), SINCE_CAP_S)
    for _, start, stop in intervals:
        block = touch[start:stop]
        index = np.where(block >= 0, np.arange(len(block)), -1)
        index = np.maximum.accumulate(index)
        seen = index >= 0
        owner[start:stop] = np.where(seen, block[np.maximum(index, 0)], -1)
        since[start:stop] = np.where(
            seen, np.minimum((np.arange(len(block)) - index) / fps, SINCE_CAP_S), SINCE_CAP_S
        )
    return np.column_stack(
        (
            (owner == 0).astype(float),
            (owner == 1).astype(float),
            since / SINCE_CAP_S,
            (touch == 0).astype(float),
            (touch == 1).astype(float),
        )
    )


def kicks(match: Match, features: Mapping[str, Array]) -> Array:
    """Ball acceleration (passes, receptions, deflections) and which team stood closest."""
    pitch = np.asarray(match.pitch)
    xy = match.tracking.ball[:, :2].astype(np.float64) * pitch
    accel, speed, lag = np.zeros(len(xy)), np.zeros(len(xy)), KICK_LAG_FRAMES
    for _, start, stop in match.intervals:
        block = xy[start:stop]
        if len(block) <= 4 * lag:
            continue
        velocity = np.full_like(block, np.nan)
        velocity[lag:-lag] = (block[2 * lag :] - block[: -2 * lag]) * match.fps / (2 * lag)
        change = np.full(len(block), np.nan)
        change[lag:-lag] = (
            np.linalg.norm(velocity[2 * lag :] - velocity[: -2 * lag], axis=1)
            * match.fps
            / (2 * lag)
        )
        accel[start:stop] = np.nan_to_num(np.minimum(change, ACCEL_CAP_MPS2)) / ACCEL_CAP_MPS2
        speed[start:stop] = (
            np.nan_to_num(np.minimum(np.linalg.norm(velocity, axis=1), BALL_SPEED_CAP_MPS))
            / BALL_SPEED_CAP_MPS
        )
    home, away = (
        np.asarray(features[k], dtype=np.float64) for k in ("nearest_home_m", "nearest_away_m")
    )
    near_home = ((home < away) & (home <= KICK_RADIUS_M)).astype(float)
    near_away = ((away < home) & (away <= KICK_RADIUS_M)).astype(float)
    columns = [accel, speed, accel * near_home, accel * near_away]
    return np.column_stack(
        columns + [np.nan_to_num(rolling_mean(c, 6, match.intervals)) for c in columns]
    )


def pressure(match: Match) -> Array:
    """Players of each team near the ball: who surrounds it says who is likely to keep it."""
    pitch = np.asarray(match.pitch)
    players = match.tracking.players[:, :, :2].astype(np.float64) * pitch
    ball = match.tracking.ball[:, None, :2].astype(np.float64) * pitch
    with np.errstate(invalid="ignore"):
        distance = np.linalg.norm(players - ball, axis=2)
    columns: list[Array] = []
    for radius in PRESSURE_RADII_M:
        near = np.nan_to_num(distance, nan=np.inf) <= radius
        for side in (0, 1):
            count = near[:, match.tracking.teams == side].sum(axis=1).astype(float) / 5.0
            columns += [count, np.nan_to_num(rolling_mean(count, 25, match.intervals))]
    return np.column_stack(columns)


def team_shape(match: Match) -> Array:
    """The team with the ball spreads out and moves up: spread around each centroid, each
    centroid's speed towards the opponent goal and the ball's progress along home's attack."""
    pitch = np.asarray(match.pitch)
    players = match.tracking.players[:, :, :2].astype(np.float64) * pitch
    ball_x = match.tracking.ball[:, 0].astype(np.float64) * pitch[0]
    lag = max(1, round(SHAPE_LAG_S * match.fps))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with a whole team missing
        centroid = [np.nanmean(players[:, match.tracking.teams == side], axis=1) for side in (0, 1)]
        spread = [
            np.nanmean(
                np.linalg.norm(players[:, match.tracking.teams == side] - c[:, None], axis=2),
                axis=1,
            )
            for side, c in zip((0, 1), centroid, strict=True)
        ]
    direction = np.zeros(len(players))  # +1 when home attacks towards growing x in this half
    for _, start, stop in match.intervals:
        gap = np.nanmean(centroid[1][start:stop, 0] - centroid[0][start:stop, 0])
        direction[start:stop] = 1.0 if not np.isfinite(gap) or gap >= 0 else -1.0
    columns: list[Array] = []
    for side, sign in ((0, 1.0), (1, -1.0)):
        x = centroid[side][:, 0]
        advance = np.zeros(len(x))
        advance[lag:] = (x[lag:] - x[:-lag]) * match.fps / lag
        advance = np.clip(np.nan_to_num(advance * direction * sign), -8.0, 8.0) / 8.0
        width = np.nan_to_num(np.minimum(spread[side], SPREAD_CAP_M), nan=0.0) / SPREAD_CAP_M
        columns += [width, advance, np.nan_to_num(rolling_mean(advance, 50, match.intervals))]
    progress = np.nan_to_num((ball_x / pitch[0] - 0.5) * direction * 2.0)
    columns += [
        columns[0] - columns[3],
        progress,
        np.nan_to_num(rolling_mean(progress, 50, match.intervals)),
    ]
    drive = np.zeros(len(ball_x))  # ball speed towards away's goal: forward passes say who plays
    drive[lag:] = (ball_x[lag:] - ball_x[:-lag]) * match.fps / lag
    drive = np.clip(np.nan_to_num(drive * direction), -20.0, 20.0) / 20.0
    columns += [
        drive,
        np.nan_to_num(rolling_mean(drive, 25, match.intervals)),
        np.nan_to_num(rolling_mean(drive, 100, match.intervals)),
    ]
    return np.column_stack(columns)


def build_design(match: Match, features: Mapping[str, Array]) -> NDArray[np.float32]:
    """The (frames, N_COLUMNS) float32 design matrix of a match."""
    return np.column_stack(
        (
            frame_design(features, match.intervals),
            *[last_touch(features, match.intervals, match.fps, radius) for radius in TOUCH_RADII_M],
            kicks(match, features),
            pressure(match),
            team_shape(match),
        )
    ).astype(np.float32)


def design_matrix(match: Match) -> NDArray[np.float32]:
    """`build_design` over the match's cached features, memoised for matches with a digest."""
    key = (match.id, match.profile, match.source, tuple(match.intervals))
    if match.source and key in _MEMO:
        return _MEMO[key]
    matrix = build_design(match, match_features(match))
    if match.source:
        _MEMO[key] = matrix
    return matrix


def clear_memo() -> None:
    """Drop the memoised design matrices."""
    _MEMO.clear()


def swapped(match: Match) -> Match:
    """The same match with home and away exchanged, so H/A evidence is learned symmetrically."""
    names = match.tracking.team_names
    tracking = Tracking(
        match.tracking.players,
        match.tracking.ball,
        (1 - match.tracking.teams).astype(np.int8),
        (names[1], names[0]),
    )
    labels = np.where(match.labels == "H", "A", np.where(match.labels == "A", "H", match.labels))
    source = f"{match.source}/swap" if match.source else ""
    return replace(match, tracking=tracking, labels=labels, source=source)


def augmented(matches: Sequence[Match]) -> list[Match]:
    """Training copies: each match, two more degradation draws of it, and all of them swapped."""
    copies = list(matches) + [
        load_match(m.id, m.profile, seed_offset=k) for m in matches for k in AUGMENT_SEEDS
    ]
    return copies + [swapped(m) for m in copies]
