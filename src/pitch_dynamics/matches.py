"""Matches as the model sees them: tracking, labels, dataset splits and tracking degradation."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pitch_dynamics import sources
from pitch_dynamics.paths import CACHE

Interval = tuple[str, int, int]  # half name, first frame, one past the last frame

# The development pool scores every model with leave-one-match-out; the holdouts are read once,
# with a frozen model, after development.
DEV = ("J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ")
HOLDOUT_DFL = ("J03WN1", "J03WR9")
HOLDOUT_LALIGA = ("skc_2841", "skc_3442")
TRAINING = DEV + HOLDOUT_DFL

SPEED_LAG_FRAMES = 3


@dataclass
class Tracking:
    """Automatic tracking: positions normalised to [0, 1] with speed in m/s, one slot per player."""

    players: NDArray[np.float32]  # (frames, slots, 3): x, y, speed
    ball: NDArray[np.float32]  # (frames, 3): x, y, speed
    teams: NDArray[np.int8]  # (slots,): 0 home, 1 away
    team_names: tuple[str, str]


@dataclass
class Match:
    """One match: tracking, frame labels ("X" everywhere when unknown) and its halves."""

    id: str
    tracking: Tracking
    labels: NDArray[np.str_]
    intervals: list[Interval]
    pitch: tuple[float, float]  # metres
    fps: int
    forced_dead: list[tuple[int, int]] = field(default_factory=list)  # half-open, always dead
    profile: str = "clean"
    source: str = ""  # digest of the input data, so caches never outlive it


@dataclass(frozen=True)
class Profile:
    """Degradation that makes clean external tracking look like the challenge tracking.

    Gaps and noise magnitudes are set a priori from the challenge documentation, not measured on
    the challenge match. Ball gaps are independent of the label, so a missing ball never teaches
    "missing = dead".
    """

    name: str
    ball_missing: float = 0.0
    ball_gap_mean_frames: float = 1.0
    player_missing: float = 0.0
    player_gap_mean_frames: float = 1.0
    ball_noise_m: float = 0.0
    player_noise_m: float = 0.0


PROFILES = {
    "clean": Profile("clean"),
    "atd_like_v1": Profile("atd_like_v1", 0.40, 50.0, 0.08, 25.0, 0.20, 0.05),
    # Broadcast tracking already loses the ball and off-screen players: only noise is added.
    "broadcast_v1": Profile("broadcast_v1", 0.0, 1.0, 0.0, 1.0, 0.10, 0.05),
}


def default_profile(match_id: str) -> str:
    """`broadcast_v1` for SkillCorner matches, `atd_like_v1` for DFL matches."""
    return "broadcast_v1" if match_id.startswith("skc_") else "atd_like_v1"


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "little")


def intervals_from_array(rows: NDArray[Any]) -> list[Interval]:
    """Half-open intervals from rows of (half, start, stop)."""
    return [(str(row[0]), int(row[1]), int(row[2])) for row in rows]


def gap_mask(
    n_frames: int, missing: float, mean_gap: float, rng: np.random.Generator
) -> NDArray[np.bool_]:
    """Boolean mask with `missing` expected share, in geometric gaps of `mean_gap` frames."""
    mask = np.zeros(n_frames, dtype=bool)
    if missing <= 0:
        return mask
    mean_run = mean_gap * (1 - missing) / missing  # observed stretches between gaps
    position = int(rng.geometric(1 / (mean_gap + mean_run)) - 1)
    while position < n_frames:
        length = int(rng.geometric(1 / mean_gap))
        mask[position : position + length] = True
        position += length + int(rng.geometric(1 / mean_run))
    return mask


def player_speeds(
    xy: NDArray[np.float32],
    pitch: tuple[float, float],
    fps: int,
    intervals: Sequence[Interval],
) -> NDArray[np.float64]:
    """Player speeds in m/s from positions; NaN where a side of the difference is unobserved."""
    lag = SPEED_LAG_FRAMES
    speed = np.full(xy.shape[:2], np.nan)
    for _, start, stop in intervals:
        if stop - start <= 2 * lag:
            continue
        block = xy[start:stop]
        delta = (block[2 * lag :] - block[: -2 * lag]) * np.asarray(pitch)
        speed[start + lag : stop - lag] = np.linalg.norm(delta, axis=2) * fps / (2 * lag)
    return speed


def degrade(
    raw: Mapping[str, Any], profile: Profile, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Players and ball of a parsed match under `profile`; deterministic for (profile, seed)."""
    rng = np.random.default_rng(seed)
    pitch = np.asarray(raw["pitch"], dtype=np.float64)
    players = raw["players"].astype(np.float32).copy()
    ball = raw["ball"].astype(np.float32).copy()
    n_frames, n_slots = players.shape[:2]
    if profile.player_noise_m:
        players[:, :, :2] += rng.normal(0, profile.player_noise_m, players[:, :, :2].shape) / pitch
    if profile.ball_noise_m:
        ball[:, :2] += rng.normal(0, profile.ball_noise_m, ball[:, :2].shape) / pitch
    hidden = gap_mask(n_frames, profile.ball_missing, profile.ball_gap_mean_frames, rng)
    ball[hidden, :2] = np.nan
    for slot in range(n_slots):
        hidden = gap_mask(n_frames, profile.player_missing, profile.player_gap_mean_frames, rng)
        players[hidden, slot, :2] = np.nan
    players[:, :, 2] = player_speeds(
        players[:, :, :2],
        (float(pitch[0]), float(pitch[1])),
        int(raw["fps"]),
        intervals_from_array(raw["intervals"]),
    )
    ball[:, 2] = np.nan  # the ATD ball speed column is not a speed: positions only
    return players, ball


def raw_path(match_id: str) -> Path:
    """Where the parsed arrays of an external match live in the cache."""
    return CACHE / "parsed" / f"{match_id}.npz"


def fetch(match_id: str) -> Path:
    """Download and parse one external match into the cache, once; returns the parsed file."""
    target = raw_path(match_id)
    if target.is_file():
        return target
    if match_id.startswith("skc_"):
        skillcorner_id = int(match_id.split("_")[1])
        sources.fetch_skillcorner(skillcorner_id)
        raw = sources.parse_skillcorner(skillcorner_id)
    else:
        sources.fetch_dfl(match_id)
        raw = sources.parse_dfl(match_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **raw)
    temporary.replace(target)
    return target


def load_raw(match_id: str) -> dict[str, NDArray[Any]]:
    """The parsed arrays of an external match, as stored in the cache."""
    path = raw_path(match_id)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `pitch-dynamics data` first")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_match(match_id: str, profile: str | None = None, seed_offset: int = 0) -> Match:
    """An external match degraded under `profile` (default per source).

    `seed_offset` draws a fresh degradation for training augmentation; evaluation always uses
    offset 0, so every model is scored on identical inputs.
    """
    chosen = PROFILES[profile or default_profile(match_id)]
    raw = load_raw(match_id)
    source = sources.sha256(raw_path(match_id))[:16]
    players, ball = degrade(raw, chosen, _seed(match_id, chosen.name, seed_offset))
    tracking = Tracking(players, ball, raw["teams"].astype(np.int8), ("Home", "Away"))
    return Match(
        match_id,
        tracking,
        raw["labels"].astype("<U1"),
        intervals_from_array(raw["intervals"]),
        (float(raw["pitch"][0]), float(raw["pitch"][1])),
        int(raw["fps"]),
        profile=chosen.name,
        source=f"{source}/{seed_offset}",
    )


def cached(
    namespace: str, key: Mapping[str, Any], compute: Callable[[], dict[str, NDArray[Any]]]
) -> dict[str, NDArray[Any]]:
    """Small npz cache for derived arrays; `key` must include everything the result depends on."""
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]
    path = CACHE / "derived" / namespace / f"{digest}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **value)  # type: ignore[arg-type]
    temporary.replace(path)
    return value


def summary(match_id: str) -> dict[str, Any]:
    """Frame count and label counts of a parsed external match."""
    labels = load_raw(match_id)["labels"]
    names, counts = np.unique(labels, return_counts=True)
    return {
        "frames": len(labels),
        "labels": {str(name): int(count) for name, count in zip(names, counts, strict=True)},
    }
