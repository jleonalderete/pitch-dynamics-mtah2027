"""Align the broadcast ATD to the tactical timeline and fill the tactical ball gaps with it.

The two recordings show the same match on different timelines, and no frame mapping is
provided: the broadcast has cuts, replays and close-ups. Both run at 25 fps, so inside a live
broadcast shot the mapping is a constant offset, `tactical = broadcast + offset`, and the whole
recording uses only a few offsets (one per continuous piece of the TV feed). Replays, slow
motion and close-ups fit none of them and stay unmatched.

1. Candidates. Each broadcast window is compared with every tactical shift through the mean
   squared distance between the two ball tracks, computed for all shifts at once with FFT
   cross-correlations.
2. Verification. A candidate is kept when the broadcast players lie close to tactical players of
   the same team. The ball alone is not trusted: an airborne ball projects onto the pitch
   differently from each camera, and a still ball fits every restart from the same spot.
3. Voting. Offsets kept by enough windows become the offsets of the recording.
4. Assignment. Each broadcast frame takes the offset its players fit, frame by frame and over a
   one-second median, so a cut never leaks a match into the replay next to it. Short runs with
   too few visible players are bridged between frames matched to the same offset.

Only tracking geometry is compared: no labels, no video. Filling replaces missing tactical ball
positions and never overwrites an observed one.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from pitch_dynamics import official
from pitch_dynamics.challenge import atd_files, load_challenge, load_tracking, provenance
from pitch_dynamics.guard import refuse_agent
from pitch_dynamics.matches import Match, Tracking
from pitch_dynamics.sources import sha256

WINDOW_FRAMES = 750  # broadcast stretch searched for one offset
HOP_FRAMES = 375
MIN_BALL_OVERLAP_FRAMES = 75  # frames where both cameras see the ball, per candidate shift
CANDIDATES = 5  # best ball shifts verified with the players, per window
CANDIDATE_SEPARATION_FRAMES = 25
VERIFY_STRIDE_FRAMES = 5
MIN_VERIFY_FRAMES = 20  # sampled frames with enough players to judge a candidate
MIN_PLAYERS = 3  # broadcast players needed to judge one frame
MATCH_RESIDUAL_M = 2.0  # median distance to the nearest tactical teammate of a true match
MIN_SUPPORT_WINDOWS = 4  # windows that must agree before an offset is trusted
OFFSET_TOLERANCE_FRAMES = 2  # window votes this close count for the same offset
SMOOTH_FRAMES = 25
BRIDGE_FRAMES = 25
CHUNK_FRAMES = 4096

PARAMETERS = {
    "window_frames": WINDOW_FRAMES,
    "hop_frames": HOP_FRAMES,
    "min_ball_overlap_frames": MIN_BALL_OVERLAP_FRAMES,
    "candidates": CANDIDATES,
    "candidate_separation_frames": CANDIDATE_SEPARATION_FRAMES,
    "verify_stride_frames": VERIFY_STRIDE_FRAMES,
    "min_verify_frames": MIN_VERIFY_FRAMES,
    "min_players": MIN_PLAYERS,
    "match_residual_m": MATCH_RESIDUAL_M,
    "min_support_windows": MIN_SUPPORT_WINDOWS,
    "offset_tolerance_frames": OFFSET_TOLERANCE_FRAMES,
    "smooth_frames": SMOOTH_FRAMES,
    "bridge_frames": BRIDGE_FRAMES,
}


@dataclass
class Alignment:
    """Where each broadcast frame sits on the tactical timeline."""

    offsets: list[tuple[int, int]]  # (tactical minus broadcast frame, supporting windows)
    tactical_frame: NDArray[np.int64]  # per broadcast frame; -1 when unmatched
    residual_m: NDArray[np.float64]  # per broadcast frame; NaN when unmatched or bridged


def recode_teams(tracking: Tracking, team_names: tuple[str, str]) -> Tracking:
    """The same tracking with teams numbered in the order of `team_names`."""
    if len(set(team_names)) != 2 or set(tracking.team_names) != set(team_names):
        raise ValueError(f"broadcast teams {tracking.team_names} differ from {team_names}")
    code = np.array([team_names.index(name) for name in tracking.team_names], dtype=np.int8)
    return Tracking(tracking.players, tracking.ball, code[tracking.teams], team_names)


def load_broadcast(team_names: tuple[str, str], data_dir: Path | None = None) -> Tracking:
    """The broadcast ATD on its own timeline, with teams numbered like `team_names`."""
    refuse_agent()
    meta, raw = atd_files("broadcast", data_dir)
    tracking = load_tracking(meta, raw, official.games().source_frames("broadcast"))
    return recode_teams(tracking, team_names)


def _median(values: NDArray[np.float64], minimum: int) -> NDArray[np.float64]:
    """Row medians that skip NaN and keep inf; NaN where a row has fewer than `minimum` values."""
    ordered = np.sort(values, axis=1)  # NaN sorts last
    count = (~np.isnan(values)).sum(axis=1)
    low = np.take_along_axis(ordered, np.maximum(count - 1, 0)[:, None] // 2, axis=1)[:, 0]
    high = np.take_along_axis(ordered, (count // 2)[:, None], axis=1)[:, 0]
    median = (low + high) / 2
    median[count < max(minimum, 1)] = np.nan
    return median


def player_residuals(
    tactical_xy: NDArray[np.float32],
    tactical_teams: NDArray[np.int8],
    broadcast_xy: NDArray[np.float32],
    broadcast_teams: NDArray[np.int8],
    frames: NDArray[np.int64],
    offset: int,
) -> NDArray[np.float64]:
    """Median distance (m) from each broadcast player to the nearest tactical teammate.

    Positions are in metres. One value per broadcast frame in `frames`, compared with tactical
    frame `frame + offset`; NaN outside the tactical timeline or with fewer than MIN_PLAYERS
    broadcast players. A broadcast player with no tactical teammate in view counts as infinitely
    far.
    """
    result = np.full(len(frames), np.nan)
    tactical = frames + offset
    inside = np.flatnonzero((tactical >= 0) & (tactical < len(tactical_xy)))
    same_team = broadcast_teams[:, None] == tactical_teams[None, :]
    for begin in range(0, len(inside), CHUNK_FRAMES):
        chunk = inside[begin : begin + CHUNK_FRAMES]
        seen = broadcast_xy[frames[chunk]]
        reference = tactical_xy[tactical[chunk]]
        distance = np.linalg.norm(seen[:, :, None] - reference[:, None], axis=3)
        distance = np.where(same_team & ~np.isnan(distance), distance, np.inf)
        nearest = distance.min(axis=2).astype(np.float64)
        nearest[~np.isfinite(seen).all(axis=2)] = np.nan
        result[chunk] = _median(nearest, MIN_PLAYERS)
    return result


def _ball_channels(ball_m: NDArray[np.float32]) -> NDArray[np.float64]:
    """Mask, squared norm, x and y of a ball track, zero where the ball is unseen."""
    seen = np.isfinite(ball_m).all(axis=1)
    xy = np.where(seen[:, None], ball_m, 0).astype(np.float64)
    return np.stack([seen.astype(np.float64), (xy**2).sum(axis=1), xy[:, 0], xy[:, 1]])


def _candidate_starts(mean_sq_m2: NDArray[np.float64]) -> list[int]:
    """Up to CANDIDATES tactical starts with the lowest ball error, kept apart from each other."""
    picked: list[int] = []
    for start in np.argsort(mean_sq_m2, kind="stable"):
        if len(picked) == CANDIDATES or not np.isfinite(mean_sq_m2[start]):
            break
        if all(abs(int(start) - other) > CANDIDATE_SEPARATION_FRAMES for other in picked):
            picked.append(int(start))
    return picked


def window_offsets(
    tactical_ball_m: NDArray[np.float32],
    broadcast_ball_m: NDArray[np.float32],
    tactical_xy: NDArray[np.float32],
    tactical_teams: NDArray[np.int8],
    broadcast_xy: NDArray[np.float32],
    broadcast_teams: NDArray[np.int8],
) -> list[int]:
    """The offset each broadcast window supports, for the windows whose ball and players fit one.

    For a window starting at broadcast frame `s` and a tactical start `u`, the squared ball
    distance summed over the window expands into cross-correlations of the ball channels, so one
    FFT pass scores every `u`. The offset is `u - s`.
    """
    n_tactical, n_broadcast = len(tactical_ball_m), len(broadcast_ball_m)
    n_starts = n_tactical - WINDOW_FRAMES + 1
    size = 1 << (n_tactical + WINDOW_FRAMES).bit_length()
    tactical = np.fft.rfft(_ball_channels(tactical_ball_m), size)
    broadcast = _ball_channels(broadcast_ball_m)
    votes: list[int] = []
    for start in range(0, n_broadcast - WINDOW_FRAMES + 1, HOP_FRAMES):
        window = broadcast[:, start : start + WINDOW_FRAMES]
        if window[0].sum() < MIN_BALL_OVERLAP_FRAMES or n_starts <= 0:
            continue
        spectrum = np.conj(np.fft.rfft(window, size))
        overlap = np.rint(np.fft.irfft(spectrum[0] * tactical[0], size)[:n_starts])
        squared = np.fft.irfft(
            spectrum[1] * tactical[0]
            + spectrum[0] * tactical[1]
            - 2 * (spectrum[2] * tactical[2] + spectrum[3] * tactical[3]),
            size,
        )[:n_starts]
        enough = overlap >= MIN_BALL_OVERLAP_FRAMES
        mean_sq = np.where(enough, squared / np.maximum(overlap, 1), np.inf)
        sampled = np.arange(start, start + WINDOW_FRAMES, VERIFY_STRIDE_FRAMES)
        best: tuple[float, int] | None = None
        for tactical_start in _candidate_starts(mean_sq):
            offset = tactical_start - start
            residuals = player_residuals(
                tactical_xy, tactical_teams, broadcast_xy, broadcast_teams, sampled, offset
            )
            judged = residuals[~np.isnan(residuals)]
            if len(judged) < MIN_VERIFY_FRAMES:
                continue
            residual = float(np.median(judged))
            if residual <= MATCH_RESIDUAL_M and (best is None or residual < best[0]):
                best = (residual, offset)
        if best is not None:
            votes.append(best[1])
    return votes


def supported_offsets(votes: Sequence[int]) -> list[tuple[int, int]]:
    """(offset, windows) for every group of nearby votes with at least MIN_SUPPORT_WINDOWS."""
    values = np.sort(np.asarray(votes, dtype=np.int64))
    if not len(values):
        return []
    groups = np.split(values, np.flatnonzero(np.diff(values) > OFFSET_TOLERANCE_FRAMES) + 1)
    supported: list[tuple[int, int]] = []
    for group in groups:
        if len(group) >= MIN_SUPPORT_WINDOWS:
            offsets, counts = np.unique(group, return_counts=True)
            supported.append((int(offsets[np.argmax(counts)]), len(group)))
    return supported


def _bridge(choice: NDArray[np.int64], unjudged: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Extend matches over short unmatched runs that cannot be judged against the same offset."""
    result = choice.copy()
    matched = np.flatnonzero(choice >= 0)
    gaps = np.diff(matched) - 1
    for index in np.flatnonzero((gaps > 0) & (gaps <= BRIDGE_FRAMES)):
        left, right = matched[index], matched[index + 1]
        offset_index = choice[left]
        if choice[right] == offset_index and unjudged[offset_index, left + 1 : right].all():
            result[left + 1 : right] = offset_index
    return result


def assign(
    tactical_xy: NDArray[np.float32],
    tactical_teams: NDArray[np.int8],
    broadcast_xy: NDArray[np.float32],
    broadcast_teams: NDArray[np.int8],
    offsets: Sequence[int],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Tactical frame (or -1) and player residual of each broadcast frame, given the offsets.

    A frame is matched to an offset when both its own residual and the median over
    SMOOTH_FRAMES around it are within MATCH_RESIDUAL_M; with several fits the lowest median wins.
    """
    frames = np.arange(len(broadcast_xy), dtype=np.int64)
    raw = np.stack(
        [
            player_residuals(
                tactical_xy, tactical_teams, broadcast_xy, broadcast_teams, frames, offset
            )
            for offset in offsets
        ]
    )
    half = SMOOTH_FRAMES // 2
    padded = np.pad(raw, ((0, 0), (half, SMOOTH_FRAMES - 1 - half)), constant_values=np.nan)
    smooth = np.stack([_median(sliding_window_view(row, SMOOTH_FRAMES), 1) for row in padded])
    fits = (raw <= MATCH_RESIDUAL_M) & (smooth <= MATCH_RESIDUAL_M)
    score = np.where(fits, smooth, np.inf)
    best = np.argmin(score, axis=0)
    choice = np.where(fits.any(axis=0), best, -1)
    choice = _bridge(choice, np.isnan(raw))
    matched = choice >= 0
    chosen = np.maximum(choice, 0)
    tactical_frame = np.where(matched, frames + np.asarray(offsets, dtype=np.int64)[chosen], -1)
    residual = np.where(matched, raw[chosen, frames], np.nan)
    return tactical_frame, residual


def align(tactical: Tracking, broadcast: Tracking, pitch: tuple[float, float]) -> Alignment:
    """Map every broadcast frame onto the tactical timeline; teams must be numbered alike."""
    scale = np.asarray(pitch, dtype=np.float32)
    tactical_xy = tactical.players[:, :, :2] * scale
    broadcast_xy = broadcast.players[:, :, :2] * scale
    votes = window_offsets(
        tactical.ball[:, :2] * scale,
        broadcast.ball[:, :2] * scale,
        tactical_xy,
        tactical.teams,
        broadcast_xy,
        broadcast.teams,
    )
    offsets = supported_offsets(votes)
    if not offsets:
        raise ValueError(
            f"no broadcast offset is supported by {MIN_SUPPORT_WINDOWS} windows "
            f"({len(votes)} windows fit any offset): the two recordings do not line up"
        )
    tactical_frame, residual = assign(
        tactical_xy,
        tactical.teams,
        broadcast_xy,
        broadcast.teams,
        [offset for offset, _ in offsets],
    )
    return Alignment(offsets, tactical_frame, residual)


def fill_ball(
    tactical_ball: NDArray[np.float32],
    broadcast_ball: NDArray[np.float32],
    tactical_frame: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Tactical ball x, y with gaps filled from matched broadcast frames, and the filled mask.

    If two broadcast frames land on the same tactical frame, the earlier one is used.
    """
    ball = tactical_ball[:, :2].astype(np.float32)
    usable = np.flatnonzero((tactical_frame >= 0) & np.isfinite(broadcast_ball[:, :2]).all(axis=1))
    target = tactical_frame[usable]
    missing = np.isnan(ball[target]).any(axis=1)
    target, first = np.unique(target[missing], return_index=True)
    ball[target] = broadcast_ball[usable[missing][first], :2]
    filled = np.zeros(len(ball), dtype=bool)
    filled[target] = True
    return ball, filled


def summary(
    alignment: Alignment,
    tactical: Tracking,
    broadcast: Tracking,
    filled: NDArray[np.bool_],
    graded: NDArray[np.bool_],
    pitch: tuple[float, float],
) -> dict[str, Any]:
    """Label-free quality numbers of an alignment and of the ball it fills."""
    frames = np.arange(len(alignment.tactical_frame))
    matched = alignment.tactical_frame >= 0
    shift = alignment.tactical_frame - frames
    pairs = np.flatnonzero(matched & np.isfinite(broadcast.ball[:, :2]).all(axis=1))
    seen_by = pairs[np.isfinite(tactical.ball[alignment.tactical_frame[pairs], :2]).all(axis=1)]
    gap = broadcast.ball[seen_by, :2] - tactical.ball[alignment.tactical_frame[seen_by], :2]
    disagreement = np.linalg.norm(gap * np.asarray(pitch), axis=1)
    before = np.isfinite(tactical.ball[:, :2]).all(axis=1) & graded
    return {
        "offsets": [
            {
                "offset_frames": offset,
                "windows": windows,
                "broadcast_frames": int((matched & (shift == offset)).sum()),
            }
            for offset, windows in alignment.offsets
        ],
        "broadcast_frames": len(frames),
        "broadcast_matched_share": float(matched.mean()),
        "player_residual_m_median": float(np.nanmedian(alignment.residual_m)),
        "ball_disagreement_m": {
            "frames": len(seen_by),
            "median": float(np.median(disagreement)) if len(seen_by) else None,
            "p90": float(np.percentile(disagreement, 90)) if len(seen_by) else None,
        },
        "graded_frames": int(graded.sum()),
        "graded_ball_before": int(before.sum()),
        "graded_ball_filled": int((filled & graded).sum()),
    }


@dataclass
class BallFill:
    """A challenge match with its ball completed from the broadcast, and how it was done."""

    match: Match
    alignment: Alignment
    filled: NDArray[np.bool_]  # per tactical frame
    summary: dict[str, Any]


def complete_ball(match: Match, broadcast: Tracking, broadcast_source: str) -> BallFill:
    """The match with ball gaps filled from the aligned broadcast tracking.

    The match source digest gains the broadcast digest, so features cached for the unfilled
    ball are never reused.
    """
    alignment = align(match.tracking, broadcast, match.pitch)
    ball, filled = fill_ball(match.tracking.ball, broadcast.ball, alignment.tactical_frame)
    graded = np.zeros(len(ball), dtype=bool)
    for _, start, stop in match.intervals:
        graded[start:stop] = True
    tracking = replace(
        match.tracking,
        ball=np.column_stack([ball, match.tracking.ball[:, 2]]).astype(np.float32),
    )
    report = {
        **summary(alignment, match.tracking, broadcast, filled, graded, match.pitch),
        "inputs": {"tactical": match.source, "broadcast": broadcast_source},
    }
    completed = replace(match, tracking=tracking, source=f"{match.source}+{broadcast_source}")
    return BallFill(completed, alignment, filled, report)


def fill_match(match: Match, data_dir: Path | None = None) -> BallFill:
    """The challenge match with ball gaps filled from the local broadcast ATD."""
    refuse_agent()
    meta, raw = atd_files("broadcast", data_dir)
    broadcast = load_broadcast(match.tracking.team_names, data_dir)
    return complete_ball(match, broadcast, sha256(meta)[:8] + sha256(raw)[:8])


def run(out_dir: Path, data_dir: Path | None = None) -> dict[str, Any]:
    """Align the broadcast to the challenge match and write `alignment.npz` and `report.json`.

    `alignment.npz` holds, per broadcast frame, `tactical_frame` and `residual_m`, and per
    tactical frame, the filled `ball` (normalised x, y) and the `filled` mask.
    """
    refuse_agent()
    result = fill_match(load_challenge(data_dir), data_dir)
    report = {**result.summary, "parameters": PARAMETERS, **provenance()}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "alignment.npz",
        tactical_frame=result.alignment.tactical_frame,
        residual_m=result.alignment.residual_m,
        ball=result.match.tracking.ball[:, :2],
        filled=result.filled,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
