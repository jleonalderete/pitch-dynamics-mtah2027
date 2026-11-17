"""The challenge match: read its local ATD tracking and write the submission file."""

import platform
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pitch_dynamics import official, paths
from pitch_dynamics.guard import refuse_agent
from pitch_dynamics.matches import Interval, Match, Tracking, player_speeds
from pitch_dynamics.sources import sha256

# The organisers' rule: this stretch of the second half is always dead ball.
FORCED_DEAD = [(110808, 113742)]

SUBMISSION_LABELS = frozenset({"H", "A", "D", "X"})


def read_metadata(path: Path) -> tuple[NDArray[np.int8], tuple[str, str]]:
    """Team (0 or 1) of every ATD slot and the two team names."""
    root = ET.parse(path).getroot()
    # Support a default XML namespace without relying on a particular URI.
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    teams = list(root.iter("Team"))
    ids = [team.get("id") for team in teams]
    if len(ids) != 2 or None in ids or len(set(ids)) != 2:
        raise ValueError("ATD metadata must contain exactly two distinct teams")
    mapping = {team_id: index for index, team_id in enumerate(ids)}
    slots = list(root.iter("Player"))
    if not slots or any(player.get("teamId") not in mapping for player in slots):
        raise ValueError("every ATD slot must reference a known team")
    names = (teams[0].findtext("Name") or str(ids[0]), teams[1].findtext("Name") or str(ids[1]))
    return np.array([mapping[player.get("teamId")] for player in slots], dtype=np.int8), names


def _observation(group: str) -> list[float]:
    parts = group.strip().split(",")
    if len(parts) != 3:
        raise ValueError("expected x,y,speed")
    values = [float(value) for value in parts]
    if any(
        np.isinf(value) or (np.isfinite(value) and abs(value) > np.finfo(np.float32).max)
        for value in values
    ):
        raise ValueError("infinite or unrepresentable observation")
    return values


def load_tracking(meta_path: Path, raw_path: Path, n_frames: int) -> Tracking:
    """Strictly parse the ATD tracking file: one record per frame, one observation per slot."""
    teams, names = read_metadata(meta_path)
    players = np.full((n_frames, len(teams), 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    count = 0
    with raw_path.open(encoding="utf-8-sig") as handle:
        for count, line in enumerate(handle, 1):
            if count > n_frames:
                raise ValueError(f"ATD has more than {n_frames} records")
            try:
                frame, groups, point = line.strip().split(":")
                if int(frame) != count:
                    raise ValueError(f"expected frame_id {count}, got {frame}")
                slots = groups.split(";")
                # Some EPTS writers terminate the slot list with a semicolon.
                if slots[-1] == "":
                    slots.pop()
                if len(slots) != len(teams):
                    raise ValueError(f"expected {len(teams)} slots, got {len(slots)}")
                players[count - 1] = [_observation(group) for group in slots]
                ball[count - 1] = _observation(point)
            except (ValueError, OverflowError) as error:
                raise ValueError(f"ATD line {count}: {error}") from error
    if count != n_frames:
        raise ValueError(f"ATD has {count} records; expected {n_frames}")
    return Tracking(players, ball, teams, names)


def atd_files(source: str, data_dir: Path | None = None) -> tuple[Path, Path]:
    """Metadata and raw ATD paths of one recording, or an error listing the missing files."""
    files = official.games().asset_paths(data_dir=str(data_dir or paths.DATA), source=source)
    missing = [files[name] for name in ("atd_meta", "atd_raw") if not Path(files[name]).is_file()]
    if missing:
        raise ValueError(
            "Missing local match assets:\n  "
            + "\n  ".join(missing)
            + "\nUnpack the official bundle or pass --data-dir /path/to/bundle."
        )
    return Path(files["atd_meta"]), Path(files["atd_raw"])


def load_challenge(data_dir: Path | None = None) -> Match:
    """The challenge match in the same shape as an external one: positions-only speeds.

    The tracking speed columns are not used: player speeds come from positions and the ball
    has none, exactly as for the degraded external matches.
    """
    refuse_agent()
    games = official.games()
    meta, raw = atd_files("tactical", data_dir)
    n_frames = games.source_frames("tactical")
    tracking = load_tracking(meta, raw, n_frames)
    intervals: list[Interval] = [
        (str(half), start, last + 1) for half, start, last in games.half_video_blocks(games.GAME)
    ]
    length, width = games.GAMES[games.GAME]["pitch"]
    pitch = (float(length), float(width))
    fps = int(games.GAMES[games.GAME]["fps"])
    players = tracking.players.astype(np.float32).copy()
    players[:, :, 2] = player_speeds(players[:, :, :2], pitch, fps, intervals)
    ball = tracking.ball.astype(np.float32).copy()
    ball[:, 2] = np.nan
    source = sha256(meta)[:8] + sha256(raw)[:8]
    return Match(
        "challenge",
        Tracking(players, ball, tracking.teams, tracking.team_names),
        np.full(n_frames, "X", dtype="<U1"),
        intervals,
        pitch,
        fps,
        forced_dead=FORCED_DEAD,
        profile="challenge",
        source=source,
    )


def write_submission(path: Path, labels: Sequence[str]) -> None:
    """Write the frame,label CSV; row i is frame i of the tactical video."""
    if not set(labels) <= SUBMISSION_LABELS:
        raise ValueError("invalid submission labels")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "frame,label\n" + "".join(f"{frame},{label}\n" for frame, label in enumerate(labels))
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _git_commit() -> str:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.ROOT, text=True, capture_output=True, check=True
        )
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return done.stdout.strip()


def provenance() -> dict[str, str]:
    """Commit, interpreter, numeric libraries and platform of the current run."""
    return {
        "commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "platform": platform.platform(),
    }
