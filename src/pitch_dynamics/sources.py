"""Download and parse the external training and holdout matches.

- DFL / IDSSE (Bassek et al., Sci Data 12, 195, 2025; CC-BY 4.0): seven Bundesliga matches with
  25 fps positions. Every ball frame carries the official ``BallStatus`` (1 alive, 0 dead) and
  ``BallPossession`` (1 home, 2 away) flags, so the H/A/D labels come from the data provider.
- SkillCorner open data (MIT licence, "Copyright (c) 2020 SkillCorner"): broadcast tracking at
  10 fps, resampled to 25 fps, for the two LaLiga Clasicos of 2019/20. Each frame carries the
  provider's team in possession; frames without one are X (not graded). There is no dead-ball flag.
"""

import hashlib
import json
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pitch_dynamics.paths import CACHE

Raw = dict[str, Any]

DFL_FPS = 25
DFL_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
DFL_MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{revision}/{filename}"
DFL_META_FILE = "DFL_02_01_matchinformation_{competition}_DFL-MAT-{match}.xml"
DFL_POSITIONS_FILE = "DFL_04_03_positions_raw_observed_{competition}_DFL-MAT-{match}.xml"
DFL_SECTIONS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}


@dataclass(frozen=True)
class DflMatch:
    """Where a match lives on the mirror and the digests its two files must have.

    Positions files are Git LFS objects, published with their SHA-256. Metadata files are plain
    git blobs, published with their git object id.
    """

    competition: str
    meta_git_blob: str
    positions_sha256: str


DFL_MATCHES = {
    "J03WMX": DflMatch(
        "DFL-COM-000001",
        "8f2a82d84428c28e0e01f0a908c8395cc8d23900",
        "d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49",
    ),
    "J03WN1": DflMatch(
        "DFL-COM-000001",
        "45d2f112aed6c4212e66db8a91e8da1b8d728090",
        "56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec",
    ),
    "J03WOH": DflMatch(
        "DFL-COM-000002",
        "a693a0cacc2b597357e65a04eec5febfdaff79aa",
        "ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f",
    ),
    "J03WOY": DflMatch(
        "DFL-COM-000002",
        "bce56e5f518216e62d1e93a38322d4b9e357e5ab",
        "a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b",
    ),
    "J03WPY": DflMatch(
        "DFL-COM-000002",
        "46f1ffe15e1b96b12aab10594b12886c503c2f68",
        "845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1",
    ),
    "J03WQQ": DflMatch(
        "DFL-COM-000002",
        "7ec8acbff5a0c6167b632cc230d2a8b0e7b44b1c",
        "cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f",
    ),
    "J03WR9": DflMatch(
        "DFL-COM-000002",
        "7a8eb84881a1631da510982c01c90a3afd1b2e23",
        "70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269",
    ),
}

SKC_COMMIT = "1b99d401ddc82e366c5c20d6860878668fd5858f"  # last commit holding the 2019/20 data
SKC_URL = (
    "https://raw.githubusercontent.com/SkillCorner/opendata/{commit}/data/matches/{match}/{name}"
)
SKC_FILES = ("match_data.json", "structured_data.json")
SKC_MATCHES = (2841, 3442)
SKC_FPS = 10
SKC_TARGET_FPS = 25


def sha256(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(path: Path) -> str:
    """Git object id of a file's content, as a blob."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(url: str, target: Path) -> Path:
    """Stream to a temporary file and rename, so an interrupted download never looks complete."""
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "pitch-dynamics"})
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle,
    ):
        shutil.copyfileobj(response, handle, 1 << 20)
        temporary = Path(handle.name)
    temporary.replace(target)
    return target


def dfl_paths(match: str, directory: Path | None = None) -> tuple[Path, Path]:
    """Local (metadata, positions) paths of a DFL match."""
    directory = directory or CACHE / "download" / "dfl"
    competition = DFL_MATCHES[match].competition
    return (
        directory / DFL_META_FILE.format(competition=competition, match=match),
        directory / DFL_POSITIONS_FILE.format(competition=competition, match=match),
    )


def fetch_dfl(match: str, directory: Path | None = None) -> None:
    """Download the two files of a DFL match from the mirror."""
    for path in dfl_paths(match, directory):
        download(DFL_MIRROR.format(revision=DFL_REVISION, filename=path.name), path)


def read_dfl_meta(path: Path) -> tuple[tuple[float, float], dict[str, int]]:
    """Pitch size in metres and a club id -> 0 (home) / 1 (away) map from the metadata file."""
    root = ET.parse(path).getroot()
    environment = root.find(".//Environment")
    if environment is None:
        raise ValueError(f"{path}: no pitch size")
    teams: dict[str, int] = {}
    for team in root.iter("Team"):
        role, team_id = team.get("Role"), team.get("TeamId")
        if role in ("home", "guest") and team_id is not None:
            teams[team_id] = 0 if role == "home" else 1
    if sorted(teams.values()) != [0, 1]:
        raise ValueError(f"{path}: expected one home and one guest team")
    return (float(environment.get("PitchX", "")), float(environment.get("PitchY", ""))), teams


def _number(value: str | None) -> float:
    return np.nan if value is None or value == "" else float(value)


def read_dfl_positions(
    path: Path, team_of: dict[str, int], pitch: tuple[float, float]
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int8],
    NDArray[np.str_],
    list[tuple[int, int, int]],
]:
    """Stream the positions XML into arrays, one contiguous block per game section.

    Coordinates are metres with the origin at the centre spot; they are normalised to [0, 1] on
    both axes. Officials and unknown objects are skipped, and frames inside a section that an
    object does not report stay NaN.
    Returns players (n, slots, 3: x, y, NaN speed), ball (n, 3: x, y, NaN), teams (slots,),
    labels (n,) and half-open intervals [(section, start, stop)].
    """
    length, width = pitch
    ball: dict[int, dict[int, tuple[float, float, str | None, str | None]]] = defaultdict(dict)
    people: dict[int, dict[str, dict[int, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    slot_team: dict[str, int] = {}
    current: tuple[str, int, str | None] | None = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            section = DFL_SECTIONS.get(element.get("GameSection", ""))
            team = element.get("TeamId")
            person = element.get("PersonId")
            if section is None:
                current = None
            elif team is not None and team.upper() == "BALL":
                current = ("ball", section, None)
            elif team in team_of and person is not None:
                slot_team[person] = team_of[team]
                current = ("person", section, person)
            else:
                current = None  # referees and anything not on either team
        elif event == "end" and element.tag == "Frame":
            if current is not None:
                frame = int(element.get("N", ""))
                x, y = _number(element.get("X")), _number(element.get("Y"))
                if current[0] == "ball":
                    ball[current[1]][frame] = (
                        x,
                        y,
                        element.get("BallStatus"),
                        element.get("BallPossession"),
                    )
                else:
                    people[current[1]][str(current[2])][frame] = (x, y)
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            current = None
            element.clear()
    if not ball:
        raise ValueError(f"{path}: no ball frames")
    slots = sorted(slot_team, key=lambda person: (slot_team[person], person))
    index = {person: i for i, person in enumerate(slots)}
    sections: list[tuple[int, int, int, int]] = []
    offset = 0
    for section in sorted(ball):
        frames = ball[section]
        first, last = min(frames), max(frames)
        sections.append((section, first, offset, offset + last - first + 1))
        offset += last - first + 1
    players = np.full((offset, len(slots), 3), np.nan, dtype=np.float32)
    balls = np.full((offset, 3), np.nan, dtype=np.float32)
    labels = np.full(offset, "X", dtype="<U1")
    for section, first, start, stop in sections:
        for frame, (x, y, status, owner) in ball[section].items():
            row = start + frame - first
            balls[row, :2] = x / length + 0.5, y / width + 0.5
            if status == "0":
                labels[row] = "D"
            elif status == "1" and owner in ("1", "2"):
                labels[row] = "H" if owner == "1" else "A"
        for person, person_frames in people[section].items():
            for frame, (x, y) in person_frames.items():
                row = start + frame - first
                if start <= row < stop:
                    players[row, index[person], :2] = x / length + 0.5, y / width + 0.5
    teams = np.array([slot_team[person] for person in slots], dtype=np.int8)
    intervals = [(section, start, stop) for section, _, start, stop in sections]
    return players, balls, teams, labels, intervals


def parse_dfl(match: str, directory: Path | None = None) -> Raw:
    """Parse a downloaded DFL match into the arrays stored in the cache, checking its digests."""
    meta_path, positions_path = dfl_paths(match, directory)
    expected = DFL_MATCHES[match]
    positions_digest = sha256(positions_path)
    if git_blob_id(meta_path) != expected.meta_git_blob or (
        positions_digest != expected.positions_sha256
    ):
        raise ValueError(f"{match}: downloaded files do not match the pinned mirror revision")
    pitch, team_of = read_dfl_meta(meta_path)
    players, ball, teams, labels, intervals = read_dfl_positions(positions_path, team_of, pitch)
    provenance = {
        "source": "DFL/IDSSE (Bassek et al. 2025), CC-BY 4.0",
        "revision": DFL_REVISION,
        "files": {meta_path.name: expected.meta_git_blob, positions_path.name: positions_digest},
    }
    return {
        "players": players,
        "ball": ball,
        "teams": teams,
        "labels": labels,
        "intervals": np.asarray(intervals, dtype=np.int64),
        "pitch": np.asarray(pitch),
        "fps": DFL_FPS,
        "provenance": json.dumps(provenance),
    }


def skc_paths(match: int) -> list[Path]:
    """Local paths of the files of a SkillCorner match."""
    return [CACHE / "download" / "skillcorner" / str(match) / name for name in SKC_FILES]


def fetch_skillcorner(match: int) -> None:
    """Download the files of a SkillCorner match from the commit-pinned repository."""
    for path in skc_paths(match):
        download(SKC_URL.format(commit=SKC_COMMIT, match=match, name=path.name), path)


def skc_meta(
    meta: dict[str, Any],
) -> tuple[tuple[float, float], list[dict[str, Any]], NDArray[np.int8]]:
    """Pitch size, players (home first) and team per slot from the match file."""
    home, away = meta["home_team"]["id"], meta["away_team"]["id"]
    side = {home: 0, away: 1}
    players = sorted(
        (p for p in meta["players"] if p["team_id"] in side),
        key=lambda p: (side[p["team_id"]], p["trackable_object"]),
    )
    teams = np.array([side[p["team_id"]] for p in players], dtype=np.int8)
    return (float(meta["pitch_length"]), float(meta["pitch_width"])), players, teams


def resample(values: NDArray[Any], factor: float) -> NDArray[Any]:
    """Linear resampling along axis 0 by `factor` (> 1). A NaN at either neighbour stays NaN."""
    n = len(values)
    if n == 0:
        return values
    t = np.arange(int(np.floor((n - 1) * factor)) + 1) / factor
    low = np.minimum(np.floor(t).astype(np.int64), n - 1)
    high = np.minimum(low + 1, n - 1)
    weight = (t - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return ((1 - weight) * values[low] + weight * values[high]).astype(values.dtype)


def skc_assemble(
    sections: list[tuple[int, NDArray[Any], NDArray[Any], NDArray[Any]]],
    pitch: tuple[float, float],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.str_], list[tuple[int, int, int]]]:
    """Sections at 10 fps -> 25 fps arrays in the DFL layout.

    `sections` is a list of (period, players (n, slots, 2) metres, ball (n, 2) metres,
    labels (n,)). Coordinates are centred metres, normalised to [0, 1] like DFL. Labels take the
    nearest 10 fps frame.
    """
    length, width = pitch
    scale = np.array([length, width], dtype=np.float64)
    factor = SKC_TARGET_FPS / SKC_FPS
    blocks: list[tuple[int, int, int]] = []
    offset = 0
    out_players, out_ball, out_labels = [], [], []
    for period, players, ball, labels in sections:
        p = resample(players.astype(np.float64), factor) / scale + 0.5
        b = resample(ball.astype(np.float64), factor) / scale + 0.5
        nearest = np.minimum(np.rint(np.arange(len(b)) / factor).astype(np.int64), len(labels) - 1)
        out_players.append(np.concatenate([p, np.full((*p.shape[:2], 1), np.nan)], axis=2))
        out_ball.append(np.column_stack([b, np.full(len(b), np.nan)]))
        out_labels.append(labels[nearest])
        blocks.append((period, offset, offset + len(b)))
        offset += len(b)
    return (
        np.concatenate(out_players).astype(np.float32),
        np.concatenate(out_ball).astype(np.float32),
        np.concatenate(out_labels).astype("<U1"),
        blocks,
    )


def _owner(possession: dict[str, Any] | None) -> str:
    group = (possession or {}).get("group")
    return "H" if group == "home team" else "A" if group == "away team" else "X"


def read_skc_tracking(
    meta: dict[str, Any], frames: list[dict[str, Any]]
) -> tuple[
    tuple[float, float],
    NDArray[np.int8],
    list[tuple[int, NDArray[Any], NDArray[Any], NDArray[Any]]],
]:
    """Per-period players, ball and possession labels at 10 fps from the match and frame files."""
    pitch, players, teams = skc_meta(meta)
    slot = {p["trackable_object"]: i for i, p in enumerate(players)}
    ball_id = meta["ball"]["trackable_object"]
    by_period: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        if frame.get("period") in (1, 2):
            by_period[frame["period"]].append(frame)
    sections = []
    for period in sorted(by_period):
        rows = sorted(by_period[period], key=lambda f: f["frame"])
        first = rows[0]["frame"]
        n = rows[-1]["frame"] - first + 1
        pl = np.full((n, len(players), 2), np.nan)
        bl = np.full((n, 2), np.nan)
        lb = np.full(n, "X", dtype="<U1")
        for frame in rows:
            i = frame["frame"] - first
            lb[i] = _owner(frame.get("possession"))
            for obj in frame.get("data") or []:
                key = obj.get("trackable_object")
                if key == ball_id:
                    bl[i] = obj["x"], obj["y"]
                elif key in slot:
                    pl[i, slot[key]] = obj["x"], obj["y"]
        sections.append((period, pl, bl, lb))
    return pitch, teams, sections


def parse_skillcorner(match: int) -> Raw:
    """Parse a downloaded SkillCorner match into the arrays stored in the cache."""
    meta_path, frames_path = skc_paths(match)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    pitch, teams, sections = read_skc_tracking(meta, frames)
    players, ball, labels, intervals = skc_assemble(sections, pitch)
    provenance = {
        "source": "SkillCorner open data (MIT licence; credit SkillCorner)",
        "commit": SKC_COMMIT,
        "files": {path.name: sha256(path) for path in (meta_path, frames_path)},
    }
    return {
        "players": players,
        "ball": ball,
        "teams": teams,
        "labels": labels,
        "intervals": np.asarray(intervals, dtype=np.int64),
        "pitch": np.asarray(pitch),
        "fps": SKC_TARGET_FPS,
        "provenance": json.dumps(provenance),
    }
