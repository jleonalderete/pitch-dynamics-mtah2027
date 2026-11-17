"""Synthetic data shared by the tests: parsed matches, tracking fixtures and an ATD bundle."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pitch_dynamics import matches
from pitch_dynamics.matches import Tracking

DFL_META = """<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest><MatchInformation>
  <General MatchId="DFL-MAT-TEST01"/>
  <Environment PitchX="105.00" PitchY="68.00"/>
  <Teams>
    <Team TeamId="DFL-CLU-HOME" Role="home"><Players><Player PersonId="H1"/></Players></Team>
    <Team TeamId="DFL-CLU-AWAY" Role="guest"><Players><Player PersonId="A1"/></Players></Team>
  </Teams>
</MatchInformation></PutDataRequest>
"""


def dfl_positions_xml() -> str:
    """Two halves, one player per team, a referee that must be skipped and a ball with flags."""

    def frames(attributes: list[tuple[int, str]]) -> str:
        return "".join(f'<Frame N="{n}" {a}/>' for n, a in attributes)

    ball_first = [
        (10000, 'X="0.0" Y="0.0" Z="0.1" BallStatus="1" BallPossession="1"'),
        (10001, 'X="52.5" Y="34.0" Z="0.1" BallStatus="1" BallPossession="2"'),
        (10002, 'X="-52.5" Y="-34.0" Z="0.1" BallStatus="0" BallPossession="2"'),
    ]
    ball_second = [
        (100000, 'X="1.0" Y="2.0" BallStatus="1" BallPossession="0"'),
        (100001, 'X="1.0" Y="2.0" BallStatus="1" BallPossession="1"'),
    ]
    home = frames([(10000, 'X="-10.5" Y="6.8"'), (10002, 'X="0" Y="0"')])
    away = frames([(10001, 'X="10.5" Y="0"')])
    referee = frames([(10000, 'X="0" Y="0"')])
    away_second = frames([(100001, 'X="0" Y="34"')])
    return f"""<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest><Positions>
  <FrameSet GameSection="firstHalf" TeamId="DFL-CLU-HOME" PersonId="H1">{home}</FrameSet>
  <FrameSet GameSection="firstHalf" TeamId="DFL-CLU-AWAY" PersonId="A1">{away}</FrameSet>
  <FrameSet GameSection="firstHalf" TeamId="referee" PersonId="R1">{referee}</FrameSet>
  <FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">{frames(ball_first)}</FrameSet>
  <FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="B">{frames(ball_second)}</FrameSet>
  <FrameSet GameSection="secondHalf" TeamId="DFL-CLU-AWAY" PersonId="A1">{away_second}</FrameSet>
</Positions></PutDataRequest>
"""


def synthetic_raw(seed: int, n_half: int = 1200, segment: int = 120) -> dict[str, Any]:
    """Parsed-match arrays: two halves of alternating H, A, D stretches.

    The owning team's player carries the ball; in D the ball is still on a touchline and
    everyone stands still.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_half
    pattern = np.array(["H", "A", "D"])[(np.arange(n) // segment + seed) % 3]
    players = np.zeros((n, 22, 3), dtype=np.float32)
    players[:, :, 2] = np.nan
    base = rng.uniform(0.2, 0.8, (22, 2))
    drift = np.cumsum(rng.normal(0, 0.002, (n, 22, 2)), axis=0)
    players[:, :, :2] = np.clip(base + drift, 0.02, 0.98)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    for frame, label in enumerate(pattern):
        if label == "D":
            ball[frame, :2] = (0.5, 0.0)
            if frame:
                players[frame, :, :2] = players[frame - 1, :, :2]
        else:
            carrier = 0 if label == "H" else 11
            ball[frame, :2] = players[frame, carrier, :2] + 0.003
    return {
        "players": players,
        "ball": ball,
        "teams": np.r_[np.zeros(11), np.ones(11)].astype(np.int8),
        "labels": pattern,
        "intervals": np.array([(1, 0, n_half), (2, n_half, n)]),
        "pitch": np.array([105.0, 68.0]),
        "fps": 25,
        "provenance": json.dumps({"source": "synthetic"}),
    }


def use_cache(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Point the match cache at `root` for one test."""
    monkeypatch.setattr(matches, "CACHE", root)


def install_matches(root: Path, match_ids: Iterable[str], n_half: int = 1200) -> None:
    """Write one synthetic parsed match per id under the cache layout of `matches.raw_path`."""
    for index, match_id in enumerate(match_ids):
        path = root / "parsed" / f"{match_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **synthetic_raw(index, n_half))


def tracking(n: int = 120, slots: int = 22) -> Tracking:
    """Home on the left, away on the right, ball with home, everyone moving at 4 m/s."""
    players = np.zeros((n, slots, 3), dtype=np.float32)
    teams = (np.arange(slots) % 2).astype(np.int8)
    players[:, :, 0] = np.where(teams == 0, 0.2, 0.8)
    players[:, :, 1] = 0.5
    players[:, :, 2] = 4
    ball = np.tile([0.2, 0.5, 1.0], (n, 1)).astype(np.float32)
    return Tracking(players, ball, teams, ("Barcelona", "Athletic Club"))


def write_atd_bundle(root: Path, n_frames: int = 199319) -> Path:
    """Full-length ATD bundle under root/tactical: home holds the ball first, away after."""
    tactical = root / "tactical"
    tactical.mkdir(parents=True)
    stem = tactical / "BAR-ATH_tactical"
    Path(f"{stem}_atd.xml").write_text(
        '<root><Team id="h"><Name>Barcelona</Name></Team>'
        '<Team id="a"><Name>Athletic Club</Name></Team>'
        '<Player teamId="h"/><Player teamId="a"/></root>'
    )
    with Path(f"{stem}_atd.txt").open("w") as handle:
        for i in range(n_frames):
            x = 0.2 if i < 110808 else 0.8
            handle.write(f"{i + 1}:0.2,0.5,4;0.8,0.5,4:{x},0.5,1\n")
    return root
