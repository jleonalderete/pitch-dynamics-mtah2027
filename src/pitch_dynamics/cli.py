"""Command line: download data, evaluate, score holdouts, align the broadcast, predict."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pitch_dynamics import broadcast, evaluation, matches, submission
from pitch_dynamics.guard import refuse_agent
from pitch_dynamics.matches import HOLDOUT_LALIGA, TRAINING
from pitch_dynamics.paths import ARTIFACTS


def _save(command: str, name: str, report: dict[str, Any]) -> Path:
    path = ARTIFACTS / command / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def _show(label: str, scores: dict[str, float]) -> None:
    print(
        f"{label:<22} spellF1 {scores['spell_f1']:.4f}  macroF1 {scores['macro_f1']:.4f}  "
        f"D-F1 {scores['d_f1']:.4f}",
        flush=True,
    )


def _data(args: argparse.Namespace) -> None:
    match_ids = (*TRAINING, *HOLDOUT_LALIGA) if args.laliga else TRAINING
    for match_id in match_ids:
        matches.fetch(match_id)
        info = matches.summary(match_id)
        labels = " ".join(f"{label} {count}" for label, count in info["labels"].items())
        print(f"{match_id}: {info['frames']} frames, {labels}", flush=True)


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluation.dev(args.seeds, on_match=_show)
    _show("mean", report["mean"])
    print(_save("evaluate", "report.json", report))


def _holdout(args: argparse.Namespace) -> None:
    refuse_agent()
    report = evaluation.holdout(args.set, on_match=_show)
    print(f"mean spellF1 {report['mean_spell_f1']:.4f}")
    print(_save("holdout", f"{args.set}.json", report))


def _align(args: argparse.Namespace) -> None:
    refuse_agent()
    out = args.out or ARTIFACTS / "align"
    report = broadcast.run(out, args.data_dir)
    for entry in report["offsets"]:
        print(
            f"offset {entry['offset_frames']:+d} frames: {entry['windows']} windows, "
            f"{entry['broadcast_frames']} broadcast frames"
        )
    print(f"broadcast frames matched: {report['broadcast_matched_share']:.1%}")
    print(f"median player residual: {report['player_residual_m_median']:.2f} m")
    disagreement = report["ball_disagreement_m"]
    if disagreement["frames"]:
        print(
            f"ball disagreement where both cameras see it: median {disagreement['median']:.2f} m, "
            f"p90 {disagreement['p90']:.2f} m over {disagreement['frames']} frames"
        )
    graded, before, filled = (
        report["graded_frames"],
        report["graded_ball_before"],
        report["graded_ball_filled"],
    )
    print(
        f"ball in graded frames: {before / graded:.1%} -> {(before + filled) / graded:.1%} "
        f"(+{filled} frames)"
    )
    print(out / "report.json")


def _predict(args: argparse.Namespace) -> None:
    refuse_agent()
    print(submission.predict(args.out or ARTIFACTS / "predict", args.data_dir))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitch-dynamics", description="Ball possession per frame from automatic tracking."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    bundle_help = "bundle root containing tactical/ and broadcast/ (default data/)"

    data = commands.add_parser("data", help="download and parse the external DFL matches")
    data.add_argument("--laliga", action="store_true", help="also fetch the LaLiga holdout")
    data.set_defaults(handler=_data)

    evaluate = commands.add_parser("evaluate", help="leave-one-match-out score on the DEV matches")
    evaluate.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(evaluation.EVAL_SEEDS),
        help="training seeds to average",
    )
    evaluate.set_defaults(handler=_evaluate)

    holdout = commands.add_parser("holdout", help="score the model once on a reserved set")
    holdout.add_argument("--set", required=True, choices=sorted(evaluation.HOLDOUT_SETS))
    holdout.set_defaults(handler=_holdout)

    align = commands.add_parser(
        "align", help="align the broadcast to the challenge match and report the ball it fills"
    )
    align.add_argument("--out", type=Path, help="default artifacts/align")
    align.add_argument("--data-dir", type=Path, help=bundle_help)
    align.set_defaults(handler=_align)

    predict = commands.add_parser("predict", help="label the challenge match: the submission")
    predict.add_argument("--out", type=Path, help="default artifacts/predict")
    predict.add_argument("--data-dir", type=Path, help=bundle_help)
    predict.set_defaults(handler=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the `pitch-dynamics` command."""
    args = _parser().parse_args(argv)
    try:
        args.handler(args)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
