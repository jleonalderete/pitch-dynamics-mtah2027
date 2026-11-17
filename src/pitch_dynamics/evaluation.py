"""Score the model with the official metrics: leave-one-match-out on DEV and the holdouts."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pitch_dynamics import mlp, official
from pitch_dynamics.guard import refuse_agent
from pitch_dynamics.matches import DEV, HOLDOUT_DFL, HOLDOUT_LALIGA, TRAINING, Match, load_match

EVAL_SEEDS = (0, 1)
BOOTSTRAP_DRAWS = 1000
# DEV scoring trains on the other DEV matches; the LaLiga holdout trains on all seven DFL matches.
HOLDOUT_SETS = {"dfl": (DEV, HOLDOUT_DFL), "laliga": (TRAINING, HOLDOUT_LALIGA)}

Report = dict[str, Any]
OnMatch = Callable[[str, dict[str, float]], None]


def score(
    truth: NDArray[np.str_], prediction: NDArray[np.str_], bootstrap_n: int = 0
) -> dict[str, float]:
    """Official metrics of one match; the bootstrap fields appear only when `bootstrap_n` > 0."""
    metrics = official.metrics()
    result = metrics.evaluate(
        metrics.label_codes(truth), metrics.label_codes(prediction), bootstrap_n=bootstrap_n
    )
    scores = {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "d_f1": result["per_class"]["D"]["f1"],
        "n_gt_spells": result["n_gt_spells"],
        "n_pred_spells": result["n_pred_spells"],
    }
    if bootstrap_n:
        scores |= {
            key: result[key]
            for key in (
                "spell_bootstrap_low",
                "spell_bootstrap_high",
                "team_acc_gt_alive",
                "team_acc_both_alive",
            )
        }
    return scores


def _unlabelled(match_id: str) -> tuple[Match, NDArray[np.str_]]:
    """A match with its labels hidden from the model, and the labels kept for scoring."""
    match = load_match(match_id)
    truth = match.labels.copy()
    match.labels = np.full(len(truth), "X", dtype="<U1")
    return match, truth


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


def dev(seeds: Sequence[int] = EVAL_SEEDS, on_match: OnMatch | None = None) -> Report:
    """Leave-one-match-out over DEV, each held-out score averaged over training `seeds`."""
    matches: dict[str, dict[str, Any]] = {}
    for held in DEV:
        match, truth = _unlabelled(held)
        training = [load_match(m) for m in DEV if m != held]
        runs = [score(truth, mlp.predict(mlp.fit(training, seed=seed), match)) for seed in seeds]
        fold: dict[str, Any] = {key: _mean([run[key] for run in runs]) for key in runs[0]}
        fold["per_seed_spell_f1"] = [run["spell_f1"] for run in runs]
        matches[held] = fold
        if on_match:
            on_match(held, fold)
    return {
        "description": mlp.DESCRIPTION,
        "seeds": list(seeds),
        "matches": matches,
        "mean": {
            key: _mean([fold[key] for fold in matches.values()])
            for key in ("spell_f1", "macro_f1", "d_f1")
        },
    }


def holdout(which: str, on_match: OnMatch | None = None) -> Report:
    """Fit on the set's training matches and score its held-out matches; run once, by a person."""
    refuse_agent()
    trained_on, held_ids = HOLDOUT_SETS[which]
    held_out = {held: _unlabelled(held) for held in held_ids}
    model = mlp.fit([load_match(m) for m in trained_on])
    scores = {}
    for held, (match, truth) in held_out.items():
        scores[held] = score(truth, mlp.predict(model, match), BOOTSTRAP_DRAWS)
        if on_match:
            on_match(held, scores[held])
    return {
        "description": mlp.DESCRIPTION,
        "set": which,
        "trained_on": list(trained_on),
        "matches": scores,
        "mean_spell_f1": _mean([m["spell_f1"] for m in scores.values()]),
    }
