"""Ensemble of two small MLPs on the design matrix, decoded with Viterbi."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from pitch_dynamics.decoding import CLASSES, change_costs, decode
from pitch_dynamics.design import N_COLUMNS, TRAIN_STRIDE, augmented, design_matrix
from pitch_dynamics.matches import Match

DESCRIPTION = "Ensemble of 2 MLPs (seeds s and s+1000), mean probability, Viterbi decoding"

Array = NDArray[Any]

HIDDEN, EPOCHS, BATCH, LR, L2 = 64, 12, 1024, 3e-3, 1e-4
ENSEMBLE = 2
PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class Member:
    """One MLP: standardisation, one tanh hidden layer and a softmax output."""

    mean: Array
    scale: Array
    w1: Array
    b1: Array
    w2: Array
    b2: Array

    def log_proba(self, x: Array) -> Array:
        logits = np.tanh(((x - self.mean) / self.scale) @ self.w1 + self.b1) @ self.w2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


@dataclass(frozen=True)
class Model:
    """MLPs that differ only in their seed, plus the Viterbi switch costs."""

    members: tuple[Member, ...]
    switch_cost: float
    dead_switch_cost: float


def fit_member(x: Array, y: Array, seed: int) -> Member:
    """Class-balanced softmax MLP, minibatch Adam, fixed seed."""
    rng = np.random.default_rng(seed)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    onehot = (y[:, None] == np.array(CLASSES)).astype(np.float64)
    weight = onehot @ (len(y) / (3 * np.maximum(onehot.sum(axis=0), 1)))
    params = [
        rng.normal(0, 1 / np.sqrt(z.shape[1]), (z.shape[1], HIDDEN)),
        np.zeros(HIDDEN),
        rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, 3)),
        np.zeros(3),
    ]
    moments = [[np.zeros_like(p), np.zeros_like(p)] for p in params]
    step = 0
    for _ in range(EPOCHS):
        order = rng.permutation(len(z))
        for start in range(0, len(z), BATCH):
            rows = order[start : start + BATCH]
            w1, b1, w2, b2 = params
            hidden = np.tanh(z[rows] @ w1 + b1)
            logits = hidden @ w2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            proba = np.exp(logits)
            proba /= proba.sum(axis=1, keepdims=True)
            delta = weight[rows, None] * (proba - onehot[rows]) / len(rows)
            back = (delta @ w2.T) * (1 - hidden**2)
            grads = [
                z[rows].T @ back + L2 * w1,
                back.sum(axis=0),
                hidden.T @ delta + L2 * w2,
                delta.sum(axis=0),
            ]
            step += 1
            for p, g, m in zip(params, grads, moments, strict=True):
                m[0] = 0.9 * m[0] + 0.1 * g
                m[1] = 0.999 * m[1] + 0.001 * g * g
                p -= LR * (m[0] / (1 - 0.9**step)) / (np.sqrt(m[1] / (1 - 0.999**step)) + 1e-8)
    return Member(mean, scale, *params)


def training_rows(copies: list[Match], stride: int) -> tuple[Array, Array, tuple[float, float]]:
    """Every `stride`-th labelled design row of the copies, its labels and the switch costs."""
    xs: list[Array] = []
    labels: list[Array] = []
    for match in copies:
        # The X row keeps a fake transition between two matches out of the change costs.
        xs += [design_matrix(match), np.zeros((1, N_COLUMNS), dtype=np.float32)]
        labels += [match.labels, np.array(["X"])]
    x, y = np.vstack(xs), np.concatenate(labels)
    mask = np.isin(y, CLASSES)
    sample = mask & (np.arange(len(y)) % stride == 0)
    return x[sample], y[sample], change_costs(y, mask)


def average_log_proba(scores: list[Array]) -> Array:
    """Log of the mean probability of several log-probability arrays, floored."""
    proba = np.mean([np.exp(score) for score in scores], axis=0)
    return np.log(np.maximum(proba, PROBABILITY_FLOOR))


def fit(matches: list[Match], seed: int = 0) -> Model:
    """Train on labelled matches, their degradation draws and their home/away swaps."""
    with threadpool_limits(1):
        x, y, costs = training_rows(augmented(matches), TRAIN_STRIDE)
        members = tuple(fit_member(x, y, seed + 1000 * k) for k in range(ENSEMBLE))
    return Model(members, *costs)


def log_proba(model: Model, match: Match) -> Array:
    """Mean-probability log scores (frames, 3) in class order H, A, D."""
    x = design_matrix(match)
    with threadpool_limits(1):
        return average_log_proba([member.log_proba(x) for member in model.members])


def predict(model: Model, match: Match) -> NDArray[np.str_]:
    """Labels for every frame of the match; never reads `match.labels`."""
    return decode(
        log_proba(model, match),
        match.intervals,
        match.forced_dead,
        model.switch_cost,
        model.dead_switch_cost,
    )
