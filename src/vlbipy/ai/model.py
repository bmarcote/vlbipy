"""The small model: a forest of shallow decision trees, in numpy only, stored as JSON.

Why trees: the decisions to learn are threshold decisions on a handful of dimensionless features (a scatter
ratio above ~2.5, a slope near 0 or 1, a level off by a factor) - exactly what a shallow tree represents, and
what the expert rules of :mod:`~vlbipy.ai.knowledge` already are. A forest of 80 depth-6 trees learns them from
labelled rows, generalises between them, needs no GPU and no dependency beyond numpy, and serialises to a JSON
file of ~200 kB that ships inside the package (``vlbipy/data/ai/model.json``). Inference is about a millisecond
per row in pure Python: the model can run on every antenna, subband and scan of an experiment on a laptop.

Missing features (a Tsys series without a raw-data check, an antenna on one source only) are handled the way
the data present them: a split sends missing values to the side that holds most of the training rows.

Training data: the labelled feature rows of :mod:`~vlbipy.ai.features`. The rule labels are the weak teacher;
human corrections (``features.annotate``) override them row by row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .features import FEATURES, matrix, read_rows

#: Where the trained model and the annotated training sets ship.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ai"
MODEL_FILE = DATA_DIR / "model.json"


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    return 0.0 if total == 0 else 1.0 - float(np.sum((counts / total) ** 2))


def _grow(X: np.ndarray, y: np.ndarray, nclass: int, depth: int, max_depth: int, min_leaf: int,
          features: np.ndarray, rng: np.random.Generator) -> dict:
    counts = np.bincount(y, minlength=nclass).astype(float)
    if depth >= max_depth or y.size < 2 * min_leaf or np.count_nonzero(counts) <= 1:
        return {"leaf": (counts / counts.sum()).round(4).tolist()}
    best = None
    parent = _gini(counts)
    for f in features:
        col = X[:, f]
        known = np.isfinite(col)
        if known.sum() < 2 * min_leaf:
            continue
        values = np.unique(col[known])
        if values.size < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        if thresholds.size > 32:
            thresholds = np.quantile(col[known], np.linspace(0.03, 0.97, 32))
        for t in thresholds:
            left = known & (col <= t)
            right = known & (col > t)
            missing_left = left.sum() >= right.sum()
            if missing_left:
                left = left | ~known
            else:
                right = right | ~known
            nl, nr = left.sum(), right.sum()
            if nl < min_leaf or nr < min_leaf:
                continue
            gain = parent - (nl * _gini(np.bincount(y[left], minlength=nclass)) +
                             nr * _gini(np.bincount(y[right], minlength=nclass))) / y.size
            if best is None or gain > best[0]:
                best = (gain, int(f), float(t), bool(missing_left), left, right)
    if best is None or best[0] <= 1e-9:
        return {"leaf": (counts / counts.sum()).round(4).tolist()}
    _, f, t, missing_left, left, right = best
    return {"feature": f, "threshold": t, "missing_left": missing_left,
            "left": _grow(X[left], y[left], nclass, depth + 1, max_depth, min_leaf, features, rng),
            "right": _grow(X[right], y[right], nclass, depth + 1, max_depth, min_leaf, features, rng)}


def _predict_tree(node: dict, x: np.ndarray) -> np.ndarray:
    while "leaf" not in node:
        value = x[node["feature"]]
        go_left = node["missing_left"] if not np.isfinite(value) else value <= node["threshold"]
        node = node["left"] if go_left else node["right"]
    return np.asarray(node["leaf"])


class Forest:
    """A bagged forest of shallow classification trees for one stage's feature set."""

    def __init__(self, stage: str, classes: list[str], trees: list[dict], feature_names: tuple) -> None:
        self.stage = stage
        self.classes = list(classes)
        self.trees = trees
        self.feature_names = tuple(feature_names)

    @classmethod
    def fit(cls, X: np.ndarray, labels: list[str], stage: str, *, n_trees: int = 80, max_depth: int = 6,
            min_leaf: int = 2, seed: int = 0, class_weight: bool = True,
            sample_weight: Optional[np.ndarray] = None) -> "Forest":
        """Train on ``X`` (n, nfeat; nan = missing) and string ``labels``.

        ``class_weight`` resamples each bootstrap so that every class is drawn equally often: the findings
        that matter (a noisy hand, a dead subband) are rare against the many healthy series. ``sample_weight``
        multiplies a row's chance of being drawn (a reviewed row counts more than a rule's guess).
        """
        classes = sorted(set(labels))
        y = np.array([classes.index(label) for label in labels])
        rng = np.random.default_rng(seed)
        nfeat = X.shape[1]
        trees = []
        p = 1.0 / np.bincount(y, minlength=len(classes))[y] if class_weight else np.ones(y.size)
        if sample_weight is not None:
            p = p * np.asarray(sample_weight, dtype=float)
        p = p / p.sum()
        for _ in range(n_trees):
            idx = rng.choice(y.size, size=y.size, replace=True, p=p)
            feats = rng.choice(nfeat, size=max(2, int(np.ceil(np.sqrt(nfeat) * 1.5))), replace=False)
            trees.append(_grow(X[idx], y[idx], len(classes), 0, max_depth, min_leaf, np.sort(feats), rng))
        return cls(stage, classes, trees, FEATURES.get(stage, tuple(f"f{i}" for i in range(nfeat))))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, (n, nclass)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros((X.shape[0], len(self.classes)))
        for tree in self.trees:
            for i, x in enumerate(X):
                out[i] += _predict_tree(tree, x)
        return out / max(len(self.trees), 1)

    def predict(self, X: np.ndarray) -> list[str]:
        """Most probable class of every row."""
        return [self.classes[k] for k in np.argmax(self.predict_proba(X), axis=1)]

    def importance(self) -> dict[str, int]:
        """How often each feature is used in a split (a rough importance)."""
        counts = np.zeros(len(self.feature_names), dtype=int)

        def walk(node):
            if "leaf" in node:
                return
            counts[node["feature"]] += 1
            walk(node["left"])
            walk(node["right"])
        for tree in self.trees:
            walk(tree)
        return dict(zip(self.feature_names, counts.tolist()))

    def to_dict(self) -> dict:
        return {"stage": self.stage, "classes": self.classes, "features": list(self.feature_names),
                "trees": self.trees}

    @classmethod
    def from_dict(cls, data: dict) -> "Forest":
        return cls(data["stage"], data["classes"], data["trees"], tuple(data["features"]))


#: How much more often a human-reviewed row is drawn into a bootstrap than a row labelled by the rules only.
REVIEW_WEIGHT = 3.0


def _stratified_folds(labels: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Fold index of every row, each class spread evenly over the ``k`` folds."""
    rng = np.random.default_rng(seed)
    folds = np.zeros(labels.size, dtype=int)
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        folds[rng.permutation(rows)] = np.arange(rows.size) % k
    return folds


def _scores(truth: np.ndarray, predicted: np.ndarray) -> dict:
    """Accuracy, balanced accuracy (mean per-class recall) and per-class recall over the rows predicted."""
    done = predicted != ""
    if not done.any():
        return {"accuracy": None, "balanced": None, "recall": {}}
    t, p = truth[done], predicted[done]
    recall = {str(c): float(np.mean(p[t == c] == c)) for c in np.unique(t)}
    return {"accuracy": float(np.mean(t == p)), "balanced": float(np.mean(list(recall.values()))), "recall": recall}


def training_files(extra: Optional[list] = None) -> list[Path]:
    """The annotated feature sets shipped in ``vlbipy/data/ai/`` plus any ``extra`` JSONL files."""
    files = sorted(DATA_DIR.glob("*.features.jsonl")) if DATA_DIR.is_dir() else []
    return files + [Path(p) for p in (extra or [])]


def train(files: Optional[list] = None, *, stages: tuple = ("tsys", "prediagnostic", "calibrated"),
          out: Optional[Path] = None, **kwargs) -> dict:
    """Train one forest per stage on the rows of ``files`` and write them to ``out`` (default: the shipped model).

    Returns
    -------
    dict
        Per stage: the number of rows, the class counts, the scores (accuracy, balanced accuracy, per-class
        recall) of a stratified 5-fold and of a leave-one-experiment-out validation, and the feature use.
    """
    rows = []
    for path in files or training_files():
        rows += read_rows(path)
    model, summary = {"version": 1, "forests": {}}, {}
    for stage in stages:
        X, labels, chosen = matrix(rows, stage)
        if len(set(labels)) < 2:
            continue
        weights = np.array([REVIEW_WEIGHT if r.get("reviewed_by") else 1.0 for r in chosen])
        forest = Forest.fit(X, labels, stage, sample_weight=weights, **kwargs)
        model["forests"][stage] = forest.to_dict()
        experiments = sorted({r["experiment"] for r in chosen})
        truth = np.array(labels)
        # Leave one experiment out: what the model does on an experiment it has never seen (pessimistic while a
        # class lives in one experiment only).
        loeo = np.full(truth.size, "", dtype=object)
        for held in experiments if len(experiments) > 1 else []:
            test = np.array([r["experiment"] == held for r in chosen])
            if len(set(truth[~test])) < 2:
                continue
            sub = Forest.fit(X[~test], list(truth[~test]), stage, sample_weight=weights[~test], **kwargs)
            loeo[test] = sub.predict(X[test])
        # Stratified 5-fold over every row: how well the features separate the classes at all.
        kfold = np.full(truth.size, "", dtype=object)
        folds = _stratified_folds(truth, 5, seed=1)
        for k in range(5):
            test = folds == k
            if test.any() and len(set(truth[~test])) >= 2:
                sub = Forest.fit(X[~test], list(truth[~test]), stage, sample_weight=weights[~test], **kwargs)
                kfold[test] = sub.predict(X[test])
        summary[stage] = {"rows": len(labels), "classes": {c: labels.count(c) for c in sorted(set(labels))},
                          "experiments": experiments, "feature_use": forest.importance(),
                          "loeo": _scores(truth, loeo), "kfold": _scores(truth, kfold),
                          "accuracy": _scores(truth, kfold)["accuracy"], "validation": "stratified 5-fold"}
    model["summary"] = summary
    out = Path(out or MODEL_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, separators=(",", ":")))
    return summary


def load(path: Optional[Path] = None) -> dict[str, Forest]:
    """The trained forests (``{}`` when no model has been trained)."""
    path = Path(path or MODEL_FILE)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {stage: Forest.from_dict(f) for stage, f in data.get("forests", {}).items()}


def predict_rows(rows: list[dict], forests: Optional[dict] = None) -> list[dict]:
    """Add ``model_label`` and ``model_p`` to every row the model has a forest for."""
    forests = forests if forests is not None else load()
    for stage, forest in forests.items():
        X, _, chosen = matrix(rows, stage)
        if not chosen:
            continue
        proba = forest.predict_proba(X)
        for row, p in zip(chosen, proba):
            k = int(np.argmax(p))
            row["model_label"], row["model_p"] = forest.classes[k], float(p[k])
    return rows
