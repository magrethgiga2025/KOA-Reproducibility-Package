"""
Bi-objective wrapper feature selection problem.

    f1 = |S| / D                      (fraction of features selected)       -> minimise
    f2 = inner-CV classification error of k-NN on the training split       -> minimise

Encoding: each decision variable x_j in [0, 1]; feature j is selected iff x_j > 0.5.
An empty selection is repaired by selecting the single feature with the largest x_j.

The data are split once per run (stratified, seed = run seed), so run k of every
algorithm sees exactly the same train/test split -> results are paired across algorithms.
The scaler is fitted on the training split only (no test leakage).
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_fs")

# ---------------------------------------------------------------- datasets ----
_BUILTIN = {
    "WDBC": load_breast_cancer,   # 30 features, 569 samples, 2 classes
    "Wine": load_wine,            # 13 features, 178 samples, 3 classes
    "Digits": load_digits,        # 64 features, 1797 samples, 10 classes
}


def load_dataset(name):
    """Return (X, y) for a built-in dataset or for data/<name>.csv.

    CSV format: one row per sample, numeric feature columns, class label in the
    LAST column, optional header row (detected automatically).
    """
    if name in _BUILTIN:
        d = _BUILTIN[name]()
        return d.data.astype(float), d.target
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{name}' is not built in and {path} does not exist. "
            "Put the CSV there (label in the last column).")
    df = pd.read_csv(path, header=None, low_memory=False)
    # header row present? (most of the first row is non-numeric)
    first = pd.to_numeric(df.iloc[0, :-1], errors="coerce")
    if first.isna().mean() > 0.5:
        df = pd.read_csv(path)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")            # drop text columns (e.g. Musk1 names)
    X = X.fillna(X.mean()).to_numpy(float)     # mean imputation for '?' entries
    y = pd.factorize(df.iloc[:, -1])[0]
    return X, y


def dataset_summary(names):
    rows = []
    for n in names:
        X, y = load_dataset(n)
        rows.append((n, X.shape[1], X.shape[0], len(np.unique(y))))
    return rows


# ----------------------------------------------------------------- problem ----
class FeatureSelectionProblem:
    """Plain numpy/sklearn problem; `evaluate(X)` maps (N, D) -> (N, 2)."""

    n_obj = 2

    def __init__(self, dataset, seed, test_size=0.3, k=5, inner_folds=5):
        X, y = load_dataset(dataset)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed)
        sc = MinMaxScaler().fit(Xtr)
        self.Xtr, self.Xte = sc.transform(Xtr), sc.transform(Xte)
        self.ytr, self.yte = ytr, yte
        self.n_var = X.shape[1]
        self.k = k
        # inner folds limited by the smallest class in the training split
        min_class = np.bincount(ytr).min()
        self.cv = StratifiedKFold(n_splits=max(2, min(inner_folds, min_class)),
                                  shuffle=True, random_state=seed)
        self.n_evals = 0          # every call counts, even when cached
        self._cache = {}

    # -- encoding -------------------------------------------------------------
    @staticmethod
    def decode(x):
        mask = np.asarray(x) > 0.5
        if not mask.any():
            mask = np.zeros_like(mask)
            mask[int(np.argmax(x))] = True
        return mask

    # -- objectives -----------------------------------------------------------
    def _train_error(self, mask):
        key = mask.tobytes()
        if key not in self._cache:
            clf = KNeighborsClassifier(n_neighbors=self.k)
            acc = cross_val_score(clf, self.Xtr[:, mask], self.ytr, cv=self.cv).mean()
            self._cache[key] = 1.0 - acc
        return self._cache[key]

    def evaluate(self, X):
        X = np.atleast_2d(X)
        F = np.empty((X.shape[0], 2))
        for i, x in enumerate(X):
            m = self.decode(x)
            F[i, 0] = m.sum() / self.n_var
            F[i, 1] = self._train_error(m)
        self.n_evals += X.shape[0]
        return F

    def test_error(self, mask):
        clf = KNeighborsClassifier(n_neighbors=self.k).fit(self.Xtr[:, mask], self.ytr)
        return 1.0 - clf.score(self.Xte[:, mask], self.yte)


# ------------------------------------------------------- pymoo wrapper --------
def as_pymoo(problem):
    """Wrap FeatureSelectionProblem as a pymoo Problem (imported lazily)."""
    from pymoo.core.problem import Problem

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=problem.n_var, n_obj=2, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = problem.evaluate(X)

    return _P()


# ------------------------------------------------------------- utilities ------
def nondominated(F):
    """Indices of non-dominated rows of F (minimisation)."""
    F = np.asarray(F)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            if dom.any():
                keep[i] = False
    return np.flatnonzero(keep)


def final_front(problem, X):
    """Unique non-dominated masks from a final population/archive, with train
    and test objectives. Returns dict of arrays."""
    masks = np.array([problem.decode(x) for x in np.atleast_2d(X)])
    masks = np.unique(masks, axis=0)
    F = np.array([[m.sum() / problem.n_var, problem._train_error(m)] for m in masks])
    nd = nondominated(F)
    masks, F = masks[nd], F[nd]
    test_err = np.array([problem.test_error(m) for m in masks])
    return {"masks": masks, "F_train": F,
            "F_test": np.column_stack([F[:, 0], test_err])}
