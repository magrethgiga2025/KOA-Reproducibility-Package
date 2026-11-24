"""
download_fs_datasets.py
=======================
Downloads the five UCI datasets used by run_feature_selection.py and saves them
as data_fs/<Name>.csv (numeric features, class label in the last column).

    pip install ucimlrepo
    python download_fs_datasets.py

Wine, WDBC and Digits are built into scikit-learn and need no download.
Source: UCI Machine Learning Repository (https://archive.ics.uci.edu), CC BY 4.0.
Cite each dataset's DOI (printed below) in the paper.

Some UCI datasets (e.g. Hill-Valley) cannot be fetched with ucimlrepo; for those the
script downloads the dataset's zip file from UCI directly. A failure on one dataset
does not stop the others. Already-saved files are skipped, so just run it again.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "data_fs"
OUT.mkdir(exist_ok=True)

# file name -> (UCI id, expected number of features, zip URL, file inside the zip, DOI)
DATASETS = {
    "Ionosphere": (52, 34, None, None, "10.24432/C5W01B"),
    "Sonar":      (151, 60, None, None, "10.24432/C5T01Q"),
    # Hill-Valley: noise-free version, training file (606 samples, header row X1..X100,class)
    "HillValley": (166, 100, "https://archive.ics.uci.edu/static/public/166/hill+valley.zip",
                   "Hill_Valley_without_noise_Training.data", "10.24432/C5JC8P"),
    "Musk1":      (74, 166, None, None, ""),
    "Arrhythmia": (5, 279, None, None, "10.24432/C5BS32"),
    "Vehicle":    (149, 18, None, None, ""),
    "SPECTF":     (96, 44, None, None, ""),
    "Libras":     (181, 90, None, None, ""),
    "Semeion":    (178, 256, None, None, ""),
    "Madelon":    (171, 500, None, None, ""),
}


def via_ucimlrepo(uid):
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=uid)
    X = ds.data.features.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    y = ds.data.targets.iloc[:, 0]
    return X, y, ds.metadata.name


def via_zip(url, member):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    names = zf.namelist()
    match = [n for n in names if n.endswith(member)]
    if not match:
        raise FileNotFoundError(f"{member} not in zip; zip contains: {names}")
    df = pd.read_csv(zf.open(match[0]))
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return df.iloc[:, :-1], df.iloc[:, -1].astype(int), member


def main():
    failed = []
    for name, (uid, n_feat, url, member, doi) in DATASETS.items():
        path = OUT / f"{name}.csv"
        if path.exists():
            print(f"skip  {name:<11} (already in {path.name})")
            continue
        try:
            try:
                X, y, src = via_ucimlrepo(uid)
            except Exception as e:
                if url is None:
                    raise
                print(f"      {name}: ucimlrepo unavailable ({type(e).__name__}); "
                      f"downloading the zip from UCI instead")
                X, y, src = via_zip(url, member)
            if X.shape[1] != n_feat:
                print(f"WARNING {name}: {X.shape[1]} numeric features, expected {n_feat}.")
            df = X.copy()
            df["class"] = y.values
            df.to_csv(path, index=False)
            print(f"saved {name:<11} {X.shape[0]} samples x {X.shape[1]} features, "
                  f"{y.nunique()} classes  <- {src}  DOI {doi}")
        except Exception as e:
            failed.append(name)
            print(f"FAIL  {name:<11} {type(e).__name__}: {e}")
    if failed:
        print(f"\nNot downloaded: {failed}. Download them manually from "
              "https://archive.ics.uci.edu (see REVISION_README.md), or leave them out with "
              "--datasets when running run_feature_selection.py.")


if __name__ == "__main__":
    main()
