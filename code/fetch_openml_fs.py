from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

OUT = Path(__file__).resolve().parent / "data_fs"
# match the header style of the files the existing script produced
first = open(OUT / "Sonar.csv").readline().split(",")[0]
has_header = not first.replace(".", "").replace("-", "").replace("e", "").strip().isdigit()

# name -> (OpenML data_id, expected number of features)
SETS = {"Libras": (299, 90), "Semeion": (1501, 256), "Madelon": (1485, 500)}

for name, (did, nfeat) in SETS.items():
    if (OUT / f"{name}.csv").exists():
        print(f"skip  {name}"); continue
    try:
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X = d.data.apply(pd.to_numeric, errors="coerce")
        y = pd.Series(pd.factorize(d.target)[0], name="class")
        df = pd.concat([X.reset_index(drop=True), y], axis=1)
        df.to_csv(OUT / f"{name}.csv", index=False, header=has_header)
        flag = "" if X.shape[1] == nfeat else f"  <-- WARNING expected {nfeat} features"
        print(f"saved {name}: {X.shape[0]} samples x {X.shape[1]} features, {y.nunique()} classes  (OpenML id {did}){flag}")
    except Exception as e:
        print(f"FAIL  {name}: {e}")
