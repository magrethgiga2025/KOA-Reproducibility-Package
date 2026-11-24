"""One-at-a-time parameter sensitivity of KOA (KOA-II in the code).

Same protocol as run_experiments.py: identical problems, shared initial
population per (problem, seed), N_GEN generations, and the same metrics.
Only one KOA parameter is changed at a time; all others stay at default.
"""
import argparse, time, traceback, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

from config import PROBLEMS, POP_SIZE, N_GEN, build_problem
from algorithms.koa2 import KOA2
from algorithms.factory import _exact_ref_dirs
from utils.metrics import compute_all_metrics, compute_reference_point
from utils.common_init import shared_initial_population

OUT = Path(__file__).resolve().parent / "results" / "sensitivity_koa2.csv"
SWEEPS = {                       # default value first
    "H":             [5, 3, 10, 20],
    "archive_ratio": [1.0, 0.5, 1.5, 2.0],
    "beta":          [0.1, 0.05, 0.2, 0.4],
    "q_lr":          [0.3, 0.1, 0.5, 0.7],
}
SENS_PROBLEMS = ["ZDT3", "DTLZ2", "DTLZ7", "WFG4", "LSMOP9", "MaF4"]


def configs():
    yield ("default", "default", {})
    for p, vals in SWEEPS.items():
        for v in vals[1:]:
            yield (p, v, {p: v})


def run_one(task):
    pname, n_obj, kind, kwargs, param, value, params, seed = task
    from pymoo.optimize import minimize
    t0 = time.time(); row = dict(problem=pname, param=param, value=value, seed=seed,
                                 HV=np.nan, IGD=np.nan, n_eval=0, status="error", error="")
    try:
        problem = build_problem(kind, kwargs)
        pf = problem.pareto_front()
        ref_point = compute_reference_point(pf)
        init_X = shared_initial_population(problem, POP_SIZE, seed)
        algo = KOA2(_exact_ref_dirs(n_obj, POP_SIZE), pop_size=POP_SIZE,
                    sampling=init_X, seed=seed, **params)
        res = minimize(problem, algo, ("n_gen", N_GEN), seed=seed, verbose=False)
        m = compute_all_metrics(res.F, pf, ref_point)
        row.update(HV=m["HV"], IGD=m["IGD"], n_eval=int(res.algorithm.evaluator.n_eval), status="ok")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    row["wall_seconds"] = round(time.time() - t0, 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=30)
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        d0 = pd.read_csv(OUT)
        d0 = d0[d0.status == "ok"]
        done = set(zip(d0.problem, d0.param, d0.value.astype(str), d0.seed))
    probs = [p for p in PROBLEMS if p[0] in SENS_PROBLEMS]
    tasks = [(pn, no, k, kw, par, val, prm, s)
             for (pn, no, k, kw) in probs for (par, val, prm) in configs()
             for s in range(1, a.n_seeds + 1) if (pn, par, str(val), s) not in done]
    print(f"Sensitivity: {len(tasks)} runs to do | problems={[p[0] for p in probs]}", flush=True)
    rows, t0 = [], time.time()
    with mp.Pool(a.n_jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, tasks), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(tasks):
                df = pd.DataFrame(rows)
                old = pd.read_csv(OUT) if OUT.exists() else None
                (pd.concat([old, df]) if old is not None else df).to_csv(OUT, index=False)
                rows = []
                print(f"[{i}/{len(tasks)}] {(time.time()-t0)/60:.1f} min  last={r['problem']}/{r['param']}={r['value']}/s{r['seed']} {r['status']}", flush=True)
    print("Done. Wrote", OUT)


if __name__ == "__main__":
    main()
