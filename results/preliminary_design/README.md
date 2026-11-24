# Preliminary design

The files in this folder belong to an earlier version of the algorithm, labelled `KOA` in the code
(`code/algorithms/koa_final_v3.py`). It contained additional mechanisms: an indicator-driven schedule
for choosing between exploration and exploitation, a geometry-adaptive niche fraction, a
stagnation-triggered archive refresh and Lévy-flight steps.

Its controlled ablation showed that only archive-based reproduction made a clear contribution, which
led to the simpler algorithm reported in the manuscript (label `KOA-II`). These results support the
paragraph "Lessons for algorithm design" in the Discussion and are not part of any table or figure.

| File | Content |
|---|---|
| `raw_runs_preliminary_KOA.csv` | the preliminary design on the 16 benchmark problems, 30 seeds, same protocol as the main comparison |
| `ablation_final_raw.csv`, `ablation_holm_tests.csv`, `ablation_ranking.csv`, `ablation_wtl.csv` | ablation of the preliminary design |
| `factorial_ablation_raw.csv`, `factorial_ranking.csv` | factorial ablation of its three main mechanisms |
| `capacity_sensitivity.csv`, `theta_sensitivity.csv`, `weight_sensitivity.csv`, `full_param_sensitivity.csv`, `delta_diagnostic.csv` | parameter studies and diagnostics of the preliminary design |
