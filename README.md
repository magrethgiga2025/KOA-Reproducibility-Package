# KOA Reproducibility Package

Source code, raw results and analysis scripts for the manuscript

> **KOA: Archive-Guided Differential Evolution with Survival-Based Adaptation for Multi-Objective Optimization and Feature Selection**
> M. D. Giga, Vinod Chandra S.S., F. Ojija, K. Gaurav
> (submitted to the *Journal of King Saud University – Computer and Information Sciences*)

Every table, statistic and figure in the results section of the manuscript can be regenerated from the raw files in `results/` with the scripts in `code/`, without rerunning any optimisation.

---

## 1. Naming used in the files

The code and the data files use internal labels that differ from the names in the manuscript. The data were **not** relabelled after the runs, so the labels below are exactly those written by the experiment scripts.

| Label in code and data | Name in the manuscript |
|---|---|
| `KOA-II` | **KOA** (the proposed algorithm, `code/algorithms/koa2.py`) |
| `KOA-II-Uniform`, `KOA-II-PopGuide`, `KOA-II-FixedAlloc`, `KOA-II-NoAdapt`, `KOA-II-DEonly`, `KOA-II-SBXonly` | the ablation variants KOA-Uniform, KOA-PopGuide, KOA-FixedAlloc, KOA-NoAdapt, KOA-DEonly, KOA-SBXonly |
| `AGE-MOEA2` | AGE-MOEA-II |
| `KOA` | the **preliminary design** mentioned in the Discussion ("Lessons for algorithm design"); it is not one of the compared algorithms. Its results are kept separately in `results/preliminary_design/`. |
| `q_lr`, `beta`, `H`, `archive_ratio` | the parameters λ, β, H and N_A/N |

## 2. Contents

```
README.md                  this file
MANIFEST.txt               SHA-256 checksum of every file
requirements.txt           Python dependencies (exact versions used)
code/
  config.py                benchmark problems, algorithms, budget (N = 100, 300 generations, 30 seeds)
  algorithms/koa2.py       KOA (label KOA-II) and the options used by its ablation variants
  algorithms/factory.py    builds KOA, its variants and the eight baselines with a shared initial population
  algorithms/koa_final_v3.py, koa_ablation_final.py, koa_v2.py, koa_star.py
                           the preliminary design (label KOA) and its ablation
  problems/                LSMOP and MaF implementations, reference directions
  utils/                   indicators, statistics, shared initialisation, many-objective problem builder
                           (including the DTLZ7 reference front for M > 3)
  run_*.py, track_convergence.py      experiment runners (Section 3 below)
  analyze_*_koa2.py        statistics and LaTeX tables for the manuscript
  make_paper_figures.py, make_extra_figures.py   all figures of the results section
  fs_problem.py, download_fs_datasets.py         feature-selection problem and dataset download
  data_fs/README.txt       where the feature-selection datasets go (they are not redistributed)
results/                   raw and processed results (Section 4 below)
figures/                   the eight result figures of the manuscript (PDF)
```

## 3. Experiments, scripts and where they appear in the manuscript

All runners are executed from `code/`. They append to the CSV files in `code/results/` and skip runs that already exist, so an interrupted job can simply be restarted. Run `python <script> --help` for every option.

| Experiment | Runner | Raw output | Analysis | Manuscript |
|---|---|---|---|---|
| Main comparison: 9 algorithms × 16 problems × 30 seeds | `run_experiments.py` | `raw_runs.csv` | `analyze_results_koa2.py` | Tables of IGD, HV and Friedman ranks; boxplot, Friedman and W/T/L figures (Sections 5.1–5.2) |
| Convergence: 9 algorithms × 16 problems × 5 seeds, IGD every 5 generations | `run_convergence_koa2.py` (wrapper of `track_convergence.py`) | `convergence_trajectories.csv` | `make_extra_figures.py` | Convergence figure (Section 5.3) |
| Many objectives: DTLZ2, DTLZ7, WFG4, MaF4 with M = 3, 5, 8, 10 | `run_many_objective.py` | `many_objective_raw.csv` | `analyze_many_objective_koa2.py` | Many-objective table (Section 5.4) |
| Feature selection: 4 algorithms × 13 datasets × 30 seeds | `run_feature_selection.py` | `feature_selection/<dataset>/<algorithm>/run_XX.npz` | `analyze_feature_selection_koa2.py` | Feature-selection table and figures (Section 5.5) |
| Ablation: KOA and 6 variants × 16 problems × 30 seeds | `run_experiments.py` | `raw_runs.csv` (labels `KOA-II-*`) | `analyze_ablation_koa2.py` | Ablation table and figure (Section 5.6) |
| Parameter sensitivity: 13 settings × 6 problems × 30 seeds | `run_sensitivity_koa2.py` | `sensitivity_koa2.csv` | `analyze_sensitivity_koa2.py` | Sensitivity figure (Section 5.7) |
| Run time | recorded in every run (`wall_seconds`) | `raw_runs.csv` | summarised in `runtime_table.csv` | Run-time table (Section 5.8) |

Example commands (from `code/`):

```bash
pip install -r ../requirements.txt

# main comparison and ablation
python run_experiments.py --algorithms NSGA-II NSGA-III MOEA/D RVEA AGE-MOEA2 CMOPSO SMS-EMOA GDE3 KOA-II --n-jobs 16
python run_experiments.py --algorithms KOA-II-Uniform KOA-II-PopGuide KOA-II-FixedAlloc KOA-II-NoAdapt KOA-II-DEonly KOA-II-SBXonly --n-jobs 16

# many objectives, convergence, sensitivity
python -u run_many_objective.py --algorithms NSGA-II NSGA-III MOEA/D RVEA AGE-MOEA2 CMOPSO SMS-EMOA GDE3 KOA-II --n-jobs 16
python -u run_convergence_koa2.py
python -u run_sensitivity_koa2.py

# feature selection (datasets first, see code/data_fs/README.txt)
python download_fs_datasets.py
python run_feature_selection.py --algorithms NSGA-II GDE3 SMS-EMOA KOA-II --n-jobs 16
```

To regenerate every table, statistic and figure from the deposited data only, copy (or link) `results/` to `code/results/` and run:

```bash
cd code
ln -s ../results results
python analyze_results_koa2.py
python analyze_ablation_koa2.py
python analyze_feature_selection_koa2.py      # the dataset table needs the datasets in data_fs/
python analyze_many_objective_koa2.py
python analyze_sensitivity_koa2.py
python make_paper_figures.py results ../figures
python make_extra_figures.py results ../figures
```

## 4. Result files

| File | Content |
|---|---|
| `raw_runs.csv` | one row per run of the main comparison and of the ablation: `problem, algorithm, seed, HV, IGD, Spacing, Spread, n_solutions, n_eval, fe_budget, wall_seconds, status, error` |
| `summary_table.csv`, `summary_table_raw.csv` | mean, standard deviation, median and bootstrap confidence intervals per problem and algorithm |
| `wilcoxon_tests.csv` | KOA against each baseline per problem and indicator: p-value, Holm-adjusted significance, A12, Cliff's delta |
| `friedman_ranking.csv` | average Friedman ranks and Friedman test over the 16 problems |
| `ablation_koa2_tests.csv`, `ablation_koa2_wtl.csv`, `ablation_koa2_ranking.csv` | the same statistics for KOA against its six variants |
| `many_objective_raw.csv` | one row per run: `problem, M, algorithm, seed, HV, IGD, Spacing, Spread, n_eval, fe_budget, wall_seconds, status` |
| `many_objective_summary.csv`, `many_objective_tests.csv`, `many_objective_friedman.csv`, `many_objective_table.tex` | statistics of the many-objective experiment (Holm over 128 comparisons per indicator) |
| `convergence_trajectories.csv` | `problem, algorithm, seed, generation, IGD` |
| `sensitivity_koa2.csv` | one row per run: `problem, param, value, seed, HV, IGD, n_eval, status, error, wall_seconds`; `param = default` marks the default configuration. A few settings were written twice (for example `10` and `10.0`); `analyze_sensitivity_koa2.py` keeps one run per setting and seed, giving 2,340 runs |
| `sensitivity_summary.csv`, `sensitivity_tests.csv`, `sensitivity_friedman.csv` | statistics of the sensitivity analysis (Holm over 72 comparisons per indicator) |
| `feature_selection/<dataset>/<algorithm>/run_XX.npz` | final non-dominated feature subsets of every run: `F_train` (selected-feature ratio, inner-CV error), `F_test` (ratio, test error), `n_evals`, `runtime` |
| `feature_selection_tables/` | `per_run_metrics.csv`, the Holm–Wilcoxon tests of each indicator and `fs_tables.tex` |
| `runtime_table.csv` | mean and standard deviation of the wall-clock time per run, per algorithm and problem |
| `preliminary_design/` | runs and analyses of the preliminary design, see the README in that folder |
| `log_*.txt` | console output of the analysis scripts when this package was built |

## 5. Protocol summary

* **Budget:** population N = 100, 30,000 evaluations per run (300 generations including the initial population); feature selection 10,000 evaluations (100 generations).
* **Pairing:** for every problem and seed (1–30), all algorithms start from the same Latin hypercube initial population.
* **Indicators:** HV with reference point 1.1 × the nadir point of the true front; IGD against a reference set on the true front. For DTLZ7 with M > 3, where pymoo provides no front, the reference set is generated in `utils/many_objective.py` (`dtlz7_reference_front`).
* **Statistics:** two-sided Wilcoxon signed-rank test over the 30 paired runs; Holm correction over all comparisons of one indicator (128 for the benchmark and many-objective studies, 96 for the ablation, 39 for feature selection, 72 for the sensitivity analysis); a significant difference counts as a win or a loss according to the mean; A12 effect size; Friedman test with the Nemenyi critical difference.
* **Feature selection:** objectives are the selected-feature ratio and the 5-fold stratified inner-CV error of k-NN (k = 5) on the training split; a feature is selected when x_j > 0.5 (an empty subset is repaired by selecting the feature with the largest x_j); stratified 70/30 train/test split per seed, Min–Max scaling fitted on the training split; HV with reference point (1.1, 1.1); IGD against the non-dominated union of the final training fronts of the four algorithms on each dataset.
* **Known failures:** 11 runs of AGE-MOEA-II on ZDT2 stop with a division-by-zero error inside pymoo; their statistics use the 19 successful runs. Two convergence runs of AGE-MOEA-II on ZDT2 fail for the same reason.

## 6. Datasets

The feature-selection datasets are public and are not redistributed here. Wine, WDBC and Digits load through scikit-learn; Libras, Semeion and Madelon come from OpenML; Vehicle, Ionosphere, SPECTF, Sonar, Hill-Valley, Musk1 and Arrhythmia come from the UCI Machine Learning Repository. `download_fs_datasets.py` writes them to `code/data_fs/` in the expected format. One malformed record of Statlog Vehicle is removed (845 samples remain), and missing values of Arrhythmia are replaced by the feature mean.

## 7. Software

Python with pymoo 0.6.1.6, pymoode (GDE3), moocore (HV cross-check), NumPy, SciPy, pandas, scikit-learn and matplotlib; exact versions are in `requirements.txt`. All runs were made on an Intel Xeon Linux server without GPU acceleration.

## 8. Name of the algorithm

The algorithm is named after the Kipunji (*Rungwecebus kipunji*), a primate endemic to Tanzania. The name is an identifier only; the design of KOA is not derived from the behaviour of the animal.

## 9. Licence and citation

Code and data: Creative Commons Attribution 4.0 International (CC BY 4.0). Please cite the manuscript above and the archived version of this package (Zenodo DOI given in the manuscript).
