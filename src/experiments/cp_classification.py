"""Classification CP results — Section 4.1.3.

Produces the inputs for both Table 3 (mean coverage and mean prediction-set
size for APS, RAPS with empty sets, and RAPS without empty sets) and Figure 3
(empirical coverage histograms for the RAPS variants) from a single sweep
over random calibration / evaluation splits.

For each alpha in {0.01, 0.05, 0.1} the script writes a JSON file containing:

  * the per-split coverage and mean-set-size arrays for APS, RAPS-with-empty
    and RAPS-without-empty (raw inputs to Figure 3, also reproduce Table 3 by
    averaging),
  * the mean coverage and mean set size for each method (Table 3 numbers).

Plotting is intentionally not part of this script — consumers can build the
histograms or any other view they need from the saved arrays.
"""
import json
import random

import numpy as np

from src.paths import RESULTS
from src.utils import cp_classification_utils as cp_cls, cp_regression_utils as cp_reg


# Number of random calibration / evaluation splits used in the paper.
N_RUNS = 1000


def _split(temp_list, split):
    cutoff = int((len(temp_list) + 1) * (1 - split))
    pred_prob = [e["probab"] for e in temp_list][:cutoff]
    pred_prob_eval = [e["probab"] for e in temp_list][cutoff:]
    gt_class = [e["gt_class"] for e in temp_list][:cutoff]
    gt_class_eval = [e["gt_class"] for e in temp_list][cutoff:]
    return pred_prob, pred_prob_eval, gt_class, gt_class_eval


def run(config):
    file_path = config["dataset_path"]
    dataset = config["dataset"]

    split = config["split"]
    random_seed = config["seed"]
    lam_reg = config["lam_reg"]
    k_reg = config["k_reg"]

    print(f"Following file will be validated: {file_path}")
    data_list = cp_reg.load_data(file_path)

    results_dir = RESULTS / dataset / "cp_classification"
    results_dir.mkdir(parents=True, exist_ok=True)

    alphas = [0.1, 0.05, 0.01]

    for alpha in alphas:
        coverage_aps, size_aps = [], []
        coverage_raps_empty, size_raps_empty = [], []
        coverage_raps, size_raps = [], []

        print(f"Number of runs: {N_RUNS} and alpha for the classes of {alpha}.")
        random.seed(random_seed)

        for iteration in range(N_RUNS):
            np.random.seed(iteration)
            temp_list = data_list.copy()
            random.shuffle(temp_list)
            pred_prob, pred_prob_eval, gt_class, gt_class_eval = _split(temp_list, split)

            # APS
            qhat_aps = cp_cls.quantile_aps(pred_prob, gt_class, alpha)
            sets_aps = cp_cls.get_prediction_set(pred_prob_eval, qhat_aps)
            coverage_aps.append(cp_cls.coverage_for_class(sets_aps, gt_class_eval))
            size_aps.append(float(np.average(sets_aps.sum(axis=1))))

            # RAPS
            qhat_raps, reg_vec = cp_cls.quantile_raps(pred_prob, gt_class, alpha, lam_reg, k_reg)

            # RAPS with empty sets allowed
            sets_raps_empty = cp_cls.get_prediction_set_raps(
                pred_prob_eval, qhat_raps, reg_vec, randomNumber=iteration)
            coverage_raps_empty.append(cp_cls.coverage_for_class(sets_raps_empty, gt_class_eval))
            size_raps_empty.append(float(np.average(sets_raps_empty.sum(axis=1))))

            # RAPS without empty sets
            sets_raps = cp_cls.get_prediction_set_raps(
                pred_prob_eval, qhat_raps, reg_vec, randomNumber=iteration, disallow_zero_sets=True)
            coverage_raps.append(cp_cls.coverage_for_class(sets_raps, gt_class_eval))
            size_raps.append(float(np.average(sets_raps.sum(axis=1))))

        payload = {
            "alpha": alpha,
            "runs": N_RUNS,
            "split": split,
            "evaluation_set_size": len(pred_prob_eval),
            "summary": {
                "APS": {
                    "mean_coverage": round(float(np.mean(coverage_aps)), 4),
                    "mean_set_size": round(float(np.mean(size_aps)), 4),
                },
                "RAPS_with_empty_sets": {
                    "mean_coverage": round(float(np.mean(coverage_raps_empty)), 4),
                    "mean_set_size": round(float(np.mean(size_raps_empty)), 4),
                },
                "RAPS_without_empty_sets": {
                    "mean_coverage": round(float(np.mean(coverage_raps)), 4),
                    "mean_set_size": round(float(np.mean(size_raps)), 4),
                },
            },
            "per_split": {
                "APS_coverage": [float(x) for x in coverage_aps],
                "APS_set_size": size_aps,
                "RAPS_with_empty_sets_coverage": [float(x) for x in coverage_raps_empty],
                "RAPS_with_empty_sets_set_size": size_raps_empty,
                "RAPS_without_empty_sets_coverage": [float(x) for x in coverage_raps],
                "RAPS_without_empty_sets_set_size": size_raps,
            },
        }

        out = results_dir / f"results_classification_alpha_{alpha}_runs_{N_RUNS}_split_{split}.json"
        with open(out, "w") as f:
            json.dump(payload, f)
        print(f"Wrote {out}")
