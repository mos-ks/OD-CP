from src.utils import cp_regression_utils as cp_func, metrics as eval_of_cp, paired_t_test as ttest
from src.paths import RESULTS
import numpy as np
import random
import json
from itertools import combinations


def run(config):
    file_path = config["dataset_path"]
    dataset = config["dataset"]

    # define params
    alpha = config["alpha_only_regression"]
    split = config["split"]
    random_seed = config["seed"]
    dc_threshold = config["dc_threshold"]
    iou_threshold = config["iou_threshold"]
    threshold = config["threshold"]

    print(f"Following file will be validated: {file_path}")
    data_list = cp_func.load_data(file_path)

    # define params
    numberOfRuns = config["runs"]
    save_t_test = config["save_t_test"]
    uncertainties = config["all_uncertainties"]
    print(f"Number of runs: {numberOfRuns} and alpha of {alpha}.")
    numbers = list(range(numberOfRuns))

    # evaluating metrics unscaled
    unscaled_IoU = []
    unscaled_coverage = []
    unscaled_dice_coefficient_list = []
    inside_with_unscaled_cp_each_run:list[int] = []
    interval_scores_unscaled = []
    previously_outside_now_inside_unscaled = []
    unscaled_iou_below_threshold = []
    unscaled_iou_below_threshold_now_inside = []

    # non CP-procedure
    gt_covered_by_pred = []
    non_cp_iou = []
    non_cp_dc = []


    # setting paths
    results_dir = RESULTS / dataset / "cp_only_regression"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = (
            results_dir /
            f"results_alpha_{alpha}_runs_{numberOfRuns}.txt"
    )

    results_unscaled = {
        "unscaled_coverage": [],
        "unscaled_IoU": [],
        "unscaled_dice_coefficient_list": [],
        "inside_with_unscaled_cp_each_run": [],
        "interval_scores_unscaled": [],
        "previously_outside_now_inside_unscaled": [],
        "gt_covered_by_pred": [],
        "unscaled_iou_below_threshold": {t: [] for t in threshold},
        "unscaled_iou_below_threshold_now_inside": {t: [] for t in threshold},
    }

    results_scaled = {
        item: {
            "scaled_coverage": [],
            "scaled_IoU": [],
            "scaled_dice_coefficient_list": [],
            "inside_with_scaled_cp_each_run": [],
            "interval_scores_scaled": [],
            "previously_outside_now_inside_scaled": [],
            "gt_covered_by_pred": [],
            "scaled_iou_below_threshold": {t: [] for t in threshold},
            "scaled_iou_below_threshold_now_inside": {t: [] for t in threshold},
        }
        for item in uncertainties
    }


    random.seed(random_seed)
    for iteration in numbers:
        # define predicted, ground truth and uncertainty lists
        temp_list = data_list.copy()
        # difference for results compared to data_list[:]
        random.shuffle(temp_list)
        pred_bbox = [entry['bbox'] for entry in temp_list][:int((len(temp_list)+1)*(1-split))]
        pred_bbox_eval = [entry['bbox'] for entry in temp_list][int((len(temp_list)+1)*(1-split)):]
        gt_bbox = [entry['gt_bbox'] for entry in temp_list][:int((len(temp_list)+1)*(1-split))]
        gt_bbox_eval = [entry['gt_bbox'] for entry in temp_list][int((len(temp_list)+1)*(1-split)):]


        # Unscaled (vanilla) CP
        iou_cp_unscaled = []
        dc_cp_unscaled = []

        # non-CP procedure
        iou_non_cp_runs = []
        dc_non_cp_runs = []

        quantile_xmin_unscaled, quantile_xmax_unscaled, quantile_ymin_unscaled, quantile_ymax_unscaled = (
            cp_func.calculate_quantiles_for_each_dim(
                cp_func.cp_scores_unscaled(pred_bbox, gt_bbox), alpha))
        bbox_outer_unscaled_cp, bbox_inner_unscaled_cp = cp_func.inner_and_outer_box_unscaled(quantile_ymin_unscaled, quantile_xmin_unscaled, quantile_ymax_unscaled, quantile_xmax_unscaled, pred_bbox_eval)

        inside_with_unscaled_cp = eval_of_cp.true_detections(bbox_outer_unscaled_cp, bbox_inner_unscaled_cp, gt_bbox_eval)

        for gt, cp_unscaled in zip(gt_bbox_eval, bbox_outer_unscaled_cp):
            iou_cp_unscaled.append(eval_of_cp.bb_intersection_over_union(gt,cp_unscaled))
            dc_cp_unscaled.append(eval_of_cp.dice_coefficient(gt,cp_unscaled))
        results_unscaled["unscaled_IoU"].append(sum(iou_cp_unscaled)/len(iou_cp_unscaled))
        results_unscaled["unscaled_coverage"].append(inside_with_unscaled_cp.count(True)/len(inside_with_unscaled_cp))
        results_unscaled["unscaled_dice_coefficient_list"].append(sum(dc_cp_unscaled)/len(dc_cp_unscaled))
        results_unscaled["inside_with_unscaled_cp_each_run"].append(inside_with_unscaled_cp.count(True))
        results_unscaled["interval_scores_unscaled"].append(
            eval_of_cp.calculate_interval_score_for_whole_bboxes(
                bbox_inner_unscaled_cp, bbox_outer_unscaled_cp, gt_bbox_eval, alpha))
        results_unscaled["previously_outside_now_inside_unscaled"].append(
            eval_of_cp.count_newly_covered_predictions(pred_bbox_eval, bbox_outer_unscaled_cp, gt_bbox_eval))

        for t in threshold:
            threshold_results_unscaled = eval_of_cp.iou_below_threshold_now_inside_cp_bboxes(pred_bbox_eval, gt_bbox_eval, t,
                                                                      bbox_outer_unscaled_cp, bbox_inner_unscaled_cp)
            results_unscaled["unscaled_iou_below_threshold"][t].append(threshold_results_unscaled[0])
            results_unscaled["unscaled_iou_below_threshold_now_inside"][t].append(threshold_results_unscaled[1])

        # evaluating non-cp procedure
        for gt, bbox_pred in zip(gt_bbox_eval, pred_bbox_eval):
            iou_non_cp_runs.append(eval_of_cp.bb_intersection_over_union(gt,bbox_pred))
            dc_non_cp_runs.append(eval_of_cp.dice_coefficient(gt,bbox_pred))
        results_unscaled["gt_covered_by_pred"].append([((pred[0] <= gt[0] and pred[1] <= gt[1] and pred[2] >= gt[2] and pred[3] >= gt[3])) for pred,gt in zip(pred_bbox_eval, gt_bbox_eval)].count(True))
        non_cp_iou.append(sum(val >= iou_threshold for val in iou_non_cp_runs))
        non_cp_dc.append(sum(val >= dc_threshold for val in dc_non_cp_runs))


        # evaluating scaled
        for item in uncertainties:
            iou_cp_scaled = []
            dc_cp_scaled = []

            uncertainty_each_coordinate = [entry[item] for entry in temp_list][:int((len(temp_list)+1)*(1-split))]
            uncertainty_each_coordinate_eval = [entry[item] for entry in temp_list][int((len(temp_list)+1)*(1-split)):]

            # calculate scaled CP
            cp_scores = cp_func.cp_scores_scaled(pred_bbox, gt_bbox, uncertainty_each_coordinate)
            [quantile_xmin, quantile_xmax, quantile_ymin, quantile_ymax] = cp_func.calculate_quantiles_for_each_dim(cp_scores, alpha)

            # calculate scaled conformalized bounding boxes
            bbox_outer_scaled_cp, bbox_inner_scaled_cp = cp_func.inner_and_outer_box_scaled(quantile_ymin, quantile_xmin, quantile_ymax, quantile_xmax, pred_bbox_eval, uncertainty_each_coordinate_eval)

            inside_with_scaled_cp = eval_of_cp.true_detections(bbox_outer_scaled_cp, bbox_inner_scaled_cp, gt_bbox_eval)

            for gt, cp_scaled in zip(gt_bbox_eval, bbox_outer_scaled_cp):
                iou_cp_scaled.append(eval_of_cp.bb_intersection_over_union(gt,cp_scaled))
                dc_cp_scaled.append(eval_of_cp.dice_coefficient(gt,cp_scaled))

            results_scaled[item]["scaled_coverage"].append(inside_with_scaled_cp.count(True)/len(inside_with_scaled_cp))
            results_scaled[item]["scaled_IoU"].append(sum(iou_cp_scaled)/len(iou_cp_scaled))
            results_scaled[item]["scaled_dice_coefficient_list"].append(sum(dc_cp_scaled)/len(dc_cp_scaled))
            results_scaled[item]["inside_with_scaled_cp_each_run"].append(inside_with_scaled_cp.count(True))
            results_scaled[item]["interval_scores_scaled"].append(
                eval_of_cp.calculate_interval_score_for_whole_bboxes(
                    bbox_inner_scaled_cp, bbox_outer_scaled_cp, gt_bbox_eval, alpha))
            results_scaled[item]["previously_outside_now_inside_scaled"].append(
                eval_of_cp.count_newly_covered_predictions(
                    pred_bbox_eval, bbox_outer_scaled_cp, gt_bbox_eval
                )
            )
            results_scaled[item]["gt_covered_by_pred"].append(
                [((pred[0] <= gt[0] and pred[1] <= gt[1] and pred[2] >= gt[2] and pred[3] >= gt[3])) for pred,gt in zip(pred_bbox_eval, gt_bbox_eval)].count(True))

            for t in threshold:
                threshold_results_scaled = eval_of_cp.iou_below_threshold_now_inside_cp_bboxes(pred_bbox_eval, gt_bbox_eval, t,
                                                                                               bbox_outer_scaled_cp, bbox_inner_scaled_cp)
                results_scaled[item]["scaled_iou_below_threshold"][t].append(threshold_results_scaled[0])
                results_scaled[item]["scaled_iou_below_threshold_now_inside"][t].append(threshold_results_scaled[1])


    with open(output_file_path, "w") as f:
        f.write(f"Using an alpha of {alpha} and {numberOfRuns} runs.\n")
        # create results unscaled
        below_threshold_means = {
            t: round(
                np.mean(results_unscaled["unscaled_iou_below_threshold"][t])
                if len(results_unscaled["unscaled_iou_below_threshold"][t]) > 0 else 0,
                2
            )
            for t in threshold
        }

        now_inside_means = {
            t: round(
                np.mean(results_unscaled["unscaled_iou_below_threshold_now_inside"][t])
                if len(results_unscaled["unscaled_iou_below_threshold_now_inside"][t]) > 0 else 0,
                2
            )
            for t in threshold
        }
        unscaled_results = {
            "type": "unscaled CP",
            "mean coverage": round(np.mean(results_unscaled["unscaled_coverage"]), 2),
            "mean IoU": round(np.mean(results_unscaled["unscaled_IoU"])*100, 2),
            "mean Dice Coefficient": round(np.mean(results_unscaled["unscaled_dice_coefficient_list"])*100, 2),
            "mean gt inside conformalized bbox": round(np.mean(results_unscaled["inside_with_unscaled_cp_each_run"]), 0),
            "mean interval score": round(np.mean(results_unscaled["interval_scores_unscaled"]), 2),
            "elements in each run": int((len(data_list)+1) * split),
            "number of gt that are covered by pred": round(np.mean(results_unscaled["gt_covered_by_pred"]), 0),
            "turning false (not inside pred) into true (only outer box)": round(np.mean(results_unscaled["previously_outside_now_inside_unscaled"]),0),
            "iou_below_threshold_mean": below_threshold_means,
            "iou_below_threshold_now_inside_mean": now_inside_means,
        }
        non_cp_results = {
            "type": "without CP",
            "threshold iou": iou_threshold,
            "mean iou of gt and pred being above threshold": round(np.mean(non_cp_iou), 0),
            "mean iou percentage": round(np.mean(non_cp_iou)/int((len(data_list)+1) * split)*100, 2),
            "threshold dc": dc_threshold,
            "mean dc of gt and pred being above threshold": round(np.mean(non_cp_dc), 0),
            "mean dc percentage": round(np.mean(non_cp_dc)/int((len(data_list)+1) * split)*100, 2),
            "elements in each run": int((len(data_list)+1) * split),
        }
        f.write(json.dumps(non_cp_results))
        f.write("\n")
        f.write(json.dumps(unscaled_results))

        for item in uncertainties:
            below_threshold_means = {
                t: round(
                    np.mean(results_scaled[item]["scaled_iou_below_threshold"][t])
                    if len(results_scaled[item]["scaled_iou_below_threshold"][t]) > 0 else 0,
                    2
                )
                for t in threshold
            }

            now_inside_means = {
                t: round(
                    np.mean(results_scaled[item]["scaled_iou_below_threshold_now_inside"][t])
                    if len(results_scaled[item]["scaled_iou_below_threshold_now_inside"][t]) > 0 else 0,
                    2
                )
                for t in threshold
            }


            scaled_results = {
                "type": f"scaled CP with {item}",
                "mean coverage": round(np.mean(results_scaled[item]["scaled_coverage"]), 2),
                "mean IoU": round(np.mean(results_scaled[item]["scaled_IoU"])*100, 2),
                "mean Dice Coefficient": round(np.mean(results_scaled[item]["scaled_dice_coefficient_list"])*100, 2),
                "mean inside gt in conformalized": round(np.mean(results_scaled[item]["inside_with_scaled_cp_each_run"]), 0),
                "mean interval score": round(np.mean(results_scaled[item]["interval_scores_scaled"]), 2),
                "elements in each run": int((len(data_list)+1) * split),
                "number of gt that are covered by pred": round(np.mean(results_scaled[item]["gt_covered_by_pred"]), 0),
                "turning false (not inside pred) into true (only outer box)": round(np.mean(results_scaled[item]["previously_outside_now_inside_scaled"]),0),
                "scaled_iou_below_threshold_mean": below_threshold_means,
                "scaled_iou_below_threshold_now_inside_mean": now_inside_means,
            }
            f.write("\n")
            f.write(json.dumps(scaled_results))

        if save_t_test:
            f.write("\n\n")
            f.write(" -------- Results for two-sample paired t-test -------- ")
            f.write("\n")
            f.write("Results for comparing scaled and unscaled")
            results_t_tests = []
            for item in uncertainties:
                t_test_results = ttest.run_paired_tests(
                    item, "unscaled CP",
                    results_scaled[item]["scaled_coverage"], results_unscaled["unscaled_coverage"],
                    results_scaled[item]["interval_scores_scaled"], results_unscaled["interval_scores_unscaled"],
                    results_scaled[item]["scaled_dice_coefficient_list"], results_unscaled["unscaled_dice_coefficient_list"],
                    results_scaled[item]["scaled_IoU"], results_unscaled["unscaled_IoU"]
                )
                results_t_tests.append(t_test_results)

            for el in results_t_tests:
                f.write("\n")
                f.write(el)

            f.write("\n")
            f.write("Results for comparing scaled and scaled")
            results_t_tests = []
            for u1, u2 in combinations(uncertainties, 2):
                t_test_results = ttest.run_paired_tests(
                    u1, u2,
                    results_scaled[u1]["scaled_coverage"], results_scaled[u2]["scaled_coverage"],
                    results_scaled[u1]["interval_scores_scaled"], results_scaled[u2]["interval_scores_scaled"],
                    results_scaled[u1]["scaled_dice_coefficient_list"], results_scaled[u2]["scaled_dice_coefficient_list"],
                    results_scaled[u1]["scaled_IoU"], results_scaled[u2]["scaled_IoU"]
                )
                results_t_tests.append(t_test_results)
            for el in results_t_tests:
                f.write("\n")
                f.write(el)
