import json
import random
import numpy as np
from src.utils import (
    cp_classification_utils as cp_func_for_classes,
    cp_regression_utils as cp_func,
    metrics as eval_of_cp,
)
from src.paths import RESULTS


def run(config):
    file_path = config["dataset_path"]
    dataset = config["dataset"]

    # define params
    alpha = config["alpha_regression"]
    alpha_class = config["alpha_class"]
    runs = config["runs"]
    split = config["split"]
    uncertainty = config["uncertainty"]
    random_seed = config["seed"]

    # for RAPS
    lam_reg = config["lam_reg"]
    k_reg = config["k_reg"]

    print(f"Following file will be validated: {file_path}")
    data_list = cp_func.load_data(file_path)
    print(f"Number of runs: {runs} and alpha for the classes of {alpha_class} and for the bbox of {alpha}.")


    # setting some dirs
    results_dir = RESULTS / dataset / "cp_for_class_and_bbox"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = (
            results_dir /
            f"results_cp_for_classes_and_bboxes_alpha_{alpha}_runs_{runs}_split_{split}.txt"
    )


    classes = [int(entry['gt_class']) for entry in data_list]
    unique_classes = np.unique(classes)

    print(f"Different classes are: {np.unique(classes)}")

    class_labels = sorted(unique_classes)
    idx_to_class = {i: cls for i, cls in enumerate(class_labels)}
    class_to_idx = {cls: i for i, cls in enumerate(class_labels)}

    grouped = {cls: [] for cls in idx_to_class}

    for entry in data_list:
        cls = int(entry['gt_class'])
        idx = class_to_idx[cls]
        grouped[idx].append(entry)

    random.seed(random_seed)


    # evaluation metrics unscaled
    unscaled_IoU: list[float] = []
    unscaled_coverage: list[float] = []
    unscaled_dice_coefficient_list: list[float] = []
    inside_with_unscaled_cp_each_run: list[int] = []
    previously_outside_now_inside_unscaled = []
    gt_covered_by_pred_unscaled = []
    interval_scores_unscaled = []


    # evaluation metrics scaled
    scaled_coverage = []
    scaled_IoU = []
    scaled_dice_coefficient_list = []
    inside_with_scaled_cp_each_run:list[int] = []
    previously_outside_now_inside_scaled = []
    gt_covered_by_pred_scaled = []
    interval_scores_scaled = []


    for run in list(range(runs)):
        iou_cp_unscaled: list[float] = []
        dc_cp_unscaled: list[float] = []

        iou_cp_scaled = []
        dc_cp_scaled = []

        calib_all = []
        eval_all = []
        cp_quantile_regression = []
        cp_quantile_regression_scaled = []

        for cls, entries in grouped.items():
            temp_list = entries.copy()
            random.shuffle(temp_list)
            cutoff = int((len(temp_list) + 1) * (1 - split))
            calib_all.append(temp_list[:cutoff])
            eval_all.append(temp_list[cutoff:])

        predicted_prob = [entry['probab']
                          for class_entries in calib_all
                          for entry in class_entries]
        predicted_prob_eval = [entry['probab']
                               for class_entries in eval_all
                               for entry in class_entries]
        gt_class = [entry['gt_class']
                    for class_entries in calib_all
                    for entry in class_entries]
        gt_class_eval = [entry['gt_class']
                         for class_entries in eval_all
                         for entry in class_entries]
        pred_bbox_eval = [entry['bbox']
                          for class_entries in eval_all
                          for entry in class_entries]
        gt_bbox_eval = [entry['gt_bbox']
                        for class_entries in eval_all
                        for entry in class_entries]
        uncertainty_each_coordinate = [entry[uncertainty]
                                       for class_entries in eval_all
                                       for entry in class_entries]

        qhat, reg_vec = cp_func_for_classes.quantile_raps(predicted_prob, gt_class, alpha_class, lam_reg, k_reg, random_seed)

        # RAPS with empty sets disallowed
        prediction_set = cp_func_for_classes.get_prediction_set_raps(predicted_prob_eval, qhat, reg_vec, randomNumber = run, disallow_zero_sets=True)

        # calib step for bbox class-wise
        for classes in calib_all:
            pred_bbox = [entry['bbox'] for entry in classes]
            gt_bbox = [entry['gt_bbox'] for entry in classes]
            uncertainty_each_coordinate_each_class = [entry[uncertainty] for entry in classes]
            # unscaled cp
            cp_quantile_regression.append(list(
                cp_func.calculate_quantiles_for_each_dim(
                    cp_func.cp_scores_unscaled(pred_bbox, gt_bbox), alpha)))
            # scaled cp
            cp_scores = cp_func.cp_scores_scaled(pred_bbox, gt_bbox, uncertainty_each_coordinate_each_class)
            cp_quantile_regression_scaled.append(list(
                cp_func.calculate_quantiles_for_each_dim(cp_scores, alpha)))

        # do cp for all
        prediction_set_eval = cp_func_for_classes.get_prediction_set_raps(predicted_prob_eval, qhat, reg_vec, randomNumber = run, disallow_zero_sets=True)
        unscaled_max_quantile_class_wise = [max(entry) for entry in cp_quantile_regression]
        scaled_max_quantile_class_wise = [max(entry) for entry in cp_quantile_regression_scaled]

        max_class_unscaled = []
        max_class_scaled = []

        # get the class with the largest quantile from all classes that are in prediction set
        for predicted_set in prediction_set_eval:

            true_indices = np.where(predicted_set)[0]

            # unscaled
            mask_values = (np.asarray(unscaled_max_quantile_class_wise))[true_indices]
            max_pos = mask_values.argmax()
            max_class_unscaled.append(int(true_indices[max_pos]))

            # scaled
            mask_values = (np.asarray(scaled_max_quantile_class_wise))[true_indices]
            max_pos = mask_values.argmax()
            max_class_scaled.append(int(true_indices[max_pos]))

        # now do the CP steps
        # start with unscaled
        bbox_outer_unscaled_cp, bbox_inner_unscaled_cp = cp_func.inner_and_outer_bbox_using_cp_for_classification_and_regression_unscaled(pred_bbox_eval, max_class_unscaled, cp_quantile_regression)
        inside_with_unscaled_cp = eval_of_cp.true_detections(bbox_outer_unscaled_cp, bbox_inner_unscaled_cp, gt_bbox_eval)

        for gt, cp_unscaled in zip(gt_bbox_eval, bbox_outer_unscaled_cp):
            iou_cp_unscaled.append(eval_of_cp.bb_intersection_over_union(gt,cp_unscaled))
            dc_cp_unscaled.append(eval_of_cp.dice_coefficient(gt,cp_unscaled))

        unscaled_coverage.append(inside_with_unscaled_cp.count(True)/len(inside_with_unscaled_cp))
        unscaled_IoU.append(sum(iou_cp_unscaled)/len(iou_cp_unscaled))
        unscaled_dice_coefficient_list.append(sum(dc_cp_unscaled)/len(dc_cp_unscaled))
        inside_with_unscaled_cp_each_run.append(inside_with_unscaled_cp.count(True))
        interval_scores_unscaled.append(
            eval_of_cp.calculate_interval_score_for_whole_bboxes(
                bbox_inner_unscaled_cp, bbox_outer_unscaled_cp, gt_bbox_eval, alpha))
        previously_outside_now_inside_unscaled.append(
            eval_of_cp.count_newly_covered_predictions(pred_bbox_eval, bbox_outer_unscaled_cp, gt_bbox_eval))
        gt_covered_by_pred_unscaled.append([((pred[0] <= gt[0] and pred[1] <= gt[1] and pred[2] >= gt[2] and pred[3] >= gt[3])) for pred,gt in zip(pred_bbox_eval, gt_bbox_eval)].count(True))

        # now scaled
        bbox_outer_scaled_cp , bbox_inner_scaled_cp = cp_func.inner_and_outer_bbox_using_cp_for_classification_and_regression_scaled(pred_bbox_eval, max_class_scaled, cp_quantile_regression_scaled, uncertainty_each_coordinate)
        inside_with_scaled_cp = eval_of_cp.true_detections(bbox_outer_scaled_cp, bbox_inner_scaled_cp, gt_bbox_eval)

        for gt,cp_scaled in zip(gt_bbox_eval, bbox_outer_scaled_cp):
            iou_cp_scaled.append(eval_of_cp.bb_intersection_over_union(gt,cp_scaled))
            dc_cp_scaled.append(eval_of_cp.dice_coefficient(gt,cp_scaled))
        scaled_coverage.append(inside_with_scaled_cp.count(True)/len(inside_with_scaled_cp))
        scaled_IoU.append(sum(iou_cp_scaled)/len(iou_cp_scaled))
        scaled_dice_coefficient_list.append(sum(dc_cp_scaled)/len(dc_cp_scaled))
        inside_with_scaled_cp_each_run.append(inside_with_scaled_cp.count(True))
        interval_scores_scaled.append(
            eval_of_cp.calculate_interval_score_for_whole_bboxes(
                bbox_inner_scaled_cp, bbox_outer_scaled_cp, gt_bbox_eval, alpha))
        previously_outside_now_inside_scaled.append(
            eval_of_cp.count_newly_covered_predictions(pred_bbox_eval, bbox_outer_scaled_cp, gt_bbox_eval))
        gt_covered_by_pred_scaled.append([((pred[0] <= gt[0] and pred[1] <= gt[1] and pred[2] >= gt[2] and pred[3] >= gt[3])) for pred,gt in zip(pred_bbox_eval, gt_bbox_eval)].count(True))



    unscaled_results = {
        "type": "unscaled CP",
        "mean coverage": round(np.mean(unscaled_coverage), 2),
        "mean IoU": round(np.mean(unscaled_IoU)*100, 2),
        "mean Dice Coefficient": round(np.mean(unscaled_dice_coefficient_list)*100, 2),
        "mean inside gt in conformalized": round(np.mean(inside_with_unscaled_cp_each_run), 0),
        "mean interval score": round(np.mean(interval_scores_unscaled), 2),
        "number of gt that are covered by pred": round(np.mean(gt_covered_by_pred_unscaled), 0),
        "turning false (not inside pred) into true (only outer box)": round(np.mean(previously_outside_now_inside_unscaled), 0),
    }


    scaled_results = {
        "type": f"scaled CP with {uncertainty}",
        "mean coverage": round(np.mean(scaled_coverage), 2),
        "mean IoU": round(np.mean(scaled_IoU)*100, 2),
        "mean Dice Coefficient": round(np.mean(scaled_dice_coefficient_list)*100, 2),
        "mean inside gt in conformalized": round(np.mean(inside_with_scaled_cp_each_run), 0),
        "mean interval score": round(np.mean(interval_scores_scaled), 2),
        "number of gt that are covered by pred": round(np.mean(gt_covered_by_pred_scaled), 0),
        "turning false (not inside pred) into true (only outer box)": round(np.mean(previously_outside_now_inside_scaled), 0),
    }
    with open(output_file_path, "w") as f:
        f.write(f"Using an alpha for the classes of {alpha_class} and for the bbox of {alpha} and {runs} runs and a {1-split} training and {split} validation split.\n")
        f.write(json.dumps(unscaled_results))
        f.write("\n")
        f.write(json.dumps(scaled_results))



