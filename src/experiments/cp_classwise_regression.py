import json
import random
import numpy as np
from src.utils import cp_regression_utils as cp_func, metrics as eval_of_cp
from src.paths import RESULTS

def run(config):
    file_path = config["dataset_path"]
    dataset = config["dataset"]

    # define params
    alpha = config["alpha_only_regression"]
    runs = config["runs"]
    split = config["split"]
    uncertainty = config["uncertainty"]
    random_seed = config["seed"]

    print(f"Following file will be validated: {file_path}")
    data_list = cp_func.load_data(file_path)

    numberOfRuns = list(range(runs))
    print(f"Number of runs: {runs} and alpha of {alpha}.")

    classes = [int(entry['gt_class']) for entry in data_list]
    unique_classes = np.unique(classes)

    grouped = {cls: [] for cls in unique_classes}
    print(f"Different classes are: {np.unique(classes)}")

    for entry in data_list:
        cls = int(entry['gt_class'])
        grouped[cls].append(entry)

    results_dir = RESULTS / dataset / "cp_only_regression"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = (
            results_dir /
            f"results_class-wise_alpha_{alpha}_runs_{runs}_split_{split}.txt"
    )


    random.seed(random_seed)


    with open(output_file_path, "w") as f:
        f.write(f"Using an alpha of {alpha} and {runs} runs and a {1-split} training and {split} validation split.")
        elements = 0
        #overall unscaled
        unscaled_weighted_coverage = 0
        unscaled_weighted_meanIoU = 0
        unscaled_weighted_DC = 0
        unscaled_sum_gt_in_cp_box = 0
        unscaled_sum_interval_score = 0
        unscaled_sum_false_into_true = 0

        #overall scaled
        scaled_weighted_coverage = 0
        scaled_weighted_meanIoU = 0
        scaled_weighted_DC = 0
        scaled_sum_gt_in_cp_box = 0
        scaled_sum_interval_score = 0
        scaled_sum_false_into_true = 0




        for cls in sorted(grouped.keys()):
            # evaluating metrics unscaled
            unscaled_IoU: list[float] = []
            unscaled_coverage: list[float] = []
            unscaled_dice_coefficient_list: list[float] = []
            inside_with_unscaled_cp_each_run: list[int] = []
            previously_outside_now_inside_unscaled = []
            gt_covered_by_pred_unscaled = []

            # evaluation metrics scaled
            scaled_coverage = []
            scaled_IoU = []
            scaled_dice_coefficient_list = []
            inside_with_scaled_cp_each_run:list[int] = []
            interval_scores_unscaled = []
            previously_outside_now_inside_scaled = []
            gt_covered_by_pred_scaled = []

            entries = grouped[cls]

            print(f"Doing class: {cls}")

            for run in numberOfRuns:
                temp_list = entries.copy()
                random.shuffle(temp_list)
                #define predicted, ground truth and uncertainty lists
                pred_bbox = [entry['bbox'] for entry in temp_list][:int((len(temp_list)+1)*(1-split))]
                pred_bbox_eval = [entry['bbox'] for entry in temp_list][int((len(temp_list)+1)*(1-split)):]
                gt_bbox = [entry['gt_bbox'] for entry in temp_list][:int((len(temp_list)+1)*(1-split))]
                gt_bbox_eval = [entry['gt_bbox'] for entry in temp_list][int((len(temp_list)+1)*(1-split)):]
                uncertainty_each_coordinate = [entry[uncertainty] for entry in temp_list][:int((len(temp_list)+1)*(1-split))]
                uncertainty_each_coordinate_eval = [entry[uncertainty] for entry in temp_list][int((len(temp_list)+1)*(1-split)):]

                # Unscaled (vanilla) CP
                iou_cp_unscaled: list[float] = []
                dc_cp_unscaled: list[float] = []
                quantile_xmin_unscaled, quantile_xmax_unscaled, quantile_ymin_unscaled, quantile_ymax_unscaled = (
                    cp_func.calculate_quantiles_for_each_dim(
                        cp_func.cp_scores_unscaled(pred_bbox, gt_bbox), alpha))
                bbox_outer_unscaled_cp, bbox_inner_unscaled_cp = (
                    cp_func.inner_and_outer_box_unscaled(
                        quantile_ymin_unscaled, quantile_xmin_unscaled,
                        quantile_ymax_unscaled, quantile_xmax_unscaled, pred_bbox_eval))
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

                #scaled CP
                iou_cp_scaled = []
                dc_cp_scaled = []
                interval_scores_scaled = []
                cp_scores = cp_func.cp_scores_scaled(pred_bbox, gt_bbox, uncertainty_each_coordinate)
                [quantile_xmin, quantile_xmax, quantile_ymin, quantile_ymax] = cp_func.calculate_quantiles_for_each_dim(cp_scores, alpha)

                # calculate scaled conformalized bounding boxes
                bbox_outer_scaled_cp , bbox_inner_scaled_cp = (
                    cp_func.inner_and_outer_box_scaled(quantile_ymin, quantile_xmin, quantile_ymax, quantile_xmax,
                                                       pred_bbox_eval, uncertainty_each_coordinate_eval))

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
                "type": f"unscaled CP                 ",
                "class": int(cls),
                "elements in class": len(temp_list),
                "mean coverage": round(np.mean(unscaled_coverage), 2),
                "mean IoU": round(np.mean(unscaled_IoU)*100, 2),
                "mean Dice Coefficient": round(np.mean(unscaled_dice_coefficient_list)*100, 2),
                "mean inside gt in conformalized": round(np.mean(inside_with_unscaled_cp_each_run), 0),
                "mean interval score": round(np.mean(interval_scores_unscaled), 2),
                "elements in each run": int((len(temp_list)+1) * split),
                "number of gt that are covered by pred": round(np.mean(gt_covered_by_pred_unscaled), 0),
                "turning false (not inside pred) into true (only outer box)": round(np.mean(previously_outside_now_inside_unscaled),0),
            }
            f.write("\n")
            f.write(json.dumps(unscaled_results))

            scaled_results = {
                "type": f"scaled CP with {uncertainty}",
                "class": int(cls),
                "elements in class": len(entries),
                "mean coverage": round(np.mean(scaled_coverage), 2),
                "mean IoU": round(np.mean(scaled_IoU)*100, 2),
                "mean Dice Coefficient": round(np.mean(scaled_dice_coefficient_list)*100, 2),
                "mean inside gt in conformalized": round(np.mean(inside_with_scaled_cp_each_run), 0),
                "mean interval score": round(np.mean(interval_scores_scaled), 2),
                "elements in each run": int((len(temp_list)+1) * split),
                "number of gt that are covered by pred": round(np.mean(gt_covered_by_pred_scaled), 0),
                "turning false (not inside pred) into true (only outer box)": round(np.mean(previously_outside_now_inside_scaled),0),
            }
            f.write("\n")
            f.write(json.dumps(scaled_results))
            elements +=  int((len(temp_list)+1) * split)

            #generate overall metrics for unscaled
            unscaled_weighted_coverage +=round(np.mean(unscaled_coverage), 2) * int((len(temp_list)+1) * split)
            unscaled_weighted_meanIoU += round(np.mean(unscaled_IoU)*100, 2) * int((len(temp_list)+1) * split)
            unscaled_weighted_DC += round(np.mean(unscaled_dice_coefficient_list)*100, 2) * int((len(temp_list)+1) * split)
            unscaled_sum_gt_in_cp_box +=round(np.mean(inside_with_unscaled_cp_each_run), 0)
            unscaled_sum_interval_score +=round(np.mean(interval_scores_unscaled), 2)
            unscaled_sum_false_into_true +=round(np.mean(previously_outside_now_inside_unscaled),0)

            #generate overall metrics for unscaled
            scaled_weighted_coverage +=round(np.mean(scaled_coverage), 2) * int((len(temp_list)+1) * split)
            scaled_weighted_meanIoU += round(np.mean(scaled_IoU)*100, 2) * int((len(temp_list)+1) * split)
            scaled_weighted_DC += round(np.mean(scaled_dice_coefficient_list)*100, 2) * int((len(temp_list)+1) * split)
            scaled_sum_gt_in_cp_box +=round(np.mean(inside_with_scaled_cp_each_run), 0)
            scaled_sum_interval_score +=round(np.mean(interval_scores_scaled), 2)
            scaled_sum_false_into_true +=round(np.mean(previously_outside_now_inside_scaled),0)

        unscaled_overall_results = {
            "Unscaled overall results": None,
            "weighted average of coverage": round((unscaled_weighted_coverage/elements), 2),
            "weighted average of meanIoU": round((unscaled_weighted_meanIoU/elements), 2),
            "weighted average of dice coefficient": round((unscaled_weighted_DC/elements), 2),
            "Sum of means of gt inside conformalized bbox": unscaled_sum_gt_in_cp_box,
            "Sum interval score": round(unscaled_sum_interval_score, 2),
            "Sum turning false into true": unscaled_sum_false_into_true,
        }
        f.write("\n")
        f.write(json.dumps(unscaled_overall_results))

        scaled_overall_results = {
            "Scaled overall results": None,
            "weighted average of coverage": round((scaled_weighted_coverage/elements), 2),
            "weighted average of meanIoU": round((scaled_weighted_meanIoU/elements), 2),
            "weighted average of dice coefficient": round((scaled_weighted_DC/elements), 2),
            "Sum of means of gt inside conformalized bbox": scaled_sum_gt_in_cp_box,
            "Sum interval score": round(scaled_sum_interval_score, 2),
            "Sum turning false into true": scaled_sum_false_into_true,
        }
        f.write("\n")
        f.write(json.dumps(scaled_overall_results))

