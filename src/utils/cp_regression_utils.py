from typing import Any
import numpy as np
import ast


def cp_scores_scaled(predicted_bbox: list[list[float]], gt_bbox: list[list[float]],
                     uncertainty: list[list[float]], eps: float=1e-8) -> list[list[float]]:
    """
    Function to calculate CP scores based on predicted, ground truth bounding boxes and the uncertainty for each coordinate individually.
    :param predicted_bbox: list of the predicted bounding boxes.
    :param gt_bbox: list of the ground truth bounding boxes.
    :param uncertainty: list of the uncertainty for each coordinate individually.
    :param eps: for safety, if uncertainty is zero --> we do not want to divide by zero
    :return: list of the CP scores.
    """
    differences = [
        [abs(a - b) for a, b in zip(b_entry, gt_entry)]
        for b_entry, gt_entry in zip(predicted_bbox, gt_bbox)
    ]
    return [[a / (b if b != 0 else eps) for a, b in zip(sub1, sub2)] for sub1, sub2 in zip(differences, uncertainty)]

def cp_scores_unscaled(predicted_bbox: list[list[float]],
                       gt_bbox: list[list[float]]) -> list[list[float]]:
    """
    Calculate vanilla CP scores
    :param predicted_bbox: list of the predicted bounding boxes.
    :param gt_bbox: list of the ground truth bounding boxes.
    :return: list of the CP scores.
    """
    differences = [
        [abs(a - b) for a, b in zip(b_entry, gt_entry)]
        for b_entry, gt_entry in zip(predicted_bbox, gt_bbox)
    ]
    return differences

def load_data(file_path: str) -> list[Any]:
    """
    Load files from Moussas format
    :param file_path: path to file
    :return: list of the entries
    """
    data_list = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    entry = ast.literal_eval(line)
                    data_list.append(entry)
                except Exception as e:
                    print(f"Skipping invalid line: {line}\nError: {e}")
    print(f"Loaded {len(data_list)} entries.")
    return data_list

def calculate_quantiles_for_each_dim(cp_scores: list[list[float]],
                                     alpha: float) -> tuple[float, float, float, float]:
    """
    Calculate quantiles for each dimension independently.
    :param cp_scores: cp scores as list
    :param alpha: parameter for the CP bound (1-alpha)
    :return:
    """
    cp_ymin = [x[0] for x in cp_scores]
    cp_xmin = [x[1] for x in cp_scores]
    cp_ymax = [x[2] for x in cp_scores]
    cp_xmax = [x[3] for x in cp_scores]
    n = len(cp_scores)
    quantile_cp = min(1, np.ceil((n + 1) * (1 - alpha)) / n)
    quantile_xmin = np.quantile(cp_xmin, quantile_cp, method="higher")
    quantile_xmax = np.quantile(cp_xmax, quantile_cp, method="higher")
    quantile_ymin = np.quantile(cp_ymin, quantile_cp, method="higher")
    quantile_ymax = np.quantile(cp_ymax, quantile_cp, method="higher")
    return float(quantile_xmin), float(quantile_xmax), float(quantile_ymin), float(quantile_ymax)

def inner_and_outer_box_scaled(quantile_ymin: float, quantile_xmin: float, quantile_ymax: float, quantile_xmax: float,
                               pred_bbox:list[list[float]], uncertainty_each_coordinate: list[list[float]]) \
        -> tuple[list[list[float]],list[list[float]]]:
    """
    Calculate inner and outer bounding box for scaled cp
    :param quantile_ymin:
    :param quantile_xmin:
    :param quantile_ymax:
    :param quantile_xmax:
    :param pred_bbox:
    :param uncertainty_each_coordinate:
    :return: two lists, containing the inner and outer bounding box for each entry
    """
    bbox_outer_scaled_cp = [[bbox[0]-uncertainty[0]*quantile_ymin,
                             bbox[1]-uncertainty[1]*quantile_xmin,
                             bbox[2]+uncertainty[2]*quantile_ymax,
                             bbox[3]+uncertainty[3]*quantile_xmax] for bbox, uncertainty in zip(pred_bbox, uncertainty_each_coordinate)]
    bbox_inner_scaled_cp = [[bbox[0]+uncertainty[0]*quantile_ymin,
                             bbox[1]+uncertainty[1]*quantile_xmin,
                             bbox[2]-uncertainty[2]*quantile_ymax,
                             bbox[3]-uncertainty[3]*quantile_xmax] for bbox, uncertainty in zip(pred_bbox, uncertainty_each_coordinate)]
    return bbox_outer_scaled_cp, bbox_inner_scaled_cp

def inner_and_outer_box_unscaled(quantile_ymin: float, quantile_xmin: float, quantile_ymax: float, quantile_xmax: float,
                                 pred_bbox: list[list[float]]) \
        -> tuple[list[list[float]],list[list[float]]]:
    """
    Calculate inner and outer bounding box for unscaled cp
    :param quantile_ymin:
    :param quantile_xmin:
    :param quantile_ymax:
    :param quantile_xmax:
    :param pred_bbox:
    :return: two lists, containing the inner and outer bounding box for each entry
    """
    bbox_outer_unscaled_cp = [[bbox[0]-quantile_ymin,
                               bbox[1]-quantile_xmin,
                               bbox[2]+quantile_ymax,
                               bbox[3]+quantile_xmax] for bbox in pred_bbox]
    bbox_inner_unscaled_cp = [[bbox[0]+quantile_ymin,
                               bbox[1]+quantile_xmin,
                               bbox[2]-quantile_ymax,
                               bbox[3]-quantile_xmax] for bbox in pred_bbox]
    return bbox_outer_unscaled_cp, bbox_inner_unscaled_cp


def inner_and_outer_bbox_using_cp_for_classification_and_regression_unscaled(pred_bbox:list[list[float]],
                                                                             max_class:list[int],
                                                                             quantiles_each_class:list[list[float]]
                                                                             ) -> tuple[list[list[float]],list[list[float]]]:
    selected_quantiles = [
        quantiles_each_class[idx] for idx in max_class
    ]
    inner = [
        (inner_and_outer_box_unscaled(q[0], q[1], q[2], q[3], [pred_bbox_single_instance]))[0][0]
        for (q, pred_bbox_single_instance) in zip(selected_quantiles, pred_bbox)
    ]
    outer = [
        (inner_and_outer_box_unscaled(q[0], q[1], q[2], q[3], [pred_bbox_single_instance]))[1][0]
        for (q, pred_bbox_single_instance) in zip(selected_quantiles, pred_bbox)
    ]

    return inner, outer


def inner_and_outer_bbox_using_cp_for_classification_and_regression_scaled(pred_bbox:list[list[float]],
                                                                           max_class:list[int],
                                                                           quantiles_each_class:list[list[float]],
                                                                           uncertainty_each_coordinate: list[list[float]],
                                                                           ) -> tuple[list[list[float]],list[list[float]]]:
    selected_quantiles = [
        quantiles_each_class[idx] for idx in max_class
    ]
    inner = [
        (inner_and_outer_box_scaled(q[0], q[1], q[2], q[3], [pred_bbox_single_instance], [uncertainty]))[0][0]
        for (q, pred_bbox_single_instance, uncertainty) in zip(selected_quantiles, pred_bbox, uncertainty_each_coordinate)
    ]
    outer = [
        (inner_and_outer_box_scaled(q[0], q[1], q[2], q[3], [pred_bbox_single_instance], [uncertainty]))[1][0]
        for (q, pred_bbox_single_instance, uncertainty) in zip(selected_quantiles, pred_bbox, uncertainty_each_coordinate)
    ]

    return inner, outer