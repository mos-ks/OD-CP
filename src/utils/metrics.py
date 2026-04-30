import numpy as np


# Adapted from https://github.com/adrian-lison/interval-scoring (matches the
# definition in scoringutils, see Gneiting & Raftery, JASA 2007).
def interval_score(observations, alpha: float, q_dict=None, q_left=None, q_right=None,
                   percent: bool = False, check_consistency: bool = True):
    """Compute interval scores for an array of observations and predicted intervals.

    Either ``q_dict`` (with the (alpha/2) and (1-(alpha/2)) quantiles) or both
    ``q_left`` and ``q_right`` must be supplied. Returns ``(total, sharpness, calibration)``.
    """
    if q_dict is None:
        if q_left is None or q_right is None:
            raise ValueError("Either quantile dictionary or left and right quantile must be supplied.")
    else:
        if q_left is not None or q_right is not None:
            raise ValueError("Either quantile dictionary OR left and right quantile must be supplied, not both.")
        q_left = q_dict.get(alpha / 2)
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha / 2}-quantile")
        q_right = q_dict.get(1 - (alpha / 2))
        if q_right is None:
            raise ValueError(f"Quantile dictionary does not include {1 - (alpha / 2)}-quantile")

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")

    sharpness = q_right - q_left
    calibration = (
        (
            np.clip(q_left - observations, a_min=0, a_max=None)
            + np.clip(observations - q_right, a_min=0, a_max=None)
        )
        * 2
        / alpha
    )
    if percent:
        sharpness = sharpness / np.abs(observations)
        calibration = calibration / np.abs(observations)
    total = sharpness + calibration
    return total, sharpness, calibration


def bb_intersection_over_union(boxA: list[float], boxB: list[float]) -> float:
    """Compute the Intersection over Union (IoU) of two axis-aligned bounding boxes."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(xB - xA, 0) * max(yB - yA, 0)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def dice_coefficient(boxA: list[float], boxB: list[float]) -> float:
    """Compute the Dice coefficient of two axis-aligned bounding boxes."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(xB - xA, 0) * max(yB - yA, 0)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return 2 * interArea / float(boxAArea + boxBArea)


def get_bins_index(sorted_list: list[object], bins: int = 10) -> list[int]:
    """Return the index of the middle element of each of `bins` equally-sized partitions."""
    n = len(sorted_list)
    results: list[int] = []
    for i in range(bins):
        start = int(i * n / bins)
        end = int((i + 1) * n / bins)
        if start < end:
            results.append((start + end) // 2)
    return results


def true_detections(bbox_outer_unscaled_cp: list[list[float]], bbox_inner_unscaled_cp: list[list[float]],
                    gt_bbox: list[list[float]]) -> list[bool]:
    return [
        (pred[0] <= gt[0] and pred[1] <= gt[1] and pred[2] >= gt[2] and pred[3] >= gt[3]
         and pred_inner[0] >= gt[0] and pred_inner[1] >= gt[1]
         and pred_inner[2] <= gt[2] and pred_inner[3] <= gt[3])
        for pred, pred_inner, gt in zip(bbox_outer_unscaled_cp, bbox_inner_unscaled_cp, gt_bbox)
    ]


def calculate_interval_score_for_whole_bboxes(inner: list[list[float]], outer: list[list[float]],
                                              gt_bbox: list[list[float]], alpha: float) -> float:
    y_min_inner = np.array([x[0] for x in inner])
    x_min_inner = np.array([x[1] for x in inner])
    y_max_inner = np.array([x[2] for x in inner])
    x_max_inner = np.array([x[3] for x in inner])

    y_min_outer = np.array([x[0] for x in outer])
    x_min_outer = np.array([x[1] for x in outer])
    y_max_outer = np.array([x[2] for x in outer])
    x_max_outer = np.array([x[3] for x in outer])

    y_min = np.array([x[0] for x in gt_bbox])
    x_min = np.array([x[1] for x in gt_bbox])
    y_max = np.array([x[2] for x in gt_bbox])
    x_max = np.array([x[3] for x in gt_bbox])

    (y_min_interval_scores, _, _) = interval_score(y_min, alpha, q_left=y_min_outer, q_right=y_min_inner)
    (x_min_interval_scores, _, _) = interval_score(x_min, alpha, q_left=x_min_outer, q_right=x_min_inner)
    (y_max_interval_scores, _, _) = interval_score(y_max, alpha, q_left=y_max_inner, q_right=y_max_outer)
    (x_max_interval_scores, _, _) = interval_score(x_max, alpha, q_left=x_max_inner, q_right=x_max_outer)
    return sum(y_min_interval_scores + x_min_interval_scores + y_max_interval_scores + x_max_interval_scores)


def count_newly_covered_predictions(pred: list[list[float]], cp_outer: list[list[float]],
                                    gt_bbox: list[list[float]]) -> int:
    count = 0
    for (p, c, gt) in zip(pred, cp_outer, gt_bbox):
        gt_covered_by_pred = (p[0] <= gt[0] and p[1] <= gt[1] and p[2] >= gt[2] and p[3] >= gt[3])
        if not gt_covered_by_pred:
            gt_covered_by_cp = (c[0] <= gt[0] and c[1] <= gt[1] and c[2] >= gt[2] and c[3] >= gt[3])
            if gt_covered_by_cp:
                count += 1
    return count


def iou_below_threshold_now_inside_cp_bboxes(pred: list[list[float]], gt_bbox: list[list[float]], threshold: float,
                                             cp_outer: list[list[float]], cp_inner: list[list[float]]) -> tuple[int, int]:
    now_inside = 0
    how_many_below = 0
    for p, gt, c_in, c_out in zip(pred, gt_bbox, cp_inner, cp_outer):
        value = bb_intersection_over_union(p, gt)
        is_in_cp = (c_out[0] <= gt[0] and c_out[1] <= gt[1] and c_out[2] >= gt[2] and c_out[3] >= gt[3]
                    and c_in[0] >= gt[0] and c_in[1] >= gt[1] and c_in[2] <= gt[2] and c_in[3] <= gt[3])
        if value <= threshold and is_in_cp:
            now_inside += 1
        if value <= threshold:
            how_many_below += 1
    return how_many_below, now_inside
