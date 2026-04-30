from typing import Any
import numpy as np
from numpy import bool, dtype, ndarray


def cp_scores_lac(probabilities: list[list[float]], gt_class: list[int]) -> list[float]:
    # using the 1 - softmax output as a CP functions
    values_for_gt_class = [(1- prob[gt_class-1]) for (prob, gt_class) in zip(probabilities, gt_class)]
    return values_for_gt_class


def quantiles_for_classes(cp_scores_classes: list[float], alpha: float) -> float:
    n = len(cp_scores_classes)
    quantile_cp = np.ceil((n + 1) * (1 - alpha)) / n
    quantile_hat = np.quantile(cp_scores_classes, quantile_cp, method="higher", axis=0) #check axis again
    return float(quantile_hat)


def quantile_aps(probabilities: list[list[float]], gt_class: list[int], alpha: float) -> float:
    gt_class_shifted = [entry-1 for entry in gt_class]
    cal_smx = np.asarray(probabilities, dtype=float)
    cal_labels = np.asarray(gt_class_shifted, dtype=int)
    n = len(cal_labels)
    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    # Get the score quantile
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    return float(qhat)


def get_prediction_set(probabilities: list[list[float]], qhat: float) -> ndarray[tuple[Any, ...], dtype[bool]]:
    val_smx = np.asarray(probabilities, dtype=float)
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    return prediction_sets


def quantile_raps(probabilities: list[list[float]], gt_class: list[int], alpha: float, lam_reg: float,
                  k_reg: int, random_number: int = None) -> tuple[float, ndarray[tuple[Any, ...], Any]]:
    if random_number is None:
        rng = np.random.default_rng(42)
    else:
        rng = np.random.default_rng(random_number)

    gt_class_shifted = [entry-1 for entry in gt_class]
    cal_smx = np.asarray(probabilities, dtype=float)
    reg_vec = np.array(k_reg*[0,] + (cal_smx.shape[1]-k_reg)*[lam_reg,])[None,:]
    cal_labels = np.asarray(gt_class_shifted, dtype=int)
    n = len(cal_labels)
    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    #cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - rng.random(n)*cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    qhat = np.quantile(
        cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher'
    )
    return float(qhat), reg_vec


def get_prediction_set_raps(probabilities: list[list[float]], qhat: float, reg_vec, randomNumber: int,
                            disallow_zero_sets: bool = False, rand: bool = True) -> ndarray[tuple[Any, ...], dtype[bool]]:
    #np.random.seed(randomNumber)
    rng = np.random.default_rng(randomNumber)
    n_val = len(probabilities)
    val_smx = np.asarray(probabilities, dtype=float)
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)  #can be deleted
    indicators = (val_srt_reg.cumsum(axis=1) - rng.random((n_val,1))*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets:
        indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    return prediction_sets


def coverage_for_class(prediction_sets: ndarray[tuple[Any, ...], dtype[bool]], gt_class: list[int]) -> float:
    gt_class_shifted = np.asarray([entry-1 for entry in gt_class], dtype=int)
    return prediction_sets[np.arange(len(gt_class)),gt_class_shifted].mean()
