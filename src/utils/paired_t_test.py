import scipy

def run_paired_tests(label_a, label_b,
                     coverage_a, coverage_b,
                     interval_a, interval_b,
                     dice_a, dice_b,
                     iou_a, iou_b):
    results = f"Paired t-test for {label_a} and {label_b}. "

    _, p = scipy.stats.ttest_rel(coverage_a, coverage_b)
    results += f"Coverage p-value: {p}. "

    _, p = scipy.stats.ttest_rel(interval_a, interval_b)
    results += f"Interval Score p-value: {p}. "

    _, p = scipy.stats.ttest_rel(dice_a, dice_b)
    results += f"Dice Coefficient p-value: {p}. "

    _, p = scipy.stats.ttest_rel(iou_a, iou_b)
    results += f"IuO p-value: {p}. "

    return results
