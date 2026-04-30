# Probabilistic Object Detection with Conformal Prediction

Code accompanying the paper [![Paper](http://img.shields.io/badge/Paper-arXiv.2605.07549-B3181B?logo=arXiv)](https://arxiv.org/pdf/2605.07549.pdf) *"Probabilistic Object Detection with Conformal
Prediction"* (Ries, Kassem Sbeyti, Bianco, Klein), **under review**.

Implementation by [Christopher Ries (@chris-adr)](https://github.com/chris-adr).

The paper applies split conformal prediction (CP) to the structured,
multi-output predictions of an EfficientDet-d0 object detector, comparing
**unscaled** intervals against intervals **scaled by per-prediction aleatoric
uncertainty** estimated via loss attenuation. Three settings are evaluated
across KITTI, BDD, and CODA (the latter under domain shift):

1. **Class-agnostic regression CP** (Section 4.1.1, Table 1, Figure 2).
2. **Class-wise regression CP** with oracle classes (Section 4.1.2, Table 2).
3. **Two-step pipeline** that conformalizes the classification head with RAPS
   and conditions the bounding-box intervals on the predicted class set
   (Section 4.1.3, Tables 3 & 4 and Figure 3).

## Repository layout

```
.
├── configs/                       # YAML configs, one per dataset
│   ├── kitti.yaml
│   ├── bdd.yaml
│   └── coda.yaml
├── data/                          # Per-prediction validation dumps (see data/README.md)
├── results/                       # Reference numerical outputs (JSON-lines / JSON)
├── src/
│   ├── main.py                    # Entry point with --config and --experiment flags
│   ├── paths.py
│   ├── experiments/
│   │   ├── cp_regression.py             # Section 4.1.1, Table 1, Figure 2
│   │   ├── cp_classwise_regression.py   # Section 4.1.2, Table 2
│   │   ├── cp_two_step.py               # Section 4.1.3, Table 4
│   │   └── cp_classification.py         # Section 4.1.3, Table 3 + Figure 3 (raw arrays)
│   └── utils/
│       ├── cp_regression_utils.py       # CP score / quantile helpers for regression
│       ├── cp_classification_utils.py   # APS, RAPS, prediction-set helpers
│       ├── metrics.py                   # IoU, Dice, interval score, recovery counts
│       └── paired_t_test.py             # Paired t-tests on coverage / IoU / etc.
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

The code targets Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

`data/prediction_data_val_unc.txt` (KITTI) and
`data/validate_results_CODA.txt` (CODA) are bundled. The BDD dump is too
large to ship; see [data/README.md](data/README.md) for instructions on
regenerating it from the companion training repository
[mos-ks/uncertainty-detection-autolabeling](https://github.com/mos-ks/uncertainty-detection-autolabeling).
That repository also contains the EfficientDet-d0 + loss-attenuation training
code that produces these dumps.

## Reproducing the results

All experiments are launched through `src/main.py` and dispatched by a config
file plus one or more `--experiment` names:

| `--experiment` flag         | Paper reference     | Notes                                                                       |
| --------------------------- | ------------------- | --------------------------------------------------------------------------- |
| `cp_regression`             | Table 1, Figure 2   | Class-agnostic regression CP, sweeps `all_uncertainties` from the config.   |
| `cp_classwise_regression`   | Table 2             | Class-wise regression CP assuming oracle GT classes.                        |
| `cp_two_step`               | Table 4             | RAPS + class-conditional regression CP (KITTI in the paper).                |
| `cp_classification`         | Table 3 + Figure 3  | Dumps per-split coverage / set-size arrays for APS, RAPS, RAPS-non-empty.   |

Examples:

```bash
# Table 1 / Figure 2 results for KITTI
python -m src.main --config configs/kitti.yaml --experiment cp_regression

# Tables 1 + 2 for CODA in one run
python -m src.main --config configs/coda.yaml --experiment cp_regression cp_classwise_regression

# Two-step pipeline (Table 4) on KITTI
python -m src.main --config configs/kitti.yaml --experiment cp_two_step
```

Each experiment writes its outputs as JSON / JSON-lines under
`results/<DATASET>/`. Reference logs from the runs reported in the paper are
checked in alongside; re-running with the seed/split configuration shipped
here should reproduce them up to floating-point noise.

## Configs

Each dataset has its own YAML in `configs/`. The fields are documented inline;
the most relevant knobs are:

- `alpha_only_regression` — per-corner miscoverage for the regression-only
  experiments (1 − 4·α is the Bonferroni bbox-level guarantee).
- `alpha_regression` / `alpha_class` — per-corner / classification miscoverage
  for the two-step pipeline.
- `runs`, `split` — number of random calibration / evaluation splits and
  evaluation fraction (default 100 runs at 80/20).
- `uncertainty` / `all_uncertainties` — names of the aleatoric uncertainty
  fields in the prediction dumps. KITTI / CODA use `uncalib_albox`,
  `iso_all_albox`, `rel_iso_perclscoo_albox`; BDD uses
  `uncalib_uncert_albox` / `iso_alluncert_albox` / `rel_iso_perclscoo_albox`.

## Citation

```
@article{RiKaSbBiKl2026,
  title={Probabilistic Object Detection with Conformal Prediction},
  author={Christopher Ries and Moussa Kassem~Sbeyti and Nicolas Bianco and Nadja Klein},
  journal={arXiv preprint arXiv:2605.07549},
  year={2026}
}
```

## Acknowledgments

This work has been partially funded by the Pilot Program for Core Informatics
(KiKIT) at the KIT of the Helmholtz Association, with support from the state
of Baden-Württemberg through bwHPC.

## License

Released under the [MIT License](LICENSE).
