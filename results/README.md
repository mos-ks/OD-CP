# Results

Reference numerical outputs for the experiments reported in the paper. Each
file is a JSON-lines log written by the corresponding script in
[src/experiments/](../src/experiments/); plots are not bundled — render them
from the values here with whatever styling you prefer.

```
results/
├── KITTI/
│   ├── cp_only_regression/                    # Tables 1 & 2; Figure 2
│   │   ├── results_alpha_0.025_runs_100.txt              # Figure 2 (alpha = 0.025)
│   │   ├── results_alpha_0.1_runs_100.txt                # Table 1, Figure 2 (alpha = 0.1)
│   │   └── results_class-wise_alpha_0.1_runs_100_split_0.2.txt   # Table 2
│   └── cp_for_class_and_bbox/                # Table 4 (two-step pipeline)
│       ├── results_cp_for_classes_and_bboxes_alpha_0.05_runs_100_split_0.2.txt
│       └── results_cp_for_classes_and_bboxes_alpha_0.1_runs_100_split_0.2.txt
├── BDD/
│   └── cp_only_regression/
│       ├── results_alpha_0.025_runs_100.txt              # Figure 2 (alpha = 0.025)
│       ├── results_alpha_0.1_runs_100.txt                # Table 1, Figure 2 (alpha = 0.1)
│       └── results_class-wise_alpha_0.1_runs_100_split_0.2.txt   # Table 2
└── CODA/
    └── cp_only_regression/
        ├── results_alpha_0.025_runs_100.txt              # Figure 2 (alpha = 0.025)
        ├── results_alpha_0.1_runs_100.txt                # Table 1, Figure 2 (alpha = 0.1)
        └── results_class-wise_alpha_0.1_runs_100_split_0.2.txt   # Table 2
```

The two-step pipeline is restricted to KITTI in the paper (see Section
4.1.3); BDD and CODA therefore have no `cp_for_class_and_bbox/` directory.
Classification CP outputs (Table 3 + Figure 3) are not pre-computed and will
be written under `results/<DATASET>/cp_classification/` the first time
`cp_classification` is run.

The metric definitions match Section 4 of the paper: coverage, IoU, Dice
coefficient, interval score, and the per-IoU-threshold recovery counts used
in Figure 2.
