# A reproducible ablation benchmark for deep learning-based leukocyte classification

Code, data splits and analysis scripts for the manuscript *"A Reproducible Ablation Benchmark for Deep Learning-Based
Leukocyte Classification"* (Engineering Applications of Artificial Intelligence, under revision).

## What is benchmarked

A transfer-learning pipeline for four-class white blood cell classification (eosinophil, lymphocyte, monocyte,
neutrophil). One factor is changed per stage:

| Stage | Change |
|---|---|
| M1 | Frozen ImageNet MobileNetV2 + trained head |
| M2 | + CLAHE on the L\* channel |
| M3 | + fine-tuning of the last 30 backbone layers |
| M4 | EfficientNetB0 instead of MobileNetV2 |
| M5 | Focal loss (γ = 2, α = 0.25) instead of cross-entropy |
| M6 / M7 | M5 + 4-fold / 6-fold test-time augmentation |
| M8 | M5 + class-balanced sampling |
| Final | M8 + 6-fold test-time augmentation |
| Controls | M5 without CLAHE; M5 with a ResNet50 backbone |
| Leave-one-out | Final pipeline with one component removed: no CLAHE, no fine-tuning, MobileNetV2 backbone, cross-entropy instead of focal loss (trained); no balanced sampling (= M7) and no TTA (= M8) |

Every trained stage is run with seeds 0, 1 and 2. Predictions are stored for all eight dihedral (D4) views of every
validation and test image, so every TTA variant, calibration measure and McNemar test is computed offline.

## Datasets

* **PBC** (primary): Acevedo et al., Data in Brief 30 (2020) 105474, https://doi.org/10.17632/snkd93bnjr.1
  (Kaggle mirror: `unclesamulus/blood-cells-image-dataset`). Four classes, 8 exact duplicates removed (MD5),
  stratified 70/15/15 split with seed 42.
* **Blood Cell Images** (second benchmark and external test): https://www.kaggle.com/datasets/paultimothymooney/blood-cells.
  Predefined test split used unchanged; validation split grouped by the original-image index in the file names.

* **Raabin-WBC** (third laboratory, external test only): Kouzehkanan et al., Sci Rep 12 (2022) 1123,
  https://www.kaggle.com/datasets/raabindata/raabin-wbc. Official Test-A without basophils (4,250 images); the
  PBC-trained networks are applied without any retraining.

The exact image lists of every split are in `splits/pbc/` and `splits/kaggle/` (PBC: file name, class and MD5 hash of every image).

## Analyses added in the revision

* **Leave-one-out from the final pipeline** (PBC, 3 seeds): `experiments/07_...`, predictions in `predictions/pbc/preds_LOO_*`.
* **External test on Raabin-WBC** (no retraining): `experiments/07_...`, predictions in `predictions/raabin/`, values from `analysis/raabin.py`.
* **Holm correction**: exact McNemar p-values are Holm-adjusted within each seed across the 12 main comparisons; the
  leave-one-out comparisons form a separate family (`p_holm` in `results/res_P.json`).
* **Few-shot adaptation to a second laboratory** (Kaggle, k = 10 and 50 labelled images per class, 3 repetitions,
  ImageNet control): `experiments/06_...`, results in `results/fewshot.json`.
* **CPU cost** (2 cores): `experiments/05_cpu_benchmark.py`, results in `results/cpu_bench.json`.

## Repository layout

| Path | Content |
|---|---|
| `experiments/01_kaggle_dataset_audit.ipynb` | Audit of the Kaggle dataset (structure, file-name indices, near-duplicates) |
| `experiments/02_kaggle_ablation.py` | All stages on the Kaggle dataset, external test on PBC (also contains exploratory stages not reported in the paper: `P_faithful`, `M9`, `B_densenet121`) |
| `experiments/03_kaggle_dense_tta.py` | Dense rotation TTA on stored Kaggle models (exploratory) |
| `experiments/04_pbc_ablation.py` | All stages on PBC, external test on the Kaggle test split |
| `experiments/05_cpu_benchmark.py` | Operations per image and inference time on a two-core CPU |
| `experiments/07_raabin_external_test_and_leave_one_out_colab.py` | External test of the PBC-trained networks on Raabin-WBC Test-A (no retraining) and leave-one-out from the final pipeline |
| `experiments/06_pbc_seed2_and_fewshot_colab.py` | Google Colab version of the PBC run (resumable) plus the few-shot adaptation experiment (PBC-trained model fine-tuned on 10 or 50 Kaggle images per class) |
| `analysis/analyse.py` | Computes every reported number (accuracy, F1, AUC, recall, ECE, Brier, temperature scaling, exact McNemar, cost) from stored predictions |
| `analysis/figures.py` | Figures 2–5 of the paper |
| `predictions/` | `raabin/`: predictions of the PBC-trained networks on Raabin-WBC Test-A (`analysis/raabin.py` computes the reported values). Stored softmax outputs for all 8 D4 views of every validation and test image (`preds_<stage>_s<seed>.npz`) and training logs, for every stage and seed on both datasets; `python analysis/analyse.py P predictions/pbc res_P.json` regenerates the PBC results (copy `splits/pbc/meta_*.csv` into the folder first) |
| `results/` | `res_P.json`, `res_K.json`: every accuracy, calibration, significance and cost value reported in the paper; `fewshot.json` and `fewshot_draws/`: few-shot adaptation results and the exact images drawn; `cpu_bench.json`: CPU timing; `faithful_reimplementation_kaggle.json`: re-implementation of the original submission's pipeline |
| `splits/` | Image lists of each split |
| `literature/screening_log.csv` | Structured literature search: 21 screened records, decisions, reasons and factor coding |

## Reproducing

1. Attach both datasets as inputs on Kaggle (or download them with `kagglehub` in Google Colab) and select a GPU.
2. Run `experiments/04_pbc_ablation.py` and `experiments/02_kaggle_ablation.py`. Each writes `preds_<stage>_s<seed>.npz`,
   weights, logs and a results bundle; finished runs are skipped on restart.
3. Run `python analysis/analyse.py P <pbc_output_dir> res_P.json` and `python analysis/analyse.py K <kaggle_output_dir> res_K.json`.
4. Run `python analysis/figures.py res_P.json res_K.json figures/`.

Environment: Python 3, TensorFlow 2 / Keras 3, OpenCV, scikit-learn, SciPy, pandas, matplotlib (see `requirements.txt`).
Stored predictions for every stage and seed are in `predictions/`. The trained Keras weights of every PBC stage and seed, including the four trained leave-one-out configurations (36 files, about 1.5 GB, too large for GitHub), are available at https://drive.google.com/drive/folders/1XlvRiwtpqAaQpMYVWB0coJK2gmflanQZ

## Licence

MIT (code). The datasets keep their own licences.
