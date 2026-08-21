# TLM: Tessellated Linear Model for Age Prediction from Voice

This is the official implementation of **Tessellated Linear Model (TLM)** from the paper: [TLM: Tessellated Linear Model for Age Prediction from Voice](https://arxiv.org/pdf/2501.09229)

## Overview

Voice biometric tasks, such as age estimation require modeling the often complex relationship between voice features and the biometric variable. While deep learning models can handle such complexity, they typically require large amounts of accurately labeled data to perform well. Such data are often scarce for biometric tasks such as voice-based age prediction. On the other hand, simpler models like linear regression can work with smaller datasets but often fail to generalize to the underlying non-linear patterns present in the data. In this paper we propose the Tessellated Linear Model (TLM), a piecewise linear approach that combines the simplicity of linear models with the capacity of non-linear functions. TLM tessellates the feature space into convex regions and fits a linear model within each region. We optimize the tessellation and the linear models using a hierarchical greedy partitioning. We evaluated TLM on the TIMIT dataset on the task of age prediction from voice, where it outperformed state-of-the-art deep learning models.

## Features

- **TLM Model**: Tessellated Linear Model with hard/soft routing
- **Oracle Mode**: Perfect routing using true age labels (upper bound performance)
- **Baseline Models**: Random Forest, K-means+Regression, Linear Regression, Neural Network, Commonsense (mean)
- **Age-based Mixup**: Data augmentation technique for improved performance

## Installation

```bash
pip install -r requirements.txt
```

## Data

Download the required data:

1. **Pre-trained TLM Model**: [Download](https://drive.google.com/file/d/1HKaq5sS6kFOJaZXcCDsw9AF6z4NqF8Oy/view?usp=drive_link)
2. **TitaNet Embeddings for TIMIT**: [Download](https://drive.google.com/drive/folders/1wi2kI-S3cMZ8vHZdvMiUfBHwEvnSNvLm?usp=drive_link)

## Usage

### Basic Usage

```bash
python main.py <data_folder> <train_metadata> <test_metadata>
```

### With Oracle Mode

```bash
python main.py <data_folder> <train_metadata> <test_metadata> --oracle
```

### Example

```bash
python main.py data/titanet/timit metadata/timit_train.txt metadata/timit_test.txt --oracle --max_depth 4
```

## Arguments

- `data_folder`: Path to folder containing embeddings (.npy files)
- `train_metadata`: Training metadata file (format: file_path|speaker_id|age)
- `test_metadata`: Test metadata file (format: file_path|speaker_id|age)
- `--oracle`: Use oracle mode (skip classifier training, use true labels)
- `--max_depth`: Maximum tree depth (default: 4)
- `--seed`: Random seed for the (stochastic) age-based mixup (default: 0)
- `--skip_baselines`: Evaluate TLM only, skip the baseline models

## Reproducing the paper results

The reported numbers use the **192-d TitaNet-Large** embeddings on **TIMIT**.

**Table I — base TLM (hard / soft routing → 4.09 / 4.02):**

```bash
python main.py data/titanet/timit metadata/timit_train.txt metadata/timit_test.txt \
    --skip_baselines --seed 1
```

`--seed 1` reproduces the paper's Table I base rows essentially exactly:

| routing | MAE | RMSE | paper |
|---------|-----|------|-------|
| hard | 4.10 | 5.62 | 4.09 / 5.62 |
| soft | 4.03 | 5.44 | 4.02 / 5.49 |

Age-based mixup draws from numpy's global RNG, so a single run is one sample from a
distribution: across seeds the hard-routing MAE lands in ~4.0–4.2 and soft-routing in
~3.95–4.05, centred on the paper's single-run values. `--seed` fixes a run.

**Table I — feature optimization (→ 3.97):**

```bash
python feature_optim.py data/titanet/timit metadata/timit_train.txt metadata/timit_test.txt \
    --seed 1
```

This grows the base tree, freezes the logistic routers, and trains a residual feature
network (`ResBlock`×2) together with the leaf regressors so the tessellation predicts
age better on the reshaped features (hard, differentiable routing; plain MSE). It runs a
fixed `--epochs` (default 1000) and evaluates on test once at the end; with `--seed 1` it
reaches **MAE 3.99 / RMSE 5.51**, reproducing the paper's 3.97 row.

**Table I — oracle upper bound (→ 0.49):** add `--oracle` to `main.py`.

> Note on data path: `data_folder` is prepended to each metadata path, so with a metadata
> line `TIMIT/TRAIN/.../SX452.WAV` and embeddings under `data/titanet/timit/TIMIT/...`,
> pass `data/titanet/timit`. Adjust the path to wherever you extracted the TitaNet `.npy`
> files (the Drive download above, or the repo's top-level `data/titanet/timit`).

## Output

The script evaluates and compares:

- **Baseline Models**: Commonsense, Random Forest, K-means+Regression, Linear Regression, Neural Network
- **TLM Normal**: Hard routing + Soft routing
- **TLM Oracle**: Hard routing with perfect classification (if --oracle flag used)

Results are printed with MAE and RMSE metrics for all methods.

## File Structure

```
tlm/
├── main.py              # Main script (base TLM: hard/soft/oracle)
├── feature_optim.py     # Feature-optimization pipeline (paper row: 3.97)
├── data_utils.py        # Data loading and age-based mixup augmentation
├── models/
│   ├── tlm.py          # TLM implementation
│   └── baselines.py    # Baseline models
├── metadata/           # Metadata files
├── README.md
└── requirements.txt
```

## Citation

```bibtex
@inproceedings{alharthi2025tessellated,
  title={Tessellated Linear Model for Age Prediction from Voice},
  author={Alharthi, Dareen and Zamani, Mahsa and Raj, Bhiksha and Singh, Rita},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--4},
  year={2025},
  organization={IEEE}
}
```
