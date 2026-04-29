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

## Output

The script evaluates and compares:

- **Baseline Models**: Commonsense, Random Forest, K-means+Regression, Linear Regression, Neural Network
- **TLM Normal**: Hard routing + Soft routing
- **TLM Oracle**: Hard routing with perfect classification (if --oracle flag used)

Results are printed with MAE and RMSE metrics for all methods.

## File Structure

```
tlm/
├── main.py              # Main script
├── data_utils.py        # Data loading and augmentation
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
