# Nanocrystal Equation Learner (NanoEQL)

This repository implements NanoEQL, a fully white-box neural network framework for predicting nanocrystal size from synthesis recipes and unraveling the underlying size-determination mechanisms. The model reveals that nanocrystal size obeys a simple linear equation governed by three physically interpretable scalars: nanocrystallization capability (−Z_p​), growth capability (Z_rea​), and external input potential (−Z_ops​). Beyond nanocrystal synthesis, the framework generalizes to organic reaction systems (e.g., Buchwald–Hartwig reactions).

## Core Files

- `dataset.py`: data loading and preprocessing, including descriptor lookup, operation-feature scaling, and morphology feature encoding.
- `model.py`: defines `EQLLayer` (with smoothed reciprocal, square-root, and cube-root operators), `AttentivePNAPooling` (temperature-gated attention), and `NanoEQLModel`.
- `train_stage1.py`: trains the neural proxy with sparse regularization and exports intermediate features.
- `extract_formulas.py`: extracts interpretable formulas and pruning statistics from the model.

## Environment Requirements

Recommended environment:

- Python 3.10+
- PyTorch with CUDA support if GPU training is needed
- `openpyxl` for reading `.xlsx` datasets

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The repository uses the following libraries:

- Core training/data stack: `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `optuna`, `openpyxl`
- Analysis and visualization: `sympy`, `matplotlib`, `tqdm`
- Traditional ML baselines: `xgboost`, `lightgbm`, `catboost`

## Workflow

### 1. Train the model

```bash
python train_stage1.py --run_name default_run
```

This stage trains the multi-branch EQL model and saves model checkpoints, metrics, and exported latent features under `results/<run_name>/`.

### 2. Extract formulas from the model

```bash
python extract_formulas.py
```

This script reconstructs sparse analytical expressions from the trained EQL layers and can optionally export additional artifacts such as pruning metrics and operator-usage statistics.

## Model Summary

The architecture contains three interpretable branches that output the scalars Z_p​, Z_rea​, and Z_ops​ :

- `h_prod`(product branch): compresses product descriptors and morphology information through an EQL network to output Z_p​ (nanocrystallization capability).
- `g_joint`(reaction branch): captures cross-reactant chemistry through early EQL dimensionality reduction, temperature-gated attention pooling, and late fusion to output Z_rea​ (growth capability).
- `g_ops` (operation branch): encodes operation descriptors (injection temperature, reaction temperature, and reaction time) through an EQL network to output Z_ops​ (external input potential).

At the pooling stage, the temperature-gated attention mechanism dynamically balances two contributions:

- a concentration-driven distribution derived from reactant amounts, and
- a reactivity-driven attention score inferred from reactant features and operation conditions.

## Outputs

Typical outputs include:

- trained weights (`eql_model.pth`)
- metrics files (`stage1_metrics.txt`)
- exported latent features (`train_features.csv`, `test_features.csv`)
- extracted symbolic formulas (`stage1_formulas.txt`)
- pruning and operator-usage statistics from `extract_formulas.py`

## Citation
If you find this project helpful, please cite our article:
```
xxxx
```
If you have any questions, please contact kai_gu94@163.com
