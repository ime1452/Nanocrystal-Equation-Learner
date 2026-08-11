# Nanocrystal Equation Learner (NanoEQL)

This repository implements a symbolic representation learning framework for predicting the final size of nanocrystals from synthesis recipes. The model follows a three-stage design:

1. learn compact representations for products and reactants,
2. aggregate inorganic and organic reactants with attentive pooling,
3. fuse the learned latent variables for final prediction and symbolic interpretation.

The project is centered around the `NanoEQLModel` neural proxy and a formula-extraction workflow that converts sparse EQL weights into explicit analytical expressions.

## Core Files

- `dataset.py`: data loading and preprocessing, including descriptor lookup, operation-feature scaling, and morphology feature encoding.
- `model.py`: defines `EQLLayer`, `AttentivePNAPooling`, and `NanoEQLModel`.
- `train_stage1.py`: trains the neural proxy with sparse regularization and exports intermediate features.
- `extract_formulas.py`: extracts interpretable formulas and pruning statistics from the trained stage-1 model.

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

### 1. Train the stage-1 neural proxy

```bash
python train_stage1.py --run_name default_run
```

This stage trains the multi-branch EQL model and saves model checkpoints, metrics, and exported latent features under `results/<run_name>/`.

### 2. Extract formulas from the trained model

```bash
python extract_formulas.py
```

This script reconstructs sparse analytical expressions from the trained EQL layers and can optionally export additional artifacts such as pruning metrics and operator-usage statistics.

## Model Summary

The architecture contains three main branches:

- `h_prod`: compresses product descriptors and morphology information
- `g_joint`: captures cross-reactant chemistry through early EQL, attentive PNA pooling, and late fusion
- `g_ops`: encodes operation descriptors such as temperature and time

At the pooling stage, the model combines:

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
