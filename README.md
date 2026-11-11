# Prediction of enantioselectivity in the hydrogen-bond-donor-catalyzed asymmetric Mannich reaction with machine learning

Data and code accompanying the manuscript

> K. V. Vasechkin, I. D. Nikitin, E. A. Shirokova, M. V. Fedorov.
> *Prediction of enantioselectivity in the hydrogen-bond-donor-catalyzed asymmetric Mannich reaction with machine learning* (submitted).

The repository contains the reaction dataset used in the study, all molecular and reaction representations,
DFT geometries and ORCA inputs, the modelling notebooks, fold assignments and the numerical results
(metrics, out-of-fold predictions, hyperparameters, SHAP importances).

## Dataset

`data/processed/mannich_dataset_hbd.csv` — 1664 organocatalytic asymmetric Mannich reactions catalysed by
hydrogen-bond donors (thioureas, ureas, squaramides, guanidines): 63 publications, 43 research groups,
202 catalysts, 296 electrophiles, 289 nucleophiles, 34 solvents. Target: |ΔΔG‡| (`ddG_kcal_mol`, kcal/mol),
computed from `ee` and the reaction temperature.

The set is the hydrogen-bond-donor subset of the published Mannich enantioselectivity dataset
(Zenodo, [10.5281/zenodo.21568582](https://doi.org/10.5281/zenodo.21568582)). Reactant–product consistency
of every record was checked manually (1677 → 1664 reactions), and reactions were atom-mapped with a
template for the Mannich bond formation (`mapped_rxn`).

Main columns: `reaction_id`; `paper_doi`, `research_group`; `electrophile`, `nucleophile`, `organocatalyst`
(SMILES; counter-ions removed), `product`, with identifiers `electrophile_id`, `nucleophile_id`, `catalyst_id`;
`axial_configuration` (R/S/none, atropisomeric catalysts — not encoded in SMILES); `ee`, `ddG_kcal_mol`, `dr`,
`syn/anti`, `yield`; conditions (`solvent`, `temperature` in °C, `time` in h, catalyst loading, additives);
`mapped_rxn` (atom-mapped reaction SMILES) and `reaction_smiles_with_ocat`
(`electrophile.nucleophile>catalyst>product`). A catalyst is identified by SMILES together with
`axial_configuration`.

## Repository layout

```
data/
  processed/          mannich_dataset_hbd.csv
  representations/    features_*.csv (one row per reaction, keyed by reaction_id), onehot_legend.csv,
                      solvent_parameters.csv
dft/
  geometries/         r2SCAN-3c optimised geometries (.xyz): catalysts/, electrophiles/, nucleophiles/
  orca_inputs/        ORCA inputs: 01 conformer ranking, 02 optimisation, 03 single points
notebooks/            analysis notebooks (see below)
splits/               fold assignments shared by all models
results/
  regression/         metrics_summary.csv, predictions_oof.csv, hyperparameters, manifest.json
  classification/     metrics_summary.csv, class_balance.csv, hyperparameters, manifest.json
  shap/               SHAP importances of the DFT models (regression and classification)
environment.yml       conda environment
```

## Installation

```bash
conda env create -f environment.yml
conda activate mannich-hbd
```

Python 3.12; package versions are pinned to those used for the reported results.
PyTorch is needed only for the chemprop D-MPNN.

## Notebooks

Run from the `notebooks/` folder; all paths are relative to it. Notebooks are stored without outputs.

| Order | Notebook | Input | Output |
|---|---|---|---|
| 1 | `mannich_representations.ipynb` | dataset | `data/representations/features_{physchem,maccs,ecfp4,ecfp4_count,drfp,onehot,conditions}.csv` |
| 2 | `mannich_regression.ipynb` | dataset, representations, `splits/` | `results/regression/` |
| 3 | `mannich_classification.ipynb` | dataset, representations, `splits/` | `results/classification/` |
| 4 | `mannich_shap.ipynb` | DFT representation, results of 2 and 3 | `results/shap/` |
| 5 | `heatmaps_*.ipynb` | `results/*/metrics_summary.csv` | `results/figures/` |

The DFT representation (`features_dft.csv`) is provided precomputed (see below).


## Representations

All representations are combined with the reaction conditions (`features_conditions.csv`: temperature,
catalyst loading, Kamlet–Taft α, β, π* and dielectric constant of the solvent; values in
`solvent_parameters.csv`, Marcus 1993).

| Representation | Columns | Description |
|---|---|---|
| `physchem` | 650 | RDKit 2D descriptors of catalyst, electrophile and nucleophile |
| `maccs` | 500 | MACCS keys of the three components |
| `ecfp4`, `ecfp4_count` | 6146 | Morgan fingerprints (radius 2, 2048 bits, chirality included), binary and count |
| `drfp` | 2050 | DRFP of `reaction_smiles_with_ocat` (catalyst as agent) |
| `onehot` | 787 | identity of catalyst, electrophile and nucleophile (legend: `onehot_legend.csv`) |
| `dft` | 50 | DFT descriptors (below) |
| CGR + catalyst | — | chemprop D-MPNN on `mapped_rxn` (condensed graph of reaction) and the catalyst graph |

Catalyst blocks contain the flags `cat_axial_R` / `cat_axial_S`.

**DFT descriptors.** Conformers: ETKDGv3 with MMFF94 pre-screening; ranking by r2SCAN-3c single points
(`dft/orca_inputs/01_*`); optimisation and frequencies of the lowest conformer at r2SCAN-3c (`02_*`);
single point at M06-2X-D3(0)/def2-TZVP with RIJCOSX (`03_*`); gas phase; ORCA 6.1.1. NPA charges from
JANPA 2.02. Electronic and steric descriptors are taken at SMARTS-defined reaction centres of each
component (48 descriptors + 2 axial flags). Sterimol parameters are used for the nucleophile only;
for catalysts and electrophiles they are undefined for part of the structures (bond in a ring, missing
donor N–H) and were excluded. The two remaining missing values (`cat_q_N_base`, `cat_d_Hch_Nbase`) are
imputed with the training-fold median. Descriptor definitions are given in the Supporting Information,
Table S1.

## Models and validation

- Regression (|ΔΔG‡|): Dummy (mean), Ridge, SVR, Gaussian process, Random Forest, XGBoost, chemprop D-MPNN.
- Classification (ee ≥ 70, 80, 90 %): logistic regression, SVC, Random Forest, XGBoost, chemprop D-MPNN.
- Splits (`splits/`): random 5-fold × 5 repeats; GroupKFold-10 by catalyst; leave-one-research-group-out
  (LORGO; `lorgo_group` in file names).
- Hyperparameters: randomized search (20 candidates, 5 inner folds following the outer split), tuned once
  per split, representation and model on the training part of the first outer fold.
- Metrics: MAE, RMSE, R², Spearman ρ and Kendall τ (regression); ROC-AUC, PR-AUC, MCC, F1 (classification);
  Pooled rank metrics under LORGO include the between-fold shift of the training mean and are not a measure
  of ranking within a research group.


## Citation

Please cite the manuscript above and, for the underlying data, the dataset paper and the Zenodo deposit
([10.5281/zenodo.21568582](https://doi.org/10.5281/zenodo.21568582)).
