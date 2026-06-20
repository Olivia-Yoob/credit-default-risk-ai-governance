# Credit Default Risk — AI Governance Portfolio

An end-to-end credit-default prediction project built **governance-first**: the model is
treated as a **High-Risk AI system** under the **EU AI Act**, and every step is documented for
a **US Fair Lending (ECOA)** review. The differentiator versus a typical ML portfolio is the
explicit **regulatory mapping + Fair Lending audit**, not just model accuracy.

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
— 10 tables (~2.5 GB), 307,511 applicants, 8.07% default rate.

---

## Headline results

| | |
|---|---|
| **Model** | XGBoost on 96 features (59 main + 37 auxiliary aggregates) |
| **Performance** | PR-AUC **0.283** · ROC-AUC **0.781** (validation) |
| **Decision threshold** | **0.10** — cost-optimal (FN:FP = 10:1), ≈ Bayes optimum |
| **Recall on defaulters** | **0.63** at the cost-optimal threshold (vs 0.04 at 0.50) |
| **Fair Lending** | Adverse impact on **age** (DI 0.65) & **education** (DI 0.76); gender TPR gap 0.14 |
| **Mitigation** | Removing `CODE_GENDER` → PR-AUC −0.4% only, gender DI 0.83→0.90 |

## Pipeline (`notebooks/`)

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_understanding` | Inventory 10 tables, schema, register protected attributes, missingness |
| 02 | `02_eda_main_table` | Feature↔TARGET signals, `EXT_SOURCE`, data-quality landmines, subgroup default rates |
| 03 | `03_preprocessing_features` | Aggregate auxiliary tables, clean, leakage-free impute/scale, save model-ready matrix |
| 04 | `04_modeling` | LogReg / RandomForest / XGBoost; cost-sensitive threshold |
| 05 | `05_evaluation_fairlending` | ⭐ Fair Lending audit: disparate impact, equal opportunity, mitigation |

## Governance documentation (`docs/`)

- [`MODEL_CARD.md`](docs/MODEL_CARD.md) — model card (Mitchell et al. structure)
- [`EU_AI_ACT_COMPLIANCE.md`](docs/EU_AI_ACT_COMPLIANCE.md) — High-Risk classification & obligations mapping
- [`FAIR_LENDING_AUDIT.md`](docs/FAIR_LENDING_AUDIT.md) — full audit with findings & mitigation
- [`AI_RISK_FRAMEWORK.md`](docs/AI_RISK_FRAMEWORK.md) — risk register (R1–R8), three lines of defense
- [`DRIFT_MONITORING.md`](docs/DRIFT_MONITORING.md) — PSI + KRI-style control-limit monitoring plan
- [`NIST_AI_RMF_MAPPING.md`](docs/NIST_AI_RMF_MAPPING.md) — mapping to NIST AI RMF (Govern/Map/Measure/Manage)

## Repository layout

```
notebooks/   01–05 analysis pipeline (run in order)
docs/        governance deliverables
results/     generated figures (PNG)
src/         (reserved for reusable modules)
data/        raw Kaggle CSVs — NOT in git (regenerable; see below)
models/      trained model artifacts — NOT in git
```

## Reproducing

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # xgboost needs libomp on macOS: brew install libomp

# Download the dataset from Kaggle and unzip into ./data/
# Then run the notebooks in order (01 -> 05).
jupyter notebook
```

> **Data is git-ignored** (10 CSVs, ~2.5 GB) and regenerable from Kaggle. Processed matrices
> and trained models are also git-ignored and reproduced by running the notebooks.

## Disclaimer

A portfolio project for educational purposes. The compliance mappings are illustrative and
not legal advice.
