# equipment-rul — Claude Code Instructions

## Project Overview

Streamlit demo for Remaining Useful Life (RUL) prediction on rotating equipment.
PyTorch LSTM trained on synthetic CMAPSS-style degradation data.

Live demo: https://royabes-equipment-rul.streamlit.app/
GitHub: https://github.com/royabes/equipment-rul

## Git Commit Rules

- NEVER include Co-Authored-By lines or any AI attribution in commits
- NEVER add Claude as contributor, co-author, or collaborator
- Roy's name only on all commits

## Running Locally

```bash
.venv/bin/streamlit run app.py --server.port 38501
```

## Retraining

```bash
# With NASA CMAPSS (download to data/CMAPSSData/ first):
.venv/bin/python train.py

# Synthetic only (always works):
.venv/bin/python train.py  # auto-detects missing NASA data
```

## Key Files

- `app.py` — Streamlit app (3 tabs: Live Demo, Upload, About)
- `model.py` — PyTorch LSTM definition
- `preprocess.py` — data loading, windowing, upload parsing
- `train.py` — training script
- `models/rul_model.pt` — pre-trained weights (commit this)
- `models/scaler.pkl` — MinMaxScaler (commit this)
- `data/sample_upload.csv` — upload template for users

## Deployment

Streamlit Cloud: share.streamlit.io → royabes/equipment-rul → app.py
