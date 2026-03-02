# app.py — Equipment RUL Prediction — Streamlit Demo
# Run: streamlit run app.py
# Deploy: streamlit.io/cloud (free)

import io
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

import streamlit as st

from model import RUL_LSTM
from preprocess import (
    generate_synthetic_cmapss, add_rul_labels, fit_scaler, scale,
    make_windows, make_test_windows, preprocess_upload,
    make_sample_upload_csv, generate_demo_windows,
    SENSOR_COLS, UPLOAD_COLS, WINDOW, MAX_RUL,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EquipmentRUL — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH  = Path("models/rul_model.pt")
SCALER_PATH = Path("models/scaler.pkl")

# ─── Colours ──────────────────────────────────────────────────────────────────
GREEN  = "#2ecc71"
YELLOW = "#f39c12"
RED    = "#e74c3c"
BLUE   = "#3498db"
BG     = "#0e1117"


# ─── Model loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_scaler():
    """Load pre-trained model + scaler, or train on synthetic data if missing."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        model = RUL_LSTM(input_size=len(SENSOR_COLS))
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model, scaler, "pre-trained"
    else:
        # Train on synthetic data (Streamlit Cloud path)
        st.toast("Training on synthetic data — first load only (~30s)")
        df = generate_synthetic_cmapss(n_units=80)
        scaler = fit_scaler(df)
        df = scale(df, scaler)
        X, y = make_windows(df)
        model = RUL_LSTM(input_size=len(SENSOR_COLS))
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        Xt = torch.FloatTensor(X)
        yt = torch.FloatTensor(y).unsqueeze(1)
        model.train()
        for _ in range(40):
            optimiser.zero_grad()
            loss_fn(model(Xt), yt).backward()
            optimiser.step()
        model.eval()
        return model, scaler, "synthetic"


@st.cache_data(show_spinner="Generating demo data...")
def get_demo_data(_scaler):
    """Generate 10 demo units sampled at various lifecycle stages."""
    X, unit_ids, true_ruls = generate_demo_windows(_scaler, n_units=10, seed=7)
    return X, unit_ids, true_ruls


def predict_units(X_windows: np.ndarray, model: RUL_LSTM) -> np.ndarray:
    with torch.no_grad():
        return model(torch.FloatTensor(X_windows)).numpy().flatten()


def rul_colour(rul: float) -> str:
    if rul > 80:   return GREEN
    if rul > 40:   return YELLOW
    return RED


def status_label(rul: float) -> str:
    if rul > 80:   return "Healthy"
    if rul > 40:   return "Monitor"
    return "Critical"


def status_icon(rul: float) -> str:
    if rul > 80:   return "✅"
    if rul > 40:   return "⚠️"
    return "🔴"


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_health_curve(group: pd.DataFrame, pred_rul: float, unit_id) -> plt.Figure:
    """Health degradation curve for one engine/unit."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=BG)

    # Left: sensor trend (first 3 sensors as proxy for health)
    ax = axes[0]
    ax.set_facecolor(BG)
    cycles = group["cycle"].values
    for i, col in enumerate(SENSOR_COLS[:3]):
        if col in group.columns:
            ax.plot(cycles, group[col].values,
                    alpha=0.7, linewidth=1.2, label=col)
    ax.set_title("Sensor Trends (last 60 cycles)", color="white", fontsize=10)
    ax.set_xlabel("Cycle", color="gray")
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333")
    ax.legend(fontsize=7, labelcolor="white", framealpha=0.2)

    # Right: RUL gauge bar
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    pct = min(pred_rul / MAX_RUL, 1.0)
    colour = rul_colour(pred_rul)
    ax2.barh(0, pct, color=colour, height=0.4)
    ax2.barh(0, 1 - pct, left=pct, color="#2a2a2a", height=0.4)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "25%", "50%", "75%", "100%"], color="gray", fontsize=8)
    ax2.set_yticks([])
    ax2.spines[:].set_color("#333")
    ax2.set_title(
        f"Predicted RUL: {pred_rul:.0f} cycles  {status_icon(pred_rul)}  {status_label(pred_rul)}",
        color=colour, fontsize=11
    )
    ax2.set_xlabel("Remaining life", color="gray")

    plt.tight_layout()
    return fig


def plot_fleet_overview(unit_ids, predictions: np.ndarray) -> plt.Figure:
    """Horizontal bar chart of all units coloured by health."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(unit_ids) * 0.45)), facecolor=BG)
    ax.set_facecolor(BG)
    sorted_idx = np.argsort(predictions)
    for rank, idx in enumerate(sorted_idx):
        rul = predictions[idx]
        ax.barh(rank, rul, color=rul_colour(rul), height=0.6)
        ax.text(rul + 1, rank, f"{rul:.0f}", va="center", color="white", fontsize=8)
    ax.set_yticks(range(len(unit_ids)))
    ax.set_yticklabels([f"Unit {unit_ids[i]}" for i in sorted_idx], color="gray", fontsize=8)
    ax.set_xlabel("Predicted RUL (cycles)", color="gray")
    ax.set_title("Fleet Health Overview", color="white")
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333")
    ax.axvline(40, color=RED,    linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(80, color=YELLOW, linestyle="--", alpha=0.5, linewidth=1)
    plt.tight_layout()
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ EquipmentRUL")
        st.caption("Predictive Maintenance • Grounded AI")
        st.divider()
        st.markdown("""
**What this does:**
Predicts Remaining Useful Life (RUL) for rotating equipment using a PyTorch LSTM trained
on NASA's CMAPSS turbofan dataset.

**Works for:**
- Compressors
- Pumps (ESP, PCP, centrifugal)
- Gas turbines
- Electric motors
- Gearboxes
        """)
        st.divider()
        st.markdown("""
**Model:** LSTM (2 layers, 64 hidden units)
**Dataset:** NASA CMAPSS FD001
**Window:** 30 cycles
**Max RUL:** 130 cycles
        """)
        st.divider()
        st.markdown("""
Built by [Roy Abes](https://royabes.com)
        """)


# ─── Tabs ─────────────────────────────────────────────────────────────────────

def tab_demo(model, scaler, X_demo: np.ndarray, unit_ids: list):
    st.subheader("Live Demo — Synthetic Equipment Fleet")
    st.caption(
        "10 virtual units sampled at various lifecycle stages — "
        "from freshly commissioned to near end-of-life."
    )

    preds = predict_units(X_demo, model)
    preds = np.clip(preds, 0, MAX_RUL)

    # Fleet overview
    st.pyplot(plot_fleet_overview(unit_ids, preds), use_container_width=True)
    st.divider()

    # Unit drill-down
    col1, col2 = st.columns([1, 3])
    with col1:
        selected = st.selectbox("Inspect unit:", unit_ids, index=int(np.argmin(preds)))
    unit_idx = unit_ids.index(selected)
    unit_rul = float(preds[unit_idx])

    with col2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted RUL", f"{unit_rul:.0f} cycles")
        c2.metric("Status", status_label(unit_rul))
        c3.metric("Life Remaining", f"{min(unit_rul / MAX_RUL * 100, 100):.0f}%")

    # Show sensor window for selected unit
    sensor_data = pd.DataFrame(
        X_demo[unit_idx], columns=SENSOR_COLS
    )
    sensor_data["cycle"] = range(1, len(sensor_data) + 1)
    st.pyplot(plot_health_curve(sensor_data, unit_rul, selected), use_container_width=True)


def tab_upload(model, scaler):
    st.subheader("Upload Your Equipment Data")
    st.markdown("""
Upload a CSV of sensor readings to get RUL predictions for your equipment.
The model will estimate how many operational cycles remain before maintenance is needed.
    """)

    # Download template
    sample_path = Path("data/sample_upload.csv")
    if not sample_path.exists():
        sample_path.parent.mkdir(exist_ok=True)
        make_sample_upload_csv(sample_path)
    with open(sample_path, "rb") as f:
        st.download_button(
            "Download CSV Template",
            f.read(),
            file_name="equipment_sensor_template.csv",
            mime="text/csv",
        )

    st.caption(f"Required columns: {', '.join(UPLOAD_COLS)}")
    st.divider()

    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            # Validate columns
            missing = [c for c in UPLOAD_COLS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.info(f"Required: {UPLOAD_COLS}")
                return

            X, unit_ids = preprocess_upload(df, scaler)
            preds = predict_units(X, model)

            st.success(f"Processed {len(unit_ids)} equipment unit(s)")
            st.divider()

            # Results table
            results = pd.DataFrame({
                "Unit": unit_ids,
                "Predicted RUL (cycles)": preds.round(1),
                "Status": [status_label(r) for r in preds],
                "Life Remaining": [f"{min(r/MAX_RUL*100, 100):.0f}%" for r in preds],
            }).sort_values("Predicted RUL (cycles)")
            st.dataframe(results, use_container_width=True, hide_index=True)

            # Fleet chart
            st.pyplot(plot_fleet_overview(unit_ids, preds), use_container_width=True)

            # Per-unit drill-down
            if len(unit_ids) > 1:
                selected = st.selectbox("Inspect unit:", unit_ids)
                unit_idx = unit_ids.index(selected)
                unit_rul = preds[unit_idx]
                unit_data = df[df["unit_id"] == selected].tail(60)
                # Rename for plotting
                unit_data = unit_data.rename(
                    columns=dict(zip(UPLOAD_COLS[2:], SENSOR_COLS))
                )
                st.pyplot(plot_health_curve(unit_data, unit_rul, selected),
                          use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)


def tab_about():
    st.subheader("About This Tool")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### What is RUL Prediction?

**Remaining Useful Life (RUL)** tells you how many operational cycles a piece of equipment
has left before it needs maintenance or will fail.

Instead of replacing parts on a fixed schedule (time-based maintenance), RUL prediction
enables **condition-based maintenance** — you service equipment exactly when it needs it.

### The Dataset

This model is trained on **NASA's CMAPSS FD001** dataset — a validated simulation of
turbofan engine degradation with 100 run-to-failure cycles, 21 sensors, and realistic noise.

The degradation patterns generalize to real rotating equipment in oil & gas:
compressors, pumps, turbines, and motors all follow similar sensor drift patterns
as they wear.

### The Model

A 2-layer **LSTM** (Long Short-Term Memory) network processes 30-cycle windows of
sensor data and predicts a single RUL value. Input: 14 sensor channels.
Training target: RUL capped at 130 cycles.
        """)

    with col2:
        st.markdown("""
### Equipment Sensor Mapping

| Upload Column | Physical Measurement | O&G Example |
|---|---|---|
| `temp_inlet` | Inlet temperature | Suction temperature |
| `temp_outlet` | Discharge temperature | Compressor discharge temp |
| `temp_exhaust` | Bearing/exhaust temp | Bearing housing temp |
| `pressure_discharge` | Discharge pressure | Discharge pressure (kPa) |
| `speed_shaft` | Primary shaft speed | RPM |
| `speed_core` | Secondary shaft | Gearbox output RPM |
| `flow_rate` | Flow ratio | Production flow ratio |
| `bypass_ratio` | Recirculation ratio | Recycle ratio |

### Typical Performance (NASA FD001)

| Metric | Value |
|---|---|
| Test RMSE | ~17–20 cycles |
| Test MAE | ~12–14 cycles |
| Training time | ~5 min (CPU) |

### Built with

PyTorch · Streamlit · scikit-learn · NASA CMAPSS · Grounded AI
        """)

    st.divider()
    st.markdown("""
**Source code:** Available on [GitHub](https://github.com/royabes)

**Article:** Predicting Equipment Failure in Oil & Gas with LSTMs — coming soon on Medium

**Contact:** [royabes.com](https://royabes.com)
    """)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    sidebar()

    st.title("⚙️ EquipmentRUL")
    st.markdown(
        "**Remaining Useful Life prediction for rotating equipment** — "
        "compressors, pumps, turbines, motors. "
        "Upload your sensor data or explore the live demo."
    )

    model, scaler, source = load_model_and_scaler()

    if source == "synthetic":
        st.info(
            "Running on synthetic training data. "
            "For full accuracy, download NASA CMAPSS and run `python train.py`.",
            icon="ℹ️"
        )

    X_demo, demo_unit_ids, demo_true_ruls = get_demo_data(scaler)

    tab1, tab2, tab3 = st.tabs(["Live Demo", "Upload Your Data", "About"])
    with tab1:
        tab_demo(model, scaler, X_demo, demo_unit_ids)
    with tab2:
        tab_upload(model, scaler)
    with tab3:
        tab_about()


if __name__ == "__main__":
    main()
