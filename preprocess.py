# preprocess.py — NASA CMAPSS + user CSV preprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Sensor columns used from CMAPSS (14 informative sensors, drop constant ones)
SENSOR_COLS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
               "s13", "s14", "s15", "s17", "s20", "s21"]

# Column names for the user-facing upload template (human-readable)
UPLOAD_COLS = [
    "unit_id", "cycle",
    "temp_inlet",         # s2  — inlet temperature
    "temp_outlet",        # s3  — outlet / discharge temperature
    "temp_exhaust",       # s4  — exhaust / downstream temperature
    "pressure_discharge", # s7  — discharge pressure
    "speed_shaft",        # s8  — primary shaft speed
    "speed_core",         # s9  — secondary / core speed
    "pressure_static",    # s11 — static pressure
    "flow_rate",          # s12 — fluid flow ratio
    "speed_corrected",    # s13 — corrected shaft speed
    "speed_secondary",    # s14 — secondary corrected speed
    "bypass_ratio",       # s15 — recirculation / bypass ratio
    "bleed_enthalpy",     # s17 — heat extracted
    "coolant_1",          # s20 — cooling flow 1
    "coolant_2",          # s21 — cooling flow 2
]

WINDOW = 30   # timesteps per sample
MAX_RUL = 130  # clip RUL at this value (standard CMAPSS approach)


# ─── CMAPSS loading ───────────────────────────────────────────────────────────

def _cmapss_colnames() -> list[str]:
    return ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]


def load_cmapss(data_dir: Path, subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load CMAPSS train, test, and RUL ground truth for a given subset."""
    cols = _cmapss_colnames()
    train = pd.read_csv(data_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=cols)
    test  = pd.read_csv(data_dir / f"test_{subset}.txt",  sep=r"\s+", header=None, names=cols)
    rul   = pd.read_csv(data_dir / f"RUL_{subset}.txt",   sep=r"\s+", header=None, names=["RUL"])
    return train, test, rul["RUL"].values


def add_rul_labels(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """Compute RUL for each row (capped at max_rul)."""
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].max()
    df["RUL"] = df.apply(lambda r: max_cycle[r["unit"]] - r["cycle"], axis=1)
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df


def fit_scaler(df: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(df[SENSOR_COLS])
    return scaler


def scale(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    df = df.copy()
    df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS])
    return df


def make_windows(df: pd.DataFrame, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences per engine unit."""
    X, y = [], []
    for unit_id, group in df.groupby("unit"):
        data = group[SENSOR_COLS].values
        rul  = group["RUL"].values
        for i in range(len(data) - window + 1):
            X.append(data[i:i + window])
            y.append(rul[i + window - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_test_windows(df: pd.DataFrame, scaler: MinMaxScaler,
                      window: int = WINDOW) -> np.ndarray:
    """One window per test engine — use last `window` cycles."""
    df = scale(df, scaler)
    X = []
    for _, group in df.groupby("unit"):
        data = group[SENSOR_COLS].values
        if len(data) >= window:
            X.append(data[-window:])
        else:
            # pad with first row if shorter than window
            pad = np.tile(data[0], (window - len(data), 1))
            X.append(np.vstack([pad, data]))
    return np.array(X, dtype=np.float32)


# ─── Synthetic data (demo without NASA files) ─────────────────────────────────

def generate_demo_windows(scaler: "MinMaxScaler", n_units: int = 10,
                          seed: int = 7) -> tuple[np.ndarray, list, list]:
    """
    Generate demo units sampled at various lifecycle stages (not just end-of-life).
    Returns (X windows, unit_ids, true_rul_list) for display purposes.
    """
    rng = np.random.default_rng(seed)
    X, unit_ids, true_ruls = [], [], []
    for uid in range(1, n_units + 1):
        total_cycles = rng.integers(150, 300)
        # Sample at 30–95% of total life for variety
        sample_pct = rng.uniform(0.3, 0.95)
        sample_at  = int(total_cycles * sample_pct)
        true_rul   = min(total_cycles - sample_at, MAX_RUL)

        rows = []
        for cycle in range(1, sample_at + 1):
            health = max(0.0, 1.0 - (cycle / total_cycles) ** 1.5)
            noise  = rng.normal(0, 0.02, 14)
            sensors = np.clip(0.5 + 0.4 * health + noise, 0, 1)
            rows.append(sensors)

        data = np.array(rows, dtype=np.float32)
        # Use scaler on the raw values
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = scaler.transform(data)
        except Exception:
            pass  # already normalised if synthetic
        window = data[-WINDOW:] if len(data) >= WINDOW else \
            np.vstack([np.tile(data[0], (WINDOW - len(data), 1)), data])
        X.append(window.astype(np.float32))
        unit_ids.append(uid)
        true_ruls.append(true_rul)

    return np.array(X, dtype=np.float32), unit_ids, true_ruls


def generate_synthetic_cmapss(n_units: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic degradation data that mimics CMAPSS FD001.
    Used when the real NASA dataset is not available (demo / Streamlit Cloud).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        total_cycles = rng.integers(150, 300)
        for cycle in range(1, total_cycles + 1):
            # Health index: 1.0 at start → 0.0 at failure
            health = max(0.0, 1.0 - (cycle / total_cycles) ** 1.5)
            noise = rng.normal(0, 0.02, 14)
            # Sensors drift as health degrades
            sensors = 0.5 + 0.4 * health + noise
            sensors = np.clip(sensors, 0.0, 1.0)
            rows.append([unit, cycle] + sensors.tolist())
    cols = ["unit", "cycle"] + SENSOR_COLS
    df = pd.DataFrame(rows, columns=cols)
    return add_rul_labels(df)


# ─── User CSV upload ───────────────────────────────────────────────────────────

def make_sample_upload_csv(path: Path, n_units: int = 3, cycles: int = 60) -> None:
    """Generate a sample upload CSV for users to use as a template."""
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, n_units + 1):
        for cycle in range(1, cycles + 1):
            health = max(0.0, 1.0 - (cycle / cycles) ** 1.5)
            noise = rng.normal(0, 0.015, 14)
            sensors = (0.5 + 0.4 * health + noise).clip(0, 1)
            # Scale back to realistic O&G ranges for readability
            readable = [
                uid, cycle,
                round(200 + sensors[0] * 150, 1),   # temp_inlet (°C)
                round(300 + sensors[1] * 200, 1),   # temp_outlet
                round(250 + sensors[2] * 100, 1),   # temp_exhaust
                round(1000 + sensors[3] * 500, 1),  # pressure_discharge (kPa)
                round(1200 + sensors[4] * 600, 1),  # speed_shaft (RPM)
                round(3000 + sensors[5] * 500, 1),  # speed_core
                round(800 + sensors[6] * 200, 1),   # pressure_static
                round(0.5 + sensors[7] * 0.4, 3),   # flow_rate (ratio)
                round(1150 + sensors[8] * 550, 1),  # speed_corrected
                round(2900 + sensors[9] * 480, 1),  # speed_secondary
                round(2.5 + sensors[10] * 1.5, 2),  # bypass_ratio
                round(300 + sensors[11] * 100, 1),  # bleed_enthalpy
                round(0.02 + sensors[12] * 0.05, 4),# coolant_1
                round(0.01 + sensors[13] * 0.04, 4),# coolant_2
            ]
            rows.append(readable)
    df = pd.DataFrame(rows, columns=UPLOAD_COLS)
    df.to_csv(path, index=False)


def preprocess_upload(df: pd.DataFrame, scaler: MinMaxScaler,
                      window: int = WINDOW) -> tuple[np.ndarray, list[int]]:
    """
    Process a user-uploaded CSV into model-ready windows.
    Expects columns: unit_id, cycle, + 14 sensor columns (UPLOAD_COLS).
    Returns (X windows, unit_ids).
    """
    df = df.copy()
    # Rename upload columns to internal CMAPSS names
    col_map = dict(zip(UPLOAD_COLS[2:], SENSOR_COLS))
    df.rename(columns={"unit_id": "unit", **col_map}, inplace=True)

    # Normalize using training scaler
    df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS])

    X, unit_ids = [], []
    for uid, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        data = group[SENSOR_COLS].values.astype(np.float32)
        if len(data) >= window:
            X.append(data[-window:])
        else:
            pad = np.tile(data[0], (window - len(data), 1))
            X.append(np.vstack([pad, data]).astype(np.float32))
        unit_ids.append(uid)
    return np.array(X, dtype=np.float32), unit_ids
