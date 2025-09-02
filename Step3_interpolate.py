import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
input_root = "Exports/Daten_Raw_Clean"
output_root = "Exports/Daten_Raw_Interpolated"
reference_dir = "Exports/Reference"
DELIMITER = ";"
AXES = ("X", "Y", "Z")

EXPERIMENT_TYPES = ["gewicht", "greifen", "präzision"]
MARKERS_TO_INTERPOLATE = [1, 2, 3, 4, 5]
FALLBACK_MARKERS = [4, 5]
SPECIAL_NEIGHBORS = {
    1: [2],
    2: [1, 3],
    3: [2]
}

# --- Helpers ---

def load_data(path: str):
    df = pd.read_csv(path, delimiter=DELIMITER, encoding="utf-8")
    frame_col = df.columns[0]
    return df[frame_col], df.drop(columns=[frame_col])

def interpolate_internal_gaps(df: pd.DataFrame, markers=None) -> pd.DataFrame:
    if markers:
        cols = [f"{m}_{a}" for m in markers for a in AXES if f"{m}_{a}" in df.columns]
        df[cols] = df[cols].astype(float).interpolate(method="linear", limit_direction="both", limit_area="inside")
    else:
        df = df.astype(float).interpolate(method="linear", limit_direction="both", limit_area="inside")
    return df

def fill_trailing_with_neighbors(df: pd.DataFrame, target_id: int, neighbor_ids):
    for axis in AXES:
        t_col = f"{target_id}_{axis}"
        neighbor_cols = [f"{nid}_{axis}" for nid in neighbor_ids if f"{nid}_{axis}" in df.columns]
        if t_col not in df.columns or not neighbor_cols:
            continue

        last_valid = df[t_col].last_valid_index()
        if last_valid is None or last_valid >= len(df) - 1:
            continue

        best_col = max(
            neighbor_cols,
            key=lambda col: df[col].iloc[last_valid+1:].notna().sum(),
            default=None
        )

        if not best_col:
            continue

        neighbor_delta = df[best_col].diff()
        value = df.at[last_valid, t_col]
        for i in range(last_valid + 1, len(df)):
            d = neighbor_delta.iat[i]
            if pd.isna(d):
                break
            value += d
            df.at[i, t_col] = value
    return df

def fill_leading_with_neighbors(df: pd.DataFrame, target_id: int, neighbor_ids):
    for axis in AXES:
        t_col = f"{target_id}_{axis}"
        neighbor_cols = [f"{nid}_{axis}" for nid in neighbor_ids if f"{nid}_{axis}" in df.columns]
        if t_col not in df.columns or not neighbor_cols:
            continue

        first_valid = df[t_col].first_valid_index()
        if first_valid is None or first_valid == 0:
            continue

        best_col = max(
            neighbor_cols,
            key=lambda col: df[col].iloc[:first_valid].notna().sum(),
            default=None
        )

        if not best_col:
            continue

        neighbor_delta = df[best_col].diff()
        value = df.at[first_valid, t_col]
        for i in range(first_valid - 1, -1, -1):
            d = neighbor_delta.iat[i + 1]
            if pd.isna(d):
                break
            value -= d
            df.at[i, t_col] = value
    return df

def fill_from_reference(df: pd.DataFrame, marker_id: int, ref_df: pd.DataFrame):
    for axis in AXES:
        col = f"{marker_id}_{axis}"
        if col not in df.columns or col not in ref_df.columns:
            continue

        ref_values = ref_df[col].dropna().values
        if len(ref_values) < 2:
            continue

        ref_deltas = np.diff(ref_values)
        ref_cumsum = np.concatenate([[0], np.cumsum(ref_deltas)])

        # --- Leading fill ---
        first_valid = df[col].first_valid_index()
        if first_valid is not None and first_valid > 0:
            stretch = np.linspace(0, len(ref_deltas)-1, first_valid).astype(int)
            deltas = ref_deltas[stretch]
            start_value = df.at[first_valid, col]
            leading_values = start_value - np.cumsum(deltas[::-1])[::-1]
            df.loc[:first_valid-1, col] = leading_values

        # --- Trailing fill ---
        last_valid = df[col].last_valid_index()
        if last_valid is not None and last_valid < len(df) - 1:
            trailing_len = len(df) - last_valid - 1
            stretch = np.linspace(0, len(ref_deltas)-1, trailing_len).astype(int)
            deltas = ref_deltas[stretch]
            start_value = df.at[last_valid, col]
            trailing_values = start_value + np.cumsum(deltas)
            df.loc[last_valid+1:, col] = trailing_values
    return df

def report_remaining_nans(df: pd.DataFrame):
    missing = df.isna().sum()
    leftovers = missing[missing > 0]
    if len(leftovers):
        print("🧓 Remaining NaNs:")
        for col, cnt in leftovers.items():
            print(f"  • {col}: {cnt}")
    else:
        print("✅ All NaNs resolved.")

# --- Experiment Detection ---

def detect_experiment_type(path: str) -> str:
    lowered = path.lower()
    for exp in EXPERIMENT_TYPES:
        if exp in lowered:
            return exp
    return None

# --- Core Processing ---

def interpolate_csv(input_path, output_path, ref_map):
    try:
        print(f"\n🔄 Processing: {input_path}")
        experiment = detect_experiment_type(input_path)

        if experiment in EXPERIMENT_TYPES:
            markers_to_use = MARKERS_TO_INTERPOLATE
            fallback_markers = FALLBACK_MARKERS
            neighbors = SPECIAL_NEIGHBORS
        else:
            markers_to_use = [1]
            fallback_markers = []
            neighbors = {}

        frames, data = load_data(input_path)
        filled = interpolate_internal_gaps(data.copy(), markers=markers_to_use)

        for tgt, nbs in neighbors.items():
            filled = fill_leading_with_neighbors(filled, tgt, nbs)
            filled = fill_trailing_with_neighbors(filled, tgt, nbs)

        if experiment in ref_map:
            for marker in fallback_markers:
                ref_df = ref_map[experiment].get(marker)
                if ref_df is not None:
                    filled = fill_from_reference(filled, marker, ref_df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.concat([frames, filled], axis=1).to_csv(
            output_path, sep=DELIMITER, index=False, encoding="utf-8"
        )
        print(f"✅ Saved: {output_path}")
        report_remaining_nans(filled)

    except Exception as e:
        print(f"❌ Failed: {input_path} → {e}")

# --- Load All Reference Files ---

def load_reference_map():
    ref_map = {}
    for exp in EXPERIMENT_TYPES:
        ref_map[exp] = {}
        for marker in FALLBACK_MARKERS:
            ref_file = os.path.join(reference_dir, f"marker{marker}_{exp}.csv")
            if os.path.exists(ref_file):
                _, ref_df = load_data(ref_file)
                ref_map[exp][marker] = ref_df
            else:
                print(f"⚠️ Missing reference: {ref_file}")
    return ref_map

# --- Run Processing ---

reference_map = load_reference_map()

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".csv"):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            interpolate_csv(input_path, output_path, reference_map)
