import os
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
import re

# --- CONFIG ---
INPUT_ROOT = "Exports/Daten_Trimmed"
OUTPUT_ROOT = "Exports/Daten_Averaged_1M"
DELIMITER = ";"
AXES = ("X", "Y", "Z")
MARKERS = [1, 2, 3, 4, 5]
NORMALIZE_START = True
EXPERIMENTS = [
    "kreis", "ptp", "ptp2", "ptp3", "zikzak", "sequentiell",
    "präzision", "greifen", "gewicht"
]

# For these experiments we only use marker #3
SINGLE_MARKER_EXPERIMENTS = {"präzision", "greifen", "gewicht"}
SINGLE_MARKER_ID = 3

# --- Utility Functions ---
def normalize_start(df):
    for marker in MARKERS:
        for axis in AXES:
            col = f"{marker}_{axis}"
            if col in df.columns:
                df[col] -= df[col].iloc[0]
    return df

def resample_df(df, target_len):
    def safe_interp(col):
        col = col.astype(float)
        if col.isna().all():
            return np.full(target_len, np.nan)
        return np.interp(
            np.linspace(0, len(col) - 1, target_len),
            np.arange(len(col)),
            col
        )
    return df.apply(safe_interp)

def get_probant_and_experiment(filename):
    name = os.path.basename(filename).lower()
    match = re.search(r"probant\d+", name)
    probant = match.group(0) if match else None
    for exp in sorted(EXPERIMENTS, key=lambda x: -len(x)):
        if f"_{exp}" in name or name.endswith(exp):
            return probant, exp
    return probant, None

# --- Collect files ---
files_by_group = defaultdict(list)
for path in glob(f"{INPUT_ROOT}/**/*.csv", recursive=True):
    probant, experiment = get_probant_and_experiment(path)
    if probant and experiment:
        files_by_group[(probant, experiment)].append(path)

# --- Process groups ---
os.makedirs(OUTPUT_ROOT, exist_ok=True)

for (probant, experiment), file_list in sorted(files_by_group.items()):
    print(f"\n📊 Processing {len(file_list)} files for {probant} - {experiment}")
    raw_dfs = []
    all_columns = set()
    original_lengths = []

    # --- Load & Normalize ---
    for file in file_list:
        try:
            df = pd.read_csv(file, delimiter=DELIMITER)
            frame_col = df.columns[0]
            df = df.drop(columns=frame_col)
            df = df.astype(float)

            if NORMALIZE_START:
                df = normalize_start(df)

            # If experiment is in single-marker mode, keep only marker #3
            if experiment in SINGLE_MARKER_EXPERIMENTS:
                keep_cols = [f"{SINGLE_MARKER_ID}_{axis}" for axis in AXES]
                df = df[[c for c in df.columns if c in keep_cols]]

            all_columns.update(df.columns)
            raw_dfs.append(df)
            original_lengths.append(len(df))
        except Exception as e:
            print(f"⚠️ Skipped file {file} due to error: {e}")

    if len(raw_dfs) < 2:
        print(f"❌ Not enough valid files for averaging: {probant} - {experiment}")
        continue

    # --- Determine average frame count ---
    avg_frame_count = int(round(np.mean(original_lengths)))
    print(f"ℹ️ Resampling all to average frame count: {avg_frame_count}")

    # --- Align and resample ---
    all_columns = sorted(all_columns)
    aligned_resampled = []

    for df in raw_dfs:
        aligned = df.reindex(columns=all_columns, fill_value=np.nan)
        resampled = resample_df(aligned, avg_frame_count)
        aligned_resampled.append(resampled)

    try:
        data_stack = np.stack([df.values for df in aligned_resampled])
    except Exception as e:
        print(f"❌ Error stacking data for {probant}-{experiment}: {e}")
        continue

    avg_data = np.nanmean(data_stack, axis=0)
    std_data = np.nanstd(data_stack, axis=0)

    frame_index = np.arange(avg_frame_count)
    avg_df = pd.DataFrame(avg_data, columns=all_columns)
    std_df = pd.DataFrame(std_data, columns=all_columns)
    avg_df.insert(0, "Frame", frame_index)
    std_df.insert(0, "Frame", frame_index)

    base = f"{probant}_{experiment}"
    avg_path = os.path.join(OUTPUT_ROOT, f"{base}_mean.csv")
    std_path = os.path.join(OUTPUT_ROOT, f"{base}_std.csv")

    avg_df.to_csv(avg_path, sep=DELIMITER, index=False)
    std_df.to_csv(std_path, sep=DELIMITER, index=False)
    print(f"✅ Saved: {avg_path}, {std_path}")
