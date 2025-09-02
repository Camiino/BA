import os
import pandas as pd
import numpy as np
from glob import glob

# --- CONFIG ---
input_root = "Exports/Daten_Raw_Interpolated"
output_root = "Exports/Daten_Trimmed"
DELIMITER = ";"
FPS = 200
WINDOW_SIZE = 60            # Number of frames for rolling window
MOVEMENT_THRESHOLD = 3      # Threshold in mm/frame

# --- Helper Function ---
def compute_trimmed_df(df):
    original_cols = df.columns.tolist()
    frame_col = df.columns[0]  # Keep first column (e.g. "Frame")
    marker_cols = [col for col in df.columns if any(axis in col for axis in ['_X', '_Y', '_Z'])]

    # For movement detection, use only marker data
    marker_df = df[marker_cols]

    def frame_distance(row1, row2):
        return np.linalg.norm(row1 - row2)

    distances = [0.0]
    for i in range(1, len(marker_df)):
        distances.append(frame_distance(marker_df.iloc[i], marker_df.iloc[i-1]))

    df["frame_distance"] = distances
    df["rolling_mean_movement"] = df["frame_distance"].rolling(WINDOW_SIZE).mean()

    threshold_crossed = df[df["rolling_mean_movement"] > MOVEMENT_THRESHOLD].index
    if len(threshold_crossed) > 0:
        last_active_idx = threshold_crossed[-1]
        df_trimmed = df.iloc[:last_active_idx + 1].copy()
        print(f"✅ Truncated at frame {last_active_idx} (of {len(df)}).")
    else:
        df_trimmed = df.copy()
        print("⚠️ No truncation — kept full recording.")

    # Drop helper columns before returning
    return df_trimmed.drop(columns=["frame_distance", "rolling_mean_movement"])

# --- Main ---
os.makedirs(output_root, exist_ok=True)
csv_files = glob(f"{input_root}/**/*.csv", recursive=True)

for path in csv_files:
    try:
        print(f"\n📄 Processing: {path}")
        df = pd.read_csv(path, delimiter=DELIMITER)
        trimmed_df = compute_trimmed_df(df)

        base_name = os.path.basename(path).replace(".csv", "_trimmed.csv")
        save_path = os.path.join(output_root, base_name)
        trimmed_df.to_csv(save_path, sep=DELIMITER, index=False)
        print(f"📁 Saved: {save_path}")
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")
