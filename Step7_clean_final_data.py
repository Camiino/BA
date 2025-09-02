import os
import pandas as pd
import numpy as np
from glob import glob
from scipy.signal import savgol_filter

# --- CONFIG ---
INPUT_DIR = "Exports/Final_Averages_1M"
OUTPUT_DIR = "Exports/Final_Cleaned"
DELIMITER = ";"
WINDOW_SIZE = 11   # must be odd, controls smoothness
POLY_ORDER = 2     # typically 2 or 3

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
input_files = glob(f"{INPUT_DIR}/*_final_mean.csv")

for path in input_files:
    try:
        print(f"\n🧹 Cleaning file: {os.path.basename(path)}")
        df = pd.read_csv(path, delimiter=DELIMITER)
        cleaned_df = df.copy()

        for col in df.columns:
            if col.lower() == "frame":
                continue
            series = df[col].astype(float)
            if series.isna().all():
                continue
            # Fill NaNs temporarily to smooth
            filled = series.interpolate(limit_direction='both')
            smoothed = savgol_filter(filled, window_length=min(WINDOW_SIZE, len(filled)//2*2+1), polyorder=POLY_ORDER)
            cleaned_df[col] = smoothed

        # Save
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(path).replace("_final_mean.csv", "_final_clean.csv"))
        cleaned_df.to_csv(out_path, sep=DELIMITER, index=False)
        print(f"✅ Cleaned and saved: {out_path}")
    except Exception as e:
        print(f"❌ Failed to clean {path}: {e}")
