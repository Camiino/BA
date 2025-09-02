import os
import pandas as pd
import numpy as np
import re

# --- CONFIG ---
input_root = "Exports/Daten_Raw_Formatted"
output_root = "Exports/Daten_Raw_Clean"

# --- Determine base_markers based on file/folder name ---
def determine_base_markers(path):
    lower = path.lower()
    if any(k in lower for k in ["gewicht", "greifen", "präzision"]):
        return [1, 2, 3, 4, 5]
    elif any(k in lower for k in ["ptp", "ptp2", "ptp3", "sequentiell", "zickzack", "kreis"]):
        return [1]
    else:
        print(f"⚠️  Unknown group for: {path} → defaulting to [1]")
        return [1]

# --- Parse custom German decimal format safely ---
def parse_val(val):
    try:
        val = str(val).replace(" ", "")
        parts = val.split(".")
        if len(parts) > 2:
            val = "".join(parts[:-1]) + "." + parts[-1]
        return float(val)
    except:
        return np.nan

# --- Clean a single CSV file ---
def clean_csv(input_path, output_path):
    base_markers = determine_base_markers(input_path)

    try:
        df_raw = pd.read_csv(input_path, sep=';', header=None, dtype=str, encoding="cp1252")
    except Exception as e:
        print(f"❌ Failed to read {input_path}: {e}")
        return

    try:
        marker_hdr_idx = df_raw.apply(lambda r: r.str.contains(r"\*1", regex=True, na=False).any(), axis=1).idxmax()
        coord_hdr_idx = marker_hdr_idx + 1
    except Exception as e:
        print(f"❌ Could not find headers in {input_path}: {e}")
        return

    marker_row = df_raw.iloc[marker_hdr_idx].fillna("")
    coord_row = df_raw.iloc[coord_hdr_idx].fillna("")

    new_cols = []
    current_marker = None
    for col_val, coord in zip(marker_row, coord_row):
        col_val = col_val.strip()
        coord = coord.strip()

        if coord.lower().startswith("field"):
            new_cols.append("Frame")
            continue

        m = re.match(r"\*(\d+)", col_val)
        if m:
            current_marker = m.group(1)

        if current_marker and coord in ("X", "Y", "Z"):
            new_cols.append(f"{current_marker}_{coord}")
        else:
            new_cols.append(f"unk_{len(new_cols)}")

    df = df_raw.iloc[coord_hdr_idx + 1:].copy()
    df.columns = new_cols

    for col in df.columns:
        if col != "Frame":
            df[col] = df[col].apply(parse_val)

    all_markers = sorted({int(c.split("_")[0]) for c in df.columns if c != "Frame"})
    extra_markers = [m for m in all_markers if m not in base_markers]

    for row_idx, row in df.iterrows():
        for e in extra_markers:
            ex, ey, ez = f"{e}_X", f"{e}_Y", f"{e}_Z"
            if pd.notna(row.get(ex)) and pd.notna(row.get(ey)) and pd.notna(row.get(ez)):
                for b in base_markers:
                    bx, by, bz = f"{b}_X", f"{b}_Y", f"{b}_Z"
                    if (
                        bx in df.columns and by in df.columns and bz in df.columns and
                        pd.isna(row.get(bx)) and pd.isna(row.get(by)) and pd.isna(row.get(bz))
                    ):
                        df.loc[row_idx, [bx, by, bz]] = [row[ex], row[ey], row[ez]]
                        break

    cols_to_drop = [
        f"{e}_{axis}" for e in extra_markers for axis in ("X", "Y", "Z")
        if f"{e}_{axis}" in df.columns
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        df.to_csv(output_path, sep=";", index=False, encoding="cp1252")
        print(f"✅ Cleaned: {input_path} → {output_path}")
    except Exception as e:
        print(f"❌ Failed to save {output_path}: {e}")

# --- Walk through all files ---
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".csv"):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            clean_csv(input_path, output_path)
