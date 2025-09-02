import pandas as pd
import numpy as np
import re

# --- CONFIGURE ---
input_file = "input.csv"
output_file = "filled_but_preserved.csv"
base_markers = list(range(1, 6))  # markers 1 to 5

# --- STEP 1: Load raw data ---
df_raw = pd.read_csv(input_file, sep=';', header=None, dtype=str)

# --- STEP 2: Locate headers ---
marker_hdr_idx = df_raw.apply(lambda r: r.str.contains(r"\*1", regex=True, na=False).any(), axis=1).idxmax()
coord_hdr_idx = marker_hdr_idx + 1

marker_row = df_raw.iloc[marker_hdr_idx].fillna("")
coord_row = df_raw.iloc[coord_hdr_idx].fillna("")

# --- STEP 3: Generate column names like '1_X', '1_Y', etc. ---
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

# --- STEP 4: Read data rows and assign headers ---
df = df_raw.iloc[coord_hdr_idx + 1:].copy()
df.columns = new_cols

# --- STEP 5: Convert number columns safely without stripping decimals ---
def parse_val(val):
    try:
        # keep decimal part, remove spaces, ignore thousand separators
        val = str(val).replace(" ", "")
        # If numbers look like "127.228.226", assume it's a German thousand separator: remove the first dots
        parts = val.split(".")
        if len(parts) > 2:
            val = "".join(parts[:-1]) + "." + parts[-1]
        return float(val)
    except:
        return np.nan

for col in df.columns:
    if col != "Frame":
        df[col] = df[col].apply(parse_val)

# --- STEP 6: Determine all marker numbers ---
all_markers = sorted({int(c.split("_")[0]) for c in df.columns if c != "Frame"})
extra_markers = [m for m in all_markers if m not in base_markers]

print("Base markers:", base_markers)
print("Extra markers:", extra_markers)

# --- STEP 7: Fill base marker gaps with extra marker data ---
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
                    # Fill missing base marker with flash data
                    df.loc[row_idx, [bx, by, bz]] = [row[ex], row[ey], row[ez]]
                    break  # Stop once filled

# --- STEP 8: Drop extra marker columns after filling ---
cols_to_drop = [
    f"{e}_{axis}" for e in extra_markers for axis in ("X", "Y", "Z")
    if f"{e}_{axis}" in df.columns
]
df.drop(columns=cols_to_drop, inplace=True)


# --- STEP 9: Save final cleaned file ---
df.to_csv(output_file, sep=";", index=False)
print(f"✅ Cleaned data saved to: {output_file}")
