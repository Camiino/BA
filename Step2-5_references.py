import os
import pandas as pd

input_root = "Exports/Daten_Raw_Clean"
reference_dir = "Exports/Reference"
DELIMITER = ";"

# Experiment categories
experiment_types = ["gewicht", "greifen", "präzision"]

# Marker columns
marker4_cols = [f"4_{a}" for a in ("X", "Y", "Z")]
marker5_cols = [f"5_{a}" for a in ("X", "Y", "Z")]
required_cols = [f"{m}_{a}" for m in range(1, 6) for a in ("X", "Y", "Z")]

# Create output folder
os.makedirs(reference_dir, exist_ok=True)

# Track best files per category and marker
best_refs = {
    exp: {
        "marker4": {"file": None, "valid_rows": 0, "df": None},
        "marker5": {"file": None, "valid_rows": 0, "df": None}
    } for exp in experiment_types
}

# Scan files
for root, _, files in os.walk(input_root):
    for file in files:
        if not file.lower().endswith(".csv"):
            continue

        path = os.path.join(root, file)

        for exp in experiment_types:
            if exp in file.lower() or exp in root.lower():
                try:
                    df = pd.read_csv(path, delimiter=DELIMITER, encoding="utf-8")

                    if not all(col in df.columns for col in required_cols):
                        continue

                    # Evaluate marker 4
                    if all(col in df.columns for col in marker4_cols):
                        valid_4 = df[marker4_cols].dropna().shape[0]
                        if valid_4 > best_refs[exp]["marker4"]["valid_rows"]:
                            best_refs[exp]["marker4"] = {
                                "file": file, "valid_rows": valid_4, "df": df
                            }

                    # Evaluate marker 5
                    if all(col in df.columns for col in marker5_cols):
                        valid_5 = df[marker5_cols].dropna().shape[0]
                        if valid_5 > best_refs[exp]["marker5"]["valid_rows"]:
                            best_refs[exp]["marker5"] = {
                                "file": file, "valid_rows": valid_5, "df": df
                            }

                except Exception as e:
                    print(f"❌ Failed to process {file}: {e}")

# Save references
for exp in experiment_types:
    for marker in ["marker4", "marker5"]:
        info = best_refs[exp][marker]
        if info["df"] is not None:
            out_path = os.path.join(reference_dir, f"{marker}_{exp}.csv")
            info["df"].to_csv(out_path, sep=DELIMITER, index=False, encoding="utf-8")
            print(f"✅ Saved {marker}_{exp}.csv from {info['file']} ({info['valid_rows']} valid rows)")
        else:
            print(f"❌ No suitable file found for {marker}_{exp}")
