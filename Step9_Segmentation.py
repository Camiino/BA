import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import defaultdict
import re

# --- CONFIG ---
AVERAGE_ROOT = "Exports/Daten_Averaged_1M"
CLUSTER_OUTPUT = "Exports/Clustered"
DELIMITER = ";"
MARKERS = [1, 2, 3, 4, 5]
AXES = ["X", "Y", "Z"]

# --- Probanden-Infos ---
probanten_dict = {
    6: {'Alter': '18-20', 'Geschlecht': 'Männlich'},
    7: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    8: {'Alter': '21-25', 'Geschlecht': 'Weiblich'},
    9: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    10: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    11: {'Alter': '25-35', 'Geschlecht': 'Männlich'},
    12: {'Alter': '18-20', 'Geschlecht': 'Weiblich'},
    13: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    14: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    15: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    16: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    17: {'Alter': '25-35', 'Geschlecht': 'Männlich'},
    18: {'Alter': '25-35', 'Geschlecht': 'Männlich'},
    19: {'Alter': '25-35', 'Geschlecht': 'Männlich'},
    20: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    21: {'Alter': '25-35', 'Geschlecht': 'Männlich'},
    22: {'Alter': '35+', 'Geschlecht': 'Weiblich'},
    23: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    24: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    25: {'Alter': '18-20', 'Geschlecht': 'Männlich'},
    26: {'Alter': '25-35', 'Geschlecht': 'Weiblich'},
    27: {'Alter': '25-35', 'Geschlecht': 'Weiblich'},
    28: {'Alter': '25-35', 'Geschlecht': 'Männlich'},
    29: {'Alter': '25-35', 'Geschlecht': 'Männlich'},
    30: {'Alter': '25-35', 'Geschlecht': 'Weiblich'},
    31: {'Alter': '25-35', 'Geschlecht': 'Männlich'},
    32: {'Alter': '35+', 'Geschlecht': 'Männlich'},
    33: {'Alter': '35+', 'Geschlecht': 'Weiblich'},
    34: {'Alter': '21-25', 'Geschlecht': 'Weiblich'},
    35: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
    36: {'Alter': '21-25', 'Geschlecht': 'Weiblich'},
    37: {'Alter': '21-25', 'Geschlecht': 'Männlich'},
}

def get_meta(probant_str):
    try:
        num = int(probant_str.replace("probant", ""))
        meta = probanten_dict.get(num, {'Alter': '21-25', 'Geschlecht': 'Männlich'})
        return meta['Geschlecht'], meta['Alter']
    except:
        return 'Männlich', '21-25'

def resample_df(df, target_len):
    df_interp = {}
    for col in df.columns:
        if col == "Frame":
            continue
        y = df[col].astype(float).values
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, target_len)
        if np.isnan(y).all():
            y_new = np.full(target_len, np.nan)
        else:
            y_new = np.interp(x_new, x_old, y)
        df_interp[col] = y_new
    return pd.DataFrame(df_interp)

def plot_derivatives_scalar_style(df, out_dir, markers=MARKERS, axes=AXES):
    frame = df["Frame"].values

    from scipy.ndimage import median_filter
    
    def compute_scalar(df, marker_id):
        vs, accs = [], []
        for axis in AXES:
            col = f"{marker_id}_{axis}"
            if col in df.columns:
                pos = df[col].astype(float).values
                v = np.gradient(pos)
                a = np.gradient(v)
                vs.append(v)
                accs.append(a)

        if len(vs) == 3:
            vel = np.sqrt(vs[0]**2 + vs[1]**2 + vs[2]**2)
            acc = np.sqrt(accs[0]**2 + accs[1]**2 + accs[2]**2)

            # --- Glättung & Spike-Entfernung für Geschwindigkeit ---
            win_v = 11 if len(vel) >= 11 else len(vel) // 2 * 2 + 1
            vel_smooth = savgol_filter(vel, window_length=win_v, polyorder=3)

            # Z-Score-Outlier-Removal
            z_v = (vel_smooth - np.nanmean(vel_smooth)) / np.nanstd(vel_smooth)
            vel_smooth[np.abs(z_v) > 2.5] = np.nan
            vel_smooth = pd.Series(vel_smooth).interpolate(limit_direction='both').values
            vel_smooth = median_filter(vel_smooth, size=7)

            # --- Glättung & Spike-Entfernung für Beschleunigung ---
            win_a = 21 if len(acc) >= 21 else len(acc) // 2 * 2 + 1
            acc_smooth = savgol_filter(acc, window_length=win_a, polyorder=3)

            z_a = (acc_smooth - np.nanmean(acc_smooth)) / np.nanstd(acc_smooth)
            acc_smooth[np.abs(z_a) > 2.5] = np.nan
            acc_smooth = pd.Series(acc_smooth).interpolate(limit_direction='both').values
            acc_smooth = median_filter(acc_smooth, size=11)

            return vel_smooth, acc_smooth
        return None, None

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Betrag von Geschwindigkeit und Beschleunigung", fontsize=14)

    for marker in markers:
        v, a = compute_scalar(df, marker)
        if v is not None and a is not None:
            ax_v.plot(frame, v, label=f"Marker {marker}")
            ax_a.plot(frame, a, label=f"Marker {marker}")

    ax_v.set_title("Geschwindigkeit (||v||)")
    ax_a.set_title("Beschleunigung (||a||)")
    ax_a.set_xlabel("Bewegungsfortschritt (%)")
    ax_v.set_ylabel("||v||")
    ax_a.set_ylabel("||a||")
    ax_v.legend()
    ax_a.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(out_dir, "velocity_acc_scalar.png"))
    plt.close()

def save_cluster_average(mode, cluster_key, experiment, dfs):
    if len(dfs) < 2:
        print(f"⚠️ Not enough data for {mode}/{cluster_key} - {experiment}")
        return
    try:
        avg_len = int(round(np.mean([len(df) for df in dfs])))

        resampled = [resample_df(df.drop(columns="Frame"), avg_len) for df in dfs]
        stack = np.stack([df.values for df in resampled])
        mean = np.nanmean(stack, axis=0)

        frame = np.arange(avg_len)
        columns = dfs[0].columns[1:]
        out_df = pd.DataFrame(mean, columns=columns)
        out_df.insert(0, "Frame", frame)

        out_dir = os.path.join(CLUSTER_OUTPUT, mode, cluster_key, experiment)
        os.makedirs(out_dir, exist_ok=True)

        out_df.to_csv(os.path.join(out_dir, "mean.csv"), sep=DELIMITER, index=False)
        plot_derivatives_scalar_style(out_df, out_dir)

        print(f"✅ Saved: {mode}/{cluster_key}/{experiment}")
    except Exception as e:
        print(f"❌ Error saving {mode}/{cluster_key} - {experiment}: {e}")

# --- Grouping ---
grouped_data = {
    'Geschlecht': defaultdict(lambda: defaultdict(list)),
    'AlterWithinGender': defaultdict(lambda: defaultdict(list)),
    'AlterOnly': defaultdict(lambda: defaultdict(list))
}

# --- Load & Cluster ---
if not os.path.exists(AVERAGE_ROOT):
    print(f"❌ Directory not found: {AVERAGE_ROOT}")
    exit()

files = [f for f in os.listdir(AVERAGE_ROOT) if f.endswith("_mean.csv")]

for file in files:
    # Capture experiment name including unicode letters (e.g., "präzision")
    match = re.search(r"(probant\d+)_([^_]+)_mean\.csv", file.lower())
    if not match:
        continue
    probant_str, experiment = match.groups()
    gender, age_group = get_meta(probant_str)

    path = os.path.join(AVERAGE_ROOT, file)
    df = pd.read_csv(path, delimiter=DELIMITER)

    grouped_data['Geschlecht'][gender][experiment].append(df)
    grouped_data['AlterWithinGender'][f"{gender}_{age_group}"][experiment].append(df)
    grouped_data['AlterOnly'][age_group][experiment].append(df)

# --- Save All ---
for mode, cluster_dict in grouped_data.items():
    for cluster_key, experiments in cluster_dict.items():
        for experiment, dfs in experiments.items():
            save_cluster_average(mode, cluster_key, experiment, dfs)
