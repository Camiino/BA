import os
import re
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt


"""
Step 11 — Best Single-Trial Plots (no averaging)

Purpose:
- Scan cleaned raw CSVs (Exports/Daten_Raw_Clean) and pick one "best" file per
  task among: ptp, zikzak, gewicht, präzision. "Best" = least missing samples for the
  marker used in analysis (marker 1 for ptp/zikzak, marker 3 for gewicht and präzision),
  with strong penalty for leading/trailing gaps (extrapolation) and internal
  gaps (interpolation).
- For each selected file, do a lightweight single-trial processing
  (interpolate internal gaps, crop away leading/trailing NaNs, smooth),
  compute scalar |v| and |a|, detect the main movement segment, and produce
  plots without any averaging.

Outputs:
- Exports/Single_Plots/<task>_single_velocity_acc_scalar.png
- Exports/Single_Plots/<task>_single_velocity_scalar.png
- Exports/Single_Plots/Step11_selection_summary.csv (metadata + scores)
- Exports/Single_Selected/<task>_single_clean.csv (cropped + smoothed positions)

Notes:
- Assumes a constant sample rate (SAMPLE_RATE_HZ) since raw files do not carry
  absolute timestamps. Positions are assumed to be in millimeters; set
  POSITION_UNIT and CONVERT_POSITION_TO_METERS as needed.
"""


# --- CONFIG ---
INPUT_ROOT = "Exports/Daten_Raw_Clean"
OUTPUT_PLOTS = "Exports/Single_Plots"
OUTPUT_SELECTED = "Exports/Single_Selected"
DELIMITER = ";"

# Map group name -> list of identifying substrings found in filenames/paths
GROUP_PATTERNS = {
    "ptp": ["_ptp", "ptp_", "-ptp", " ptp", "ptp", "ptp", os.sep + "ptp"],
    "ptp2": ["ptp2", "ptp-2", "ptp_2", "ptp 2", "ptpii"],
    "präzision": ["präzision", "präzision_", "-präzision", " präzision", "präzision", "präzision", os.sep + "präzision"],
    "zikzak": ["zikzak", "zickzack", "zik-zak"],
    "gewicht": ["gewicht"],
}

# Marker used per group for kinematics
MARKER_OF_INTEREST = {
    "ptp": 1,
    "ptp2": 1,
    "zikzak": 1,
    "gewicht": 3,
    "präzision": 3,
}

# Time base assumptions
SAMPLE_RATE_HZ = 200.0  # frames per second

# Units
POSITION_UNIT = "mm"         # "mm" or "m"
CONVERT_POSITION_TO_METERS = True

# Savitzky–Golay parameters (positions first, then derivatives)
SG_POLY_POS = 3
SG_WINDOW_POS = 21
SG_WINDOW_DER = 21

# Optional outlier guard before smoothing (Hampel-like)
HAMPEL_WINDOW = 7
HAMPEL_SIGMAS = 3.0

# Emphasis smoothing on magnitudes for nicer display
EMPHASIZE_V = True
EMPHASIZE_A = True
PEAK_V_WINDOW_FRAC = 0.33
PEAK_A_WINDOW_FRAC = 0.25
PEAK_V_WINDOW = 41
PEAK_A_WINDOW = 41
PEAK_POLY = 2
PEAK_RESCALE_TO_MAX = True

# Optional strong smoothing for magnitude via zero-phase Butterworth
USE_BUTTER_SMOOTH = True
BUTTER_ORDER = 4
VEL_CUTOFF_HZ = 6.0
ACC_CUTOFF_HZ = 8.0


# --- Utils ---
def _ensure_odd(n: int, *, min_val: int = 5) -> int:
    n = int(max(n, min_val))
    return n if n % 2 == 1 else n + 1


def _cap_window(target: int, series_len: int, *, min_val: int = 5) -> int:
    if series_len <= 1:
        return _ensure_odd(min_val)
    max_allowed = series_len - 1 if series_len % 2 == 0 else series_len
    return _ensure_odd(min(target, max_allowed), min_val=min_val)


def _hampel(x: np.ndarray, k: int, n_sigmas: float) -> np.ndarray:
    if k <= 0:
        return x
    y = x.astype(float).copy()
    n = len(y)
    for i in range(n):
        i0 = max(0, i - k)
        i1 = min(n, i + k + 1)
        w = y[i0:i1]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) + 1e-12
        thresh = n_sigmas * 1.4826 * mad
        if abs(y[i] - med) > thresh:
            y[i] = med
    return y


def _emphasize_peak(y: np.ndarray, base_window: int, poly: int, *, rescale_to_max: bool = True, frac: float | None = None) -> np.ndarray:
    if len(y) < 5:
        return y
    w = base_window or 5
    if frac is not None and np.isfinite(frac) and frac > 0:
        w = max(w, int(round(frac * len(y))))
    w = _cap_window(w, len(y), min_val=poly + 2)
    ys = savgol_filter(y, window_length=w, polyorder=poly, mode="interp")
    if rescale_to_max and np.max(ys) > 0:
        ys = ys * (np.max(y) / (np.max(ys) + 1e-12))
    return ys


def _find_main_segment(v_ref: np.ndarray, enter_frac: float = 0.05, exit_frac: float = 0.03, min_len_frac: float = 0.1):
    n = len(v_ref)
    if n == 0:
        return 0, -1
    peak = float(np.nanmax(v_ref))
    if not np.isfinite(peak) or peak <= 0:
        return 0, n - 1
    thr_enter = enter_frac * peak
    thr_exit = exit_frac * peak

    mask = v_ref > thr_exit
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0, n - 1

    cuts = np.where(np.diff(idx) > 1)[0]
    segments = []
    start = 0
    for c in cuts:
        segments.append((idx[start], idx[c]))
        start = c + 1
    segments.append((idx[start], idx[-1]))

    peak_i = int(np.argmax(v_ref))
    chosen = None
    for s, e in segments:
        if s <= peak_i <= e:
            chosen = (s, e)
            break
    if chosen is None:
        chosen = max(segments, key=lambda se: se[1] - se[0])

    s, e = chosen
    i = s
    while i > 0 and v_ref[i - 1] >= thr_enter:
        i -= 1
    s = i
    j = e
    while j < n - 1 and v_ref[j + 1] >= thr_enter:
        j += 1
    e = j

    min_len = max(5, int(min_len_frac * n))
    if (e - s + 1) < min_len:
        half = (min_len - (e - s + 1)) // 2
        s = max(0, s - half)
        e = min(n - 1, e + half)

    return s, e


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, delimiter=DELIMITER, encoding=enc)
        except Exception:
            continue
    # last resort
    return pd.read_csv(path, delimiter=DELIMITER)


def _detect_group(path: str) -> str | None:
    lowered = path.lower()
    for group, pats in GROUP_PATTERNS.items():
        for p in pats:
            if p in lowered:
                return group
    return None


def _marker_cols(marker_id: int) -> list[str]:
    return [f"{marker_id}_X", f"{marker_id}_Y", f"{marker_id}_Z"]


def _valid_mask(df: pd.DataFrame, marker_id: int) -> np.ndarray:
    cols = [c for c in _marker_cols(marker_id) if c in df.columns]
    if len(cols) < 3:
        # If any axis is missing, treat as invalid everywhere
        return np.zeros(len(df), dtype=bool)
    return df[cols].notna().all(axis=1).values


def score_file(path: str, group: str) -> dict:
    df = _read_csv_any(path)
    # Ensure numeric for all non-first columns
    frame_col = df.columns[0]
    num = df.drop(columns=[frame_col], errors="ignore").apply(pd.to_numeric, errors="coerce")
    marker_id = MARKER_OF_INTEREST[group]
    valid = _valid_mask(num, marker_id)

    n = len(valid)
    if n == 0:
        return {"score": float("inf"), "reason": "empty", "path": path}

    if not valid.any():
        return {"score": float("inf"), "reason": "no-valid-samples", "path": path}

    first_valid = int(np.argmax(valid))
    last_valid = int(n - 1 - np.argmax(valid[::-1]))
    leading = first_valid
    trailing = n - 1 - last_valid

    inner = valid[first_valid:last_valid + 1]
    inner_missing = int((~inner).sum())
    # count contiguous False runs inside
    if len(inner) > 1:
        gaps = int(np.sum((~inner)[1:] & inner[:-1])) + int((~inner)[0])
    else:
        gaps = 0

    # total rows where any axis missing
    missing_rows = int((~valid).sum())

    # Longest contiguous valid span length
    best_run = 0
    cur = 0
    for v in valid:
        if v:
            cur += 1
            best_run = max(best_run, cur)
        else:
            cur = 0

    # Heuristic score: penalize extrapolation strongly, internal gaps next
    score = 1000 * (leading + trailing) + 50 * inner_missing + 10 * gaps + 5 * missing_rows + 0.001 * (n - best_run)

    return {
        "path": path,
        "score": float(score),
        "n": int(n),
        "leading": int(leading),
        "trailing": int(trailing),
        "inner_missing": int(inner_missing),
        "gaps": int(gaps),
        "missing_rows": int(missing_rows),
        "best_run": int(best_run),
        "valid_fraction": float(valid.sum() / max(1, n)),
    }


def interpolate_and_crop(num: pd.DataFrame, marker_id: int) -> pd.DataFrame:
    """Fill internal gaps; then crop to first..last valid sample to avoid extrapolation."""
    # Fill internal gaps only
    num = num.apply(pd.to_numeric, errors="coerce")
    num = num.astype(float).interpolate(method="linear", limit_area="inside", limit_direction="both")
    # Crop to valid span for the marker of interest
    valid = _valid_mask(num, marker_id)
    if not valid.any():
        return num.iloc[0:0].copy()
    s = int(np.argmax(valid))
    e = int(len(valid) - 1 - np.argmax(valid[::-1]))
    return num.iloc[s:e + 1].reset_index(drop=True)


def compute_va(num: pd.DataFrame, marker_id: int, dt: float, pos_scale: float):
    cols = _marker_cols(marker_id)
    for c in cols:
        if c not in num.columns:
            return None, None, None

    # Outlier guard
    X = {}
    for axis in ("X", "Y", "Z"):
        x = num[f"{marker_id}_{axis}"].values.astype(float) * pos_scale
        if HAMPEL_WINDOW > 0:
            x = _hampel(x, k=HAMPEL_WINDOW, n_sigmas=HAMPEL_SIGMAS)
        w_pos = _cap_window(SG_WINDOW_POS, len(x), min_val=SG_POLY_POS + 2)
        xs = savgol_filter(x, window_length=w_pos, polyorder=SG_POLY_POS, mode="interp")
        w_der = _cap_window(SG_WINDOW_DER, len(xs), min_val=SG_POLY_POS + 2)
        v = savgol_filter(xs, window_length=w_der, polyorder=SG_POLY_POS, deriv=1, delta=dt, mode="interp")
        a = savgol_filter(xs, window_length=w_der, polyorder=SG_POLY_POS, deriv=2, delta=dt, mode="interp")
        X[axis] = (xs, v, a)

    vx, ax = X["X"][1], X["X"][2]
    vy, ay = X["Y"][1], X["Y"][2]
    vz, az = X["Z"][1], X["Z"][2]
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    amag = np.sqrt(ax**2 + ay**2 + az**2)

    if USE_BUTTER_SMOOTH and dt > 0:
        fs = 1.0 / dt
        try:
            wc_v = min(max(VEL_CUTOFF_HZ / (0.5 * fs), 1e-4), 0.99)
            b_v, a_v = butter(BUTTER_ORDER, wc_v, btype="low")
            vmag = filtfilt(b_v, a_v, vmag, method="gust")
            wc_a = min(max(ACC_CUTOFF_HZ / (0.5 * fs), 1e-4), 0.99)
            b_a, a_a = butter(BUTTER_ORDER, wc_a, btype="low")
            amag = filtfilt(b_a, a_a, amag, method="gust")
        except Exception:
            pass

    return vmag, amag, X


def plot_single(group: str, path: str, out_dir: str):
    df = _read_csv_any(path)
    frame_col = df.columns[0]
    num = df.drop(columns=[frame_col], errors="ignore")

    marker_id = MARKER_OF_INTEREST[group]
    num_c = interpolate_and_crop(num, marker_id)
    if len(num_c) < 5:
        raise ValueError(f"{group}: not enough valid samples after cropping in {path}")

    # Time base and units
    dt = 1.0 / float(SAMPLE_RATE_HZ)
    pos_scale = 1.0
    unit_out = "a.u."
    if POSITION_UNIT == "mm":
        pos_scale = 1e-3 if CONVERT_POSITION_TO_METERS else 1.0
        unit_out = "m" if CONVERT_POSITION_TO_METERS else "mm"
    elif POSITION_UNIT == "m":
        pos_scale = 1.0
        unit_out = "m"

    v, a, X = compute_va(num_c, marker_id, dt=dt, pos_scale=pos_scale)
    if v is None or a is None:
        raise ValueError(f"{group}: could not compute v/a for {path}")

    # Tangential acceleration: robustly compute as time derivative of speed
    # a_t = d|v|/dt. This avoids blow-ups when |v| is very small.
    w_at = _cap_window(SG_WINDOW_DER, len(v), min_val=SG_POLY_POS + 2)
    a_t = savgol_filter(v, window_length=w_at, polyorder=SG_POLY_POS, deriv=1, delta=dt, mode="interp")
    # Optional additional smoothing similar to |a| pipeline
    if USE_BUTTER_SMOOTH and dt > 0:
        fs = 1.0 / dt
        try:
            wc_a = min(max(ACC_CUTOFF_HZ / (0.5 * fs), 1e-4), 0.99)
            b_a, a_a = butter(BUTTER_ORDER, wc_a, btype="low")
            a_t = filtfilt(b_a, a_a, a_t, method="gust")
        except Exception:
            pass

    v_ref = v.copy()
    if EMPHASIZE_V:
        v_ref = _emphasize_peak(v_ref, base_window=PEAK_V_WINDOW, poly=PEAK_POLY, rescale_to_max=PEAK_RESCALE_TO_MAX, frac=PEAK_V_WINDOW_FRAC)
    s, e = _find_main_segment(v_ref, enter_frac=0.05, exit_frac=0.03, min_len_frac=0.1)

    x_seconds = np.arange(len(v), dtype=float) * dt
    x_seg = x_seconds[s:e + 1] - x_seconds[s]

    # Emphasize for display
    v_disp = v
    # use tangential acceleration for display/analysis
    a_disp = a_t
    if EMPHASIZE_V:
        v_disp = _emphasize_peak(v_disp, base_window=PEAK_V_WINDOW, poly=PEAK_POLY, rescale_to_max=PEAK_RESCALE_TO_MAX, frac=PEAK_V_WINDOW_FRAC)
    if EMPHASIZE_A:
        a_disp = _emphasize_peak(a_disp, base_window=PEAK_A_WINDOW, poly=PEAK_POLY, rescale_to_max=PEAK_RESCALE_TO_MAX, frac=PEAK_A_WINDOW_FRAC)

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_v.plot(x_seg, v_disp[s:e + 1], label=f"Marker {marker_id}", linewidth=1.4)
    ax_a.plot(x_seg, a_disp[s:e + 1], label=f"Marker {marker_id}", linewidth=1.2)
    ax_v.set_title("Geschwindigkeit (||v||)")
    ax_a.set_title("Tangentialbeschleunigung (a_t)")
    ax_a.set_xlabel("Zeit (s)")
    ax_v.set_ylabel(f"||v|| [{unit_out}/s]")
    ax_a.set_ylabel(f"a_t [{unit_out}/s²]")
    ax_v.grid(True, alpha=0.2)
    ax_a.grid(True, alpha=0.2)
    ax_v.legend(); ax_a.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{group}_single_velocity_acc_scalar.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    fig_vo, ax_vo = plt.subplots(1, 1, figsize=(10, 4))
    ax_vo.plot(x_seg, v_disp[s:e + 1], label=f"Marker {marker_id}", linewidth=1.4)
    ax_vo.set_title("Geschwindigkeit (||v||)")
    ax_vo.set_xlabel("Zeit (s)")
    ax_vo.set_ylabel(f"||v|| [{unit_out}/s]")
    ax_vo.grid(True, alpha=0.2)
    ax_vo.legend()
    fig_vo.tight_layout()
    fig_vo.savefig(os.path.join(out_dir, f"{group}_single_velocity_scalar.png"), dpi=160)
    plt.close(fig_vo)

    # Save cropped, smoothed positions for reproducibility
    os.makedirs(OUTPUT_SELECTED, exist_ok=True)
    # Re-assemble into a CSV with a synthetic 0..N-1 frame index
    out_df = num_c.copy()
    out_df.insert(0, "Frame", np.arange(len(out_df)))
    out_df.to_csv(os.path.join(OUTPUT_SELECTED, f"{group}_single_clean.csv"), index=False, sep=DELIMITER)

    # Return summary
    v_seg = v[s:e + 1]
    peak_idx = int(np.argmax(v_seg)) if len(v_seg) else 0
    return {
        "Group": group,
        "SourcePath": path,
        "SamplesUsed": int(len(num_c)),
        "PeakTime_s": float(x_seg[peak_idx]) if len(x_seg) else float("nan"),
        "MaxVelocity": float(np.max(v_seg)) if len(v_seg) else float("nan"),
        "VelocityUnits": f"{unit_out}/s",
    }


def main():
    # --- Discover candidates ---
    all_paths = [p for p in glob(f"{INPUT_ROOT}/**/*.csv", recursive=True)]
    by_group: dict[str, list[str]] = {g: [] for g in GROUP_PATTERNS}
    for p in all_paths:
        g = _detect_group(p)
        if g in by_group:
            by_group[g].append(p)

    # --- Score and select best per group ---
    selections = {}
    score_rows = []
    for group, paths in by_group.items():
        if not paths:
            continue
        scored = [score_file(p, group) for p in paths]
        scored = [s for s in scored if np.isfinite(s.get("score", np.inf))]
        if not scored:
            continue
        scored.sort(key=lambda d: d["score"])  # lowest is best
        best = scored[0]
        selections[group] = best["path"]
        for r in scored:
            r_out = r.copy()
            r_out["Group"] = group
            score_rows.append(r_out)

    if not selections:
        print("No candidates found. Ensure Exports/Daten_Raw_Clean exists and file names contain ptp/zikzak/gewicht.")
        return

    # --- Process selections and plot ---
    os.makedirs(OUTPUT_PLOTS, exist_ok=True)
    summaries = []
    for group, path in selections.items():
        try:
            print(f"\n▶ Processing best file for {group}: {path}")
            summary = plot_single(group, path, OUTPUT_PLOTS)
            summaries.append(summary)
        except Exception as e:
            print(f"❌ Failed for {group}: {e}")

    # --- Persist selection summary ---
    if score_rows:
        pd.DataFrame(score_rows).to_csv(
            os.path.join(OUTPUT_PLOTS, "Step11_selection_summary.csv"), index=False, sep=DELIMITER
        )
    if summaries:
        pd.DataFrame(summaries).to_csv(
            os.path.join(OUTPUT_PLOTS, "Step11_run_summary.csv"), index=False, sep=DELIMITER
        )


if __name__ == "__main__":
    main()
