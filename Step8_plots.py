import os
import re
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt


# --- CONFIG ---
INPUT_DIR = "Exports/Final_Cleaned"
OUTPUT_DIR = "Exports/Plots"
DELIMITER = ";"

# If empty, marker IDs are inferred from columns like "3_X", "3_Y", "3_Z"
MARKERS = []
AXES = ["X", "Y", "Z"]

# Savitzky–Golay parameters
SG_POLY_POS = 3
SG_WINDOW_POS = 21  # smooth positions first (odd)
SG_WINDOW_DER = 21  # window for SG differentiation (odd)

# Optional robust outlier guard before smoothing (Hampel-like)
HAMPEL_WINDOW = 7     # half-window; set 0 to disable
HAMPEL_SIGMAS = 3.0

# Emphasis to suppress smaller bumps (post-processing on magnitudes)
# Apply strong, adaptive smoothing to velocity/acceleration magnitudes
EMPHASIZE_VELOCITY = True
EMPHASIZE_ACCELERATION = False

# Fraction of series length used to adapt the smoothing window
# e.g., 0.33 -> window ≈ 33% of series length (capped to valid odd length)
PEAK_V_WINDOW_FRAC = 0.33
PEAK_A_WINDOW_FRAC = 0.25

# Minimum/base windows (will be increased by *_FRAC if that yields larger window)
PEAK_V_WINDOW = 41
PEAK_A_WINDOW = 41

# Polyorder for the emphasis smoother
PEAK_POLY = 2

# Keep the original peak height after emphasis smoothing
PEAK_RESCALE_TO_MAX = True

# --- Units & Time Base ---
TIME_COLUMN: str | None = None  # e.g., "Time"
SAMPLE_RATE_HZ: float | None = 200.0
TIME_AXIS_MODE: str = "percent"

POSITION_UNIT: str | None = "m"
CONVERT_POSITION_TO_METERS: bool = False

# Optional zero-phase low-pass for magnitudes to suppress noise strongly
USE_BUTTER_SMOOTH = True
BUTTER_ORDER = 4
VEL_CUTOFF_HZ = 6.0
ACC_CUTOFF_HZ = 8.0


# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
files = glob(f"{INPUT_DIR}/*_final_clean.csv")
summary_rows = []  # collect per-aufgabe velocity + acceleration stats


def _ensure_odd(n: int, *, min_val: int = 5) -> int:
    n = int(max(n, min_val))
    return n if n % 2 == 1 else n + 1


def _cap_window(target: int, series_len: int, *, min_val: int = 5) -> int:
    if series_len <= 1:
        return _ensure_odd(min_val)
    max_allowed = series_len - 1 if series_len % 2 == 0 else series_len
    return _ensure_odd(min(target, max_allowed), min_val=min_val)


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


def _infer_markers(columns) -> list[int]:
    ids = set()
    for c in columns:
        m = re.match(r"^(\d+)_([XYZ])$", str(c))
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def compute_scalar_velocity_and_acceleration(df: pd.DataFrame, marker_id: int, dt: float, pos_scale: float = 1.0):
    """
    Compute smoothed ||v||, ||a|| and tangential acceleration a_tang = d||v||/dt for a marker.
    Returns: (vel_smooth, acc_smooth, a_tang)
    """
    pos_axes = {}
    for axis in AXES:
        col = f"{marker_id}_{axis}"
        if col not in df.columns:
            return None, None, None
        x = df[col].astype(float).values * float(pos_scale)
        if HAMPEL_WINDOW > 0:
            x = _hampel(x, k=HAMPEL_WINDOW, n_sigmas=HAMPEL_SIGMAS)
        w_pos = _cap_window(SG_WINDOW_POS, len(x), min_val=SG_POLY_POS + 2)
        x_s = savgol_filter(x, window_length=w_pos, polyorder=SG_POLY_POS, mode="interp")
        w_der = _cap_window(SG_WINDOW_DER, len(x_s), min_val=SG_POLY_POS + 2)
        v = savgol_filter(x_s, window_length=w_der, polyorder=SG_POLY_POS, deriv=1, delta=dt, mode="interp")
        a = savgol_filter(x_s, window_length=w_der, polyorder=SG_POLY_POS, deriv=2, delta=dt, mode="interp")
        pos_axes[axis] = (x_s, v, a)

    vx, ax = pos_axes["X"][1], pos_axes["X"][2]
    vy, ay = pos_axes["Y"][1], pos_axes["Y"][2]
    vz, az = pos_axes["Z"][1], pos_axes["Z"][2]

    vel_norm = np.sqrt(vx**2 + vy**2 + vz**2)
    acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
    a_tang = savgol_filter(vel_norm, window_length=_cap_window(SG_WINDOW_DER, len(vel_norm), min_val=SG_POLY_POS + 2),
                           polyorder=SG_POLY_POS, deriv=1, delta=dt, mode="interp")

    if USE_BUTTER_SMOOTH and dt > 0 and np.isfinite(dt):
        fs = 1.0 / dt
        try:
            wc_v = min(max(VEL_CUTOFF_HZ / (0.5 * fs), 1e-4), 0.99)
            b_v, a_v = butter(BUTTER_ORDER, wc_v, btype="low")
            vel_smooth = filtfilt(b_v, a_v, vel_norm, method="gust")

            wc_a = min(max(ACC_CUTOFF_HZ / (0.5 * fs), 1e-4), 0.99)
            b_a, a_a = butter(BUTTER_ORDER, wc_a, btype="low")
            acc_smooth = filtfilt(b_a, a_a, acc_norm, method="gust")
        except Exception:
            w_vmag = _cap_window(9, len(vel_norm), min_val=5)
            w_amag = _cap_window(31, len(acc_norm), min_val=5)
            vel_smooth = savgol_filter(vel_norm, window_length=w_vmag, polyorder=2, mode="interp")
            acc_smooth = savgol_filter(acc_norm, window_length=w_amag, polyorder=2, mode="interp")
    else:
        w_vmag = _cap_window(9, len(vel_norm), min_val=5)
        w_amag = _cap_window(31, len(acc_norm), min_val=5)
        vel_smooth = savgol_filter(vel_norm, window_length=w_vmag, polyorder=2, mode="interp")
        acc_smooth = savgol_filter(acc_norm, window_length=w_amag, polyorder=2, mode="interp")

    return vel_smooth, acc_smooth, a_tang


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


def _renormalize_percent(t_segment: np.ndarray) -> np.ndarray:
    if len(t_segment) == 0:
        return t_segment
    t0, t1 = float(t_segment[0]), float(t_segment[-1])
    if t1 == t0:
        return np.zeros_like(t_segment)
    return (t_segment - t0) / (t1 - t0) * 100.0


# --- Plotting ---
for path in files:
    name = os.path.basename(path).replace("_final_clean.csv", "")
    print(f"\nPlotting scalar curves for: {name}")

    try:
        df = pd.read_csv(path, delimiter=DELIMITER)

        # --- Determine time base and dt ---
        use_seconds = False
        t_seconds = None
        dt = 1.0
        if TIME_COLUMN and TIME_COLUMN in df.columns:
            t_seconds = df[TIME_COLUMN].astype(float).values
            if len(t_seconds) > 1:
                dt = float(np.mean(np.diff(t_seconds)))
            use_seconds = True
        elif SAMPLE_RATE_HZ is not None:
            try:
                sr = float(SAMPLE_RATE_HZ)
            except Exception:
                sr = None
            if sr and np.isfinite(sr) and sr > 0:
                t_seconds = np.arange(len(df), dtype=float) / sr
                dt = 1.0 / sr
                use_seconds = True
        else:
            if "Frame" in df.columns:
                t_percent = df["Frame"].astype(float).values
                if len(t_percent) > 1:
                    dt = float(np.mean(np.diff(t_percent)))

        # Infer marker IDs if not provided
        markers_to_use = MARKERS or _infer_markers(df.columns)
        if not markers_to_use:
            raise ValueError("No marker columns like '<id>_X|Y|Z' found")

        # Compute v,a for all markers first
        per_marker = {}
        v_stack = []
        a_tang_stack = []
        pos_scale = 1.0
        pos_unit_out = "a.u."
        if use_seconds:
            if POSITION_UNIT == "mm":
                if CONVERT_POSITION_TO_METERS:
                    pos_scale = 1e-3
                    pos_unit_out = "m"
                else:
                    pos_scale = 1.0
                    pos_unit_out = "mm"
            elif POSITION_UNIT == "m":
                pos_scale = 1.0
                pos_unit_out = "m"

        for marker_id in markers_to_use:
            v, a, a_tang = compute_scalar_velocity_and_acceleration(df, marker_id, dt=dt, pos_scale=pos_scale)
            if v is None or a is None or a_tang is None:
                continue
            per_marker[marker_id] = (v, a, a_tang)
            v_stack.append(v)
            a_tang_stack.append(a_tang)

        if not per_marker:
            raise ValueError("No valid marker velocities computed")

        v_stack = np.vstack(v_stack)
        a_tang_stack = np.vstack(a_tang_stack)

        # Reference velocity as median across markers (unemphasized)
        v_ref_raw = np.median(v_stack, axis=0)
        v_ref = v_ref_raw.copy()

        # Median tangential acceleration across markers for peak detection
        a_tang_ref_raw = np.median(a_tang_stack, axis=0)

        # OPTIONAL: Apply emphasis on the reference velocity before segment detection
        if EMPHASIZE_VELOCITY:
            v_ref = _emphasize_peak(
                v_ref,
                base_window=PEAK_V_WINDOW,
                poly=PEAK_POLY,
                rescale_to_max=PEAK_RESCALE_TO_MAX,
                frac=PEAK_V_WINDOW_FRAC,
            )

        # Detect main movement segment via hysteresis thresholds on velocity
        s, e = _find_main_segment(v_ref, enter_frac=0.05, exit_frac=0.03, min_len_frac=0.1)

        # Time handling for plotting axis
        if TIME_AXIS_MODE.lower() == "seconds" and use_seconds and t_seconds is not None:
            x_plot = t_seconds[s:e + 1] - t_seconds[s]
            xlabel = "Zeit (s)"
        elif TIME_AXIS_MODE.lower() == "frames":
            x_plot = np.arange(e - s + 1, dtype=float)
            xlabel = "Frames"
        else:
            if use_seconds and t_seconds is not None:
                t_base = t_seconds
            else:
                t_base = df["Frame"].astype(float).values if "Frame" in df.columns else np.arange(len(df), dtype=float)
            time_seg = t_base[s:e + 1]
            x_plot = _renormalize_percent(time_seg)
            xlabel = "Bewegungsfortschritt (%)"

        # --- Velocity scalar stats (within segment, on UNemphasized ref) ---
        v_seg = v_ref_raw[s:e + 1]
        max_vel = float(np.max(v_seg)) if len(v_seg) > 0 else float("nan")
        median_vel = float(np.median(v_seg)) if len(v_seg) > 0 else float("nan")

        if use_seconds and t_seconds is not None:
            t_base = t_seconds
        else:
            t_base = df["Frame"].astype(float).values if "Frame" in df.columns else np.arange(len(df), dtype=float)
        seg_time = t_base[s:e + 1]
        seg_pct = _renormalize_percent(seg_time)
        if len(v_seg) > 0:
            peak_local_idx = int(np.argmax(v_seg))
            peak_pct = float(seg_pct[peak_local_idx])
        else:
            peak_pct = float("nan")

        # --- Tangential acceleration peak stats (within the same segment) ---
        a_tang_seg = a_tang_ref_raw[s:e + 1]
        if len(a_tang_seg) > 0:
            pos_idx = int(np.argmax(a_tang_seg))
            neg_idx = int(np.argmin(a_tang_seg))
            pos_accel_peak = float(a_tang_seg[pos_idx])
            neg_accel_peak = float(a_tang_seg[neg_idx])
            pos_accel_pct = float(seg_pct[pos_idx])
            neg_accel_pct = float(seg_pct[neg_idx])
        else:
            pos_accel_peak = float("nan")
            neg_accel_peak = float("nan")
            pos_accel_pct = float("nan")
            neg_accel_pct = float("nan")

        vel_units_here = (f"{pos_unit_out}/s" if use_seconds else "a.u./%")
        acc_units_here = (f"{pos_unit_out}/s²" if use_seconds else "a.u./%²")

        summary_rows.append({
            "Aufgabe": name,
            "MaxVelocity": max_vel,
            "MedianVelocity": median_vel,
            "VelocityUnits": vel_units_here,
            "PeakVelocityPercent": peak_pct,
            "PosAccelPeak": pos_accel_peak,
            "NegAccelPeak": neg_accel_peak,
            "AccelUnits": acc_units_here,
            "PosAccelPeakPercent": pos_accel_pct,
            "NegAccelPeakPercent": neg_accel_pct,
        })

        # --- Plotting ---
        fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"{name} | Betrag von Geschwindigkeit und Beschleunigung", fontsize=14)

        trim = max(1, int(0.01 * (e - s + 1)))  # 1 % Rand weg
        sl = slice(s + trim, e + 1 - trim)

        for marker_id, (v, a, a_tang) in per_marker.items():
            if EMPHASIZE_VELOCITY:
                v = _emphasize_peak(v, base_window=PEAK_V_WINDOW, poly=PEAK_POLY,
                                    rescale_to_max=PEAK_RESCALE_TO_MAX, frac=PEAK_V_WINDOW_FRAC)
            ax_v.plot(x_plot[trim:-trim] if trim > 0 else x_plot, v[sl], label=f"Marker {marker_id}", linewidth=1.4)
            ax_a.plot(x_plot[trim:-trim] if trim > 0 else x_plot, a_tang[sl], label=f"Marker {marker_id}", linewidth=1.2)

        if use_seconds:
            vel_units = f"{pos_unit_out}/s"
            acc_units = f"{pos_unit_out}/s²"
        else:
            vel_units = "a.u./%"
            acc_units = "a.u./%²"

        ax_v.set_title("Geschwindigkeit (||v||)")
        ax_a.set_title("Tangentiale Beschleunigung (d||v||/dt)")
        ax_a.set_xlabel(xlabel)
        ax_v.set_ylabel(f"||v|| [{vel_units}]")
        ax_a.set_ylabel(f"a_tang [{acc_units}]")
        ax_v.grid(True, alpha=0.2)
        ax_a.grid(True, alpha=0.2)
        ax_v.legend()
        ax_a.legend()
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = os.path.join(OUTPUT_DIR, f"{name}_velocity_acc_scalar.png")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot: {out_path}")
        print(
            f"Summary for {name}: max_v={max_vel:.3f} {vel_units_here} @ {peak_pct:.1f}%, "
            f"median_v={median_vel:.3f} {vel_units_here}, "
            f"pos_a={pos_accel_peak:.3f} {acc_units_here} @ {pos_accel_pct:.1f}%, "
            f"neg_a={neg_accel_peak:.3f} {acc_units_here} @ {neg_accel_pct:.1f}%"
        )

        # --- Velocity-only plot ---
        fig_vo, ax_vo = plt.subplots(1, 1, figsize=(10, 4))
        for marker_id, (v, _a, _a_tang) in per_marker.items():
            if EMPHASIZE_VELOCITY:
                v = _emphasize_peak(
                    v,
                    base_window=PEAK_V_WINDOW,
                    poly=PEAK_POLY,
                    rescale_to_max=PEAK_RESCALE_TO_MAX,
                    frac=PEAK_V_WINDOW_FRAC,
                )
            ax_vo.plot(x_plot, v[s:e + 1], label=f"Marker {marker_id}", linewidth=1.4)
        ax_vo.set_title("Geschwindigkeit (||v||)")
        ax_vo.set_xlabel(xlabel)
        ax_vo.set_ylabel(f"||v|| [{vel_units}]")
        ax_vo.grid(True, alpha=0.2)
        ax_vo.legend()
        fig_vo.tight_layout()
        out_vo_path = os.path.join(OUTPUT_DIR, f"{name}_velocity_scalar.png")
        fig_vo.savefig(out_vo_path, dpi=160)
        plt.close(fig_vo)
        print(f"Saved velocity-only plot: {out_vo_path}")

    except Exception as e:
        print(f"Failed for {name}: {e}")

# Write summary CSV for all Aufgaben
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "velocity_summary.csv")
    summary_df.to_csv(summary_path, index=False, sep=DELIMITER)
    print(f"\nSaved velocity summary: {summary_path}")
