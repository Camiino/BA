import os
import re
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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
PEAK_A_WINDOW_FRAC = None

# Minimum/base windows (will be increased by *_FRAC if that yields larger window)
PEAK_V_WINDOW = 41
PEAK_A_WINDOW = 41

# Polyorder for the emphasis smoother
PEAK_POLY = 2

# Keep the original peak height after emphasis smoothing
PEAK_RESCALE_TO_MAX = True


# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
files = glob(f"{INPUT_DIR}/*_final_clean.csv")


def _ensure_odd(n: int, *, min_val: int = 5) -> int:
    n = int(max(n, min_val))
    return n if n % 2 == 1 else n + 1


def _cap_window(target: int, series_len: int, *, min_val: int = 5) -> int:
    if series_len <= 1:
        return _ensure_odd(min_val)
    # keep below length and odd
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


def compute_scalar_velocity_and_acceleration(df: pd.DataFrame, marker_id: int):
    """
    Returns smoothed ||v|| and ||a|| for a given marker using SG smoothing + SG derivatives.
    Time base in percent (0..100), used only to scale derivatives via delta.
    """
    t = df["Frame"].astype(float).values
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0

    pos_axes = {}
    for axis in AXES:
        col = f"{marker_id}_{axis}"
        if col not in df.columns:
            return None, None
        x = df[col].astype(float).values
        # Outlier guard
        if HAMPEL_WINDOW > 0:
            x = _hampel(x, k=HAMPEL_WINDOW, n_sigmas=HAMPEL_SIGMAS)
        # Smooth positions first
        w_pos = _cap_window(SG_WINDOW_POS, len(x), min_val=SG_POLY_POS + 2)
        x_s = savgol_filter(x, window_length=w_pos, polyorder=SG_POLY_POS, mode="interp")
        # Derivatives via Savitzky–Golay
        w_der = _cap_window(SG_WINDOW_DER, len(x_s), min_val=SG_POLY_POS + 2)
        v = savgol_filter(x_s, window_length=w_der, polyorder=SG_POLY_POS, deriv=1, delta=dt, mode="interp")
        a = savgol_filter(x_s, window_length=w_der, polyorder=SG_POLY_POS, deriv=2, delta=dt, mode="interp")
        pos_axes[axis] = (x_s, v, a)

    vx, ax = pos_axes["X"][1], pos_axes["X"][2]
    vy, ay = pos_axes["Y"][1], pos_axes["Y"][2]
    vz, az = pos_axes["Z"][1], pos_axes["Z"][2]

    vel_norm = np.sqrt(vx**2 + vy**2 + vz**2)
    acc_norm = np.sqrt(ax**2 + ay**2 + az**2)

    # Light final smoothing on magnitudes to suppress residual ripples (keep peak shape)
    w_vmag = _cap_window(9, len(vel_norm), min_val=5)
    w_amag = _cap_window(15, len(acc_norm), min_val=5)
    vel_smooth = savgol_filter(vel_norm, window_length=w_vmag, polyorder=2, mode="interp")
    acc_smooth = savgol_filter(acc_norm, window_length=w_amag, polyorder=2, mode="interp")

    return vel_smooth, acc_smooth


def _find_main_segment(v_ref: np.ndarray, enter_frac: float = 0.05, exit_frac: float = 0.03, min_len_frac: float = 0.1):
    n = len(v_ref)
    if n == 0:
        return 0, -1
    peak = float(np.nanmax(v_ref))
    if not np.isfinite(peak) or peak <= 0:
        return 0, n - 1
    thr_enter = enter_frac * peak
    thr_exit = exit_frac * peak

    # Hysteresis base mask on exit threshold
    mask = v_ref > thr_exit
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0, n - 1

    # Split into contiguous segments
    cuts = np.where(np.diff(idx) > 1)[0]
    segments = []
    start = 0
    for c in cuts:
        segments.append((idx[start], idx[c]))
        start = c + 1
    segments.append((idx[start], idx[-1]))

    # Choose segment containing global peak
    peak_i = int(np.argmax(v_ref))
    chosen = None
    for s, e in segments:
        if s <= peak_i <= e:
            chosen = (s, e)
            break
    if chosen is None:
        # fallback: longest segment
        chosen = max(segments, key=lambda se: se[1] - se[0])

    s, e = chosen
    # Expand to enter threshold crossings (hysteresis)
    i = s
    while i > 0 and v_ref[i - 1] >= thr_enter:
        i -= 1
    s = i
    j = e
    while j < n - 1 and v_ref[j + 1] >= thr_enter:
        j += 1
    e = j

    # Ensure minimum duration
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

        # Infer marker IDs if not provided
        markers_to_use = MARKERS or _infer_markers(df.columns)
        if not markers_to_use:
            raise ValueError("No marker columns like '<id>_X|Y|Z' found")

        # Compute v,a for all markers first
        per_marker = {}
        v_stack = []
        for marker_id in markers_to_use:
            v, a = compute_scalar_velocity_and_acceleration(df, marker_id)
            if v is None or a is None:
                continue
            per_marker[marker_id] = (v, a)
            v_stack.append(v)

        if not per_marker:
            raise ValueError("No valid marker velocities computed")

        v_stack = np.vstack(v_stack)
        v_ref = np.median(v_stack, axis=0)

        # OPTIONAL: Apply emphasis on the reference velocity before segment detection
        if EMPHASIZE_VELOCITY:
            v_ref = _emphasize_peak(
                v_ref,
                base_window=PEAK_V_WINDOW,
                poly=PEAK_POLY,
                rescale_to_max=PEAK_RESCALE_TO_MAX,
                frac=PEAK_V_WINDOW_FRAC,
            )

        # Detect main movement segment via hysteresis thresholds
        s, e = _find_main_segment(v_ref, enter_frac=0.05, exit_frac=0.03, min_len_frac=0.1)

        # Time handling: slice to segment and re-normalize to 0..100 for plotting
        time_full = df["Frame"].astype(float).values
        time_seg = time_full[s:e + 1]
        time_pct = _renormalize_percent(time_seg)

        fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"{name} | Betrag von Geschwindigkeit und Beschleunigung", fontsize=14)

        for marker_id, (v, a) in per_marker.items():
            # OPTIONAL: Emphasize individual curves for display
            if EMPHASIZE_VELOCITY:
                v = _emphasize_peak(
                    v,
                    base_window=PEAK_V_WINDOW,
                    poly=PEAK_POLY,
                    rescale_to_max=PEAK_RESCALE_TO_MAX,
                    frac=PEAK_V_WINDOW_FRAC,
                )
            if EMPHASIZE_ACCELERATION:
                a = _emphasize_peak(
                    a,
                    base_window=PEAK_A_WINDOW,
                    poly=PEAK_POLY,
                    rescale_to_max=PEAK_RESCALE_TO_MAX,
                    frac=PEAK_A_WINDOW_FRAC,
                )

            ax_v.plot(time_pct, v[s:e + 1], label=f"Marker {marker_id}", linewidth=1.4)
            ax_a.plot(time_pct, a[s:e + 1], label=f"Marker {marker_id}", linewidth=1.2)

        ax_v.set_title("Geschwindigkeit (||v||)")
        ax_a.set_title("Beschleunigung (||a||)")
        ax_a.set_xlabel("Bewegungsfortschritt (%)")
        ax_v.set_ylabel("||v|| [a.u./%]")
        ax_a.set_ylabel("||a|| [a.u./%²]")
        ax_v.grid(True, alpha=0.2)
        ax_a.grid(True, alpha=0.2)
        ax_v.legend()
        ax_a.legend()
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = os.path.join(OUTPUT_DIR, f"{name}_velocity_acc_scalar.png")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    except Exception as e:
        print(f"Failed for {name}: {e}")
