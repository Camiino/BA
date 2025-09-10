#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import subprocess
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
from matplotlib.animation import FuncAnimation


def sniff_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(1024)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])  # type: ignore
        return dialect.delimiter
    except Exception:
        # Fallback: guess by presence
        if ";" in sample and sample.count(";") >= sample.count(","):
            return ";"
        return ","


def load_series(path: str, marker_id: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delim = sniff_delimiter(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)

        def idx(col: str) -> Optional[int]:
            try:
                return header.index(col)
            except ValueError:
                return None

        i_frame = idx("Frame") or idx("frame") or idx("frame_%")
        i_x = idx(f"{marker_id}_X")
        i_y = idx(f"{marker_id}_Y")
        i_z = idx(f"{marker_id}_Z")
        if i_x is None or i_y is None or i_z is None:
            raise ValueError(
                f"Columns for marker {marker_id} not found. Expected {marker_id}_X,{marker_id}_Y,{marker_id}_Z"
            )

        frames: List[float] = []
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        for row in reader:
            if not row:
                continue
            try:
                x = float(row[i_x])
                y = float(row[i_y])
                z = float(row[i_z])
                xs.append(x)
                ys.append(y)
                zs.append(z)
                if i_frame is not None:
                    frames.append(float(row[i_frame]))
            except Exception:
                # Skip malformed rows
                continue

        n = len(xs)
        if n == 0:
            raise ValueError("No valid rows found in CSV")
        if i_frame is None:
            frames = [0.0 if n == 1 else (100.0 * i / (n - 1)) for i in range(n)]

        return np.array(frames), np.array(xs), np.array(ys), np.array(zs)


def set_axes_equal(ax, x, y, z):
    # Enforce equal aspect ratio for 3D plots
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)
    max_range = max(x_range, y_range, z_range, 1.0)

    x_mid = (np.max(x) + np.min(x)) / 2.0
    y_mid = (np.max(y) + np.min(y)) / 2.0
    z_mid = (np.max(z) + np.min(z)) / 2.0

    half = max_range / 2.0
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def build_projection(x, y, z, mode: str, plane_z: Optional[float] = None):
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    if mode == "xy":
        zp = np.min(z) if plane_z is None else float(plane_z)
        return x, y, np.full_like(x, zp)
    if mode == "xz":
        yp = np.min(y)
        return x, np.full_like(x, yp), z
    if mode == "yz":
        xp = np.min(x)
        return np.full_like(y, xp), y, z
    raise ValueError("projection must be one of: none, xy, xz, yz")


def _ask_choice() -> str:
    print("Which trajectory to view?\n  [1] Kreis\n  [2] ZickZack")
    while True:
        choice = input("Enter 1/2 (or 'k'/'z', 'q' to quit): ").strip().lower()
        if choice in ("1", "k", "kreis"):  
            return "kreis"
        if choice in ("2", "z", "zz", "zickzack", "zickzak", "zikzak"):  
            return "zickzack"
        if choice in ("q", "quit", "exit"):
            raise SystemExit(0)
        print("Invalid choice, please enter 1, 2, k, z, or q.")


def _ensure_musterverlauf(kind: str, script_dir: str) -> str:
    assert kind in ("kreis", "zickzack")
    filename = f"Musterverlauf_{'Kreis' if kind=='kreis' else 'ZickZack'}.csv"
    p1 = os.path.join(script_dir, filename)
    p2 = os.path.join(os.getcwd(), filename)
    if os.path.isfile(p1):
        return p1
    if os.path.isfile(p2):
        return p2

    # Try to generate via create_musterverlauf.py
    gen = os.path.join(script_dir, "create_musterverlauf.py")
    if os.path.isfile(gen):
        try:
            print(f"Generating {filename} via create_musterverlauf.py ...")
            cmd = [sys.executable or "python", gen, "--outdir", script_dir]
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"Warning: could not auto-generate: {e}")
    if os.path.isfile(p1):
        return p1
    if os.path.isfile(p2):
        return p2
    raise SystemExit(f"Could not find or generate {filename}. Please create it first.")


def main():
    parser = argparse.ArgumentParser(description="Animate 3D point with path and projection from Musterverlauf CSV")
    parser.add_argument("csv", nargs="?", default=None, help="Input CSV path or shortcut: 'kreis' or 'zickzack' (default: ask)")
    parser.add_argument("--marker-id", type=int, default=1, help="Marker ID to animate (default: 1)")
    parser.add_argument("--projection", choices=["none", "xy", "xz", "yz"], default="none", help="Projection plane for path shadow (default: none)")
    parser.add_argument("--plane-z", type=float, default=None, help="Z height for XY projection plane (default: z-min)")
    parser.add_argument("--interval", type=int, default=30, help="Animation frame interval in ms (default: 30)")
    parser.add_argument("--fps", type=int, default=30, help="Save FPS if --save is used (default: 30)")
    parser.add_argument("--save", default=None, help="Optional path to save animation (mp4/gif)")
    parser.add_argument("--dpi", type=int, default=120, help="DPI for saving (default: 120)")
    parser.add_argument("--elev", type=float, default=None, help="Fixed elevation angle in degrees (locks camera if set)")
    parser.add_argument("--azim", type=float, default=None, help="Fixed azimuth angle in degrees (locks camera if set)")
    parser.add_argument("--lock-view", action="store_true", help="Lock view: no autoscale; use --elev/--azim if provided")
    parser.add_argument("--full-path", action="store_true", help="Show full 3D path as faint line (default: off)")
    parser.add_argument("--projection-path", action="store_true", help="Draw full projected path (default: off)")
    parser.add_argument("--projection-trail", action="store_true", help="Draw projected trail (default: off)")
    parser.add_argument("--projection-point", action="store_true", help="Draw projected moving point (default: off)")
    # XY top-down lock (default on); use --free-view to disable
    parser.add_argument("--xy-view", dest="xy_view", action="store_true", default=True, help="Lock to top-down XY view (default: on)")
    parser.add_argument("--free-view", dest="xy_view", action="store_false", help="Disable XY lock and allow free 3D rotation")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = args.csv

    if path is None or path.strip().lower() in ("kreis", "k", "1", "zickzack", "zickzak", "zikzak", "z", "zz", "2"):
        # Interactive selection if not explicitly provided
        if path is None or path.strip().lower() in ("kreis", "k", "1"):
            kind = "kreis" if path else _ask_choice()
            if path and kind != "kreis":
                # If user typed e.g. '2' we already handled above; keep robust
                kind = "kreis"
        elif path.strip().lower() in ("zickzack", "zickzak", "zikzak", "z", "zz", "2"):
            kind = "zickzack"
        else:
            kind = _ask_choice()
        path = _ensure_musterverlauf(kind, script_dir)
    else:
        # Resolve given path or try script-relative
        if not os.path.isfile(path):
            alt = os.path.join(script_dir, path)
            if os.path.isfile(alt):
                path = alt
            else:
                raise SystemExit(f"File not found: {args.csv}")

    t, x, y, z = load_series(path, marker_id=args.marker_id)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(os.path.basename(path))

    # Optional full 3D path (faint)
    line_full = None
    if args.full_path:
        line_full, = ax.plot(x, y, z, color="#999999", linewidth=1.0, alpha=0.5, label="path")

    # Trail of traveled segment
    line_trail, = ax.plot([], [], [], color="#1f77b4", linewidth=2.0, label="trail")

    # Current point
    point, = ax.plot([], [], [], "o", color="#d62728", markersize=6, label="point")

    # Projection (shadow) lines and point
    proj = build_projection(x, y, z, args.projection, args.plane_z)
    if proj is not None:
        xp, yp, zp = proj
        proj_full = None
        if args.projection_path:
            proj_full, = ax.plot(xp, yp, zp, color="#cccccc", linewidth=1.0, alpha=0.6, label=f"{args.projection}-proj")
        proj_trail = None
        if args.projection_trail:
            proj_trail, = ax.plot([], [], [], color="#2ca02c", linewidth=1.5, alpha=0.9, label="proj trail")
        proj_point = None
        if args.projection_point:
            proj_point, = ax.plot([], [], [], "o", color="#2ca02c", markersize=4, alpha=0.9)
    else:
        proj_full = proj_trail = proj_point = None

    # Axes limits and equal aspect
    set_axes_equal(ax, x, y, z)
    # Camera/view handling: XY top-down lock by default
    if args.xy_view:
        try:
            ax.set_proj_type('ortho')  # orthographic (no perspective skew)
        except Exception:
            pass
        ax.view_init(elev=90, azim=-90)  # top-down onto XY plane
        ax.set_autoscale_on(False)
        # Optional: hide Z ticks since z is not relevant
        try:
            ax.set_zticks([])
        except Exception:
            pass
    else:
        if args.elev is not None or args.azim is not None:
            ax.view_init(elev=args.elev if args.elev is not None else ax.elev,
                         azim=args.azim if args.azim is not None else ax.azim)
        if args.lock_view:
            ax.set_autoscale_on(False)
    ax.legend(loc="upper right")

    max_frames = len(x)

    def update(frame):
        # Update point
        point.set_data([x[frame]], [y[frame]])
        point.set_3d_properties([z[frame]])

        # Update trail (0..frame)
        line_trail.set_data(x[:frame + 1], y[:frame + 1])
        line_trail.set_3d_properties(z[:frame + 1])

        # Projection updates
        artists = [line_trail, point]
        if proj is not None:
            if proj_trail is not None:
                proj_trail.set_data(xp[:frame + 1], yp[:frame + 1])
                proj_trail.set_3d_properties(zp[:frame + 1])
                artists.append(proj_trail)
            if proj_point is not None:
                proj_point.set_data([xp[frame]], [yp[frame]])
                proj_point.set_3d_properties([zp[frame]])
                artists.append(proj_point)

        return tuple(artists)

    ani = FuncAnimation(fig, update, frames=max_frames, interval=args.interval, blit=True)

    if args.save:
        out = args.save
        root, ext = os.path.splitext(out)
        if ext.lower() in (".mp4", ".m4v"):
            try:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=args.fps, bitrate=1800)
                ani.save(out, writer=writer, dpi=args.dpi)
                print(f"Saved MP4: {out}")
            except Exception as e:
                print(f"Could not save MP4 (ffmpeg missing?): {e}")
        elif ext.lower() in (".gif",):
            try:
                ani.save(out, writer="pillow", dpi=args.dpi, fps=args.fps)
                print(f"Saved GIF: {out}")
            except Exception as e:
                print(f"Could not save GIF: {e}")
        else:
            print("Use .mp4 or .gif for --save output")
    else:
        plt.show()


if __name__ == "__main__":
    main()
