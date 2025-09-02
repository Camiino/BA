import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import argparse
import os
import re

# --- CONFIG ---
AXES = ("X", "Y", "Z")
DELIMITER = ";"


def detect_markers(columns):
    """Detect available marker IDs based on column names like '1_X', '2_Y'."""
    marker_ids = set()
    pattern = re.compile(r"(\d+)_X")  # we assume each marker has an _X column
    for col in columns:
        match = pattern.match(col)
        if match:
            marker_ids.add(int(match.group(1)))
    return sorted(marker_ids)


def load_marker_data(csv_path):
    df = pd.read_csv(csv_path, delimiter=DELIMITER)

    # Convert all data to numeric (coerce errors to NaN)
    df = df.apply(pd.to_numeric, errors='coerce')

    marker_ids = detect_markers(df.columns)
    marker_data = {}

    for marker_id in marker_ids:
        try:
            x = df[f"{marker_id}_X"].values
            y = df[f"{marker_id}_Y"].values
            z = df[f"{marker_id}_Z"].values
            coords = np.vstack((x, y, z)).T

            # Remove invalid rows
            coords = coords[~np.isnan(coords).any(axis=1)]
            coords = coords[~np.isinf(coords).any(axis=1)]

            if coords.size == 0:
                print(f"⚠️ Alle Werte von Marker {marker_id} sind ungültig.")
                continue

            marker_data[marker_id] = coords
        except KeyError:
            print(f"❗ Marker {marker_id} fehlt in Datei: {csv_path}")
    return marker_data


def animate_markers(marker_data, title="Marker Trajectories"):
    if not marker_data:
        print("❌ Keine gültigen Markerdaten zum Anzeigen.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Concatenate all coordinates into one array (even if only one marker)
    all_coords = np.concatenate([v for v in marker_data.values()], axis=0)

    if np.isnan(all_coords).any() or np.isinf(all_coords).any():
        print("❌ Ungültige Werte (NaN/Inf) in Koordinatendaten!")
        print(all_coords)
        return

    ax.set_xlim(np.min(all_coords[:, 0]), np.max(all_coords[:, 0]))
    ax.set_ylim(np.min(all_coords[:, 1]), np.max(all_coords[:, 1]))
    ax.set_zlim(np.min(all_coords[:, 2]), np.max(all_coords[:, 2]))

    points = {mid: ax.plot([], [], [], 'o', label=f'Marker {mid}')[0] for mid in marker_data}
    trails = {mid: ax.plot([], [], [], '-', alpha=0.4)[0] for mid in marker_data}

    max_frames = min(len(data) for data in marker_data.values())

    def update(frame):
        for mid, data in marker_data.items():
            if frame < len(data):
                x, y, z = data[frame]
                trail_data = data[:frame + 1]
                points[mid].set_data([x], [y])
                points[mid].set_3d_properties([z])
                trails[mid].set_data(trail_data[:, 0], trail_data[:, 1])
                trails[mid].set_3d_properties(trail_data[:, 2])
        return list(points.values()) + list(trails.values())

    ax.legend()
    ani = FuncAnimation(fig, update, frames=max_frames, interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize marker trajectories from CSV.")
    parser.add_argument("csv", help="Path to the averaged CSV file")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"❌ Datei nicht gefunden: {args.csv}")
        exit(1)

    print(f"🎬 Visualisiere: {args.csv}")
    marker_data = load_marker_data(args.csv)
    animate_markers(marker_data, title=os.path.basename(args.csv))
