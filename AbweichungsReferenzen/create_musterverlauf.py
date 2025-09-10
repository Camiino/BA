#!/usr/bin/env python3
import csv
import math
import os
import argparse
from typing import Tuple, List


def read_markers_from_reference_csv(path: str) -> List[Tuple[float, float, float]]:
    """
    Reads a reference CSV from AbweichungsReferenzen with header lines like
    
        ,*1,,,*2,,,*3
        Field #,X,Y,Z,X,Y,Z,X,Y,Z
        1,x1,y1,z1,x2,y2,z2,x3,y3,z3

    Returns a list of marker coordinates in order: [(x1,y1,z1), (x2,y2,z2), ...].
    """
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        # Skip first two header rows
        try:
            next(reader)
            next(reader)
        except StopIteration:
            raise ValueError(f"CSV {path} does not contain expected header rows")

        # Read first data row
        try:
            row = next(reader)
        except StopIteration:
            raise ValueError(f"CSV {path} has no data row")

        if not row:
            raise ValueError(f"CSV {path} has empty data row")

        # First column is field index, then groups of 3 for each marker
        values = row[1:]
        if len(values) % 3 != 0:
            raise ValueError(
                f"CSV {path} data row length ({len(values)}) is not a multiple of 3"
            )

        markers = []
        for i in range(0, len(values), 3):
            try:
                x = float(values[i])
                y = float(values[i + 1])
                z = float(values[i + 2])
            except Exception as e:
                raise ValueError(
                    f"CSV {path} contains non-numeric value near index {i}: {e}"
                )
            markers.append((x, y, z))
        return markers


def lerp(a: Tuple[float, float, float], b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def vec_sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_len(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_norm(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    l = vec_len(v)
    if l == 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def zickzack_path(m1, m2, m3, samples: int, *, lock_z: float | None = None) -> List[Tuple[float, float, float]]:
    """Piecewise-linear path from m1->m2->m3 with constant speed by arc length."""
    # If lock_z is specified, use that Z value for all points
    if lock_z is not None:
        m1 = (m1[0], m1[1], lock_z)
        m2 = (m2[0], m2[1], lock_z)
        m3 = (m3[0], m3[1], lock_z)
    
    d1 = vec_len(vec_sub(m2, m1))
    d2 = vec_len(vec_sub(m3, m2))
    total = d1 + d2
    if total == 0:
        return [m1 for _ in range(samples)]

    points = []
    for i in range(samples):
        s = i / (samples - 1) if samples > 1 else 0.0  # 0..1
        dist = s * total
        if dist <= d1:
            t = dist / d1 if d1 > 0 else 0.0
            p = lerp(m1, m2, t)
        else:
            rem = dist - d1
            t = rem / d2 if d2 > 0 else 0.0
            p = lerp(m2, m3, t)
        points.append(p)
    return points


def derive_extra_anchors(m1, m2, m3, *, lock_z: float | None = None):
    """Derive anchor 4 and 5 from 1..3 without changing 1..3.
    - anchor4: x = x3, y = y1
    - anchor5: x = x1, y = 2*y1 - y2 (same |Δy| from 1 as anchor 2)
    Z is taken from lock_z if provided, else from anchor 1.
    Returns (m4, m5).
    """
    x1, y1, z1 = m1
    x2, y2, z2 = m2
    x3, y3, z3 = m3
    z_use = lock_z if lock_z is not None else z1
    m4 = (x3, y1, z_use)
    y5 = 2 * y1 - y2
    m5 = (x1, y5, z_use)
    return m4, m5


def piecewise_path(anchors: List[Tuple[float, float, float]], samples: int, *, lock_z: float | None = None) -> List[Tuple[float, float, float]]:
    """Interpolate along a polyline defined by anchors with constant speed overall."""
    if len(anchors) == 0:
        return []
    if lock_z is not None:
        anchors = [(x, y, lock_z) for (x, y, _z) in anchors]

    # Compute segment lengths
    seg_lengths = []
    for i in range(len(anchors) - 1):
        seg_lengths.append(vec_len(vec_sub(anchors[i + 1], anchors[i])))
    total = sum(seg_lengths) if seg_lengths else 0.0
    if total == 0:
        # All anchors identical
        return [anchors[0] for _ in range(samples)]

    # Cumulative lengths
    cum = [0.0]
    for L in seg_lengths:
        cum.append(cum[-1] + L)

    points: List[Tuple[float, float, float]] = []
    for i in range(samples):
        s = (i / (samples - 1)) * total if samples > 1 else 0.0
        # Find segment index j such that cum[j] <= s <= cum[j+1]
        j = 0
        while j < len(seg_lengths) - 1 and s > cum[j + 1]:
            j += 1
        if seg_lengths[j] == 0:
            t = 0.0
        else:
            t = (s - cum[j]) / seg_lengths[j]
        p = lerp(anchors[j], anchors[j + 1], t)
        if lock_z is not None:
            p = (p[0], p[1], lock_z)
        points.append(p)
    return points


def derive_straightened_anchors(m1, m2, m3, *, lock_z: float | None = None) -> List[Tuple[float, float, float]]:
    """
    Build anchors satisfying:
    - 2 -> 3: 90° (axis-aligned) by forcing y3 = y2 (horizontal segment)
    - 3 -> 1: 45° in XY by setting |x1 - x3| = |y1 - y3|
      choosing the x3 sign nearest to original x3
    - 1 -> 4: 90° by setting 4 = (x3, y1)
    - 4 -> 5: 45° by setting 5 = (x1, y1 ± |x1 - x3|) with sign opposite to sign(y2 - y1)
    - 5 -> 6: 90° with x6 = x4 and y6 = y5 (horizontal)
    Returns anchors ordered for traversal: [2, 3, 1, 4, 5, 6].
    Z for 4/5/6 uses lock_z if provided, else m1.z.
    """
    x1, y1, z1 = m1
    x2, y2, z2 = m2
    x3o, y3o, z3 = m3
    z_use = lock_z if lock_z is not None else z1

    # 2 -> 3 horizontal (y3 = y2)
    y3p = y2
    dx45 = abs(y1 - y3p)
    # choose x3 to satisfy 45deg to 1 and be close to original x3
    cand_a = x1 + dx45
    cand_b = x1 - dx45
    x3p = cand_a if abs(cand_a - x3o) <= abs(cand_b - x3o) else cand_b

    m2p = (x2, y2, z_use)
    m3p = (x3p, y3p, z_use)
    m1p = (x1, y1, z_use)

    # 1 -> 4 vertical/horizontal: (x3, y1)
    m4 = (x3p, y1, z_use)

    # 4 -> 5 45°: delta y magnitude equals |x1 - x3p|
    # pick direction opposite to sign(y2 - y1) to mirror across 1
    sign = -1 if (y2 - y1) > 0 else (1 if (y2 - y1) < 0 else 1)
    y5 = y1 + sign * abs(x1 - x3p)
    m5 = (x1, y5, z_use)

    # 5 -> 6 90° with x6 = x4 and y6 = y5 (horizontal)
    m6 = (x3p, y5, z_use)

    return [m2p, m3p, m1p, m4, m5, m6]


def compute_plane_normal(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float] | None:
    """Given >=3 points, compute a robust plane normal using the first non-colinear triple."""
    n = len(points)
    if n < 3:
        return None
    p0 = points[0]
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            a = vec_sub(points[i], p0)
            b = vec_sub(points[j], p0)
            nvec = vec_cross(a, b)
            if vec_len(nvec) > 1e-9:
                return vec_norm(nvec)
    return None


def circle_path(center, on_circumference, samples: int, *, plane_normal: Tuple[float, float, float] | None = None, lock_z: float | None = None) -> List[Tuple[float, float, float]]:
    """
    Generates a full circle trajectory using:
    - center = marker 1
    - on_circumference = marker 2 (defines radius and start angle)

    The circle plane is defined by:
    - If plane_normal is provided: the plane whose normal is plane_normal, using
      the radius direction projected into this plane.
    - Else: an automatically chosen plane spanned by the radius direction and a
      perpendicular axis based on a global up-vector heuristic.
    """
    c = center
    r_vec = vec_sub(on_circumference, center)
    
    # If lock_z is specified, project everything to that Z plane
    if lock_z is not None:
        c = (center[0], center[1], lock_z)
        on_circumference = (on_circumference[0], on_circumference[1], lock_z)
        r_vec = vec_sub(on_circumference, c)
    
    r = vec_len(r_vec)
    if r == 0:
        return [c for _ in range(samples)]

    if plane_normal is not None and vec_len(plane_normal) > 0 and lock_z is None:
        n_hat = vec_norm(plane_normal)
        # Project radius vector into the plane
        dotp = vec_dot(r_vec, n_hat)
        rp = (r_vec[0] - dotp * n_hat[0], r_vec[1] - dotp * n_hat[1], r_vec[2] - dotp * n_hat[2])
        rp_len = vec_len(rp)
        if rp_len == 0:
            # Fall back: pick any unit vector in plane as u_hat
            candidates = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            base = candidates[0]
            for cnd in candidates:
                if abs(vec_dot(cnd, n_hat)) < 0.99:
                    base = cnd
                    break
            u_hat = vec_norm(vec_cross(n_hat, base))
            radius = r
        else:
            u_hat = (rp[0] / rp_len, rp[1] / rp_len, rp[2] / rp_len)
            radius = rp_len
        v_hat = vec_norm(vec_cross(n_hat, u_hat))
    else:
        # When lock_z is specified, force circle in XY plane
        if lock_z is not None:
            u_hat = vec_norm((r_vec[0], r_vec[1], 0.0))  # direction in XY plane
            v_hat = (-u_hat[1], u_hat[0], 0.0)  # perpendicular in XY plane
            radius = math.sqrt(r_vec[0]**2 + r_vec[1]**2)  # radius in XY plane
        else:
            # Fallback: plane spanned by radius direction and a perpendicular axis
            u_hat = vec_norm(r_vec)  # direction from center to start point (theta=0)
            # Choose a stable auxiliary up-vector not parallel to u_hat
            up_candidates = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
            up = up_candidates[0]
            for cand in up_candidates:
                if abs(vec_dot(u_hat, cand)) < 0.99:
                    up = cand
                    break
            # Construct an orthonormal basis (u_hat, v_hat) spanning the circle plane
            n_hat = vec_norm(vec_cross(u_hat, up))
            # If u is parallel to up (edge case), pick an alternative normal
            if vec_len(n_hat) == 0:
                up = up_candidates[1]
                n_hat = vec_norm(vec_cross(u_hat, up))
                if vec_len(n_hat) == 0:
                    up = up_candidates[2]
                    n_hat = vec_norm(vec_cross(u_hat, up))
            v_hat = vec_norm(vec_cross(n_hat, u_hat))
            radius = r

    points = []
    for i in range(samples):
        t = i / (samples - 1) if samples > 1 else 0.0
        theta = 2.0 * math.pi * t
        p = (
            c[0] + radius * (math.cos(theta) * u_hat[0] + math.sin(theta) * v_hat[0]),
            c[1] + radius * (math.cos(theta) * u_hat[1] + math.sin(theta) * v_hat[1]),
            c[2] + radius * (math.cos(theta) * u_hat[2] + math.sin(theta) * v_hat[2]),
        )
        # Ensure Z is locked if specified
        if lock_z is not None:
            p = (p[0], p[1], lock_z)
        points.append(p)
    return points


def write_series_csv(path: str, series: List[Tuple[float, float, float]], delimiter: str = ";") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["Frame", "1_X", "1_Y", "1_Z"])
        n = len(series)
        for i, (x, y, z) in enumerate(series):
            # Frame as percent 0..100
            frame = 0.0 if n <= 1 else (100.0 * i / (n - 1))
            writer.writerow([f"{frame:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])


def write_series_csv_multi(path: str, series_by_marker: List[List[Tuple[float, float, float]]], delimiter: str = ";") -> None:
    """Write multiple series as markers 1..N into one CSV (Frame, 1_X.., 2_X.., ...).
    Pads shorter series with their last value.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lengths = [len(s) for s in series_by_marker]
    if not lengths or max(lengths) == 0:
        raise ValueError("No series to write")
    n = max(lengths)

    header = ["Frame"]
    for mid in range(1, len(series_by_marker) + 1):
        header += [f"{mid}_X", f"{mid}_Y", f"{mid}_Z"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)
        for i in range(n):
            frame = 0.0 if n <= 1 else (100.0 * i / (n - 1))
            row = [f"{frame:.6f}"]
            for s in series_by_marker:
                if i < len(s):
                    x, y, z = s[i]
                else:
                    x, y, z = s[-1]
                row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Create Musterverlauf for ZickZack and Kreis from reference markers.")
    parser.add_argument("--zickzack", default="ZickZack.csv", help="Input CSV for ZickZack markers (default: ZickZack.csv)")
    parser.add_argument("--kreis", default="Kreis.csv", help="Input CSV for Kreis markers (default: Kreis.csv)")
    parser.add_argument("--samples", type=int, default=101, help="Number of samples per trajectory (default: 101)")
    parser.add_argument("--zz-samples", type=int, default=None, help="Number of samples for ZickZack only (overrides --samples for ZickZack)")
    parser.add_argument("--outdir", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--prefix", default="Musterverlauf_", help="Output filename prefix (default: Musterverlauf_)")
    parser.add_argument("--delimiter", default=";", choices=[",", ";", "\t"], help="Output delimiter (default: ;)")
    parser.add_argument("--table", default="Table.csv", help="Optional Table CSV to define plane for circle (default: Table.csv if present)")
    parser.add_argument("--lock-z", type=float, default=None, help="Lock Z coordinate to this value for all trajectories")

    args = parser.parse_args()

    # Use specific ZickZack samples if provided, otherwise use general samples
    zz_samples = args.zz_samples if args.zz_samples is not None else args.samples

    # Resolve relative to script directory for inputs when not absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- ZickZack (single moving marker over straightened 6 anchors) ---
    try:
        zz_path = args.zickzack if os.path.isabs(args.zickzack) else os.path.join(script_dir, args.zickzack)
        markers = read_markers_from_reference_csv(zz_path)
        if len(markers) < 3:
            raise ValueError("ZickZack requires 3 markers (1,2,3)")

        m1, m2, m3 = markers[0], markers[1], markers[2]
        anchors = derive_straightened_anchors(m1, m2, m3, lock_z=args.lock_z)

        series = piecewise_path(anchors, zz_samples, lock_z=args.lock_z)
        out_zz = os.path.join(args.outdir, f"{args.prefix}ZickZack.csv")
        write_series_csv(out_zz, series, delimiter=args.delimiter)
        print(f"✅ ZickZack Musterverlauf saved: {out_zz} (order 2→3→1→4→5→6, {zz_samples} samples)")
    except Exception as e:
        print(f"❌ Failed to create ZickZack Musterverlauf: {e}")

    # --- Kreis ---
    try:
        k_path = args.kreis if os.path.isabs(args.kreis) else os.path.join(script_dir, args.kreis)
        markers = read_markers_from_reference_csv(k_path)
        if len(markers) < 2:
            raise ValueError("Kreis requires 2 markers (center=1, on-diameter=2)")
        # Try to obtain plane normal from Table.csv (optional)
        plane_normal = None
        if args.table:
            t_path = args.table if os.path.isabs(args.table) else os.path.join(script_dir, args.table)
            if os.path.exists(t_path):
                try:
                    table_markers = read_markers_from_reference_csv(t_path)
                    plane_normal = compute_plane_normal(table_markers)
                except Exception:
                    plane_normal = None
        series = circle_path(markers[0], markers[1], args.samples, plane_normal=plane_normal, lock_z=args.lock_z)
        out_k = os.path.join(args.outdir, f"{args.prefix}Kreis.csv")
        write_series_csv(out_k, series, delimiter=args.delimiter)
        print(f"✅ Kreis Musterverlauf saved: {out_k}")
    except Exception as e:
        print(f"❌ Failed to create Kreis Musterverlauf: {e}")


if __name__ == "__main__":
    main()
