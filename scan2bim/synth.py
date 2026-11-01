"""Generate a synthetic indoor scan with known ground truth.

The room is L-shaped, with a door, two windows, a table and a tall cabinet
standing against one wall. The cabinet hides part of the wall from the
scanner, which leaves a hole in the scan that looks like a door: a good
test of whether the pipeline can tell "missing because open" from
"missing because occluded".

The whole scene is rotated about the vertical axis by a random angle, as a
real scan would be, so the pipeline has to find the building's axes itself.

Usage:
    python -m scan2bim.synth --out data/synthetic_room.ply
"""
import argparse
import json

import numpy as np

from .io import write_ply

HEIGHT = 2.70
# L-shaped footprint, counter-clockwise, metres
FOOTPRINT = [(0, 0), (6, 0), (6, 3), (3.5, 3), (3.5, 5.5), (0, 5.5)]
# openings: (wall index, distance along wall from its start, width, sill height, height, kind)
OPENINGS = [
    (0, 1.0, 0.90, 0.00, 2.10, "door"),
    (1, 0.9, 1.20, 0.90, 1.20, "window"),
    (5, 2.0, 1.50, 0.90, 1.30, "window"),
]
# a tall cabinet pushed against wall 3 (x = 3.5): x range, y range, height
CABINET = ((2.9, 3.5), (3.8, 4.8), 2.0)
TABLE = ((1.0, 2.6), (1.2, 2.2), 0.75)


def _sample_rect(origin, u, v, w, h, density, rng):
    n = max(1, int(w * h * density))
    a, b = rng.random(n) * w, rng.random(n) * h
    return origin + a[:, None] * u + b[:, None] * v


def _in_opening(wall_idx, s, z):
    for wi, s0, w, sill, h, _ in OPENINGS:
        if wi == wall_idx and s0 <= s <= s0 + w and sill <= z <= sill + h:
            return True
    return False


def _box_surfaces(xr, yr, h, density, rng, skip_face=None):
    """Points on the visible faces of an axis-aligned box standing on the floor."""
    (x0, x1), (y0, y1) = xr, yr
    faces = {
        "top": ((x0, y0, h), (1, 0, 0), (0, 1, 0), x1 - x0, y1 - y0),
        "x0": ((x0, y0, 0), (0, 1, 0), (0, 0, 1), y1 - y0, h),
        "x1": ((x1, y0, 0), (0, 1, 0), (0, 0, 1), y1 - y0, h),
        "y0": ((x0, y0, 0), (1, 0, 0), (0, 0, 1), x1 - x0, h),
        "y1": ((x0, y1, 0), (1, 0, 0), (0, 0, 1), x1 - x0, h),
    }
    out = []
    for name, (o, u, v, w, hh) in faces.items():
        if name == skip_face:
            continue
        out.append(_sample_rect(np.array(o, float), np.array(u, float), np.array(v, float), w, hh, density, rng))
    return np.vstack(out)


def make_scene(seed=0, density=600, noise=0.008):
    rng = np.random.default_rng(seed)
    fp = np.array(FOOTPRINT, float)
    pts = []

    # walls, with openings cut out and the cabinet's shadow removed
    for i in range(len(fp)):
        a, b = fp[i], fp[(i + 1) % len(fp)]
        length = np.linalg.norm(b - a)
        d = (b - a) / length
        s = rng.random(int(length * HEIGHT * density)) * length
        z = rng.random(s.size) * HEIGHT
        keep = np.array([not _in_opening(i, si, zi) for si, zi in zip(s, z)])
        p = np.column_stack([a[0] + s * d[0], a[1] + s * d[1], z])[keep]
        (cx0, cx1), (cy0, cy1), ch = CABINET
        hidden = (np.abs(p[:, 0] - cx1) < 1e-6) & (p[:, 1] > cy0) & (p[:, 1] < cy1) & (p[:, 2] < ch)
        pts.append(p[~hidden])

    # floor and ceiling: sample the bounding box, keep points inside the L
    def inside(x, y):
        return ~((x > 3.5) & (y > 3.0))

    for zc in (0.0, HEIGHT):
        n = int(6 * 5.5 * density)
        x, y = rng.random(n) * 6, rng.random(n) * 5.5
        m = inside(x, y)
        if zc == 0.0:  # the floor under the cabinet and table legs is not visible
            (cx0, cx1), (cy0, cy1), _ = CABINET
            m &= ~((x > cx0) & (x < cx1) & (y > cy0) & (y < cy1))
        pts.append(np.column_stack([x[m], y[m], np.full(m.sum(), zc)]))

    # clutter
    pts.append(_box_surfaces(*CABINET, density, rng, skip_face="x1"))
    tx, ty, th = TABLE
    pts.append(_sample_rect(np.array([tx[0], ty[0], th]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                            tx[1] - tx[0], ty[1] - ty[0], density, rng))

    pts = np.vstack(pts)
    pts += rng.normal(0, noise, pts.shape)
    # a few stray points, as every real scan has
    outliers = rng.random((300, 3)) * [6, 5.5, HEIGHT]
    pts = np.vstack([pts, outliers])

    # rotate and shift as an arbitrary scanner frame would
    theta = rng.uniform(-np.pi, np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    shift = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-1, 1)])
    pts = pts @ R.T + shift

    gt = {
        "rotation_deg": float(np.degrees(theta)),
        "shift": shift.tolist(),
        "height": HEIGHT,
        "floor_z": float(shift[2]),
        "corners": [(R[:2, :2] @ c + shift[:2]).tolist() for c in fp],
        "floor_area_m2": 6 * 5.5 - 2.5 * 2.5,
        "openings": [],
    }
    for wi, s0, w, sill, h, kind in OPENINGS:
        a, b = fp[wi], fp[(wi + 1) % len(fp)]
        d = (b - a) / np.linalg.norm(b - a)
        c = a + d * (s0 + w / 2)
        gt["openings"].append({
            "kind": kind, "width": w, "height": h, "sill": sill,
            "center_xy": (R[:2, :2] @ c + shift[:2]).tolist(),
        })
    return pts, gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic_room.ply")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.008, help="sensor noise std-dev in metres")
    args = ap.parse_args()
    pts, gt = make_scene(args.seed, noise=args.noise)
    write_ply(args.out, pts)
    gt_path = args.out.rsplit(".", 1)[0] + "_groundtruth.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"wrote {len(pts):,} points to {args.out} and ground truth to {gt_path}")


if __name__ == "__main__":
    main()
