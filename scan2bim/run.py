"""Scan-to-BIM: point cloud of a room in, IFC model out.

Usage:
    python -m scan2bim.run data/synthetic_room.ply --out output/
    python -m scan2bim.run my_scan.ply --up y --voxel 0.02 --out output/my_room/

If a file named <scan>_groundtruth.json sits next to the scan (the
synthetic generator writes one), the result is scored against it.
"""
import argparse
import json
import os
import time

from .evaluate import evaluate
from .ifc_export import export_ifc
from .io import read_ply, remove_outliers, voxel_downsample
from .layout import build_layout
from .planes import detect_planes
from .visualize import elevations, ifc_3d, plan_view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="input point cloud (.ply)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--up", choices=["z", "y"], default="z", help="vertical axis of the input")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply coordinates, e.g. 0.001 for mm")
    ap.add_argument("--voxel", type=float, default=0.03, help="downsampling voxel size in metres (0 = off)")
    ap.add_argument("--threshold", type=float, default=0.03, help="RANSAC inlier distance in metres")
    ap.add_argument("--min-points", type=int, default=400, help="smallest plane to keep, in points")
    ap.add_argument("--cell", type=float, default=0.10, help="grid cell for opening detection, metres")
    ap.add_argument("--thickness", type=float, default=0.20, help="assumed wall thickness, metres")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    pts = read_ply(args.scan, up=args.up, scale=args.scale)
    n_raw = len(pts)
    pts = remove_outliers(voxel_downsample(pts, args.voxel))
    print(f"loaded {n_raw:,} points, {len(pts):,} after downsampling and outlier removal")

    planes = detect_planes(pts, threshold=args.threshold, min_points=args.min_points)
    print(f"found {len(planes)} planes")

    layout, aligned = build_layout(pts, planes, cell=args.cell, threshold=args.threshold)
    n_open = sum(len(w.openings) for w in layout.walls)
    n_occ = sum(len(w.occluded) for w in layout.walls)
    print(f"walls: {len(layout.walls)}, openings: {n_open}, occluded regions rejected: {n_occ}, "
          f"footprint {'closed' if layout.footprint else 'NOT closed'}")

    ifc_path = os.path.join(args.out, "model.ifc")
    export_ifc(layout, ifc_path, thickness=args.thickness)
    plan_view(aligned, layout, os.path.join(args.out, "plan.png"), thickness=args.thickness)
    elevations(aligned, layout, os.path.join(args.out, "elevations.png"), threshold=args.threshold)
    ifc_3d(ifc_path, os.path.join(args.out, "model_3d.png"))
    print(f"wrote {ifc_path} and figures in {time.time() - t0:.1f} s")

    gt_path = args.scan.rsplit(".", 1)[0] + "_groundtruth.json"
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            scores = evaluate(layout, json.load(f))
        with open(os.path.join(args.out, "evaluation.json"), "w") as f:
            json.dump(scores, f, indent=2)
        print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
