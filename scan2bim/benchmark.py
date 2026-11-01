"""Score the pipeline on many synthetic scenes at increasing sensor noise.

Usage:
    python -m scan2bim.benchmark --seeds 10
"""
import argparse

import numpy as np

from .evaluate import evaluate
from .io import remove_outliers, voxel_downsample
from .layout import build_layout
from .planes import detect_planes
from .synth import make_scene


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03])
    args = ap.parse_args()

    print("| noise (cm) | runs | footprint closed | corner error (cm) | openings found / true | false openings | opening size error (cm) |")
    print("|---|---|---|---|---|---|---|")
    for noise in args.noise:
        closed, corner, found, true, false, size = 0, [], 0, 0, 0, []
        for seed in range(args.seeds):
            pts, gt = make_scene(seed, noise=noise)
            pts = remove_outliers(voxel_downsample(pts, 0.03))
            thr = max(0.03, 3 * noise)
            try:
                layout, _ = build_layout(pts, detect_planes(pts, threshold=thr, seed=seed), threshold=thr)
            except RuntimeError:
                true += len(gt["openings"])
                continue
            r = evaluate(layout, gt)
            if "corner_error_mean_cm" in r:
                closed += 1
                corner.append(r["corner_error_mean_cm"])
            found += r["openings_matched"]
            true += r["openings_true"]
            false += r["false_openings"]
            size += [abs(m["width_error_cm"]) for m in r["opening_details"]]
            size += [abs(m["height_error_cm"]) for m in r["opening_details"]]
        mean = lambda v: f"{np.mean(v):.1f}" if v else "n/a"
        print(f"| {noise * 100:.1f} | {args.seeds} | {closed}/{args.seeds} | {mean(corner)} | {found}/{true} | {false} | {mean(size)} |")


if __name__ == "__main__":
    main()
