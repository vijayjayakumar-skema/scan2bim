"""Compare a reconstructed layout against the synthetic scene's ground truth."""
import numpy as np


def evaluate(layout, gt, match_dist=0.4):
    res = {}
    # corners: map ours back to the scan frame, then nearest-neighbour distance to each true corner
    if layout.footprint:
        ours = layout.to_scan_frame(np.array(layout.footprint))
        true = np.array(gt["corners"])
        d = np.linalg.norm(true[:, None] - ours[None], axis=2).min(axis=1)
        res["corners_true"] = len(true)
        res["corners_found"] = len(ours)
        res["corner_error_mean_cm"] = round(100 * float(d.mean()), 1)
        res["corner_error_max_cm"] = round(100 * float(d.max()), 1)
        res["floor_area_true_m2"] = gt["floor_area_m2"]
        res["floor_area_found_m2"] = round(layout.floor_area(), 2)
    res["height_true_m"] = gt["height"]
    res["height_found_m"] = round(layout.height, 3)

    # openings: match by centre position in plan
    found = []
    for w in layout.walls:
        for op in w.openings:
            uc = (op.u0 + op.u1) / 2
            xy = np.array([w.c, uc]) if w.axis == "x" else np.array([uc, w.c])
            found.append((op, layout.to_scan_frame(xy[None])[0]))
    matches, used = [], set()
    for g in gt["openings"]:
        best = None
        for j, (op, xy) in enumerate(found):
            dist = np.linalg.norm(xy - np.array(g["center_xy"]))
            if j not in used and op.kind == g["kind"] and dist < match_dist and (best is None or dist < best[0]):
                best = (dist, j)
        if best:
            used.add(best[1])
            op = found[best[1]][0]
            matches.append({"kind": g["kind"],
                            "width_error_cm": round(100 * (op.width - g["width"]), 1),
                            "height_error_cm": round(100 * (op.height - g["height"]), 1),
                            "position_error_cm": round(100 * best[0], 1)})
    res["openings_true"] = len(gt["openings"])
    res["openings_found"] = len(found)
    res["openings_matched"] = len(matches)
    res["false_openings"] = len(found) - len(matches)
    res["opening_details"] = matches
    return res
