"""Reading and writing point clouds (PLY), plus basic preprocessing."""
import numpy as np
from plyfile import PlyData, PlyElement


def read_ply(path, up="z", scale=1.0):
    """Load an (N, 3) float array of XYZ points from a PLY file.

    up:    which input axis points up ("z" or "y"). Phone apps such as
           Polycam often export y-up; everything downstream assumes z-up.
    scale: multiply coordinates by this (e.g. 0.001 if the file is in mm).
    """
    v = PlyData.read(path)["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    if up == "y":
        # (x, y, z) y-up  ->  (x, -z, y) z-up, keeps a right-handed frame
        pts = np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1)
    return pts * scale


def write_ply(path, pts, colors=None):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(len(pts), dtype=dtype)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if colors is not None:
        arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


def voxel_downsample(pts, voxel):
    """Keep one point (the centroid) per voxel of size `voxel` metres."""
    if voxel <= 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((counts.size, 3))
    np.add.at(out, inv, pts)
    return out / counts[:, None]


def remove_outliers(pts, voxel=0.1, min_count=3):
    """Drop isolated points: those in voxels holding fewer than `min_count` points."""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return pts[counts[inv.ravel()] >= min_count]
