"""Plane detection with sequential RANSAC, written in plain NumPy so every step is visible."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray   # unit normal (a, b, c)
    d: float             # plane: normal . x + d = 0
    idx: np.ndarray      # indices of inlier points in the original cloud
    rms: float           # RMS distance of inliers to the fitted plane

    @property
    def is_horizontal(self):
        return abs(self.normal[2]) > 0.95

    @property
    def is_vertical(self):
        return abs(self.normal[2]) < 0.1


def fit_plane(pts):
    """Least-squares plane through points (smallest singular vector of the centred cloud)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, -n @ c


def ransac_plane(pts, threshold, iterations, rng):
    best_count, best = 0, None
    n_pts = len(pts)
    for _ in range(iterations):
        p = pts[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -n @ p[0]
        count = np.count_nonzero(np.abs(pts @ n + d) < threshold)
        if count > best_count:
            best_count, best = count, (n, d)
    return best


def detect_planes(pts, threshold=0.03, min_points=400, max_planes=30, iterations=400, seed=0):
    """Repeatedly find the plane with most support, remove its points, and continue.

    Returns planes sorted by size. Each plane is refined by least squares on
    its RANSAC inliers, and split along its extent if it covers several
    disconnected surfaces (two walls that happen to be coplanar).
    """
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(pts))
    planes = []
    while len(remaining) > min_points and len(planes) < max_planes:
        found = ransac_plane(pts[remaining], threshold, iterations, rng)
        if found is None:
            break
        n, d = found
        inl = remaining[np.abs(pts[remaining] @ n + d) < threshold]
        if len(inl) < min_points:
            break
        n, d = fit_plane(pts[inl])
        inl = remaining[np.abs(pts[remaining] @ n + d) < threshold]
        if abs(n[2]) > 0.5 and n[2] < 0:  # horizontal planes: normal points up
            n, d = -n, -d
        remaining = np.setdiff1d(remaining, inl, assume_unique=True)
        for part in _split_connected(pts, inl, n):
            if len(part) >= min_points:
                rms = float(np.sqrt(np.mean((pts[part] @ n + d) ** 2)))
                planes.append(Plane(n, float(d), part, rms))
    planes.sort(key=lambda p: -len(p.idx))
    return planes


def _split_connected(pts, idx, n, cell=0.15):
    """Split a plane's inliers into connected patches on a 2D grid in the plane."""
    from scipy import ndimage

    u = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    uv = np.column_stack([pts[idx] @ u, pts[idx] @ v])
    ij = np.floor((uv - uv.min(axis=0)) / cell).astype(int)
    grid = np.zeros(ij.max(axis=0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    # close small gaps (window frames, noise) so one surface stays one patch
    grid = ndimage.binary_closing(grid, iterations=2, border_value=0) | grid
    labels, n_lab = ndimage.label(grid)
    lab = labels[ij[:, 0], ij[:, 1]]
    return [idx[lab == k] for k in range(1, n_lab + 1)]
