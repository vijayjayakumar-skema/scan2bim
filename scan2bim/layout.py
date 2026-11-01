"""Turn detected planes into building elements: floor, ceiling, walls, doors and windows."""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Opening:
    kind: str            # "door" or "window"
    u0: float            # along the wall, in the aligned frame
    u1: float
    z0: float            # above the floor
    z1: float

    @property
    def width(self):
        return self.u1 - self.u0

    @property
    def height(self):
        return self.z1 - self.z0


@dataclass
class Wall:
    axis: str            # "x": the wall lies in the plane x = c; "y": plane y = c
    c: float             # position of the scanned (interior) face
    u0: float            # extent along the other horizontal axis
    u1: float
    inside: int          # +1 if the room is on the +c side, -1 otherwise
    n_points: int
    rms: float
    openings: list = field(default_factory=list)
    occluded: list = field(default_factory=list)  # holes rejected because something stood in front

    def endpoints(self):
        if self.axis == "x":
            return np.array([self.c, self.u0]), np.array([self.c, self.u1])
        return np.array([self.u0, self.c]), np.array([self.u1, self.c])


@dataclass
class Layout:
    floor_z: float
    ceiling_z: float
    walls: list
    footprint: list      # closed polygon of corner points, or [] if walls did not close
    angle: float         # rotation (rad) applied about z to align walls with the axes
    origin: np.ndarray   # translation applied after the rotation

    @property
    def height(self):
        return self.ceiling_z - self.floor_z

    def floor_area(self):
        if not self.footprint:
            return None
        p = np.array(self.footprint)
        return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))

    def to_scan_frame(self, xy):
        """Map aligned-frame XY back to the original scan's coordinates."""
        xy = np.asarray(xy) + self.origin[:2]
        c, s = np.cos(-self.angle), np.sin(-self.angle)
        return xy @ np.array([[c, -s], [s, c]]).T


def dominant_angle(planes):
    """Main wall direction, from vertical plane normals, modulo 90 degrees.

    Averaging 4*theta on the unit circle treats directions 90 degrees apart
    as the same, which is the Manhattan-world assumption most buildings meet.
    """
    vert = [p for p in planes if p.is_vertical]
    th = np.array([np.arctan2(p.normal[1], p.normal[0]) for p in vert])
    w = np.array([len(p.idx) for p in vert], float)
    return np.arctan2((w * np.sin(4 * th)).sum(), (w * np.cos(4 * th)).sum()) / 4


def rotate_z(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return pts @ R.T


def build_layout(pts, planes, wall_height_ratio=0.85, snap=0.6, cell=0.10, threshold=0.03):
    # 1. floor and ceiling: the lowest and highest large horizontal planes
    horiz = [p for p in planes if p.is_horizontal]
    if len(horiz) < 2:
        raise RuntimeError("need at least a floor and a ceiling plane; is the scan z-up? try --up y")
    big = max(len(p.idx) for p in horiz)
    heights = sorted((-p.d / p.normal[2], p) for p in horiz if len(p.idx) > 0.2 * big)
    floor_z, ceiling_z = heights[0][0], heights[-1][0]
    storey = ceiling_z - floor_z

    # 2. align the walls with the x and y axes, put the floor at z = 0
    angle = -dominant_angle(planes)
    al = rotate_z(pts, angle)
    origin = np.array([al[:, 0].min(), al[:, 1].min(), floor_z])
    al = al - origin

    # 3. walls: vertical planes spanning most of the storey height
    walls = []
    for p in planes:
        if not p.is_vertical:
            continue
        q = al[p.idx]
        zlo, zhi = np.percentile(q[:, 2], [2, 98])
        if zhi - zlo < wall_height_ratio * storey:
            continue
        n = rotate_z(p.normal[None], angle)[0]
        axis = "x" if abs(n[0]) > abs(n[1]) else "y"
        k, o = (0, 1) if axis == "x" else (1, 0)
        c = float(np.median(q[:, k]))
        u0, u1 = np.percentile(q[:, o], [0.5, 99.5])
        walls.append(Wall(axis, c, float(u0), float(u1), 0, len(p.idx), p.rms))
    walls = _merge_walls(walls)

    # 4. which side of each wall is the room? look at the floor points next to it
    floor_pts = al[np.abs(al[:, 2]) < threshold]
    for w in walls:
        k, o = (0, 1) if w.axis == "x" else (1, 0)
        near = floor_pts[(floor_pts[:, o] > w.u0) & (floor_pts[:, o] < w.u1) & (np.abs(floor_pts[:, k] - w.c) < 1.0)]
        w.inside = 1 if np.median(near[:, k] - w.c) > 0 else -1

    # 5. close the corners: snap each wall end onto the perpendicular wall it meets
    links = _snap_corners(walls, snap)
    footprint = _chain(walls, links)

    # 6. openings in each wall
    for w in walls:
        _find_openings(al, w, storey, cell, threshold)

    return Layout(0.0, storey, walls, footprint, angle, origin), al


def _merge_walls(walls, tol=0.10):
    walls = sorted(walls, key=lambda w: (w.axis, w.c, w.u0))
    out = []
    for w in walls:
        m = next((o for o in out if o.axis == w.axis and abs(o.c - w.c) < tol and w.u0 < o.u1 + tol and o.u0 < w.u1 + tol), None)
        if m is None:
            out.append(w)
        else:
            tot = m.n_points + w.n_points
            m.c = (m.c * m.n_points + w.c * w.n_points) / tot
            m.u0, m.u1, m.n_points = min(m.u0, w.u0), max(m.u1, w.u1), tot
    return out


def _snap_corners(walls, snap):
    links = {}
    for i, w in enumerate(walls):
        for end in (0, 1):
            u = w.u0 if end == 0 else w.u1
            best = None
            for j, v in enumerate(walls):
                if v.axis == w.axis:
                    continue
                if abs(v.c - u) < snap and v.u0 - snap < w.c < v.u1 + snap:
                    dist = abs(v.c - u)
                    if best is None or dist < best[0]:
                        best = (dist, j)
            if best is None:
                continue
            j = best[1]
            v = walls[j]
            if end == 0:
                w.u0 = v.c
            else:
                w.u1 = v.c
            # and the other wall's nearer end onto this one
            if abs(v.u0 - w.c) < abs(v.u1 - w.c):
                v.u0, vend = w.c, 0
            else:
                v.u1, vend = w.c, 1
            links[(i, end)] = (j, vend)
            links[(j, vend)] = (i, end)
    return links


def _chain(walls, links):
    """Follow wall-to-wall links round the room; return the corner polygon if it closes."""
    if not walls or len(links) < 2 * len(walls):
        return []
    poly, i, end, seen = [], 0, 1, set()
    while i not in seen:
        seen.add(i)
        poly.append(walls[i].endpoints()[end].tolist())
        j, jend = links[(i, end)]
        i, end = j, 1 - jend
    return poly if len(seen) == len(walls) else []


def _find_openings(al, w, storey, cell, threshold):
    k, o = (0, 1) if w.axis == "x" else (1, 0)
    dist = (al[:, k] - w.c) * w.inside            # > 0 on the room side
    in_range = (al[:, o] > w.u0) & (al[:, o] < w.u1)
    on_wall = in_range & (np.abs(dist) < threshold)
    front = in_range & (dist > 0.05) & (dist < 1.0) & (al[:, 2] > 0.02)

    nu = max(1, int(np.ceil((w.u1 - w.u0) / cell)))
    nz = max(1, int(np.ceil(storey / cell)))

    def grid(mask):
        g = np.zeros((nu, nz), bool)
        q = al[mask]
        iu = np.clip(((q[:, o] - w.u0) / cell).astype(int), 0, nu - 1)
        iz = np.clip((q[:, 2] / cell).astype(int), 0, nz - 1)
        g[iu, iz] = True
        return g

    occ, blocked = grid(on_wall), grid(front)
    empty = ndimage.binary_opening(~occ, structure=np.ones((3, 3)))  # ignore single missing cells
    labels, n = ndimage.label(empty)
    wall_pts = al[on_wall]
    for lab in range(1, n + 1):
        cells = np.argwhere(labels == lab)
        (a0, b0), (a1, b1) = cells.min(axis=0), cells.max(axis=0) + 1
        if b1 >= nz - 1 or a0 == 0 or a1 >= nu:   # touches the ceiling or a wall end
            continue
        if (a1 - a0) * cell < 0.4 or (b1 - b0) * cell < 0.4:
            continue
        if len(cells) / ((a1 - a0) * (b1 - b0)) < 0.75:  # not rectangular
            continue
        u0, u1 = w.u0 + a0 * cell, w.u0 + a1 * cell
        z0, z1 = b0 * cell, b1 * cell
        u0, u1, z0, z1 = _refine(wall_pts, o, u0, u1, z0, z1, cell)
        box = blocked[a0:a1, b0:b1]
        if box.mean() > 0.3:
            w.occluded.append(Opening("occluded", u0, u1, z0, z1))
            continue
        kind = "door" if z0 < 0.15 else "window"
        if kind == "door":
            z0 = 0.0
        w.openings.append(Opening(kind, u0, u1, z0, z1))


def _refine(wall_pts, o, u0, u1, z0, z1, cell):
    """Move each edge of a grid-sized hole to the nearest scanned wall points."""
    u, z = wall_pts[:, o], wall_pts[:, 2]
    band_z = (z > z0 + cell) & (z < z1 - cell)
    band_u = (u > u0 + cell) & (u < u1 - cell)
    left = u[band_z & (u < u0 + cell) & (u > u0 - cell)]
    right = u[band_z & (u > u1 - cell) & (u < u1 + cell)]
    bottom = z[band_u & (z < z0 + cell) & (z > z0 - cell)]
    top = z[band_u & (z > z1 - cell) & (z < z1 + cell)]
    return (left.max() if left.size else u0, right.min() if right.size else u1,
            bottom.max() if bottom.size else z0, top.min() if top.size else z1)
