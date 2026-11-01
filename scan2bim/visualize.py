"""Figures for the README and for checking results: plan view, wall elevations, 3D view of the IFC."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"door": "#2a9d8f", "window": "#3a86ff", "occluded": "#e63946"}


def plan_view(al, layout, path, thickness=0.2):
    fig, ax = plt.subplots(figsize=(7, 7))
    sub = al[:: max(1, len(al) // 40000)]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.3, c=sub[:, 2], cmap="Greys", alpha=0.5)
    for i, w in enumerate(layout.walls):
        a, b = w.endpoints()
        ext = np.array([-w.inside * thickness / 2, 0]) if w.axis == "x" else np.array([0, -w.inside * thickness / 2])
        ax.plot(*np.array([a + ext, b + ext]).T, color="#333", lw=5, solid_capstyle="butt")
        mid = (a + b) / 2 + ext * 4
        ax.text(*mid, f"W{i + 1}", ha="center", va="center", fontsize=8)
        for op in w.openings + w.occluded:
            if w.axis == "x":
                seg = np.array([[w.c, op.u0], [w.c, op.u1]]) + ext
            else:
                seg = np.array([[op.u0, w.c], [op.u1, w.c]]) + ext
            ax.plot(*seg.T, color=COLORS[op.kind], lw=7, solid_capstyle="butt",
                    ls="-" if op.kind != "occluded" else (0, (1, 1)))
    ax.legend(handles=[mpatches.Patch(color=c, label=k) for k, c in COLORS.items()]
              + [mpatches.Patch(color="#333", label="wall")], loc="upper right", fontsize=8)
    area = layout.floor_area()
    ax.set_title(f"Plan view: {len(layout.walls)} walls, height {layout.height:.2f} m"
                 + (f", floor area {area:.1f} m²" if area else ""))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def elevations(al, layout, path, threshold=0.03):
    walls = [w for w in layout.walls if w.openings or w.occluded]
    if not walls:
        return
    fig, axes = plt.subplots(len(walls), 1, figsize=(8, 2.6 * len(walls)), squeeze=False)
    for ax, w in zip(axes[:, 0], walls):
        k, o = (0, 1) if w.axis == "x" else (1, 0)
        dist = (al[:, k] - w.c) * w.inside
        rng = (al[:, o] > w.u0) & (al[:, o] < w.u1)
        on = rng & (np.abs(dist) < threshold)
        front = rng & (dist > 0.05) & (dist < 1.0) & (al[:, 2] > 0.05) & (al[:, 2] < layout.height - 0.05)
        front &= (al[:, o] > w.u0 + 0.1) & (al[:, o] < w.u1 - 0.1)  # skip the floor, ceiling and side walls
        ax.scatter(al[on, o], al[on, 2], s=0.5, c="#888")
        ax.scatter(al[front, o], al[front, 2], s=0.5, c="#f4a261", alpha=0.4)
        for op in w.openings + w.occluded:
            ax.add_patch(mpatches.Rectangle((op.u0, op.z0), op.width, op.height, fill=False,
                                            ec=COLORS[op.kind], lw=2, ls="-" if op.kind != "occluded" else "--"))
            label = op.kind if op.kind == "occluded" else f"{op.kind} {op.width:.2f}×{op.height:.2f} m"
            ax.text(op.u0, op.z1 + 0.05, label, color=COLORS[op.kind], fontsize=8)
        i = layout.walls.index(w) + 1
        ax.set_title(f"W{i} elevation (grey: wall points, orange: objects in front of the wall)", fontsize=9)
        ax.set_aspect("equal")
        ax.set_ylim(-0.1, layout.height + 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ifc_3d(ifc_path, path):
    import ifcopenshell
    import ifcopenshell.geom
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    f = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    colors = {"IfcWall": "#d9d9d9", "IfcSlab": "#bfbfbf", "IfcDoor": COLORS["door"], "IfcWindow": COLORS["window"]}
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    allv = []
    for cls, col in colors.items():
        for el in f.by_type(cls):
            if el.Name == "Ceiling slab":
                continue
            shape = ifcopenshell.geom.create_shape(settings, el)
            v = np.array(shape.geometry.verts).reshape(-1, 3)
            tri = np.array(shape.geometry.faces).reshape(-1, 3)
            allv.append(v)
            ax.add_collection3d(Poly3DCollection(v[tri], facecolor=col, edgecolor=(0, 0, 0, 0.08),
                                                 linewidths=0.2, alpha=0.9 if cls != "IfcWall" else 0.75))
    v = np.vstack(allv)
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = (hi - lo).max()
    ax.set_xlim(lo[0], lo[0] + span)
    ax.set_ylim(lo[1], lo[1] + span)
    ax.set_zlim(lo[2], lo[2] + span * 0.6)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=35, azim=-60)
    ax.set_axis_off()
    ax.set_title("Generated IFC model (ceiling hidden)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
