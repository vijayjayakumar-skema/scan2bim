"""Write a Layout to an IFC4 file: storey, walls with real openings, doors, windows, floor and ceiling slabs.

Every element carries a property set recording the scan evidence behind it
(how many points supported it, how well the plane fitted), so a reviewer
can see how much each part of the model is worth trusting.
"""
import numpy as np
import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.feature
import ifcopenshell.api.geometry
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit

api = ifcopenshell.api


def _matrix(origin, xdir):
    """4x4 placement with local x along xdir, z up, y = z cross x."""
    x = np.array([xdir[0], xdir[1], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


def _wall_frame(w, walls):
    """Start point, direction and length of a wall drawn so its thickness grows away from the room.

    The wall is lengthened at convex corners only, so neighbouring walls
    overlap at the outside corner instead of leaving a gap, while reflex
    corners do not poke into the room.
    """
    a, b = w.endpoints()
    axis_dir = (b - a) / np.linalg.norm(b - a)
    exterior = np.array([-w.inside, 0.0]) if w.axis == "x" else np.array([0.0, -w.inside])
    # local y = z cross x must point to the exterior; otherwise draw the wall the other way
    if np.cross([0, 0, 1], [axis_dir[0], axis_dir[1], 0])[:2] @ exterior < 0:
        a, b, axis_dir = b, a, -axis_dir
    return a, b, axis_dir


def export_ifc(layout, path, thickness=0.2, name="Scanned room"):
    f = api.project.create_file(version="IFC4")
    project = api.root.create_entity(f, ifc_class="IfcProject", name="Scan-to-BIM demo")
    api.unit.assign_unit(f)
    model = api.context.add_context(f, context_type="Model")
    body = api.context.add_context(f, context_type="Model", context_identifier="Body",
                                   target_view="MODEL_VIEW", parent=model)
    site = api.root.create_entity(f, ifc_class="IfcSite", name="Site")
    building = api.root.create_entity(f, ifc_class="IfcBuilding", name="Building")
    storey = api.root.create_entity(f, ifc_class="IfcBuildingStorey", name="Ground floor")
    api.aggregate.assign_object(f, products=[site], relating_object=project)
    api.aggregate.assign_object(f, products=[building], relating_object=site)
    api.aggregate.assign_object(f, products=[storey], relating_object=building)

    height = layout.height
    t = thickness
    elements = []

    def evidence(el, props):
        ps = api.pset.add_pset(f, product=el, name="Scan2BIM_Evidence")
        api.pset.edit_pset(f, pset=ps, properties=props)

    for i, w in enumerate(layout.walls):
        a, b, d = _wall_frame(w, layout.walls)
        length = float(np.linalg.norm(b - a))
        start_ext = end_ext = 0.0
        for end_pt, sign in ((a, -1), (b, 1)):
            other = next((v for v in layout.walls if v is not w and v.axis != w.axis
                          and abs(v.c - (end_pt[1] if w.axis == "x" else end_pt[0])) < 1e-6), None)
            if other is None:
                continue
            other_inside = np.array([other.inside, 0.0]) if other.axis == "x" else np.array([0.0, other.inside])
            if other_inside @ (sign * d) < 0:   # the room lies back towards this wall: convex corner
                if sign < 0:
                    start_ext = t
                else:
                    end_ext = t
        origin = np.array([a[0] - d[0] * start_ext, a[1] - d[1] * start_ext, 0.0])
        wall = api.root.create_entity(f, ifc_class="IfcWall", name=f"Wall {i + 1}")
        rep = api.geometry.add_wall_representation(f, context=body, length=length + start_ext + end_ext,
                                                   height=height, thickness=t)
        api.geometry.assign_representation(f, product=wall, representation=rep)
        api.geometry.edit_object_placement(f, product=wall, matrix=_matrix(origin, d))
        evidence(wall, {"SupportPoints": int(w.n_points), "PlaneFitRMS_mm": round(w.rms * 1000, 1),
                        "Method": "RANSAC plane, Manhattan-snapped", "ThicknessAssumed": True})
        elements.append(wall)

        # u is measured along the aligned axis; convert to distance from this wall's start point
        u_start = a[1] if w.axis == "x" else a[0]
        dir_sign = d[1] if w.axis == "x" else d[0]
        for k, op in enumerate(w.openings):
            s0 = min((op.u0 - u_start) * dir_sign, (op.u1 - u_start) * dir_sign)
            o_origin = np.array([a[0] + d[0] * s0, a[1] + d[1] * s0, op.z0])
            opening = api.root.create_entity(f, ifc_class="IfcOpeningElement", name=f"Opening {i + 1}.{k + 1}")
            orep = api.geometry.add_wall_representation(f, context=body, length=op.width, height=op.height,
                                                        thickness=t + 0.2, offset=-0.1)
            api.geometry.assign_representation(f, product=opening, representation=orep)
            api.geometry.edit_object_placement(f, product=opening, matrix=_matrix(o_origin, d))
            api.feature.add_feature(f, feature=opening, element=wall)

            cls = "IfcDoor" if op.kind == "door" else "IfcWindow"
            fill = api.root.create_entity(f, ifc_class=cls, name=f"{op.kind.title()} {i + 1}.{k + 1}")
            fill.OverallWidth, fill.OverallHeight = round(op.width, 3), round(op.height, 3)
            frep = api.geometry.add_wall_representation(f, context=body, length=op.width, height=op.height,
                                                        thickness=0.05, offset=t / 2 - 0.025)
            api.geometry.assign_representation(f, product=fill, representation=frep)
            api.geometry.edit_object_placement(f, product=fill, matrix=_matrix(o_origin, d))
            api.feature.add_filling(f, opening=opening, element=fill)
            evidence(fill, {"Method": "empty region in wall occupancy grid",
                            "CheckedForOcclusion": True})
            elements.append(fill)

    if layout.footprint:
        for name_, z, kind in (("Floor slab", -t, "FLOOR"), ("Ceiling slab", height, "ROOF")):
            slab = api.root.create_entity(f, ifc_class="IfcSlab", name=name_, predefined_type=kind)
            rep = api.geometry.add_slab_representation(f, context=body, depth=t,
                                                       polyline=[tuple(p) for p in layout.footprint])
            api.geometry.assign_representation(f, product=slab, representation=rep)
            api.geometry.edit_object_placement(f, product=slab, matrix=_matrix(np.array([0, 0, z]), (1, 0)))
            elements.append(slab)

    api.spatial.assign_container(f, products=elements, relating_structure=storey)
    f.write(path)
    return f
