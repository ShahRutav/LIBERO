"""Opt-in, geometry-derived vertical placement for native ``On`` resets.

Only movable-object On(workspace) and On(object) heights are corrected. Site/In
placements retain their established articulated-site sampler. Bounds are a
conservative support envelope, not a concave-surface contact or stability proof;
callers must still qualify settling. No mesh assets or sampled XY/rotations change.
"""
from copy import deepcopy
import math

import numpy as np


def validate_height_policy(mode, clearance):
    if mode not in ('legacy', 'collision_bounds'):
        raise ValueError('Unknown placement height mode')
    if isinstance(clearance, bool) or not isinstance(clearance, (int, float)) or not math.isfinite(clearance) or clearance < 0:
        raise ValueError('Placement clearance must be finite and nonnegative')


def _rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid placement quaternion (expected wxyz)')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _geom_z_bounds(model, geom, center, rotation):
    """Exact vertical support for finite MuJoCo primitives and compiled meshes."""
    kind = int(model.geom_type[geom])
    direction = rotation[2]
    size = np.asarray(model.geom_size[geom], dtype=float)
    # MuJoCo's public mjtGeom enum: plane=0, hfield=1, sphere=2,
    # capsule=3, ellipsoid=4, cylinder=5, box=6, mesh=7.
    if kind == 7:
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = np.asarray(model.mesh_vert[start:start+count], dtype=float)
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError('Invalid compiled collision mesh')
        # The compiler has already scaled/recentered vertices; geom_xmat and
        # geom_xpos include its mesh-frame correction. Applying mesh_pos/scale
        # again here would double-transform collision geometry.
        values = vertices @ direction + center[2]
        return float(values.min()), float(values.max())
    if kind == 2:
        extent = size[0]
    elif kind == 3:
        extent = size[0] + size[1] * abs(direction[2])
    elif kind == 4:
        extent = np.linalg.norm(size * direction)
    elif kind == 5:
        extent = size[0] * np.linalg.norm(direction[:2]) + size[1] * abs(direction[2])
    elif kind == 6:
        extent = np.abs(direction) @ size
    else:
        raise ValueError(f'Unsupported collision geometry type {kind} for placement bounds')
    if not np.isfinite(extent) or extent < 0:
        raise ValueError('Invalid collision geometry bounds')
    return float(center[2] - extent), float(center[2] + extent)


def object_vertical_bounds(sim, obj, quaternion):
    """Bounds relative to the NEW root; retain current descendant articulation.

    Kinematics must be refreshed after sampling articulated joints. Every geom
    is pulled out of the previous world/root frame, then rotated into the new
    sampled root frame. This avoids stale reset poses and supports child bodies.
    """
    model = sim.model._model
    data = sim.data._data
    root = sim.model.body_name2id(obj.root_body)
    old_position = np.asarray(data.xpos[root], dtype=float)
    old_rotation = np.asarray(data.xmat[root], dtype=float).reshape(3, 3)
    transform = _rotation(quaternion) @ old_rotation.T
    bounds = []
    geom_ids = []
    explicit_pairs = set(map(int, getattr(model, "pair_geom1", []))) | set(map(int, getattr(model, "pair_geom2", [])))
    for geom in range(model.ngeom):
        # Explicit contact pairs can activate otherwise mask-disabled geoms.
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0 and geom not in explicit_pairs:
            continue
        body = int(model.geom_bodyid[geom])
        while body not in (0, root):
            body = int(model.body_parentid[body])
        if body != root:
            continue
        center = transform @ (np.asarray(data.geom_xpos[geom]) - old_position)
        rotation = transform @ np.asarray(data.geom_xmat[geom]).reshape(3, 3)
        if not np.isfinite(center).all() or not np.isfinite(rotation).all():
            raise ValueError('Nonfinite collision geometry transform')
        bounds.append(_geom_z_bounds(model, geom, center, rotation))
        geom_ids.append(geom)
    if not bounds:
        raise ValueError(f'No supported collision geometry for {obj.name}')
    return min(x[0] for x in bounds), max(x[1] for x in bounds), geom_ids


def correct_on_placements(sim, placements, initial_state, regions, movable_names,
                          fixture_names, workspace_z, *, clearance=0.001):
    """Return corrected copies and auditable heights, without RNG or simulation.

    Dependency ordering is independent of BDDL statement order. Sites and In
    retain legacy semantics and are listed as unchanged, never treated as a
    whole cabinet's upper envelope. Unknown On supports fail explicitly.
    """
    validate_height_policy('collision_bounds', clearance)
    if not math.isfinite(float(workspace_z)):
        raise ValueError('Workspace height must be finite')
    movable_names, fixture_names = set(movable_names), set(fixture_names)
    relations, unchanged = {}, []
    for state in initial_state:
        if len(state) < 3 or state[0] not in ('on', 'in'):
            continue
        predicate, child, target = state[:3]
        if child not in movable_names:
            continue
        if predicate == 'in':
            unchanged.append({'relation': list(state), 'reason': 'site/In sampler unchanged'})
            continue
        if target in regions:
            support = regions[target]['target']
            if support in movable_names or support in fixture_names:
                unchanged.append({'relation': list(state), 'reason': 'site/In sampler unchanged'})
                continue
            support = None  # Existing workspace sampler uses workspace_offset.
        elif target in movable_names:
            support = target
        else:
            raise ValueError(f'Unsupported On placement support: {target}')
        if child in relations:
            raise ValueError(f'Multiple On support relations for {child}')
        relations[child] = support
    corrected = dict(placements)
    corrected_bounds, corrections, pending = {}, [], dict(relations)

    def bounds(name):
        if name not in corrected:
            raise ValueError(f'Missing sampled placement for {name}')
        if name not in corrected_bounds:
            corrected_bounds[name] = object_vertical_bounds(sim, corrected[name][2], corrected[name][1])
        return corrected_bounds[name]

    while pending:
        progressed = False
        for child, support in list(pending.items()):
            if support in pending:
                continue
            minimum, maximum, geoms = bounds(child)
            support_top = float(workspace_z)
            if support is not None:
                support_max = bounds(support)[1]
                support_top = float(corrected[support][0][2]) + support_max
            old_position, quaternion, obj = corrected[child]
            position = np.asarray(old_position, dtype=float).copy()
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Invalid sampled object position')
            position[2] = support_top - minimum + clearance
            corrected[child] = (tuple(position), deepcopy(quaternion), obj)
            corrections.append({'object': child, 'support': support or 'workspace',
                                'old_z': float(old_position[2]), 'new_z': float(position[2]),
                                'support_top_z': support_top, 'collision_min_z': minimum,
                                'collision_max_z': maximum, 'geom_ids': geoms})
            del pending[child]
            progressed = True
        if not progressed:
            raise ValueError('Cyclic On placement dependencies')
    return corrected, {'version': 1, 'mode': 'collision_bounds', 'clearance_m': float(clearance),
                       'adjustments': corrections, 'ignored_relations': unchanged}
