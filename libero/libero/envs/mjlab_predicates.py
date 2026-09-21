"""Batched LIBERO-90 goal predicates using the native predicate conventions.

Region formulas intentionally retain SiteObject's matrix convention and strict
bounds. Contact membership follows robosuite, including nonpenetrating contacts.
"""
import numpy as np
import torch


def compile_program(batch):
    native = batch.env.env
    model = native.sim.model
    program, bodies = [], set()

    def object_id(name):
        body = native.obj_body_id[name]
        bodies.add(int(body))
        return int(body)

    def geom_ids(name):
        return torch.tensor([model.geom_name2id(g) for g in native.get_object(name).contact_geoms],
                            dtype=torch.long, device=batch.device)

    for goal in native.parsed_problem['goal_state']:
        kind, *names = goal
        kind = kind.lower()
        item = dict(kind=kind, names=names)
        if kind in ('on', 'in') and len(names) == 2:
            top, target = names
            if native.object_states_dict[top].object_state_type != 'object':
                raise NotImplementedError(f'Nonobject source: {goal}')
            item['top'] = object_id(top)
            state = native.object_states_dict[target]
            if state.object_state_type == 'object':
                if kind != 'on':
                    raise NotImplementedError(f'Object containment requires a separate native contract: {goal}')
                item.update(bottom=object_id(target), top_geoms=geom_ids(top), bottom_geoms=geom_ids(target))
            elif state.object_state_type == 'site':
                site = native.object_sites_dict[target]
                item.update(site=int(model.site_name2id(target)),
                            size=torch.tensor(site.size, device=batch.device, dtype=torch.float32))
                if kind == 'on':
                    if not hasattr(site, 'under'):
                        item['always_true'] = True
                    elif native.get_object(state.parent_name) is not None:
                        item.update(top_geoms=geom_ids(top), bottom_geoms=geom_ids(state.parent_name))
                if state.parent_name in native.obj_body_id:
                    object_id(state.parent_name)
            else:
                raise NotImplementedError(f'Unknown state type: {goal}')
        elif kind in ('open', 'close', 'turnon', 'turnoff') and len(names) == 1:
            name = names[0]
            state = native.object_states_dict[name]
            is_site = state.object_state_type == 'site'
            obj = native.get_object(state.parent_name if is_site else name)
            joints = native.object_sites_dict[name].joints if is_site else obj.joints
            addresses = [model.get_joint_qpos_addr(joint) for joint in joints]
            if any(not isinstance(a, (int, np.integer)) for a in addresses):
                raise NotImplementedError(f'Articulated predicate requires scalar joints: {goal}')
            item['qpos'] = addresses
            object_id(state.parent_name if is_site else name)
            properties = obj.object_properties['articulation']
            class_name = type(obj).__name__
            if kind in ('open', 'close'):
                if class_name in ('Microwave', 'WoodenCabinet', 'WhiteCabinet'):
                    item.update(operator='lt' if kind == 'open' else 'gt',
                                threshold=max(properties['default_open_ranges']) if kind == 'open'
                                else min(properties['default_close_ranges']))
                elif class_name in ('ShortCabinet', 'ShortFridge'):
                    item.update(operator='gt' if kind == 'open' else 'lt',
                                threshold=min(properties['default_open_ranges']) if kind == 'open'
                                else max(properties['default_close_ranges']))
                else:
                    raise NotImplementedError(f'Unqualified articulation class: {class_name}')
            elif class_name == 'FlatStove' and not is_site:
                item.update(operator='ge' if kind == 'turnon' else 'lt',
                            threshold=min(properties['default_turnon_ranges']) if kind == 'turnon'
                            else max(properties['default_turnoff_ranges']))
            else:
                raise NotImplementedError(f'Unqualified switch predicate: {goal}')
            item['any_joint'] = kind in ('open', 'turnon')
        else:
            raise NotImplementedError(f'GPU success predicate unsupported: {goal}')
        program.append(item)
    if not program:
        raise ValueError('Task has no goal predicates')
    batch.goal_body_ids = sorted(bodies)
    return program


def geometry(item, xpos, site_xpos, site_xmat, qpos):
    """One boolean per world, before any required collision membership."""
    if item.get('always_true'):
        return torch.ones(len(qpos), dtype=torch.bool, device=qpos.device)
    if 'qpos' in item:
        values = qpos[:, item['qpos']]
        threshold = item['threshold']
        flags = {'lt': torch.lt, 'gt': torch.gt, 'ge': torch.ge}[item['operator']](values, threshold)
        return flags.any(-1) if item['any_joint'] else flags.all(-1)
    top = xpos[:, item['top']]
    if 'bottom' in item:
        bottom = xpos[:, item['bottom']]
        return (top[:, 2] >= bottom[:, 2]) & (torch.linalg.vector_norm(top[:, :2] - bottom[:, :2], dim=-1) < .03)
    position = site_xpos[:, item['site']]
    matrix = site_xmat[:, item['site']].reshape(-1, 3, 3)
    size = item['size'].to(dtype=position.dtype)
    if item['kind'] == 'in':
        extent = torch.abs(matrix @ size)
        lower, upper = position - extent, position + extent
        lower = lower.clone()
        lower[:, 2] -= .01
        return ((top > lower) & (top < upper)).all(-1)
    delta = (matrix @ (top - position).unsqueeze(-1)).squeeze(-1)
    return ((delta[:, 2] > size[2] - .005) & (delta[:, 2] < size[2] + .10)
            & (torch.abs(delta[:, :2]) < size[:2]).all(-1))


def success(batch):
    import warp as wp
    data = batch.engine.data
    result = torch.ones(batch.num_envs, dtype=torch.bool, device=batch.device)
    needs_contact = any('top_geoms' in item for item in batch._predicate_program)
    if needs_contact:
        contacts = batch.engine.wp_data.contact
        geom = wp.to_torch(contacts.geom).long()
        world = wp.to_torch(contacts.worldid).long()
        count = wp.to_torch(batch.engine.wp_data.nacon).reshape(-1)[0]
        if bool(count > len(world)):
            raise RuntimeError('GPU contact buffer overflow; increase nconmax')
        valid = (torch.arange(len(world), device=batch.device) < count) & (world >= 0) & (world < batch.num_envs)
    for item in batch._predicate_program:
        value = geometry(item, data.xpos[:], data.site_xpos[:], data.site_xmat[:], data.qpos[:])
        if 'top_geoms' in item:
            top, bottom = item['top_geoms'], item['bottom_geoms']
            matches = ((torch.isin(geom[:, 0], top) & torch.isin(geom[:, 1], bottom))
                       | (torch.isin(geom[:, 1], top) & torch.isin(geom[:, 0], bottom))) & valid
            counts = torch.zeros(batch.num_envs, dtype=torch.long, device=batch.device)
            counts.scatter_add_(0, world.clamp(0, batch.num_envs - 1), matches.long())
            value &= counts > 0
        result &= value
    return result
