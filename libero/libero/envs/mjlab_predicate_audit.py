"""Independent native predicate evaluation on GPU states and GPU contacts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def predicate_parity(batch, *, diagnostics_path=None, context=None, raise_on_mismatch=True):
    import warp as wp
    from robosuite.utils.binding_utils import MjSim
    native = batch.env.env
    sim = MjSim(copy.copy(batch.model))
    old_sim, old_contact = native.sim, native.check_contact
    states = {k: v.cpu().numpy() for k, v in batch.export_state().items()}
    with batch._scope():
        actual = batch._success().cpu().tolist()
        count = int(wp.to_torch(batch.engine.wp_data.nacon).reshape(-1)[0])
        contact = batch.engine.wp_data.contact
        pairs = wp.to_torch(contact.geom)[:count].cpu().numpy()
        worlds = wp.to_torch(contact.worldid)[:count].cpu().numpy()
        distances = wp.to_torch(contact.dist)[:count].cpu().numpy()
        positions = batch.engine.data.xpos[:].cpu().numpy()
        sites = batch.engine.data.site_xpos[:].cpu().numpy()
        site_matrices = batch.engine.data.site_xmat[:].cpu().numpy()

    def geoms(obj):
        names = obj.contact_geoms if hasattr(obj, 'contact_geoms') else obj
        if isinstance(names, str):
            names = [names]
        return {sim.model.geom_name2id(g) for g in names}

    expected, mismatches = [], []
    try:
        for i in range(batch.num_envs):
            sim.model._model.body_pos[:] = states['model_body_pos'][i]
            sim.model._model.body_quat[:] = states['model_body_quat'][i]
            sim.data.qpos[:] = states['qpos'][i]
            sim.data.qvel[:] = states['qvel'][i]
            if batch.model.na:
                sim.data.act[:] = states['act'][i]
            sim.forward()
            native.sim, native.check_contact = sim, old_contact
            expected.append(bool(batch.env.check_success()))
            selected = worlds == i
            gpu_pairs = pairs[selected]

            def gpu_contact(first, second=None):
                a = geoms(first)
                b = geoms(second) if second is not None else None
                return any((int(g[0]) in a and (b is None or int(g[1]) in b))
                           or (int(g[1]) in a and (b is None or int(g[0]) in b)) for g in gpu_pairs)

            data = SimpleNamespace(
                body_xpos=positions[i], qpos=states['qpos'][i],
                get_site_xpos=lambda name: sites[i, sim.model.site_name2id(name)],
                get_site_xmat=lambda name: site_matrices[i, sim.model.site_name2id(name)].reshape(3, 3),
            )
            native.sim = SimpleNamespace(model=sim.model, data=data)
            native.check_contact = gpu_contact
            logical = bool(batch.env.check_success())
            if logical != actual[i]:
                raise AssertionError(f'GPU predicate logic differs from native functions on identical GPU data: world {i}, goals {native.parsed_problem["goal_state"]}')
            if expected[-1] != actual[i]:
                mismatches.append(dict(world=i, native_success=expected[-1], gpu_success=actual[i],
                    goals=native.parsed_problem['goal_state'],
                    body_position_max_error=float(np.max(np.abs(sim.data.body_xpos - positions[i]))),
                    site_position_max_error=float(np.max(np.abs(sim.data.site_xpos - sites[i]))),
                    gpu_contacts=[dict(geoms=g.tolist(), distance=float(d))
                                  for g, d in zip(gpu_pairs, distances[selected])],
                    native_contacts=[dict(geoms=[int(c.geom1), int(c.geom2)], distance=float(c.dist))
                                     for c in sim.data.contact[:sim.data.ncon]],
                    qpos=states['qpos'][i].tolist()))
    finally:
        native.sim, native.check_contact = old_sim, old_contact
    if mismatches:
        record = dict(context=context, mismatches=mismatches, gpu_logic_matches=True,
                      protocol='native_functions_on_gpu_state_v1')
        if diagnostics_path:
            path = Path(diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        if raise_on_mismatch:
            raise AssertionError(f'Native/GPU predicate boundaries differ in {len(mismatches)} worlds')
    return actual if raise_on_mismatch else dict(native=expected, gpu=actual,
                                                mismatches=mismatches, gpu_logic_matches=True)
