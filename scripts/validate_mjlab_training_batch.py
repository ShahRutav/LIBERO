"""GPU-pod batch correctness gate. Writes compact report; fails on divergence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def prepare(dataset, demo):
    import h5py
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from scripts.replay_mjlab import localize_xml
    with h5py.File(dataset, "r") as f:
        data = f["data"]
        group = data[demo]
        xml = localize_xml(group.attrs["model_file"])
        states, actions = group["states"][:], group["actions"][:]
        kwargs = json.loads(data.attrs["env_args"])["env_kwargs"]
        bddl = Path(data.attrs["bddl_file_name"])
        bddl = Path(get_libero_path("bddl_files")) / bddl.parent.name / bddl.name
    kwargs.update(bddl_file_name=str(bddl), backend="mujoco", use_camera_obs=False,
                  has_renderer=False, has_offscreen_renderer=False, hard_reset=False, ignore_done=True)
    env = ControlEnv(**kwargs)
    env.reset()
    env.reset_from_xml_string(xml)
    env.env.deterministic_reset = True
    try:
        env.reset()
    finally:
        env.env.deterministic_reset = False
    env.sim.reset()
    env.set_init_state(states[0])
    env.robots[0].controller.update(force=True)
    env.robots[0].controller.reset_goal()
    return env, states, actions


def assert_close(a, b, label, atol=1e-5):
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=0 if atol == 0 else 1e-5)
    except AssertionError as error:
        diff = (a-b).abs()
        index = torch.nonzero(diff == diff.max())[0].tolist()
        raise AssertionError(f"{label}: max_abs={float(diff.max())}, index={index}, "
                             f"actual={float(a[tuple(index)])}, expected={float(b[tuple(index)])}\n{error}") from error
    return float((a-b).abs().max()) if a.numel() else 0.


def field_errors(batch, actual, expected):
    result, offset = {}, 0
    for field in batch.observation_spec["fields"]:
        width = int(np.prod(field["shape"]))
        if width:
            result[field["name"]] = float((actual[:, offset:offset+width]-expected[:, offset:offset+width]).abs().max())
        offset += width
    return result


def compare_observations(batch, actual, expected, label, report, diagnostic):
    errors = field_errors(batch, actual, expected)
    maxima = report.setdefault(label, {})
    for name, error in errors.items():
        maxima[name] = max(maxima.get(name, 0.), error)
    # Compare physical units separately. Contact summation order changes tiny
    # resting-body velocities even when positions agree to sub-micrometers.
    # These are bounded short-rollout checks, not long-contact reproducibility.
    tolerances = {name: (1e-3 if "velocity" in name or name == "qvel" else 1e-5)
                  for name in errors}
    report["field_absolute_tolerances"] = tolerances
    if not diagnostic:
        for name, error in errors.items():
            if error > tolerances[name]:
                raise AssertionError(f"{label}/{name}: max_abs={error} exceeds {tolerances[name]}")


def predicate_parity(batch):
    """Evaluate native predicates on the exact GPU states, excluding dynamics drift."""
    import mujoco
    from robosuite.utils.binding_utils import MjSim
    sim = MjSim(batch.model)
    native = batch.env.env
    old = native.sim
    states = batch.export_state()
    expected = []
    saved_pos, saved_quat = batch.model.body_pos.copy(), batch.model.body_quat.copy()
    try:
        native.sim = sim
        for i in range(batch.num_envs):
            batch.model.body_pos[:] = states["model_body_pos"][i].cpu().numpy()
            batch.model.body_quat[:] = states["model_body_quat"][i].cpu().numpy()
            sim.data.qpos[:] = states["qpos"][i].cpu().numpy()
            sim.data.qvel[:] = states["qvel"][i].cpu().numpy()
            if batch.model.na:
                sim.data.act[:] = states["act"][i].cpu().numpy()
            sim.forward()
            expected.append(bool(batch.env.check_success()))
    finally:
        native.sim = old
        batch.model.body_pos[:] = saved_pos
        batch.model.body_quat[:] = saved_quat
    with batch._scope():
        actual = batch._success().cpu().tolist()
    if expected != actual:
        raise AssertionError(f"Native/GPU predicate mismatch: {expected} != {actual}")
    return actual


def run(args):
    from libero.libero.envs.mjlab_batch import LiberoBatchEnv
    env, states, actions = prepare(args.dataset, args.demo)
    # Controlled, heterogeneous starts avoid testing chaotic release of an
    # in-flight grasp with intentionally missing prefix controller state.
    # Contact-rich recorded states are independently checked against predicates.
    bank = np.repeat(states[:1], 4, axis=0)
    arm_qpos = int(env.robots[0].controller.qpos_index[0])
    bank[:, 1+arm_qpos] += np.arange(4)*.002
    model = env.sim.model._model
    body_pos = np.repeat(model.body_pos[None], len(bank), axis=0)
    body_quat = np.repeat(model.body_quat[None], len(bank), axis=0)
    fixtures = [name for name in env.env.fixtures_dict if name in env.env.obj_body_id]
    if fixtures:
        body = env.env.obj_body_id[fixtures[0]]
        body_pos[:, body, 0] += np.arange(len(bank))*.001
    pose_kwargs = dict(initial_body_pos=body_pos, initial_body_quat=body_quat)
    batch = LiberoBatchEnv(env, bank, 4, device=args.device, **pose_kwargs)
    one = LiberoBatchEnv(env, bank, 1, device=args.device, **pose_kwargs)
    report = {}
    try:
        ids = torch.arange(4, device=args.device)
        permutation = torch.tensor([2, 0, 3, 1], device=args.device)
        generator = torch.Generator(device=args.device).manual_seed(17)
        sequence = torch.rand((args.steps, 4, 7), device=args.device, generator=generator)*.2-.1
        baseline_reset = batch.reset(state_ids=ids).clone()
        with batch._scope():
            torque = batch.controller.control(sequence[0].double(), True).clone()
            goals = batch.controller.goal_pos.clone()
            orientation = batch.controller.goal_ori.clone()
            grip = batch.controller.grip.clone()
        batch.reset(state_ids=ids[permutation])
        with batch._scope():
            torque_permuted = batch.controller.control(sequence[0, permutation].double(), True).clone()
            report["controller_permutation_torque"] = assert_close(torque_permuted, torque[permutation], "controller torque", atol=1e-7)
            assert_close(batch.controller.goal_pos, goals[permutation], "controller position", atol=1e-10)
            assert_close(batch.controller.goal_ori, orientation[permutation], "controller orientation", atol=1e-10)
            assert_close(batch.controller.grip, grip[permutation], "controller grip", atol=0)
        batch.reset(state_ids=ids)
        # Include positive final placement and negative initial snapshots.
        with batch._scope():
            batch._inject(ids, torch.as_tensor(states[np.linspace(0, len(states)-1, 4, dtype=int)], device=args.device))
        report["recorded_predicate_values"] = predicate_parity(batch)
        if not any(report["recorded_predicate_values"]) or all(report["recorded_predicate_values"]):
            raise AssertionError("Predicate gate requires native-positive and native-negative snapshots")
        batch.reset(state_ids=ids)
        baseline = []
        for a in sequence:
            baseline.append(batch.step(a)[0].clone())
            predicate_parity(batch)
        batch.reset(state_ids=ids)
        for i, a in enumerate(sequence):
            compare_observations(batch, batch.step(a)[0], baseline[i], "repeat_rollout", report, args.diagnostic)
        batch.reset(state_ids=ids[permutation])
        for i, a in enumerate(sequence):
            compare_observations(batch, batch.step(a[permutation])[0], baseline[i][permutation], "permutation", report, args.diagnostic)
        one.reset(state_ids=ids[2:3])
        for i, a in enumerate(sequence):
            compare_observations(batch, one.step(a[2:3])[0], baseline[i][2:3], "single_vs_batch", report, args.diagnostic)
        report["repeat_reset_max_abs"] = assert_close(batch.reset(state_ids=ids), baseline_reset, "repeat reset")
        for a in sequence[:2]:
            batch.step(a)
        before = batch.export_state()
        batch.reset(env_ids=ids[:1], state_ids=ids[:1])
        after = batch.export_state()
        for key in before:
            assert_close(after[key][1:], before[key][1:], "partial reset "+key, atol=0)
        report["partial_reset_isolation"] = True
        # Fast conversion must preserve the live controller's memory without
        # advancing physics, including zero-rotation and saturated gripper cases.
        for action in actions[:min(args.steps, len(actions))]:
            one.reset(state_ids=torch.zeros(1, device=args.device, dtype=torch.long))
            one.step(torch.as_tensor(action, device=args.device).reshape(1, -1))
            expected_memory = one.export_state()
            one.reset(state_ids=torch.zeros(1, device=args.device, dtype=torch.long))
            before_memory = one.export_state()
            with one._scope():
                before_time = one.engine.data.time[:].clone()
                one.controller.advance_memory(torch.as_tensor(action, device=args.device, dtype=torch.float64).reshape(1, -1).clamp(-1, 1), one.substeps)
                assert_close(one.engine.data.time[:], before_time, "fast prefix time", atol=0)
            actual_memory = one.export_state()
            for key in ("controller_goal_position", "controller_goal_orientation", "controller_grip"):
                assert_close(actual_memory[key], expected_memory[key], "fast prefix "+key, atol=0)
            for key in ("qpos", "qvel"):
                assert_close(actual_memory[key], before_memory[key], "fast prefix unchanged "+key, atol=0)
        report["fast_controller_prefix_exact_parity"] = True
        # Restore action-prefix memory and check online/offline observation path.
        one.reset(state_ids=torch.zeros(1, device=args.device, dtype=torch.long))
        for i in range(min(args.steps, len(states))):
            offline = one.restore_recorded_state(states[i], None if i == 0 else actions[i-1])
            assert_close(offline, one.observe(), "offline-online", atol=0)
        report["offline_online_observations"] = True
        report["schema_hash"] = batch.observation_schema_hash
        report["observation_dim"] = batch.num_obs
        # Existing matched-state GPU OSC/native controller parity gate.
        from scripts.benchmark_gpu_osc import validate_controller
        report["controller_parity"] = validate_controller(env, states[0], actions[:args.steps], batch.substeps, args.device)
        report["passed"] = not args.diagnostic
    except Exception:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        np.savez_compressed(args.output.with_suffix(".failure.npz"),
                            **{k:v.cpu().numpy() for k,v in batch.export_state().items()},
                            actions=sequence.cpu().numpy())
        raise
    finally:
        one.close()
        batch.close()
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnostic", action="store_true", help="Measure field errors without accepting rollout equivalence")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
