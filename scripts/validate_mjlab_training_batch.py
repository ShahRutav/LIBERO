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
    torch.testing.assert_close(a, b, atol=atol, rtol=1e-5, msg=label)
    return float((a-b).abs().max())


def predicate_parity(batch):
    """Evaluate native predicates on the exact GPU states, excluding dynamics drift."""
    import mujoco
    from robosuite.utils.binding_utils import MjSim
    sim = MjSim(batch.model)
    native = batch.env.env
    old = native.sim
    states = batch.export_state()
    expected = []
    try:
        native.sim = sim
        for i in range(batch.num_envs):
            sim.data.qpos[:] = states["qpos"][i].cpu().numpy()
            sim.data.qvel[:] = states["qvel"][i].cpu().numpy()
            if batch.model.na:
                sim.data.act[:] = states["act"][i].cpu().numpy()
            sim.forward()
            expected.append(bool(batch.env.check_success()))
    finally:
        native.sim = old
    with batch._scope():
        actual = batch._success().cpu().tolist()
    if expected != actual:
        raise AssertionError(f"Native/GPU predicate mismatch: {expected} != {actual}")
    return actual


def run(args):
    from libero.libero.envs.mjlab_batch import LiberoBatchEnv
    env, states, actions = prepare(args.dataset, args.demo)
    # Midtrajectory states here intentionally stress indexing; their controller
    # resets define new episodes and do not claim expert-prefix restoration.
    bank = states[np.linspace(0, len(states)-1, 4, dtype=int)]
    batch = LiberoBatchEnv(env, bank, 4, device=args.device)
    one = LiberoBatchEnv(env, bank, 1, device=args.device)
    report = {}
    try:
        ids = torch.arange(4, device=args.device)
        permutation = torch.tensor([2, 0, 3, 1], device=args.device)
        generator = torch.Generator(device=args.device).manual_seed(17)
        sequence = torch.rand((args.steps, 4, 7), device=args.device, generator=generator)*.2-.1
        baseline_reset = batch.reset(state_ids=ids).clone()
        batch.reset(state_ids=ids)
        baseline = []
        for a in sequence:
            baseline.append(batch.step(a)[0].clone())
            predicate_parity(batch)
        batch.reset(state_ids=ids[permutation])
        for i, a in enumerate(sequence):
            report["permutation_max_abs"] = assert_close(batch.step(a[permutation])[0], baseline[i][permutation], "permutation")
        one.reset(state_ids=ids[2:3])
        for i, a in enumerate(sequence):
            report["single_vs_batch_max_abs"] = assert_close(one.step(a[2:3])[0], baseline[i][2:3], "single world", atol=5e-5)
        report["repeat_reset_max_abs"] = assert_close(batch.reset(state_ids=ids), baseline_reset, "repeat reset")
        for a in sequence[:2]:
            batch.step(a)
        before = batch.export_state()
        batch.reset(env_ids=ids[:1], state_ids=ids[:1])
        after = batch.export_state()
        for key in before:
            assert_close(after[key][1:], before[key][1:], "partial reset "+key, atol=0)
        report["partial_reset_isolation"] = True
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
        report["passed"] = True
    except Exception:
        args.output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
