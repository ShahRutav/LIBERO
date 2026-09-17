"""Validate and time batched GPU OSC plus mjlab physics on the GPU pod."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np
import torch
import warp as wp

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from scripts.replay_mjlab import localize_xml


from scripts.benchmark_mjlab_batch import Batch as CPUBatch
from libero.libero.envs.gpu_osc import GPUOSC
import libero.libero.envs.gpu_osc as gpu_osc_module


class Batch(CPUBatch):
    def __init__(self, *args, **kwargs):
        self.controller = None
        super().__init__(*args, **kwargs)
        if self.engine is not None:
            with torch.cuda.stream(self.stream):
                self.controller = GPUOSC(self.engine, self.env.robots[0])

    def reset(self):
        super().reset()
        if self.controller is not None:
            with torch.cuda.stream(self.stream):
                self.controller.reset(self.env.robots[0].controller)

    def close(self):
        self.controller = None
        self.engine = None
        self.robots.clear()
        self.hosts.clear()
        gc.collect()

    def step_controlled(self, action, policy_step):
        if self.engine is None:
            return super().step_controlled(action, policy_step)
        with torch.cuda.stream(self.stream):
            self.engine.forward()
            self.controller.control(action, policy_step)
            self.engine.step()


def validate_controller(env, initial, actions, substeps, device):
    """103 native trajectory states: matched inputs and GPU-derived dynamics."""
    native = CPUBatch(env, 1, "mujoco", initial, device)
    gpu = Batch(env, 1, "mjlab", initial, device)
    errors = dict(torque=0., goal_position=0., goal_orientation=0., gripper=0.,
                  gpu_position=0., gpu_orientation=0., gpu_jacobian=0., gpu_mass=0., gpu_bias=0.)
    try:
        for action in actions:
            robot = native.robots[0]
            robot.sim.forward()
            c = robot.controller
            c.update(force=True)
            values = (c.ee_pos, c.ee_ori_mat, c.J_full, c.mass_matrix,
                      c.joint_pos, c.joint_vel, c.torque_compensation)
            with torch.cuda.stream(gpu.stream):
                gc = gpu.controller
                gc.reset(c)
                gc.grip[:] = gc.tensor(robot.gripper.current_action)
                for name in ("qpos", "qvel"):
                    getattr(gpu.engine.data,name)[:] = gc.tensor(getattr(robot.sim.data,name))
                gpu.engine.forward()
                actual = gc.state()
                for name, value, target in zip(("gpu_position","gpu_orientation","gpu_jacobian","gpu_mass",None,None,"gpu_bias"),actual,values):
                    if name:
                        errors[name] = max(errors[name],float(np.max(np.abs(value[0].cpu().numpy()-target))))
                original = gc.state
                gc.state = lambda: tuple(gc.tensor(v)[None] for v in values)
                try:
                    gt = gc.control(gc.tensor(action), True)[0].cpu().numpy()
                finally:
                    gc.state = original
                robot.control(action, policy_step=True)
                for name, value, target in (("torque",gt,c.torques),
                        ("goal_position",gc.goal_pos[0].cpu().numpy(),c.goal_pos),
                        ("goal_orientation",gc.goal_ori[0].cpu().numpy(),c.goal_ori),
                        ("gripper",gpu.engine.data.ctrl[0,gc.gids].cpu().numpy(),robot.sim.data.ctrl[gc.gids.cpu().numpy()])):
                    errors[name] = max(errors[name],float(np.max(np.abs(value-target))))
            robot.sim.step()
            for _ in range(substeps-1):
                native.step_controlled(action,False)
        if any(errors[k] > limit for k, limit in {"torque":1e-4,"gpu_mass":1e-4,"gpu_jacobian":1e-4,"goal_position":1e-8,"goal_orientation":1e-6,"gripper":1e-7,"gpu_position":1e-5,"gpu_orientation":1e-5,"gpu_bias":1e-3}.items()):
            raise RuntimeError(f"Controller parity failed: {errors}")
        print("controller_parity",errors,flush=True)
        return {"states":len(actions),"max_absolute_errors":errors}
    finally:
        native.close()
        gpu.close()


def controlled(batch, actions, substeps, collect_torques=False):
    controls = []
    if batch.engine is not None:
        with torch.cuda.stream(batch.stream):
            actions = torch.as_tensor(actions, dtype=torch.float64, device=batch.device)
    batch.synchronize()
    start = time.perf_counter()
    for action in actions:
        for substep in range(substeps):
            batch.step_controlled(action, policy_step=substep == 0)
            if collect_torques:
                controls.append(batch.hosts[0].data.ctrl.copy())
    batch.synchronize()
    return time.perf_counter() - start, np.asarray(controls)


def timing_record(backend, count, scope, seconds, actions, substeps, outcomes):
    median = float(np.median(seconds))
    return {"backend": backend, "worlds": count, "scope": scope,
            "seconds_trials": seconds, "batch_seconds_median": median,
            "amortized_episode_seconds": median / count,
            "batch_control_step_ms": 1000 * median / actions,
            "amortized_control_step_ms": 1000 * median / (count * actions),
            "aggregate_env_control_steps_per_second": count * actions / median,
            "aggregate_env_physics_steps_per_second": count * actions * substeps / median,
            "outcomes_trials": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 50])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--demo", default="demo_0")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sizes) < 1:
        parser.error("sizes and repeats must be positive")
    torch.set_num_threads(1)
    np.random.seed(0)
    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        demo = data[args.demo]
        xml = localize_xml(demo.attrs["model_file"])
        initial = demo["states"][0]
        actions = demo["actions"][:]
        kwargs = json.loads(data.attrs["env_args"])["env_kwargs"]
        bddl = Path(data.attrs["bddl_file_name"])
        bddl = Path(get_libero_path("bddl_files")) / bddl.parent.name / bddl.name
    kwargs.update(bddl_file_name=str(bddl), backend="mujoco", use_camera_obs=False,
                  has_renderer=False, has_offscreen_renderer=False, hard_reset=False,
                  ignore_done=True)
    env = ControlEnv(**kwargs)
    report = {"demo": args.demo, "dataset": str(args.dataset),
              "source_revision": os.environ.get("LIBERO_SOURCE_REVISION", "not recorded"),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "controller_source_sha256": hashlib.sha256(Path(gpu_osc_module.__file__).read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(args.device),
              "versions": {p: importlib.metadata.version(p) for p in
                           ("mujoco", "mjlab", "mujoco-warp", "torch")},
              "identical_demo_in_independent_worlds": True, "cameras": False,
              "cpu_threads": 1, "actions_per_episode": len(actions), "rows": []}
    try:
        env.reset()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.set_init_state(initial)
        # Match replay_mjlab's reset protocol before collecting the reference.
        env.env.deterministic_reset = True
        try:
            env.reset()
        finally:
            env.env.deterministic_reset = False
        env.sim.reset()
        env.set_init_state(initial)
        env.robots[0].controller.update(force=True)
        env.robots[0].controller.reset_goal()
        substeps = int(env.env.control_timestep / env.env.model_timestep)
        report["physics_steps_per_action"] = substeps
        # Produce a physically meaningful torque workload outside all timings.
        reference = Batch(env, 1, "mujoco", initial, args.device)
        _, torques = controlled(reference, actions, substeps, collect_torques=True)
        reference_outcomes, reference_states = reference.outcomes()
        report["reference"] = reference_outcomes
        ref_eef = reference.hosts[0].data.get_site_xpos(env.robots[0].controller.eef_name).copy()
        object_ids = list(env.env.obj_body_id.values())
        ref_objects = reference.hosts[0].data.body_xpos[object_ids].copy()
        reference.close()
        if not reference_outcomes["success_count"]:
            raise RuntimeError("Native reference must complete the expert task before benchmarking")
        # Verify that removing observation bookkeeping did not change the
        # native physics/controller trajectory used as the timing baseline.
        for action in actions:
            env.step(action)
        original_state = env.get_sim_state()
        report["native_loop_vs_original_env_max_abs"] = float(np.max(
            np.abs(original_state - reference_states[0])))
        if report["native_loop_vs_original_env_max_abs"] > 1e-7:
            raise RuntimeError(f"Benchmark loop differs from LIBERO by {report['native_loop_vs_original_env_max_abs']}")
        env.env.deterministic_reset = True
        try:
            env.reset()
        finally:
            env.env.deterministic_reset = False
        env.sim.reset()
        env.set_init_state(initial)
        env.robots[0].controller.update(force=True)
        env.robots[0].controller.reset_goal()
        report["controller_parity"] = validate_controller(env, initial, actions, substeps, args.device)
        for count in args.sizes:
            for backend in ("mjlab",):
                batch = Batch(env, count, backend, initial, args.device)
                try:
                    for scope in ("gpu_osc_and_physics",):
                        # Warm the entire workload before reset and timing.
                        batch.reset()
                        controlled(batch, actions, substeps)
                        seconds, outcomes = [], []
                        for trial in range(args.repeats):
                            batch.reset()
                            elapsed, _ = controlled(batch, actions, substeps)
                            outcome, states = batch.outcomes()
                            outcome["max_final_qpos_error_vs_native"] = float(np.max(
                                np.abs(states[:, 1:1 + batch.model.nq] - reference_states[:, 1:1 + batch.model.nq])))
                            eef = np.stack([s.data.get_site_xpos(env.robots[0].controller.eef_name) for s in batch.hosts])
                            objects = np.stack([s.data.body_xpos[object_ids] for s in batch.hosts])
                            outcome["max_final_eef_distance_m"] = float(np.linalg.norm(eef-ref_eef,axis=-1).max())
                            outcome["max_final_object_distance_m"] = float(np.linalg.norm(objects-ref_objects,axis=-1).max())
                            free, total = torch.cuda.mem_get_info(args.device)
                            outcome["device_used_mib"] = (total-free)/2**20
                            if not outcome["finite"]:
                                raise RuntimeError("Non-finite simulation invalidates timing")
                            seconds.append(elapsed)
                            outcomes.append(outcome)
                            print(backend, count, scope, trial, round(elapsed, 3),
                                  "success", outcome["success_count"], flush=True)
                        row = timing_record(backend, count, scope, seconds, len(actions), substeps, outcomes)
                        report["rows"].append(row)
                        (args.output / "report.json").write_text(json.dumps(report, indent=2))
                finally:
                    batch.close()
    finally:
        env.close()
    print(json.dumps([{k: v for k, v in row.items() if k != "outcomes_trials"}
                      for row in report["rows"]], indent=2), flush=True)


if __name__ == "__main__":
    main()
