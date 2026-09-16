"""GPU-pod benchmark of shared-model LIBERO worlds at batch sizes 1 and 50.

Two scopes: original CPU OSC plus physics, and physics-only torque replay.
No cameras, observation manager, policy inference, or reset/setup in timings.
"""

import argparse
import copy
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

from mjlab.sim import Simulation, SimulationCfg
from robosuite.utils.binding_utils import MjSim

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.envs.mjlab_sim import _PreserveOptions
from scripts.replay_mjlab import localize_xml


class Batch:
    """Shared immutable model, independent MjData/controllers, one GPU batch.

    CPU mirrors are a benchmark bridge, not a vector environment API. During
    rollout only controls change on the host; integrated state stays on GPU.
    """

    def __init__(self, env, count, backend, initial, device):
        self.env, self.count, self.backend = env, count, backend
        self.device = device
        self.model = env.sim.model._model
        self.hosts = [MjSim(self.model) for _ in range(count)]
        self.initial = initial
        self.engine = None
        self.stream = None
        if backend == "mjlab":
            self.engine = Simulation(count, SimulationCfg(mujoco=_PreserveOptions()),
                                     model=self.model, device=device)
            self.stream = torch.cuda.ExternalStream(
                wp.get_stream(self.engine.wp_device).cuda_stream, device=device)
        self.reset()

    def reset(self):
        template = self.env.robots[0]
        self.robots = []
        for sim in self.hosts:
            sim.reset()
            sim.set_state_from_flattened(self.initial)
            sim.forward()
            robot = copy.copy(template)
            robot.sim = sim
            robot.gripper = copy.copy(template.gripper)
            robot.gripper.current_action = np.zeros_like(template.gripper.current_action)
            for name, value in vars(template).items():
                if name.startswith("recent_"):
                    setattr(robot, name, copy.deepcopy(value))
            # Clone the controller's complete reset state, including cached
            # end-effector pose and update flag, before swapping its simulator.
            # Reconstructing it at the injected demo state changes action 0.
            robot.controller = copy.deepcopy(
                template.controller, {id(template.controller.sim): sim})
            self.robots.append(robot)
        if self.engine is not None:
            with torch.cuda.stream(self.stream):
                self.engine.reset()
                for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart",
                             "qfrc_applied", "xfrc_applied", "eq_active", "history",
                             "userdata", "mocap_pos", "mocap_quat"):
                    values = np.stack([getattr(s.data._data, name) for s in self.hosts])
                    if values.size:
                        target = getattr(self.engine.data, name)
                        target.copy_(torch.as_tensor(values, device=self.device))
                self.engine.data.time[:] = float(self.initial[0])
            self.synchronize()

    def synchronize(self):
        if self.engine is not None:
            wp.synchronize_device(self.engine.wp_device)

    def download(self):
        with torch.cuda.stream(self.stream):
            for name in ("qpos", "qvel", "act", "qacc_warmstart"):
                if getattr(self.hosts[0].data._data, name).size:
                    values = getattr(self.engine.data, name).cpu().numpy()
                    for i, sim in enumerate(self.hosts):
                        getattr(sim.data._data, name)[:] = values[i]
            times = self.engine.data.time.cpu().numpy()
            for i, sim in enumerate(self.hosts):
                sim.data.time = float(times[i])

    def step_controlled(self, action, policy_step):
        for robot in self.robots:
            robot.sim.forward()
            robot.control(action, policy_step=policy_step)
        if self.engine is None:
            for sim in self.hosts:
                sim.step()
        else:
            controls = np.stack([s.data.ctrl for s in self.hosts])
            with torch.cuda.stream(self.stream):
                self.engine.data.ctrl.copy_(torch.as_tensor(controls, device=self.device))
                self.engine.step()
                self.download()

    def outcomes(self):
        if self.engine is not None:
            self.download()
        saved_sim = self.env.env.sim
        successes = []
        try:
            for sim in self.hosts:
                sim.forward()
                self.env.env.sim = sim
                successes.append(bool(self.env.check_success()))
        finally:
            self.env.env.sim = saved_sim
        states = np.stack([s.get_state().flatten() for s in self.hosts])
        return {"finite": bool(np.isfinite(states).all()),
                "success_count": sum(successes), "worlds": self.count,
                "success_by_world": successes}, states

    def close(self):
        self.engine = None
        for sim in self.hosts:
            sim.free()


def controlled(batch, actions, substeps, collect_torques=False):
    controls = []
    batch.synchronize()
    start = time.perf_counter()
    for action in actions:
        for substep in range(substeps):
            batch.step_controlled(action, policy_step=substep == 0)
            if collect_torques:
                controls.append(batch.hosts[0].data.ctrl.copy())
    batch.synchronize()
    return time.perf_counter() - start, np.asarray(controls)


def physics_only(batch, torques):
    """Same torque trace for every world, resident on GPU before timing."""
    if batch.engine is not None:
        with torch.cuda.stream(batch.stream):
            gpu_torques = torch.as_tensor(torques, dtype=torch.float32, device=batch.device)
    batch.synchronize()
    start = time.perf_counter()
    if batch.engine is None:
        for ctrl in torques:
            for sim in batch.hosts:
                sim.data.ctrl[:] = ctrl
                sim.step()
    else:
        with torch.cuda.stream(batch.stream):
            for ctrl in gpu_torques:
                batch.engine.data.ctrl[:] = ctrl
                batch.engine.step()
    batch.synchronize()
    return time.perf_counter() - start


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
    parser.add_argument("--backends", nargs="+", choices=["mujoco", "mjlab"],
                        default=["mujoco", "mjlab"])
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
        for count in args.sizes:
            for backend in args.backends:
                batch = Batch(env, count, backend, initial, args.device)
                try:
                    for scope in ("cpu_osc_and_physics", "physics_only_torque_replay"):
                        # Warm the entire workload before reset and timing.
                        batch.reset()
                        if scope == "cpu_osc_and_physics":
                            controlled(batch, actions, substeps)
                        else:
                            physics_only(batch, torques)
                        seconds, outcomes = [], []
                        for trial in range(args.repeats):
                            batch.reset()
                            if scope == "cpu_osc_and_physics":
                                elapsed, _ = controlled(batch, actions, substeps)
                            else:
                                elapsed = physics_only(batch, torques)
                            outcome, states = batch.outcomes()
                            outcome["max_final_qpos_error_vs_native"] = float(np.max(
                                np.abs(states[:, 1:1 + batch.model.nq] - reference_states[:, 1:1 + batch.model.nq])))
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
