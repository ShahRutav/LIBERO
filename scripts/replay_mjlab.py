"""Compare real expert action replay on native MuJoCo and mjlab. Run on a GPU.

python -m scripts.replay_mjlab --dataset /path/to/task_demo.hdf5 --output /path/to/report
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def localize_xml(xml):
    import robosuite
    from libero.libero import get_libero_path

    tree = ET.fromstring(xml)
    for element in tree.iter():
        old = element.get("file")
        if not old:
            continue
        parts = Path(old).parts
        if "robosuite" in parts:
            index = max(i for i, part in enumerate(parts) if part == "robosuite")
            new = Path(robosuite.__file__).parent.joinpath(*parts[index + 1:])
        elif "assets" in parts:
            index = max(i for i, part in enumerate(parts) if part == "assets")
            new = Path(get_libero_path("assets")).joinpath(*parts[index + 1:])
        else:
            raise ValueError(f"Cannot relocate demo asset: {old}")
        if not new.is_file():
            raise FileNotFoundError(new)
        element.set("file", str(new.resolve()))
    return ET.tostring(tree, encoding="unicode")


def replay(backend, bddl, xml, states, actions, kwargs, cameras, device):
    from libero.libero.envs.env_wrapper import ControlEnv

    options = dict(kwargs)
    options.pop("bddl_file_name", None)
    options.update(
        bddl_file_name=str(bddl), backend=backend, mjlab_device=device,
        has_renderer=False, has_offscreen_renderer=cameras, use_camera_obs=cameras,
        hard_reset=False, ignore_done=True,
    )
    env = ControlEnv(**options)
    try:
        env.reset()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.set_init_state(states[0])
        reset_error = float(np.max(np.abs(env.get_sim_state() - states[0])))
        initial_objects = {
            name: env.sim.data.body_xpos[index].copy()
            for name, index in env.env.obj_body_id.items()
        }
        # Test reset after physics has run, including controller and warm-start
        # reset. Merely reading back a CPU state does not verify a GPU reset.
        def reset_demo():
            # Preserve the recorded XML's fixture poses rather than resampling.
            previous = env.env.deterministic_reset
            env.env.deterministic_reset = True
            try:
                env.reset()
            finally:
                env.env.deterministic_reset = previous
            env.sim.reset()
            env.set_init_state(states[0])

        reset_demo()
        env.step(actions[0])
        first = env.get_sim_state().copy()
        reset_demo()
        env.step(actions[0])
        repeat_error = float(np.max(np.abs(first - env.get_sim_state())))
        reset_demo()
        actual, eef, objects, successes, frames = [], [], [], [], []
        camera_shapes = {}
        start = time.perf_counter()
        for i, action in enumerate(actions):
            # Recorded states are pre-action states, not post-action states.
            actual.append(env.get_sim_state().copy())
            eef.append(env.sim.data.get_site_xpos(env.robots[0].controller.eef_name).copy())
            objects.append([env.sim.data.body_xpos[env.env.obj_body_id[n]].copy()
                            for n in initial_objects])
            obs, reward, done, info = env.step(action)
            successes.append(bool(env.check_success()))
            if cameras:
                for camera in env.env.camera_names:
                    pixels = obs[f"{camera}_image"]
                    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
                        raise ValueError(f"Invalid RGB observation for {camera}: {pixels.shape}")
                    if not np.any(pixels):
                        raise ValueError(f"Black RGB observation for {camera}")
                    camera_shapes[camera] = list(pixels.shape)
                frames.append(obs["agentview_image"].copy())
            if i % 25 == 0:
                print(f"{backend}: {i + 1}/{len(actions)} success={successes[-1]}", flush=True)
        elapsed = time.perf_counter() - start
        actual = np.asarray(actual)
        report = {
            "backend": backend,
            "success_any": any(successes), "success_final": successes[-1],
            "first_success_step": next((i for i, x in enumerate(successes) if x), None),
            "reset_state_max_abs": reset_error,
            "reset_repeat_step_max_abs": repeat_error,
            "recorded_state_rmse": float(np.sqrt(np.mean((actual[:, 1:] - states[:, 1:]) ** 2))),
            "recorded_state_max_abs": float(np.max(np.abs(actual[:, 1:] - states[:, 1:]))),
            "finite": bool(np.isfinite(actual).all()),
            "steps": len(actions), "seconds": elapsed,
            "mjlab_physics_steps": getattr(env.sim, "physics_steps", 0),
            "object_names": list(initial_objects),
            "model_nq": env.sim.model.nq, "model_nv": env.sim.model.nv,
            "timestep": float(env.sim.model.opt.timestep),
            "solver": int(env.sim.model.opt.solver),
            "integrator": int(env.sim.model.opt.integrator),
            "cone": int(env.sim.model.opt.cone),
            "disableflags": int(env.sim.model.opt.disableflags),
            "camera_shapes": camera_shapes,
        }
        return report, dict(states=actual, eef=np.asarray(eef), objects=np.asarray(objects),
                            successes=np.asarray(successes), frames=np.asarray(frames))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    from libero.libero import get_libero_path

    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        demo = data[args.demo]
        states = demo["states"][:]
        actions = demo["actions"][:]
        xml = localize_xml(demo.attrs["model_file"])
        kwargs = json.loads(data.attrs["env_args"])["env_kwargs"]
        task_path = Path(data.attrs["bddl_file_name"])
        bddl = Path(get_libero_path("bddl_files")) / task_path.parent.name / task_path.name
        recorded_success = bool(np.any(demo["dones"][:]))
    if len(states) != len(actions) or actions.ndim != 2 or not np.isfinite(actions).all():
        raise ValueError("Expected finite actions and one pre-action state per action")
    if args.max_steps is not None:
        states, actions = states[:args.max_steps], actions[:args.max_steps]
    digest = hashlib.sha256()
    with args.dataset.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = {
        "dataset": str(args.dataset), "demo": args.demo, "bddl": str(bddl),
        "dataset_sha256": digest.hexdigest(),
        "source_revision": os.environ.get("LIBERO_SOURCE_REVISION", "not recorded"),
        "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "recorded_success": recorded_success, "truncated": args.max_steps is not None,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("mujoco", "mujoco-warp", "mjlab", "robosuite", "torch")},
    }
    arrays = {}
    for backend in ("mujoco", "mjlab"):
        report[backend], arrays[backend] = replay(
            backend, bddl, xml, states, actions, kwargs, args.cameras, args.device)
        np.savez_compressed(args.output / f"{backend}.npz", **arrays[backend])
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    delta = arrays["mjlab"]["states"][:, 1:] - arrays["mujoco"]["states"][:, 1:]
    report["comparison"] = {
        "state_rmse": float(np.sqrt(np.mean(delta ** 2))),
        "state_max_abs": float(np.max(np.abs(delta))),
        "eef_max_distance_m": float(np.max(np.linalg.norm(
            arrays["mjlab"]["eef"] - arrays["mujoco"]["eef"], axis=-1))),
        "object_max_distance_m": float(np.max(np.linalg.norm(
            arrays["mjlab"]["objects"] - arrays["mujoco"]["objects"], axis=-1))),
    }
    report["acceptance"] = {
        "require_final_success_both_backends": True,
        "reset_state_max_abs": 1e-10,
        "reset_repeat_step_max_abs": 1e-5,
        "native_vs_mjlab_eef_max_distance_m": 0.005,
        "native_vs_mjlab_object_max_distance_m": 0.010,
    }
    report["passed"] = all(
        report[b]["finite"] and report[b]["success_final"]
        and report[b]["reset_state_max_abs"] < 1e-10
        and report[b]["reset_repeat_step_max_abs"] < 1e-5
        for b in ("mujoco", "mjlab")
    ) and report["mjlab"]["mjlab_physics_steps"] > 0 and not report["truncated"]
    report["passed"] = (report["passed"]
                        and report["comparison"]["eef_max_distance_m"] < 0.005
                        and report["comparison"]["object_max_distance_m"] < 0.010)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
