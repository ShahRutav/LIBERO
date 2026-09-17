"""Privileged-state GPU vector environment; no cameras or automatic resets.

Construction and all simulation must run on a GPU pod. The caller prepares a
native ControlEnv with the exact source XML and supplies compatible reset states.
Different models require separate instances, never merely matching dimensions.
"""
from contextlib import contextmanager
import hashlib
import json
import numpy as np
import torch

from .gpu_osc import GPUOSC


def schema_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class LiberoBatchEnv:
    action_dim = 7
    action_spec = {"version": 1, "name": "libero_osc_pose_v1", "shape": [7],
                   "range": [-1, 1], "controller": "fixed_delta_OSC_POSE_PandaGripper"}

    @torch.inference_mode(False)
    def __init__(self, env, initial_states, num_envs, device="cuda:0", horizon=500, seed=0,
                 initial_body_pos=None, initial_body_quat=None):
        import mujoco
        import warp as wp
        from mjlab.sim import Simulation, SimulationCfg
        from .mjlab_sim import _PreserveOptions
        if num_envs < 1 or horizon < 1:
            raise ValueError("num_envs and horizon must be positive")
        self.env, self.model = env, env.sim.model._model
        self.num_envs, self.device, self.horizon = num_envs, torch.device(device), horizon
        self.initial_states = np.asarray(initial_states, dtype=np.float64)
        width = 1 + self.model.nq + self.model.nv + self.model.na
        if self.initial_states.ndim != 2 or self.initial_states.shape[1] != width or not len(self.initial_states):
            raise ValueError(f"Expected nonempty reset states [S,{width}]")
        if not np.isfinite(self.initial_states).all():
            raise ValueError("Nonfinite reset states")
        self.engine = Simulation(num_envs, SimulationCfg(mujoco=_PreserveOptions()), model=self.model, device=str(device))
        if (initial_body_pos is None) != (initial_body_quat is None):
            raise ValueError("Both body pose banks are required")
        self._body_pose_banks = {}
        if initial_body_pos is not None:
            for name, values, dim in (("body_pos", initial_body_pos, 3), ("body_quat", initial_body_quat, 4)):
                array = np.asarray(values)
                if array.shape != (len(self.initial_states), self.model.nbody, dim) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid {name} reset bank")
                self._body_pose_banks[name] = torch.as_tensor(array, device=device, dtype=torch.float32)
            self.engine.expand_model_fields(("body_pos", "body_quat"))
        self.stream = torch.cuda.ExternalStream(wp.get_stream(self.engine.wp_device).cuda_stream, device=device)
        self.rng = torch.Generator(device=device).manual_seed(seed)
        ratio = env.env.control_timestep / env.env.model_timestep
        self.substeps = int(ratio)
        if self.substeps < 1 or abs(ratio-self.substeps) > 1e-8:
            raise ValueError("Control timestep must be an integer multiple of physics timestep")
        self.object_names = sorted(env.env.obj_body_id)
        self.body_ids = [int(env.env.obj_body_id[n]) for n in self.object_names]
        self.model_spec = {
            "nq": self.model.nq, "nv": self.model.nv, "na": self.model.na,
            "joints": [{"name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j),
                        "qpos_adr": int(self.model.jnt_qposadr[j]),
                        "qvel_adr": int(self.model.jnt_dofadr[j]), "type": int(self.model.jnt_type[j])}
                       for j in range(self.model.njnt)],
            "objects": dict(zip(self.object_names, self.body_ids)),
        }
        with self._scope(), torch.inference_mode(False):
            self.controller = GPUOSC(self.engine, env.robots[0])
            self.elapsed = torch.zeros(num_envs, dtype=torch.long, device=device)
            self.previous_action = torch.zeros((num_envs, self.action_dim), device=device)
            self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._reset_bank = torch.as_tensor(self.initial_states, device=device)
        native_controller = env.robots[0].controller
        self.action_spec = dict(type(self).action_spec)
        self.action_spec.update({
            "control_dt": float(env.env.control_timestep),
            "physics_dt": float(env.env.model_timestep), "substeps": self.substeps,
            "uncoupling": bool(native_controller.uncoupling),
            "gripper_speed": float(env.robots[0].gripper.speed),
            "clipping": "normalized_action_clipped_to_minus1_plus1_then_controller_input_bounds",
            "gripper_accumulation": "sign_of_action_every_physics_substep_clamped_minus1_plus1",
            "zero_rotation_action": "retain_previous_orientation_goal",
        })
        for name in ("input_min", "input_max", "output_min", "output_max", "kp", "kd", "initial_joint"):
            self.action_spec[name] = np.asarray(getattr(native_controller, name)).tolist()
        self.action_spec["position_limits"] = (None if native_controller.position_limits is None
                                                 else np.asarray(native_controller.position_limits).tolist())
        self._goals = self._compile_goals()
        fields = [("qpos", [self.model.nq]), ("qvel", [self.model.nv]), ("act", [self.model.na])]
        for name in self.object_names:
            fields.extend((f"object/{name}/{key}", [dim]) for key, dim in
                          (("position", 3), ("rotation6d", 6), ("linear_velocity", 3), ("angular_velocity", 3)))
        fields += [("eef_position", [3]), ("eef_rotation6d", [6]), ("eef_velocity", [6]),
                   ("controller_goal_position", [3]), ("controller_goal_rotation6d", [6]),
                   ("controller_grip", [2]), ("previous_action", [self.action_dim])]
        self.observation_spec = {"version": 1, "name": "privileged_state_v1", "frame": "world",
                                 "units": "SI", "rotation6d": "first_two_matrix_columns_row_major",
                                 "qpos_quaternion_order": "wxyz", "model": self.model_spec,
                                 "fields": [{"name": n, "shape": shape} for n, shape in fields]}
        self.observation_schema_hash = schema_hash(self.observation_spec)
        self.num_obs = sum(int(np.prod(shape)) for _, shape in fields)
        self.reset()

    @contextmanager
    def _scope(self):
        caller = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream), torch.no_grad():
            yield
        caller.wait_stream(self.stream)

    def _compile_goals(self):
        goals = []
        for goal in self.env.env.parsed_problem["goal_state"]:
            if len(goal) != 3 or goal[0].lower() != "on":
                raise NotImplementedError(f"GPU success predicate unsupported: {goal}")
            _, top, bottom = goal
            states = self.env.env.object_states_dict
            if any(states[n].object_state_type != "object" for n in (top, bottom)):
                raise NotImplementedError("Site On predicate needs a dedicated implementation")
            geoms = []
            for name in (top, bottom):
                obj = self.env.env.get_object(name)
                geoms.append(torch.tensor([self.env.sim.model.geom_name2id(g) for g in obj.contact_geoms],
                                          device=self.device, dtype=torch.long))
            goals.append((self.env.env.obj_body_id[top], self.env.env.obj_body_id[bottom], *geoms))
        if not goals:
            raise ValueError("Task has no supported goal")
        return goals

    def _success(self):
        import warp as wp
        contact = self.engine.wp_data.contact
        geom = wp.to_torch(contact.geom).long()
        world = wp.to_torch(contact.worldid).long()
        nacon = wp.to_torch(self.engine.wp_data.nacon).reshape(-1)[0]
        if bool(nacon > len(world)):
            raise RuntimeError("GPU contact buffer overflow; increase nconmax")
        valid = (torch.arange(len(world), device=self.device) < nacon) & (world >= 0) & (world < self.num_envs)
        success = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for top, bottom, top_geoms, bottom_geoms in self._goals:
            matches = ((torch.isin(geom[:, 0], top_geoms) & torch.isin(geom[:, 1], bottom_geoms)) |
                       (torch.isin(geom[:, 1], top_geoms) & torch.isin(geom[:, 0], bottom_geoms))) & valid
            counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            counts.scatter_add_(0, world.clamp(0, self.num_envs-1), matches.long())
            p = self.engine.data.xpos[:, top]
            q = self.engine.data.xpos[:, bottom]
            success &= (counts > 0) & (p[:, 2] >= q[:, 2]) & (torch.linalg.vector_norm(p[:, :2]-q[:, :2], dim=-1) < .03)
        return success

    def _inject(self, ids, states):
        d, m = self.engine.data, self.model
        d.time[ids] = states[:, 0].to(d.time.dtype)
        d.qpos[ids] = states[:, 1:1+m.nq].to(d.qpos.dtype)
        d.qvel[ids] = states[:, 1+m.nq:1+m.nq+m.nv].to(d.qvel.dtype)
        if m.na:
            d.act[ids] = states[:, -m.na:].to(d.act.dtype)
        self.engine.forward()

    def set_reset_bank(self, initial_states, initial_body_pos=None, initial_body_quat=None):
        """Replace reset candidates after caller verifies model compatibility.

        Does not reset live worlds. Only fixed-fixture body poses may differ;
        equal array dimensions do not establish XML/model compatibility.
        """
        states = np.asarray(initial_states, dtype=np.float64)
        width = 1 + self.model.nq + self.model.nv + self.model.na
        if states.ndim != 2 or not len(states) or states.shape[1] != width or not np.isfinite(states).all():
            raise ValueError(f"Expected finite nonempty reset states [S,{width}]")
        if (initial_body_pos is None) != (initial_body_quat is None):
            raise ValueError("Both body pose banks are required")
        poses = {}
        if initial_body_pos is not None:
            for name, values, dim in (("body_pos", initial_body_pos, 3), ("body_quat", initial_body_quat, 4)):
                array = np.asarray(values)
                if array.shape != (len(states), self.model.nbody, dim) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid {name} reset bank")
                poses[name] = array
        elif self._body_pose_banks:
            raise ValueError("Replacing a fixture-aware bank requires body poses")
        with self._scope():
            if poses and not self._body_pose_banks:
                self.engine.expand_model_fields(("body_pos", "body_quat"))
            self.initial_states = states.copy()
            self._reset_bank = torch.as_tensor(states, device=self.device)
            self._body_pose_banks = {name: torch.as_tensor(array, device=self.device, dtype=torch.float32)
                                     for name, array in poses.items()}

    def reset(self, env_ids=None, state_ids=None):
        with self._scope():
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if ids.ndim != 1 or len(ids.unique()) != len(ids):
                raise ValueError("env_ids must be a vector without duplicates")
            if bool(((ids < 0) | (ids >= self.num_envs)).any()):
                raise ValueError("env_ids outside batch")
            if state_ids is None:
                state_ids = torch.randint(len(self.initial_states), (len(ids),), generator=self.rng, device=self.device)
            state_ids = torch.as_tensor(state_ids, device=self.device, dtype=torch.long)
            if state_ids.shape != ids.shape:
                raise ValueError("state_ids must match env_ids")
            if bool(((state_ids < 0) | (state_ids >= len(self.initial_states))).any()):
                raise ValueError("state_ids outside reset bank")
            self.engine.reset(ids)
            for name, bank in self._body_pose_banks.items():
                getattr(self.engine.model, name)[ids] = bank[state_ids]
            self._inject(ids, self._reset_bank[state_ids])
            self.controller.reset_indices(ids)
            self.previous_action[ids] = 0
            self.elapsed[ids] = 0
            self.done[ids] = False
            return self._observe()

    def _observe(self):
        d, c = self.engine.data, self.controller
        parts = [d.qpos[:], d.qvel[:], d.act[:]]
        for body in self.body_ids:
            # cvel is angular/linear about the subtree COM, not the body origin.
            cv = d.cvel[:, body]
            offset = d.xpos[:, body] - d.subtree_com[:, int(self.model.body_rootid[body])]
            linear = cv[:, 3:] + torch.linalg.cross(cv[:, :3], offset, dim=-1)
            parts += [d.xpos[:, body], d.xmat[:, body].reshape(-1, 3, 3)[:, :, :2].reshape(self.num_envs, 6), linear, cv[:, :3]]
        pos, ori, jac, _, _, vel, _ = c.state()
        parts += [pos, ori[:, :, :2].reshape(self.num_envs, 6), (jac @ vel[:, :, None]).squeeze(-1),
                  c.goal_pos, c.goal_ori[:, :, :2].reshape(self.num_envs, 6), c.grip, self.previous_action]
        return torch.cat([p.reshape(self.num_envs, -1).float() for p in parts], dim=-1)

    def observe(self):
        with self._scope():
            return self._observe()

    def step(self, actions):
        with self._scope():
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float64).detach()
            if actions.shape != (self.num_envs, self.action_dim):
                raise ValueError(f"Expected actions {(self.num_envs, self.action_dim)}, got {tuple(actions.shape)}")
            if not torch.isfinite(actions).all():
                raise ValueError("Nonfinite actions")
            actions = actions.clamp(-1, 1)
            for substep in range(self.substeps):
                self.engine.forward()
                self.controller.control(actions, policy_step=substep == 0)
                self.engine.step()
            self.engine.forward()
            self.previous_action.copy_(actions)
            self.elapsed += 1
            obs = self._observe()
            finite = torch.isfinite(obs).all(-1)
            success = self._success() & finite
            terminated = success | ~finite
            truncated = (self.elapsed >= self.horizon) & ~terminated
            self.done.copy_(terminated | truncated)
            return obs, success.float(), terminated, truncated, {"success": success, "invalid_state": ~finite}

    def restore_recorded_state(self, raw_state, previous_action=None):
        """Teacher-forced reference state with action-prefix controller memory."""
        if self.num_envs != 1:
            raise ValueError("Recorded-state conversion uses one world")
        with self._scope():
            if previous_action is not None:
                action = torch.as_tensor(previous_action, device=self.device, dtype=torch.float64).detach().reshape(1, -1)
                if action.shape != (1, self.action_dim) or not torch.isfinite(action).all():
                    raise ValueError("Invalid previous action")
                action = action.clamp(-1, 1)
                self.controller.advance_memory(action, self.substeps)
                self.previous_action.copy_(action)
            raw = torch.as_tensor(raw_state, dtype=torch.float64, device=self.device).reshape(1, -1)
            if not torch.isfinite(raw).all():
                raise ValueError("Nonfinite recorded state")
            if raw.shape[1] != self._reset_bank.shape[1]:
                raise ValueError("Recorded-state dimension mismatch")
            self._inject(torch.zeros(1, device=self.device, dtype=torch.long), raw)
            return self._observe()

    def export_state(self):
        with self._scope():
            d, c = self.engine.data, self.controller
            angular = d.cvel[:, self.body_ids, :3]
            offset = d.xpos[:, self.body_ids] - d.subtree_com[:, self.model.body_rootid[self.body_ids]]
            linear = d.cvel[:, self.body_ids, 3:] + torch.linalg.cross(angular, offset, dim=-1)
            import warp as wp
            body_pos = wp.to_torch(self.engine.wp_model.body_pos)
            body_quat = wp.to_torch(self.engine.wp_model.body_quat)
            if body_pos.shape[0] == 1:
                body_pos = body_pos.expand(self.num_envs, -1, -1)
                body_quat = body_quat.expand(self.num_envs, -1, -1)
            return {"model_body_pos": body_pos.clone(), "model_body_quat": body_quat.clone(), "qpos": d.qpos[:].clone(), "qvel": d.qvel[:].clone(), "act": d.act[:].clone(),
                    "object_position": d.xpos[:, self.body_ids].clone(),
                    "object_linear_velocity": linear.clone(), "object_angular_velocity": angular.clone(),
                    "object_quaternion_wxyz": d.xquat[:, self.body_ids].clone(),
                    "controller_goal_position": c.goal_pos.clone(), "controller_goal_orientation": c.goal_ori.clone(),
                    "controller_grip": c.grip.clone(), "previous_action": self.previous_action.clone()}

    def close(self):
        self.controller = None
        self.engine = None
