"""Single-world mjlab physics with robosuite's existing CPU-facing interfaces.

The CPU MjData is a mirror for controllers, predicates, and rendering. Only
mjlab advances simulation time. This bridge prioritizes compatibility over
throughput; it is not a vectorized manager-based training environment.
"""

import mujoco
import numpy as np
from robosuite.utils.binding_utils import MjSim


class _PreserveOptions:
    """Do not silently replace the task's solver, timestep, or integrator."""

    def apply(self, model):
        pass


class MjlabSim(MjSim):
    def __init__(self, model, device="cuda:0"):
        super().__init__(model)
        self.device = device
        self.engine = None
        self.physics_steps = 0
        self._model_pose = None

    @classmethod
    def from_xml_string(cls, xml, device="cuda:0"):
        return cls(mujoco.MjModel.from_xml_string(xml), device=device)

    def _ensure_engine(self):
        from mjlab.sim import Simulation, SimulationCfg

        model = self.model._model
        # LIBERO's reset sampler changes fixed fixture poses in MjModel.
        pose = (model.body_pos.copy(), model.body_quat.copy())
        changed = self._model_pose is not None and any(
            not np.array_equal(old, new)
            for old, new in zip(self._model_pose, pose)
        )
        if self.engine is None or changed:
            self.engine = Simulation(
                num_envs=1,
                cfg=SimulationCfg(mujoco=_PreserveOptions()),
                model=model,
                device=self.device,
            )
            self._model_pose = pose

    def step(self, with_udd=True):
        import torch

        self._ensure_engine()
        host = self.data._data
        # Host arrays are writable through robosuite. Upload every integration
        # input, including external forces and warm starts, before stepping.
        for name in (
            "qpos", "qvel", "act", "ctrl", "qacc_warmstart",
            "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
            "eq_active",
        ):
            value = getattr(host, name)
            if value.size:
                target = getattr(self.engine.data, name)
                target[0].copy_(torch.as_tensor(value, device=self.device))
        self.engine.data.time[0] = host.time
        self.engine.step()
        for name in ("qpos", "qvel", "act", "qacc_warmstart"):
            value = getattr(host, name)
            if value.size:
                value[:] = getattr(self.engine.data, name)[0].cpu().numpy()
        host.time = float(self.engine.data.time[0].item())
        # Refresh CPU Jacobians, mass matrix, contacts, and render geometry.
        # This computes derived quantities but does not integrate time.
        mujoco.mj_forward(self.model._model, host)
        self.physics_steps += 1

    def reset(self):
        super().reset()
        # All integration inputs are uploaded on the next step. The GPU model
        # is rebuilt only if the subsequent fixture sampling changes its poses.

    def free(self):
        self.engine = None
        super().free()
