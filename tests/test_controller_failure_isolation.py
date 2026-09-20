"""Pure tensor controller-failure tests; no MuJoCo, Warp, mjlab, or GPU."""
import contextlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # Keep the repository's pure-Python suite usable without torch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_gpu_osc():
    spec = importlib.util.spec_from_file_location(
        "standalone_gpu_osc", ROOT / "libero/libero/envs/gpu_osc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GPUOSC


def load_batch_env(gpu_osc):
    package = ModuleType("failure_test_envs")
    package.__path__ = []
    gpu_module = ModuleType("failure_test_envs.gpu_osc")
    gpu_module.GPUOSC = gpu_osc
    saved = {name: sys.modules.get(name) for name in (package.__name__, gpu_module.__name__)}
    sys.modules[package.__name__] = package
    sys.modules[gpu_module.__name__] = gpu_module
    try:
        spec = importlib.util.spec_from_file_location(
            "failure_test_envs.mjlab_batch", ROOT / "libero/libero/envs/mjlab_batch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.LiberoBatchEnv
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@unittest.skipIf(torch is None, "torch unavailable")
class ControllerFailureIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.GPUOSC = load_gpu_osc()

    def controller(self, mass):
        worlds, dofs = mass.shape[:2]
        c = self.GPUOSC.__new__(self.GPUOSC)
        c.dtype = torch.float64
        c.device = torch.device("cpu")
        c.eye = torch.eye(dofs, dtype=c.dtype)
        c.task_eye = torch.eye(6, dtype=c.dtype)
        c.uncoupling = False
        c.kp = torch.arange(1, 7, dtype=c.dtype)
        c.kd = torch.arange(7, 13, dtype=c.dtype) / 10
        c.initial_joint = torch.linspace(-.2, .2, dofs, dtype=c.dtype)
        c.low = torch.full((dofs,), -100., dtype=c.dtype)
        c.high = torch.full((dofs,), 100., dtype=c.dtype)
        c.aids = torch.arange(dofs)
        c.gids = torch.tensor([dofs, dofs + 1])
        c.actuator_ids = torch.cat((c.aids, c.gids))
        c.grip = torch.zeros((worlds, 2), dtype=c.dtype)
        c.grip_delta = torch.tensor([-.01, .01], dtype=c.dtype)
        c.grip_bias = torch.zeros(2, dtype=c.dtype)
        c.grip_weight = torch.ones(2, dtype=c.dtype)
        c.invalid_controller = torch.zeros(worlds, dtype=torch.bool)
        c.engine = SimpleNamespace(data=SimpleNamespace(ctrl=torch.full((worlds, dofs + 2), 17.)))
        generator = torch.Generator().manual_seed(4)
        pos = torch.randn(worlds, 3, generator=generator, dtype=c.dtype) / 10
        ori = torch.eye(3, dtype=c.dtype).expand(worlds, -1, -1).clone()
        jac = torch.randn(worlds, 6, dofs, generator=generator, dtype=c.dtype)
        q = torch.randn(worlds, dofs, generator=generator, dtype=c.dtype) / 10
        vel = torch.randn(worlds, dofs, generator=generator, dtype=c.dtype) / 10
        bias = torch.randn(worlds, dofs, generator=generator, dtype=c.dtype) / 10
        c.goal_pos = pos + .03
        c.goal_ori = ori.clone()
        c.state = lambda: (pos, ori, jac, mass, q, vel, bias)
        return c

    def original_controls(self, c, state, action):
        pos, ori, jac, mass, q, vel, bias = state
        error_ori = .5 * torch.linalg.cross(
            ori.transpose(1, 2), c.goal_ori.transpose(1, 2), dim=-1).sum(1)
        desired = torch.cat((c.goal_pos - pos, error_ori), -1) * c.kp
        desired -= (jac @ vel[:, :, None]).squeeze(-1) * c.kd
        minv = torch.linalg.inv(mass)
        jt = jac.transpose(1, 2)
        linv = jac @ minv @ jt
        lam = torch.linalg.pinv(linv, rtol=1e-15)
        wrench = lam @ desired[:, :, None]
        null = c.eye - minv @ jt @ lam @ jac
        pose_torque = mass @ (
            10 * (c.initial_joint - q) - 2 * 10 ** .5 * vel)[:, :, None]
        torques = (jt @ wrench + null.transpose(1, 2) @ pose_torque).squeeze(-1) + bias
        arm = torques.clamp(c.low, c.high).float()
        grip = c.grip_bias + c.grip_weight * (
            c.grip + c.grip_delta * torch.sign(action[:, 6:7])).clamp(-1, 1)
        return torch.cat((arm, grip.float()), dim=-1)

    def test_singular_world_is_zeroed_without_changing_healthy_controls(self):
        raw = torch.randn(3, 7, 7, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
        mass = raw @ raw.transpose(1, 2) + torch.eye(7, dtype=torch.float64)
        mass[1].zero_()
        action = torch.full((3, 7), .2, dtype=torch.float64)

        batched = self.controller(mass.clone())
        batched_state = batched.state()
        healthy = torch.tensor([0, 2])
        healthy_state = tuple(value[healthy] for value in batched_state)
        reference = self.controller(mass[healthy].clone())
        reference.goal_pos = batched.goal_pos[healthy].clone()
        reference.goal_ori = batched.goal_ori[healthy].clone()
        expected_controls = self.original_controls(reference, healthy_state, action[healthy])
        batched.control(action, policy_step=False)

        self.assertEqual(batched.invalid_controller.tolist(), [False, True, False])
        torch.testing.assert_close(batched.engine.data.ctrl[1], torch.zeros(9))
        torch.testing.assert_close(
            batched.engine.data.ctrl[healthy], expected_controls, rtol=1e-12, atol=1e-12)

    def test_failure_mask_accumulates_until_only_that_world_is_reset(self):
        mass = torch.eye(7, dtype=torch.float64).expand(3, -1, -1).clone()
        mass[0, 0, 0] = float("nan")
        c = self.controller(mass)
        c.control(torch.zeros(3, 7, dtype=torch.float64), policy_step=False)
        self.assertEqual(c.invalid_controller.tolist(), [True, False, False])
        c.site = 0
        c.engine.data.site_xpos = torch.zeros(3, 1, 3)
        c.engine.data.site_xmat = torch.eye(3).expand(3, 1, 3, 3)
        c.reset_indices(torch.tensor([2]))
        self.assertEqual(c.invalid_controller.tolist(), [True, False, False])
        c.reset_indices(torch.tensor([0]))
        self.assertEqual(c.invalid_controller.tolist(), [False, False, False])

    def test_pinv_linalg_error_retries_per_world_and_isolates_failure(self):
        mass = torch.eye(7, dtype=torch.float64).expand(3, -1, -1).clone()
        action = torch.zeros(3, 7, dtype=torch.float64)
        c = self.controller(mass)
        state = c.state()
        healthy = torch.tensor([0, 2])
        reference = self.controller(mass[healthy])
        reference.goal_pos = c.goal_pos[healthy].clone()
        reference.goal_ori = c.goal_ori[healthy].clone()
        expected = self.original_controls(
            reference, tuple(value[healthy] for value in state), action[healthy])
        real_pinv = torch.linalg.pinv
        per_world_calls = 0

        def failing_pinv(matrix, *args, **kwargs):
            nonlocal per_world_calls
            if matrix.ndim == 3:
                raise torch.linalg.LinAlgError("batched SVD did not converge")
            per_world_calls += 1
            if per_world_calls == 2:
                raise torch.linalg.LinAlgError("world SVD did not converge")
            return real_pinv(matrix, *args, **kwargs)

        with patch.object(torch.linalg, "pinv", side_effect=failing_pinv):
            c.control(action, policy_step=False)

        self.assertEqual(c.invalid_controller.tolist(), [False, True, False])
        self.assertEqual(per_world_calls, 3)
        torch.testing.assert_close(c.engine.data.ctrl[1], torch.zeros(9))
        torch.testing.assert_close(
            c.engine.data.ctrl[healthy], expected, rtol=1e-12, atol=1e-12)

    def test_latched_world_uses_benign_matrices_before_linalg(self):
        mass = torch.eye(7, dtype=torch.float64).expand(3, -1, -1).clone()
        mass[1].fill_(37)
        c = self.controller(mass)
        c.invalid_controller[1] = True
        real_inv_ex, real_pinv = torch.linalg.inv_ex, torch.linalg.pinv
        seen = {}

        def recording_inv_ex(matrix, *args, **kwargs):
            seen["mass"] = matrix.clone()
            return real_inv_ex(matrix, *args, **kwargs)

        def recording_pinv(matrix, *args, **kwargs):
            seen["task"] = matrix.clone()
            return real_pinv(matrix, *args, **kwargs)

        with patch.object(torch.linalg, "inv_ex", side_effect=recording_inv_ex), \
                patch.object(torch.linalg, "pinv", side_effect=recording_pinv):
            c.control(torch.zeros(3, 7, dtype=torch.float64), policy_step=False)

        torch.testing.assert_close(seen["mass"][1], c.eye)
        torch.testing.assert_close(seen["task"][1], c.task_eye)
        self.assertTrue(c.invalid_controller[1])
        torch.testing.assert_close(c.engine.data.ctrl[1], torch.zeros(9))

    def test_unrelated_pinv_runtime_error_propagates(self):
        c = self.controller(torch.eye(7, dtype=torch.float64)[None])
        with patch.object(torch.linalg, "pinv", side_effect=RuntimeError("infrastructure")), \
                self.assertRaisesRegex(RuntimeError, "infrastructure"):
            c.control(torch.zeros(1, 7, dtype=torch.float64), policy_step=False)


@unittest.skipIf(torch is None, "torch unavailable")
class BatchFailureLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gpu_osc = load_gpu_osc()
        cls.BatchEnv = load_batch_env(gpu_osc)

    def test_controller_failure_accumulates_across_substeps_and_terminates(self):
        env = self.BatchEnv.__new__(self.BatchEnv)
        env.num_envs, env.action_dim, env.substeps = 3, 7, 2
        env.device, env.stepping_mode, env.horizon = torch.device("cpu"), "full", 20
        env._scope = lambda: contextlib.nullcontext()
        env.engine = SimpleNamespace(forward=lambda: None, step=lambda: None)
        env.previous_action = torch.zeros(3, 7)
        env.elapsed = torch.zeros(3, dtype=torch.long)
        env.done = torch.zeros(3, dtype=torch.bool)
        env._observe = lambda: torch.zeros(3, 4)
        env._success = lambda: torch.zeros(3, dtype=torch.bool)
        masks = iter((torch.tensor([False, True, False]), torch.tensor([False, False, True])))
        controller = SimpleNamespace(invalid_controller=torch.zeros(3, dtype=torch.bool))

        def control(actions, policy_step):
            controller.invalid_controller |= next(masks)

        controller.control = control
        env.controller = controller
        _, reward, terminated, truncated, info = env.step(torch.zeros(3, 7))
        self.assertEqual(info["invalid_state"].tolist(), [False, True, True])
        self.assertEqual(terminated.tolist(), [False, True, True])
        self.assertEqual(reward.tolist(), [0., 0., 0.])
        self.assertFalse(truncated.any())


if __name__ == "__main__":
    unittest.main()
