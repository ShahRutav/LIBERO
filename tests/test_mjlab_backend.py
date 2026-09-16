"""Run on the GPU pod: python -m unittest discover -s tests -v."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.utils.errors import RandomizationError


TASK = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"


class ResetErrorsTest(unittest.TestCase):
    def make_wrapper(self, side_effect):
        wrapper = ControlEnv.__new__(ControlEnv)
        wrapper.env = Mock()
        wrapper.env.reset.side_effect = side_effect
        return wrapper

    def test_real_failure_propagates(self):
        wrapper = self.make_wrapper(ValueError("invalid model"))
        with self.assertRaisesRegex(ValueError, "invalid model"):
            wrapper.reset()
        self.assertEqual(wrapper.env.reset.call_count, 1)

    def test_placement_retry_returns_observation(self):
        wrapper = self.make_wrapper([RandomizationError("retry"), {"ok": 1}])
        self.assertEqual(wrapper.reset(), {"ok": 1})
        self.assertEqual(wrapper.env.reset.call_count, 2)

    def test_placement_retry_is_bounded(self):
        wrapper = self.make_wrapper(RandomizationError("no placement"))
        with self.assertRaisesRegex(RuntimeError, "100 reset attempts"):
            wrapper.reset()
        self.assertEqual(wrapper.env.reset.call_count, 100)


class MjlabIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ControlEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / "libero_spatial" / f"{TASK}.bddl"),
            backend="mjlab", use_camera_obs=False, has_offscreen_renderer=False,
            hard_reset=False,
        )
        cls.initial_states = torch.load(
            Path(get_libero_path("init_states")) / "libero_spatial" / f"{TASK}.pruned_init",
            weights_only=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_benchmark_initial_states_and_gpu_step(self):
        for state in self.initial_states[:3]:
            self.env.reset()
            obs = self.env.set_init_state(state)
            np.testing.assert_allclose(self.env.get_sim_state(), state, atol=0, rtol=0)
            self.assertIn("robot0_eef_pos", obs)
            before = self.env.sim.physics_steps
            with patch("mujoco.mj_step", side_effect=AssertionError("CPU integration is forbidden")):
                self.env.step(np.zeros(7))
            self.assertEqual(self.env.sim.physics_steps - before, 25)
            np.testing.assert_allclose(
                self.env.sim.engine.data.qpos[0].cpu().numpy(), self.env.sim.data.qpos,
                atol=0, rtol=0,
            )
            self.assertTrue(np.isfinite(self.env.get_sim_state()).all())

    def test_fixture_pose_reaches_gpu(self):
        self.env.reset()
        self.env.step(np.zeros(7))
        old_engine = self.env.sim.engine
        fixture = self.env.env.obj_body_id["wooden_cabinet_1"]
        self.env.sim.model.body_pos[fixture, 0] += 0.01
        self.env.sim.forward()
        self.env.step(np.zeros(7))
        self.assertIsNot(self.env.sim.engine, old_engine)
        np.testing.assert_allclose(
            self.env.sim.engine.model.body_pos.cpu().numpy().reshape(-1, 3)[fixture],
            self.env.sim.model.body_pos[fixture], atol=1e-7,
        )

    def test_soft_reset_clears_gripper_command(self):
        self.env.reset()
        action = np.zeros(7)
        action[-1] = 1
        self.env.step(action)
        self.assertTrue(np.any(self.env.robots[0].gripper.current_action != 0))
        self.env.reset()
        np.testing.assert_array_equal(self.env.robots[0].gripper.current_action, 0)

    def test_transfer_and_step_share_warp_stream(self):
        import warp as wp
        self.env.reset()
        self.env.step(np.zeros(7))
        engine = self.env.sim.engine
        original_step = engine.step

        def checked_step():
            self.assertEqual(
                torch.cuda.current_stream(self.env.sim.device).cuda_stream,
                wp.get_stream(engine.wp_device).cuda_stream,
            )
            original_step()

        caller_stream = torch.cuda.Stream(device=self.env.sim.device)
        with torch.cuda.stream(caller_stream), patch.object(engine, "step", side_effect=checked_step) as step:
            self.env.step(np.zeros(7))
        self.assertEqual(step.call_count, 25)

    def test_hard_reset_keeps_backend_and_controller_references(self):
        from libero.libero.envs.mjlab_sim import MjlabSim
        self.env.env.hard_reset = True
        try:
            self.env.reset()
            self.assertIsInstance(self.env.sim, MjlabSim)
            self.assertIs(self.env.robots[0].controller.sim, self.env.sim)
            self.env.step(np.zeros(7))
            self.assertEqual(self.env.sim.physics_steps, 25)
        finally:
            self.env.env.hard_reset = False


if __name__ == "__main__":
    unittest.main()
