"""Pure tensor tests: no MuJoCo models, simulation, or GPU required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

spec = importlib.util.spec_from_file_location("standalone_gpu_osc", Path(__file__).parents[1] / "libero/libero/envs/gpu_osc.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GPUOSC = module.GPUOSC


class IndexedResetTest(unittest.TestCase):
    def controller(self):
        c = GPUOSC.__new__(GPUOSC)
        c.dtype = torch.float64
        c.site = 0
        c.goal_pos = torch.full((4, 3), -7., dtype=c.dtype)
        c.goal_ori = torch.full((4, 3, 3), -8., dtype=c.dtype)
        c.grip = torch.full((4, 2), .5, dtype=c.dtype)
        c.engine = SimpleNamespace(data=SimpleNamespace(
            site_xpos=torch.arange(12).reshape(4, 1, 3).float(),
            site_xmat=torch.eye(3).expand(4, 1, 3, 3)))
        return c

    def test_partial_reset_uses_each_selected_world(self):
        c = self.controller()
        c.reset_indices(torch.tensor([3, 1]))
        torch.testing.assert_close(c.goal_pos[[3, 1]], c.engine.data.site_xpos[[3, 1], 0].double())
        torch.testing.assert_close(c.goal_pos[[0, 2]], torch.full((2, 3), -7., dtype=c.dtype))
        torch.testing.assert_close(c.grip[[3, 1]], torch.zeros((2, 2), dtype=c.dtype))
        torch.testing.assert_close(c.grip[[0, 2]], torch.full((2, 2), .5, dtype=c.dtype))

    def test_prefix_controller_memory_can_be_restored(self):
        c = self.controller()
        p, r, g = torch.ones(1, 3), torch.eye(3)[None], torch.tensor([[-.2, .2]])
        c.reset_indices(torch.tensor([2]), goal_pos=p, goal_ori=r, grip=g)
        torch.testing.assert_close(c.goal_pos[2], p[0].double())
        torch.testing.assert_close(c.goal_ori[2], r[0].double())
        torch.testing.assert_close(c.grip[2], g[0].double())


if __name__ == "__main__":
    unittest.main()
