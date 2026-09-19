"""Pure-Python split-stepping guard tests: no GPU, MuJoCo, Warp, or mjlab.

Run anywhere: python -m unittest tests.test_split_stepping -v

The module under test only reads integrator/enableflag identities and the
mujoco_warp entry points, so stubs are enough here. The real enum member names
(mjINT_EULER, mjINT_RK4, mjINT_IMPLICIT, mjINT_IMPLICITFAST, mjENBL_SLEEP) are
exercised on the pod, where importing this module against real MuJoCo fails
loudly if a name is wrong. Only the distinctness of the values matters below.
"""

import enum
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "libero" / "libero" / "envs" / "mjlab_split_step.py"


class Integrator(enum.IntEnum):
    mjINT_EULER = 0
    mjINT_RK4 = 1
    mjINT_IMPLICIT = 2
    mjINT_IMPLICITFAST = 3


class EnableBit(enum.IntEnum):
    mjENBL_SLEEP = 1 << 4


class BaseSimulation:
    """Stand-in for mjlab.sim.Simulation's capture lifecycle."""

    def __init__(self):
        self.base_create_graph_calls = 0
        self.use_cuda_graph = False

    def create_graph(self):
        self.base_create_graph_calls += 1


def load_module():
    """Import the candidate module against stub dependencies, unpolluted."""
    mujoco = types.ModuleType("mujoco")
    mujoco.mjtIntegrator, mujoco.mjtEnableBit = Integrator, EnableBit
    mjwarp = types.ModuleType("mujoco_warp")
    mjwarp.step1, mjwarp.step2 = (lambda m, d: None), (lambda m, d: None)
    warp = types.ModuleType("warp")
    mjlab_sim = types.ModuleType("mjlab.sim")
    mjlab_sim.Simulation = BaseSimulation
    mjlab = types.ModuleType("mjlab")
    mjlab.sim = mjlab_sim
    stubs = {"mujoco": mujoco, "mujoco_warp": mjwarp, "warp": warp,
             "mjlab": mjlab, "mjlab.sim": mjlab_sim}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("candidate_split_step", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module, mjwarp


class GuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.mjwarp = load_module()

    def make(self, integrator=Integrator.mjINT_EULER, enableflags=0, timestep=.002,
             callbacks=None, use_cuda_graph=False):
        sim = self.module.SplitStepSimulation.__new__(self.module.SplitStepSimulation)
        sim.use_cuda_graph = use_cuda_graph
        sim.base_create_graph_calls = 0
        callback = types.SimpleNamespace(passive=None, control=None, act_dyn=None,
                                         act_gain=None, act_bias=None, sensor=None,
                                         contactfilter=None)
        for name, value in (callbacks or {}).items():
            setattr(callback, name, value)
        sim.wp_model = types.SimpleNamespace(
            opt=types.SimpleNamespace(integrator=int(integrator), enableflags=int(enableflags)),
            callback=callback)
        sim.mj_model = types.SimpleNamespace(opt=types.SimpleNamespace(
            timestep=timestep, integrator=int(integrator)))
        return sim

    def test_euler_accepted(self):
        sim = self.make()
        sim.create_graph()
        self.assertEqual(sim.base_create_graph_calls, 1)
        self.assertIsNone(sim.step1_graph)
        self.assertIsNone(sim.step2_graph)

    def test_implicit_integrators_accepted_for_future_combination(self):
        # Warp's step2 integrates these exactly as step does.
        for integrator in (Integrator.mjINT_IMPLICITFAST, Integrator.mjINT_IMPLICIT):
            with self.subTest(integrator=integrator):
                self.make(integrator=integrator).create_graph()

    def test_rk4_rejected_and_never_downgraded(self):
        sim = self.make(integrator=Integrator.mjINT_RK4)
        with self.assertRaisesRegex(NotImplementedError, "RK4"):
            sim.create_graph()
        self.assertEqual(sim.base_create_graph_calls, 0)

    def test_unknown_integrator_rejected(self):
        sim = self.make(integrator=97)
        with self.assertRaisesRegex(NotImplementedError, "does not support integrator 97"):
            sim.create_graph()

    def test_sleeping_rejected(self):
        # forward() wakes and updates sleeping worlds; step1/step2 do not.
        with self.assertRaisesRegex(NotImplementedError, "sleeping"):
            self.make(enableflags=EnableBit.mjENBL_SLEEP).create_graph()

    def test_callbacks_rejected_and_named(self):
        with self.assertRaisesRegex(NotImplementedError, "passive"):
            self.make(callbacks={"passive": lambda m, d: None}).create_graph()

    def test_non_fixed_timestep_rejected(self):
        with self.assertRaisesRegex(ValueError, "dt=.002"):
            self.make(timestep=.001).create_graph()

    def test_missing_warp_entry_points_rejected(self):
        original = self.mjwarp.step2
        del self.mjwarp.step2
        try:
            with self.assertRaisesRegex(NotImplementedError, "no step2"):
                self.make().create_graph()
        finally:
            self.mjwarp.step2 = original

    def test_graphs_recaptured_and_cleared_before_capture(self):
        sim = self.make(use_cuda_graph=True)
        captured = []

        class Capture:
            def __init__(self, tag):
                self.graph = tag

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def scoped_capture():
            capture = Capture(f"graph-{len(captured)}")
            captured.append(capture)
            return capture

        self.module.wp.ScopedCapture = scoped_capture
        self.module.wp.ScopedDevice = lambda device: _null()
        sim.wp_device, sim.wp_data = "cuda:0", object()
        try:
            sim.create_graph()
            self.assertEqual((sim.step1_graph, sim.step2_graph), ("graph-0", "graph-1"))
            first = (sim.step1_graph, sim.step2_graph)
            # Recapture after model expansion must replace both graphs.
            sim.create_graph()
            self.assertEqual((sim.step1_graph, sim.step2_graph), ("graph-2", "graph-3"))
            self.assertNotEqual((sim.step1_graph, sim.step2_graph), first)
        finally:
            del self.module.wp.ScopedCapture, self.module.wp.ScopedDevice


class SpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, _ = load_module()

    def spec(self, integrator=Integrator.mjINT_EULER, substeps=25):
        model = types.SimpleNamespace(opt=types.SimpleNamespace(
            integrator=int(integrator), timestep=.002))
        return self.module.split_stepping_spec(model, substeps)

    def test_solve_and_call_counts_are_derived(self):
        spec = self.spec()
        self.assertEqual(spec["constraint_solves_per_control_step"], {"full": 51, "split": 26})
        self.assertEqual(spec["engine_calls_per_control_step"], {
            "full": {"forward": 26, "step": 25, "control": 25},
            "split": {"forward": 1, "step1": 25, "step2": 25, "control": 25}})

    def test_counts_track_substeps(self):
        spec = self.spec(substeps=4)
        self.assertEqual(spec["constraint_solves_per_control_step"], {"full": 9, "split": 5})
        self.assertEqual(spec["engine_calls_per_control_step"]["split"]["step2"], 4)

    def test_measured_flag_only_true_for_euler(self):
        self.assertTrue(self.spec()["measured_integrator"])
        implicitfast = self.spec(integrator=Integrator.mjINT_IMPLICITFAST)
        self.assertTrue(implicitfast["supported"])
        self.assertFalse(implicitfast["measured_integrator"])

    def test_rk4_reported_unsupported(self):
        self.assertFalse(self.spec(integrator=Integrator.mjINT_RK4)["supported"])


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DispatchTest(unittest.TestCase):
    """LiberoBatchEnv's substep loop contract, without constructing the env."""

    def loop(self, stepping_mode, substeps=25):
        engine, controller, calls = Mock(), Mock(), []
        for name in ("forward", "step", "step1", "step2"):
            getattr(engine, name).side_effect = lambda _n=name: calls.append(_n)
        controller.control.side_effect = lambda *a, **kw: calls.append("control")
        for substep in range(substeps):
            if stepping_mode == "split":
                engine.step1()
            else:
                engine.forward()
            controller.control(None, policy_step=substep == 0)
            if stepping_mode == "split":
                engine.step2()
            else:
                engine.step()
        engine.forward()
        return calls

    def test_full_mode_call_counts(self):
        calls = self.loop("full")
        self.assertEqual(calls.count("forward"), 26)
        self.assertEqual(calls.count("step"), 25)
        self.assertEqual(calls.count("control"), 25)
        self.assertEqual(calls.count("step1") + calls.count("step2"), 0)

    def test_split_mode_call_counts(self):
        calls = self.loop("split")
        self.assertEqual(calls.count("forward"), 1)
        self.assertEqual(calls.count("step1"), 25)
        self.assertEqual(calls.count("step2"), 25)
        self.assertEqual(calls.count("control"), 25)
        self.assertEqual(calls.count("step"), 0)

    def test_controller_reads_between_the_two_halves(self):
        # Every control call must sit after a step1 and before its step2.
        calls = self.loop("split")
        self.assertEqual(calls[:4], ["step1", "control", "step2", "step1"])
        self.assertEqual(calls[-1], "forward")


if __name__ == "__main__":
    unittest.main()
