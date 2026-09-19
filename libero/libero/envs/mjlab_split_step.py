"""Opt-in MuJoCo-Warp split stepping for the fixed-dt LIBERO GPU controller.

step1 refreshes positions, mass, velocities, bias forces and position/velocity
sensors before OSC. step2 solves constraints, updates acceleration sensors and
integrates. Acceleration sensors are not current between these two calls.
The caller must retain its final forward before observations and success tests.

step2 integrates Euler, implicitfast and implicit exactly as step does, so all
three are accepted. RK4 is rejected: step2 would silently run Euler instead.
Only Euler has measured paired and replay evidence; see split_stepping_spec.
"""
import gc
from contextlib import contextmanager

import mujoco
import mujoco_warp as mjwarp
import warp as wp
from mjlab.sim import Simulation


@contextmanager
def _suspend_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


_ONLY_MEASURED_INTEGRATOR = int(mujoco.mjtIntegrator.mjINT_EULER)

# Integrators whose step2 integration stage is identical to step's. RK4 is
# excluded deliberately: Warp's step2 would run Euler instead.
SPLIT_SUPPORTED_INTEGRATORS = frozenset({
    int(mujoco.mjtIntegrator.mjINT_EULER),
    int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
    int(mujoco.mjtIntegrator.mjINT_IMPLICIT),
})


def split_stepping_spec(mj_model, substeps):
    """Provenance for a runtime report. Never a qualification marker."""
    integrator = int(mj_model.opt.integrator)
    return {
        "version": 1,
        "integrator": integrator,
        "supported": integrator in SPLIT_SUPPORTED_INTEGRATORS,
        # Only Euler has measured paired/replay evidence on this pod so far.
        "measured_integrator": integrator == _ONLY_MEASURED_INTEGRATOR,
        "physics_dt": float(mj_model.opt.timestep),
        # forward+step per substep, then the caller's final forward.
        "constraint_solves_per_control_step": {"full": 2 * substeps + 1, "split": substeps + 1},
        "engine_calls_per_control_step": {
            "full": {"forward": substeps + 1, "step": substeps, "control": substeps},
            "split": {"forward": 1, "step1": substeps, "step2": substeps, "control": substeps},
        },
    }


class SplitStepSimulation(Simulation):
    """Simulation with split graphs that follow mjlab's recapture lifecycle.

    Uses the same Warp device/stream as Simulation.forward/step. LIBERO's
    _scope supplies the Torch stream dependency on either side of OSC.
    Model options and callbacks are static after capture, as in base mjlab.
    """

    def _validate_split(self):
        opt = self.wp_model.opt
        integrator = int(opt.integrator)
        # Warp step2 integrates Euler, implicitfast and implicit exactly as
        # step does. It silently falls back to Euler for RK4, so RK4 is always
        # rejected rather than downgraded.
        if integrator == int(mujoco.mjtIntegrator.mjINT_RK4):
            raise NotImplementedError(
                "LIBERO split stepping cannot express RK4; Warp step2 would silently downgrade it to Euler")
        if integrator not in SPLIT_SUPPORTED_INTEGRATORS:
            raise NotImplementedError(f"LIBERO split stepping does not support integrator {integrator}")
        # forward() wakes and updates sleeping worlds; step1/step2 do not.
        if int(opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
            raise NotImplementedError("LIBERO split stepping does not support sleeping")
        for name in ("step1", "step2"):
            if not callable(getattr(mjwarp, name, None)):
                raise NotImplementedError(f"Installed mujoco_warp has no {name}")
        active = [name for name, value in vars(self.wp_model.callback).items() if value is not None]
        if active:
            raise NotImplementedError(f"LIBERO split stepping does not support callbacks: {active}")
        if abs(float(self.mj_model.opt.timestep) - .002) > 1e-12:
            raise ValueError("LIBERO split stepping requires physics dt=.002")

    def create_graph(self):
        self._validate_split()
        self.step1_graph = self.step2_graph = None
        super().create_graph()
        if self.use_cuda_graph:
            with _suspend_gc(), wp.ScopedDevice(self.wp_device):
                with wp.ScopedCapture() as capture:
                    mjwarp.step1(self.wp_model, self.wp_data)
                self.step1_graph = capture.graph
                with wp.ScopedCapture() as capture:
                    mjwarp.step2(self.wp_model, self.wp_data)
                self.step2_graph = capture.graph

    def step1(self):
        with wp.ScopedDevice(self.wp_device):
            if self.use_cuda_graph and self.step1_graph is not None:
                wp.capture_launch(self.step1_graph)
            else:
                mjwarp.step1(self.wp_model, self.wp_data)

    def step2(self):
        with wp.ScopedDevice(self.wp_device), self.nan_guard.watch(self.data):
            if self.use_cuda_graph and self.step2_graph is not None:
                wp.capture_launch(self.step2_graph)
            else:
                mjwarp.step2(self.wp_model, self.wp_data)
