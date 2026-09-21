import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def load_helper():
    path = ROOT / "libero/libero/envs/mjlab_seed.py"
    spec = importlib.util.spec_from_file_location("reset_seed_test_mjlab_seed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_simulation_from_reset_seed


def test_simulation_uses_reset_seed_but_restores_reference_qpos():
    helper = load_helper()
    model = SimpleNamespace(nq=3, qpos0=np.array([10.0, 20.0, 30.0]))
    observed = {}

    class WarpArray:
        def __init__(self, value):
            self.value = np.asarray(value).copy()

        def assign(self, value):
            self.value[:] = value

    class Simulation:
        def __init__(self, num_envs, cfg, *, model, device):
            observed.update(num_envs=num_envs, cfg=cfg, device=device,
                            constructor_qpos0=model.qpos0.copy())
            self.wp_model = SimpleNamespace(qpos0=WarpArray(model.qpos0))

    engine = helper(Simulation, "cfg", model, np.array([0.0, 1.0, 2.0, 3.0]),
                    1024, "cuda:7")

    np.testing.assert_array_equal(observed["constructor_qpos0"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(model.qpos0, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(engine.wp_model.qpos0.value, [10.0, 20.0, 30.0])
    assert observed["num_envs"] == 1024
    assert observed["device"] == "cuda:7"


def test_constructor_failure_still_restores_reference_qpos():
    helper = load_helper()
    model = SimpleNamespace(nq=2, qpos0=np.array([4.0, 5.0]))

    class FailingSimulation:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("compile failed")

    try:
        helper(FailingSimulation, None, model, np.array([0.0, 7.0, 8.0]), 1, "cpu")
    except RuntimeError as error:
        assert str(error) == "compile failed"
    else:
        raise AssertionError("Expected construction failure")
    np.testing.assert_array_equal(model.qpos0, [4.0, 5.0])
