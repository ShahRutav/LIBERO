"""Pure-Python tests; no simulator or GPU initialization."""

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


path = Path(__file__).resolve().parents[1] / "libero/libero/envs/model_compat.py"
spec = importlib.util.spec_from_file_location("model_compat", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LegacyDefaultsTest(unittest.TestCase):
    def test_absent_option_gets_original_collision_behavior(self):
        root = ET.fromstring(module.preserve_libero_defaults("<mujoco/>"))
        self.assertEqual(root.find("option/flag").get("multiccd"), "disable")

    def test_preserves_explicit_override_and_physics(self):
        xml = '<mujoco><option timestep="0.001" cone="elliptic"><flag multiccd="enable" gravity="disable"/></option></mujoco>'
        root = ET.fromstring(module.preserve_libero_defaults(xml))
        self.assertEqual(root.find("option").attrib, {"timestep": "0.001", "cone": "elliptic"})
        self.assertEqual(root.find("option/flag").attrib, {"multiccd": "enable", "gravity": "disable"})

    def test_preserves_other_flags_and_is_idempotent(self):
        xml = '<mujoco><option><flag gravity="disable"/></option></mujoco>'
        first = module.preserve_libero_defaults(xml)
        self.assertEqual(first, module.preserve_libero_defaults(first))
        root = ET.fromstring(first)
        self.assertEqual(root.find("option/flag").get("gravity"), "disable")
        self.assertEqual(len(root.findall("option/flag")), 1)


if __name__ == "__main__":
    unittest.main()
