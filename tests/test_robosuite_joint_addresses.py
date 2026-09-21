"""Joint address compatibility without a simulator or GPU."""
from enum import Enum
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Joint(Enum):
    mjJNT_FREE = 0
    mjJNT_BALL = 1
    mjJNT_SLIDE = 2
    mjJNT_HINGE = 3

    def __int__(self):
        return self.value


class JointAddressTest(unittest.TestCase):
    def test_scalar_and_multidof_addresses_with_noninteger_enums(self):
        path = Path(__file__).resolve().parents[1] / 'libero/libero/envs/robosuite_compat.py'
        spec = importlib.util.spec_from_file_location('joint_compat_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'mujoco': SimpleNamespace(mjtJoint=Joint),
                                      'numpy': SimpleNamespace()}):
            spec.loader.exec_module(module)
        model = SimpleNamespace(jnt_type=[0, 1, 2, 3],
                                jnt_qposadr=[0, 7, 11, 12],
                                jnt_dofadr=[0, 6, 9, 10],
                                joint_name2id=lambda name: int(name))
        self.assertNotEqual(3, Joint.mjJNT_HINGE)
        for i, expected in enumerate([(0, 7), (7, 11), 11, 12]):
            self.assertEqual(module._joint_qpos_addr(model, str(i)), expected)
        for i, expected in enumerate([(0, 6), (6, 9), 9, 10]):
            self.assertEqual(module._joint_qvel_addr(model, str(i)), expected)


if __name__ == '__main__':
    unittest.main()
