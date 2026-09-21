"""Reject unpinned or modified GPU packages before starting a simulation."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'libero/libero/envs/mujoco_warp_pin.py'
spec = importlib.util.spec_from_file_location('warp_pin_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PinTest(unittest.TestCase):
    def verify(self, origin, data=b'tested source', solver_data=None):
        dist = SimpleNamespace(
            read_text=lambda _: json.dumps(origin) if origin else None,
            locate_file=lambda name: SimpleNamespace(read_bytes=lambda:
                solver_data if name.endswith('/solver.py') and solver_data is not None else data))
        with patch.object(module.metadata, 'distribution', return_value=dist), \
             patch.object(module, 'CONSTRAINT_SHA256', hashlib.sha256(b'tested source').hexdigest()), \
             patch.object(module, 'SOLVER_SHA256', hashlib.sha256(b'tested source').hexdigest()):
            return module.verify()

    def test_exact_source_is_accepted(self):
        self.verify({'url': module.REPOSITORY, 'vcs_info': {'commit_id': module.COMMIT}})

    def test_pypi_or_wrong_git_revision_is_rejected(self):
        for origin in [None, {'url': module.REPOSITORY, 'vcs_info': {'commit_id': 'other'}}]:
            with self.subTest(origin=origin), self.assertRaises(RuntimeError):
                self.verify(origin)

    def test_modified_source_with_correct_git_metadata_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.verify({'url': module.REPOSITORY, 'vcs_info': {'commit_id': module.COMMIT}}, b'modified')

    def test_modified_solver_only_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.verify({'url': module.REPOSITORY, 'vcs_info': {'commit_id': module.COMMIT}},
                        solver_data=b'modified solver')


if __name__ == '__main__':
    unittest.main()
