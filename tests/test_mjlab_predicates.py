"""Predicate geometry regression tests; no simulator or model construction."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
PATH = Path(__file__).resolve().parents[1] / 'libero/libero/envs/mjlab_predicates.py'
spec = importlib.util.spec_from_file_location('predicate_math', PATH)
predicates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predicates)


def evaluate(item, points, matrix=None, qpos=None):
    points = torch.as_tensor(points, dtype=torch.float64).reshape(-1, 1, 3)
    n = len(points)
    rotation = torch.as_tensor(np.eye(3) if matrix is None else matrix, dtype=torch.float64)
    joints = torch.zeros(n, 2) if qpos is None else torch.tensor(qpos)
    return predicates.geometry(item, points, torch.zeros_like(points),
        rotation.expand(n, 1, 3, 3), joints).tolist()


def test_containment_uses_native_rotated_extent_and_lower_z_slack():
    angle = .7
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    size = np.array([.2, .07, .05])
    extent = np.abs(matrix @ size)
    points = [[0, 0, -.059], [0, 0, -.061], [extent[0] + .001, 0, 0],
              [extent[0] - .001, 0, 0], [0, 0, .051]]
    assert evaluate(dict(kind='in', top=0, site=0, size=torch.tensor(size)), points, matrix) == [True, False, False, True, False]


def test_site_on_matches_native_matrix_direction_and_strict_bounds():
    angle = .6
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    size = torch.tensor([.2, .03, .05], dtype=torch.float64)
    # Native under() rotates world displacement by R, not R.T.
    accepted = matrix.T @ np.array([.15, .0, .06])
    rejected = matrix @ np.array([.15, .0, .06])
    item = dict(kind='on', top=0, site=0, size=size)
    assert evaluate(item, [accepted, rejected], matrix) == [True, False]
    assert evaluate(item, [[0, 0, .044], [0, 0, .046], [0, 0, .151]]) == [False, True, False]


@pytest.mark.parametrize('operator,any_joint,expected', [
    ('lt', True, [True, False, False]), ('gt', False, [False, False, True]),
    ('ge', True, [True, True, True]),
])
def test_articulation_any_all_and_switch_boundary(operator, any_joint, expected):
    item = dict(qpos=[0, 1], threshold=0., operator=operator, any_joint=any_joint)
    assert evaluate(item, np.zeros((3, 3)), qpos=[[-.1, .1], [0., 0.], [.1, .2]]) == expected
