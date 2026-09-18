"""Pure coordinate regressions; extract real sampler methods without simulator imports."""
import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'libero/libero/envs/regions/base_region_sampler.py'


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def load_methods():
    tree = ast.parse(SOURCE.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_site_frame_in_new_reference':
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name in ('SiteRegionRandomSampler', 'InSiteRegionRandomSampler'):
            sample = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
            sample.name = node.name
            nodes.append(sample)
    # Tests represent yaw quaternions faithfully; production uses robosuite T.
    def quat2mat(q):
        x, y, z, w = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    namespace = dict(np=np, copy=copy, RandomizationError=RuntimeError,
                     T=SimpleNamespace(convert_quat=lambda q, **_: np.asarray(q)[[1, 2, 3, 0]],
                                       quat2mat=quat2mat))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


def sim_at(old_position, old_rotation, local_site, local_rotation=np.eye(3)):
    return SimpleNamespace(data=SimpleNamespace(
        get_body_xpos=lambda name: old_position,
        get_body_xmat=lambda name: old_rotation,
        get_site_xpos=lambda name: old_position + old_rotation @ local_site,
        get_site_xmat=lambda name: old_rotation @ local_rotation))


@pytest.mark.parametrize('name', ['SiteRegionRandomSampler', 'InSiteRegionRandomSampler'])
@pytest.mark.parametrize('pitched', [False, True])
@pytest.mark.parametrize('on_top', [False, True])
@pytest.mark.parametrize('old_position,old_yaw', [(np.zeros(3), 0.), (np.array([.3, -.2, .9]), 1.2),
                                               (np.array([-.1, .4, 1.1]), -2.)])
def test_site_placement_independent_of_previous_root_pose(name, old_position, old_yaw, pitched, on_top):
    functions = load_methods()
    # Descendant drawer offset is already included in current site kinematics;
    # using raw model.site_pos would miss this translated, rotated child frame.
    drawer_offset = np.array([.12, -.04, .2])
    child_rotation = (np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
                      if pitched else rotation(.4))
    site_local_in_child = np.array([.01, .02, .03])
    site_in_root = drawer_offset + child_rotation @ site_local_in_child
    sim = sim_at(old_position, rotation(old_yaw), site_in_root, child_rotation)
    reference = SimpleNamespace(root_body='cabinet_root', top_offset=np.array([0., 0., .045]))
    obj = SimpleNamespace(name='bowl', horizontal_radius=.01, bottom_offset=np.array([0., 0., -.02]))
    sample = SimpleNamespace(mujoco_objects=[obj], num_ranges=1, z_offset=.005,
                             ensure_valid_placement=False, _sample_x=lambda _: .03,
                             _sample_y=lambda _: -.01, _sample_quat=lambda: np.array([1., 0., 0., 0.]))
    new_yaw = -.7
    quat = np.array([np.cos(new_yaw/2), 0., 0., np.sin(new_yaw/2)])
    new_position = np.array([-.2, .1, .85])
    fixtures = {'cabinet': (new_position.copy(), quat, reference)}
    result = functions[name](sample, sim, fixtures, reference='cabinet', site_name='drawer_site', on_top=on_top)
    expected = new_position + rotation(new_yaw) @ (site_in_root + child_rotation @ np.array([.03, -.01, 0.]))
    # Preserve existing vertical clearance/top-offset conventions intentionally.
    expected[2] += .005 + (.02 if on_top else 0) + (.045 if on_top and name == 'SiteRegionRandomSampler' else 0)
    np.testing.assert_allclose(result['bowl'][0], expected, atol=1e-12)
    np.testing.assert_array_equal(fixtures['cabinet'][0], new_position)


def test_nonfinite_live_frame_fails_explicitly():
    sim = sim_at(np.array([np.nan, 0., 0.]), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match='finite'):
        load_methods()['_site_frame_in_new_reference'](sim, SimpleNamespace(root_body='root'), [1, 0, 0, 0], 'site')
