"""GPU integration tests. Run on a pod with LIBERO_GPU_TESTS=1."""
import os
import unittest


@unittest.skipUnless(os.environ.get('LIBERO_GPU_TESTS') == '1', 'requires GPU pod')
class FixtureGeometryTest(unittest.TestCase):
    def test_nested_rotation_partial_refresh_and_collision(self):
        from types import SimpleNamespace
        import numpy as np
        import torch
        import mujoco as mj
        import mujoco_warp as mjw
        import warp as wp
        from libero.libero.envs.mjlab_fixture_geometry import refresh_static_fixture_geometry
        wp.init()
        model = mj.MjModel.from_xml_string('''<mujoco>
          <option gravity="0 0 0" cone="elliptic"/>
          <worldbody>
            <body name="fixture" pos="0 0 .5"><body name="nested" pos=".1 0 0" euler="15 0 0">
              <geom name="support" type="box" size=".1 .1 .05"/>
              <body name="hinge" pos="0 0 .3"><joint type="hinge"/><geom name="moving" type="sphere" size=".03"/></body>
            </body></body>
            <body name="mocap" mocap="true" pos="0 0 2"><geom name="mocap_geom" type="sphere" size=".03"/></body>
            <body name="ball" pos="0 0 3"><freejoint/><geom name="ball_geom" type="sphere" size=".05"/></body>
          </worldbody></mujoco>''')
        m = mjw.put_model(model, batch_sizes={'body_pos': 3, 'body_quat': 3})
        d = mjw.make_data(model, nworld=3, nconmax=64, njmax=128)
        engine = SimpleNamespace(wp_model=m, wp_data=d, wp_device=wp.get_device('cuda:0'))
        fixture = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'fixture')
        support = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'support')
        moving = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'moving')
        mocap = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'mocap_geom')
        ball = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'ball_geom')
        body_pos = wp.to_torch(m.body_pos)
        body_quat = wp.to_torch(m.body_quat)
        body_pos[0, fixture] += torch.tensor([.5, -.2, .1], device='cuda')
        body_pos[2, fixture] += torch.tensor([-.4, .3, -.1], device='cuda')
        angle = .7
        body_quat[0, fixture] = torch.tensor([np.cos(angle/2), 0, 0, np.sin(angle/2)], device='cuda')
        body_quat[2, fixture] = torch.tensor([np.cos(-angle/2), 0, 0, np.sin(-angle/2)], device='cuda')
        before_pos = wp.to_torch(d.geom_xpos).clone()
        before_mat = wp.to_torch(d.geom_xmat).clone()
        torch.cuda.synchronize()
        refresh_static_fixture_geometry(engine, torch.tensor([2, 0], device='cuda'))
        wp.synchronize()
        after_pos = wp.to_torch(d.geom_xpos).clone()
        after_mat = wp.to_torch(d.geom_xmat).clone()
        self.assertTrue(torch.equal(before_pos[1], after_pos[1]))
        self.assertTrue(torch.equal(before_mat[1], after_mat[1]))
        for g in (moving, mocap, ball):
            self.assertTrue(torch.equal(before_pos[:, g], after_pos[:, g]))
            self.assertTrue(torch.equal(before_mat[:, g], after_mat[:, g]))
        refresh_static_fixture_geometry(engine, torch.tensor([], dtype=torch.long, device='cuda'))
        self.assertTrue(torch.equal(after_pos, wp.to_torch(d.geom_xpos)))
        positions = body_pos.cpu().numpy(); rotations = body_quat.cpu().numpy()
        qpos = wp.to_torch(d.qpos).cpu().numpy().copy()
        # Place the free sphere 1 mm into the refreshed support, so stale bounds
        # would miss the pair. Check contact generation after regular forward.
        native_contacts = []
        for world in range(3):
            model.body_pos[:] = positions[world]; model.body_quat[:] = rotations[world]
            nd = mj.MjData(model); nd.qpos[:] = qpos[world]; mj.mj_kinematics(model, nd)
            if world != 1:
                np.testing.assert_allclose(after_pos[world, support].cpu(), nd.geom_xpos[support], atol=2e-7)
                np.testing.assert_allclose(after_mat[world, support].cpu(), nd.geom_xmat[support].reshape(3,3), atol=2e-7)
            qpos[world, 1:4] = nd.geom_xpos[support] + nd.geom_xmat[support].reshape(3,3)[:, 2] * .099
            nd.qpos[:] = qpos[world]; mj.mj_forward(model, nd)
            native_contacts.append(any(set(c.geom) == {support, ball} for c in nd.contact))
        wp.to_torch(d.qpos).copy_(torch.as_tensor(qpos, device='cuda'))
        torch.cuda.synchronize()
        mjw.forward(m, d)
        wp.synchronize()
        n = int(wp.to_torch(d.nacon)[0]); geoms = wp.to_torch(d.contact.geom).cpu().numpy()[:n]
        worlds = wp.to_torch(d.contact.worldid).cpu().numpy()[:n]
        gpu_contacts = [any(int(w) == world and set(g) == {support, ball} for w,g in zip(worlds,geoms)) for world in range(3)]
        self.assertEqual(native_contacts, [True, True, True])
        self.assertEqual(gpu_contacts, native_contacts)


if __name__ == '__main__':
    unittest.main()
