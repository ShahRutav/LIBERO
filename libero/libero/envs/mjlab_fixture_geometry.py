"""Refresh Warp's cached world-welded geom transforms after fixture resets."""
import warp as wp
from mujoco_warp._src import math


@wp.kernel
def _refresh_static_geoms(
    env_ids: wp.array[wp.int64],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_mocapid: wp.array[int],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
    geom_bodyid: wp.array[int],
    geom_pos: wp.array2d[wp.vec3],
    geom_quat: wp.array2d[wp.quat],
    geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
):
    index, geom = wp.tid()
    world = int(env_ids[index])
    body = geom_bodyid[geom]
    # Match Warp's static-geom cache condition. Articulated and mocap geoms
    # remain the responsibility of the following normal forward pass.
    if body_weldid[body] != 0 or body_mocapid[body_rootid[body]] != -1:
        return
    pos = geom_pos[world % geom_pos.shape[0], geom]
    quat = geom_quat[world % geom_quat.shape[0], geom]
    # These ancestors are jointless. Compose their current model poses directly
    # so refreshing a partial reset never writes another world's kinematics.
    while body != 0:
        rotation = body_quat[world % body_quat.shape[0], body]
        pos = body_pos[world % body_pos.shape[0], body] + math.rot_vec_quat(pos, rotation)
        quat = math.mul_quat(rotation, quat)
        body = body_parentid[body]
    geom_xpos[world, geom] = pos
    geom_xmat[world, geom] = math.quat_to_mat(quat)


def refresh_static_fixture_geometry(engine, env_ids):
    """Update selected worlds before collision detection reads their geom poses.

    Warp 3.13 computes world-welded geometry once at make_data, so changing
    model body poses followed by forward alone leaves those collision geoms
    stale. Local geom bounds do not change; broadphase reads the updated global
    geom transforms during the subsequent forward.
    """
    if not len(env_ids):
        return
    m, d = engine.wp_model, engine.wp_data
    with wp.ScopedDevice(engine.wp_device):
        wp.launch(
            _refresh_static_geoms, dim=(len(env_ids), m.ngeom),
            inputs=[wp.from_torch(env_ids.contiguous(), dtype=wp.int64),
                    m.body_parentid, m.body_rootid, m.body_weldid, m.body_mocapid,
                    m.body_pos, m.body_quat, m.geom_bodyid, m.geom_pos, m.geom_quat],
            outputs=[d.geom_xpos, d.geom_xmat],
        )
