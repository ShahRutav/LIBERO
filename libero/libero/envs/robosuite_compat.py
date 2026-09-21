# State update adapted from robosuite 1.4.0 base_controller.py.
# MIT License
#
# Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Compatibility for robosuite 1.4 controllers on MuJoCo 3.11.

MuJoCo 3.11 removed MjData.qM and changed mj_fullM's arguments. The control
law is unchanged; only the state-reading method needs an API adaptation.
"""

import mujoco
import numpy as np


def _update(self, force=False):
    if not (self.new_update or force):
        return
    self.sim.forward()
    data = self.sim.data
    site_id = self.sim.model.site_name2id(self.eef_name)
    self.ee_pos = np.array(data.site_xpos[site_id])
    self.ee_ori_mat = np.array(data.site_xmat[site_id].reshape(3, 3))
    self.ee_pos_vel = np.array(data.get_site_xvelp(self.eef_name))
    self.ee_ori_vel = np.array(data.get_site_xvelr(self.eef_name))
    self.joint_pos = np.array(data.qpos[self.qpos_index])
    self.joint_vel = np.array(data.qvel[self.qvel_index])
    self.J_pos = np.array(data.get_site_jacp(self.eef_name).reshape(3, -1)[:, self.qvel_index])
    self.J_ori = np.array(data.get_site_jacr(self.eef_name).reshape(3, -1)[:, self.qvel_index])
    self.J_full = np.vstack([self.J_pos, self.J_ori])
    mass = np.empty((self.sim.model.nv, self.sim.model.nv), dtype=np.float64)
    mujoco.mj_fullM(self.sim.model._model, data._data, mass)
    self.mass_matrix = mass[self.qvel_index, :][:, self.qvel_index]
    self.new_update = False


def install_controller_compat():
    """Apply the necessary shared robosuite API fix once, for either backend."""
    if tuple(map(int, mujoco.__version__.split(".")[:2])) >= (3, 11):
        from robosuite.controllers.base_controller import Controller
        Controller.update = _update
    # Newer MuJoCo enums do not compare equal to NumPy integer scalars.
    # Robosuite 1.4 compares jnt_type array entries directly with these enums.
    if tuple(map(int, mujoco.__version__.split(".")[:2])) >= (3, 12):
        from robosuite.utils.binding_utils import MjModel
        MjModel.get_joint_qpos_addr = _joint_qpos_addr
        MjModel.get_joint_qvel_addr = _joint_qvel_addr


def _joint_addr(model, name, *, position):
    joint_id = model.joint_name2id(name)
    kind = int(model.jnt_type[joint_id])
    if kind == int(mujoco.mjtJoint.mjJNT_FREE):
        size = 7 if position else 6
    elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
        size = 4 if position else 3
    elif kind in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        size = 1
    else:
        raise ValueError(f"Unsupported joint type {kind} for {name}")
    address = int((model.jnt_qposadr if position else model.jnt_dofadr)[joint_id])
    return address if size == 1 else (address, address + size)


def _joint_qpos_addr(self, name):
    return _joint_addr(self, name, position=True)


def _joint_qvel_addr(self, name):
    return _joint_addr(self, name, position=False)
