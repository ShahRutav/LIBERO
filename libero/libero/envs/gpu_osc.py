"""Batched GPU equivalent of robosuite 1.4 fixed delta OSC_POSE + Panda gripper.

Call on the simulation's Warp CUDA stream after forward(). No state readback
occurs during control. Unsupported controller configurations fail explicitly.
"""
import numpy as np
import torch


class GPUOSC:
    def __init__(self, engine, robot):
        self.engine = engine
        self.device = engine.data.qpos.device
        self.dtype = torch.float64  # robosuite computes its controller in float64
        c = robot.controller
        if not (c.name == 'OSC_POSE' and c.impedance_mode == 'fixed' and c.use_delta
                and c.interpolator_pos is None and c.interpolator_ori is None
                and c.orientation_limits is None):
            raise NotImplementedError('GPUOSC supports fixed delta OSC_POSE without interpolation/orientation limits')
        if type(robot.gripper).__name__ != 'PandaGripper':
            raise NotImplementedError('GPUOSC supports PandaGripper')
        self.uncoupling = c.uncoupling
        self.qids = torch.tensor(c.qpos_index, device=self.device)
        self.vids = torch.tensor(c.qvel_index, device=self.device)
        self.aids = torch.tensor(robot._ref_joint_actuator_indexes, device=self.device)
        self.gids = torch.tensor([robot.sim.model.actuator_name2id(a) for a in robot.gripper.actuators], device=self.device)
        self.site = robot.sim.model.site_name2id(c.eef_name)
        model = robot.sim.model._model
        body = model.site_bodyid[self.site]
        self.root = int(model.body_rootid[body])
        # The seven Panda arm DOFs all influence its grip site.
        ancestors = set()
        while body:
            ancestors.add(body)
            body = model.body_parentid[body]
        if any(model.dof_bodyid[i] not in ancestors for i in c.qvel_index):
            raise ValueError('Arm DOFs must be ancestors of the controlled site')
        wm = engine.wp_model
        adr, nnz, col = wm.M_rowadr.numpy(), wm.M_rownnz.numpy(), wm.M_colind.numpy()
        lookup = {}
        for i in range(model.nv):
            for k in range(int(adr[i]), int(adr[i] + nnz[i])):
                lookup[i, int(col[k])] = k
        self.midx = torch.tensor([[lookup.get((i,j), lookup.get((j,i), -1)) for j in c.qvel_index]
                                 for i in c.qvel_index], device=self.device)
        if (self.midx < 0).any().item():
            raise ValueError('Missing arm inertia entries')
        for name in ('input_min','input_max','output_min','output_max','kp','kd','initial_joint'):
            setattr(self, name, self.tensor(getattr(c,name)))
        self.low, self.high = map(self.tensor, robot.torque_limits)
        ranges = self.tensor(model.actuator_ctrlrange[self.gids.cpu().numpy()])
        self.grip_bias = ranges.mean(-1)
        self.grip_weight = (ranges[:,1] - ranges[:,0]) / 2
        self.grip_delta = self.tensor([-1,1]) * robot.gripper.speed
        self.position_limits = None if c.position_limits is None else self.tensor(c.position_limits)
        self.eye = torch.eye(len(c.qvel_index), dtype=self.dtype, device=self.device)
        self.reset(c)

    def tensor(self, x):
        return torch.as_tensor(np.asarray(x), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def reset(self, controller):
        n = self.engine.num_envs
        # Persistent buffers must remain normal tensors even when setup/reset
        # is called inside an evaluation inference context.
        with torch.inference_mode(False):
            if not hasattr(self, "goal_pos"):
                self.goal_pos = torch.empty((n, 3), device=self.device, dtype=self.dtype)
                self.goal_ori = torch.empty((n, 3, 3), device=self.device, dtype=self.dtype)
                self.grip = torch.empty((n, 2), device=self.device, dtype=self.dtype)
                self.invalid_controller = torch.zeros(n, device=self.device, dtype=torch.bool)
            self.goal_pos.copy_(self.tensor(controller.goal_pos).expand(n, -1))
            self.goal_ori.copy_(self.tensor(controller.goal_ori).expand(n, -1, -1))
            self.grip.zero_()
            self.invalid_controller.zero_()

    @torch.no_grad()
    def reset_indices(self, env_ids, goal_pos=None, goal_ori=None, grip=None):
        """Reset selected worlds after forward, preserving all other controllers."""
        d = self.engine.data
        self.goal_pos[env_ids] = (d.site_xpos[env_ids, self.site].to(self.dtype)
                                 if goal_pos is None else torch.as_tensor(goal_pos, device=self.goal_pos.device, dtype=self.dtype))
        self.goal_ori[env_ids] = (d.site_xmat[env_ids, self.site].reshape(-1, 3, 3).to(self.dtype)
                                 if goal_ori is None else torch.as_tensor(goal_ori, device=self.goal_ori.device, dtype=self.dtype))
        self.grip[env_ids] = 0 if grip is None else torch.as_tensor(grip, device=self.grip.device, dtype=self.dtype)
        self.invalid_controller[env_ids] = False

    def state(self):
        d = self.engine.data
        pos = d.site_xpos[:,self.site].to(self.dtype)
        ori = d.site_xmat[:,self.site].reshape(-1,3,3).to(self.dtype)
        cdof = d.cdof[:,self.vids].to(self.dtype)
        offset = pos - d.subtree_com[:,self.root].to(self.dtype)
        jr = cdof[:,:,:3].transpose(1,2)
        jp = (cdof[:,:,3:] + torch.linalg.cross(cdof[:,:,:3], offset[:,None,:], dim=-1)).transpose(1,2)
        jac = torch.cat((jp,jr), dim=1)
        mass = d.M[:,self.midx].to(self.dtype)
        q = d.qpos[:,self.qids].to(self.dtype)
        v = d.qvel[:,self.vids].to(self.dtype)
        bias = d.qfrc_bias[:,self.vids].to(self.dtype)
        return pos, ori, jac, mass, q, v, bias

    @torch.no_grad()
    def set_goal(self, action, pos, ori):
        """Policy-step controller memory, shared by live control and conversion."""
        scaled = (action[:,:6].clamp(self.input_min,self.input_max) - (self.input_max+self.input_min)/2)
        scaled = scaled * (self.output_max-self.output_min).abs() / (self.input_max-self.input_min).abs() + (self.output_max+self.output_min)/2
        self.goal_pos.copy_(pos + scaled[:,:3])
        if self.position_limits is not None:
            self.goal_pos.clamp_(self.position_limits[0],self.position_limits[1])
        aa = scaled[:,3:6]
        theta = torch.linalg.vector_norm(aa,dim=-1,keepdim=True)
        axis = aa / theta.clamp_min(1e-30)
        x,y,z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
        rot = torch.eye(3,device=self.device,dtype=self.dtype) + torch.sin(theta)[:,:,None]*skew + (1-torch.cos(theta))[:,:,None]*(skew@skew)
        self.goal_ori.copy_(torch.where((theta > 0)[:,:,None],rot@ori,self.goal_ori))

    @torch.no_grad()
    def advance_memory(self, action, substeps):
        """Advance only persistent memory at a recorded state, without physics."""
        d = self.engine.data
        pos = d.site_xpos[:, self.site].to(self.dtype)
        ori = d.site_xmat[:, self.site].reshape(-1, 3, 3).to(self.dtype)
        self.set_goal(action, pos, ori)
        # Match repeated float64 accumulator additions exactly; saturation is
        # monotonic within an action but one multiplied increment can round.
        for _ in range(substeps):
            self.grip.copy_((self.grip + self.grip_delta*torch.sign(action[:, 6:7])).clamp(-1, 1))
        self.engine.data.ctrl[:, self.gids] = (self.grip_bias+self.grip_weight*self.grip).float()

    @torch.no_grad()
    def control(self, action, policy_step):
        pos, ori, jac, mass, q, v, bias = self.state()
        if action.ndim == 1:
            action = action.expand(pos.shape[0],-1)
        if policy_step:
            self.set_goal(action, pos, ori)
        error_ori = 0.5 * torch.linalg.cross(ori.transpose(1,2),self.goal_ori.transpose(1,2),dim=-1).sum(1)
        desired = torch.cat((self.goal_pos-pos,error_ori),-1)*self.kp - (jac@v[:,:,None]).squeeze(-1)*self.kd
        # Match np.linalg.pinv default rcond=1e-15, including singular cases.
        finite_state = (
            torch.isfinite(pos).all(-1) & torch.isfinite(ori).all((-2, -1)) &
            torch.isfinite(jac).all((-2, -1)) & torch.isfinite(mass).all((-2, -1)) &
            torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1) &
            torch.isfinite(bias).all(-1) & torch.isfinite(desired).all(-1)
        )
        # inv_ex reports a singular world without aborting healthy worlds.
        # Substitution is confined to temporary controller math: simulator
        # qpos/qvel and contact integration remain untouched.
        safe_mass = torch.where(finite_state[:, None, None], mass, self.eye)
        minv, info = torch.linalg.inv_ex(safe_mass, check_errors=False)
        valid = finite_state & (info == 0) & torch.isfinite(minv).all((-2, -1))
        jt = jac.transpose(1,2)
        linv = jac@minv@jt
        valid &= torch.isfinite(linv).all((-2, -1))
        task_eye = torch.eye(linv.shape[-1], dtype=self.dtype, device=self.device)
        safe_linv = torch.where(valid[:, None, None], linv, task_eye)
        lam = torch.linalg.pinv(safe_linv,rtol=1e-15)
        if self.uncoupling:
            wrench = torch.cat((torch.linalg.pinv(safe_linv[:,:3,:3],rtol=1e-15)@desired[:,:3,None],
                                 torch.linalg.pinv(safe_linv[:,3:,3:],rtol=1e-15)@desired[:,3:,None]),1)
        else:
            wrench = lam@desired[:,:,None]
        null = self.eye - minv@jt@lam@jac
        pose_torque = mass@(10*(self.initial_joint-q)-2*np.sqrt(10)*v)[:,:,None]
        torques = (jt@wrench + null.transpose(1,2)@pose_torque).squeeze(-1)+bias
        valid &= torch.isfinite(torques).all(-1)
        failed = ~valid
        self.invalid_controller |= failed
        self.engine.data.ctrl[:,self.aids] = torques.clamp(self.low,self.high).float()
        self.grip.copy_((self.grip + self.grip_delta*torch.sign(action[:,6:7])).clamp(-1,1))
        self.engine.data.ctrl[:,self.gids] = (self.grip_bias+self.grip_weight*self.grip).float()
        actuator_ids = torch.cat((self.aids, self.gids))
        controls = self.engine.data.ctrl[:, actuator_ids]
        self.engine.data.ctrl[:, actuator_ids] = torch.where(
            self.invalid_controller[:, None], 0, controls)
        return torques
