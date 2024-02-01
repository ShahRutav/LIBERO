import numpy as np
import robosuite.utils.transform_utils as trans
from .base_skill import BaseSkill

class ReachSkill(BaseSkill):

    STATES = ['INIT', 'PRE_LIFT_H', 'HOVERING', 'REACHED']

    def __init__(
            self,
            skill_type,
            use_gripper_params=True,
            use_ori_params=False,
            max_ac_calls=15,
            **config
    ):
        super().__init__(
            skill_type,
            use_gripper_params=use_gripper_params,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            **config,
        )

    def get_param_dim(self, base_param_dim):
        return base_param_dim

    def update_state(self, info):
        cur_pos = info['cur_ee_pos']
        goal_pos = self._get_reach_pos(info)

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori = self._reached_goal_ori(info)
        #print(reached_lift, reached_xy, reached_xyz, reached_ori)

        if reached_xyz and reached_ori:
            self._state = 'REACHED'
        else:
            if reached_xy and reached_ori:
                self._state = 'HOVERING'
            else:
                if reached_lift:
                    self._state = 'PRE_LIFT_H'
                else:
                    self._state = 'INIT'

        assert self._state in ReachSkill.STATES

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self._config['lift_height'] = self._params[2] + 0.1
        self._num_steps_steps = 0

    def get_pos_ac(self, info):
        cur_pos = info['cur_ee_pos']
        goal_pos = self._get_reach_pos(info)
        #print('cur_pos', cur_pos, 'goal_pos', goal_pos, 'state', self._state)

        is_delta = False
        if self._state == 'INIT': # go to the pre-lift height
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'PRE_LIFT_H': # go the x,y position while keeping the pre-lift height
            pos = goal_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'HOVERING': # go to the goal position by moving downwards
            pos = goal_pos.copy()
        elif self._state == 'REACHED':
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos, is_delta

    def get_ori_ac(self, info):
        ori = super().get_ori_ac(info)
        if self._config['use_ori_params']:
            if self._state == 'INIT':
                ori[:] = [0.0 for _ in range(len(ori))]
                is_delta = True
            else:
                is_delta = False
        else:
            ori[:] = [0.0 for _ in range(len(ori))]
            is_delta = True
        return ori, is_delta

    def get_gripper_ac(self, info):
        gripper_action = super().get_gripper_ac(info)
        if not self._config['use_gripper_params']:
            gripper_action[:] = [0.0 for _ in range(len(gripper_action))]

        return gripper_action

    def _get_reach_pos(self, info):
        params = self._params
        #pos = self._get_unnormalized_pos(
        #    params[:3], self._config['global_xyz_bounds'])
        pos = params[:3]
        return pos

    def is_success(self, info):
        return self._state == 'REACHED'

    def _get_aff_centers(self, info):
        aff_centers = info.get('reach_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

