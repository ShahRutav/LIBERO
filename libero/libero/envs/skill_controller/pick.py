import numpy as np
import robosuite.utils.transform_utils as trans
from .base_skill import BaseSkill
from termcolor import colored

class PickSkill(BaseSkill):

    STATES = ['INIT', 'PRE_LIFT_H', 'HOVERING', 'REACHED', 'GRASPED', 'LIFTING', 'LIFTED']

    def __init__(
            self,
            skill_type,
            use_gripper_params=True,
            use_ori_params=False,
            max_ac_calls=15,
            **config
    ):
        assert use_gripper_params == True
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
        gripper_state = info['gripper_state']
        goal_pos = self._get_reach_pos(info)

        th = self._config['reach_threshold']
        # gripper is open when gripper_state is near 0.04
        #print("gripper_state:", gripper_state, np.abs(gripper_state[0] - gripper_state[1]), np.abs(np.abs(gripper_state[0] - gripper_state[1]) - 0.08))
        # the gap between the gripper fingers is 0.08 for gripper open
        gripper_close = not(np.abs(np.abs(gripper_state[0] - gripper_state[1]) - 0.08) < 0.01)
        reached_pre_lift_h = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_post_lift_h = np.abs(cur_pos[2] - self._config['lift_height']) < th
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < th/5)
        # reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        print(goal_pos, cur_pos)
        reached_xyz = (reached_xy) and (np.abs(cur_pos[2] - goal_pos[2]) < 1e-3)
        reached_ori = self._reached_goal_ori(info)
        print(self._state, "gripper_close", gripper_close, "reached_pre_lift_h", reached_pre_lift_h, "reached_xy", reached_xy, "reached_xyz", reached_xyz, "reached_ori", reached_ori, "reached_post_lift_h", reached_post_lift_h)
        #print(cur_pos, goal_pos, np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]))
        self._prev_state = self._state
        if gripper_close and reached_xy and reached_ori and reached_post_lift_h:
            self._state = 'LIFTED'
        elif gripper_close and reached_xy and reached_ori and (not reached_xyz):
            self._state = 'LIFTING'
        elif gripper_close and reached_xy and reached_ori:
            self._state = 'GRASPED'
        elif reached_xyz and reached_ori:
            print(colored(f'reach_xyz, {reached_xyz}, {np.linalg.norm(cur_pos - goal_pos)}, {cur_pos[2]-goal_pos[2]}', 'green'))
            self._state = 'REACHED'
        elif reached_xy and reached_ori:
            self._state = 'HOVERING'
        elif reached_pre_lift_h:
            self._state = 'PRE_LIFT_H'
        else:
            self._state = 'INIT'
        # print in red color the state  and previous state
        print('\033[91m' + 'state changed from {} to {} while number of steps {}'.format(self._prev_state, self._state, self._num_state_steps) + '\033[0m')

        if self._state != self._prev_state:
            # add the print statement in red color
            #print('\033[91m' + 'state changed from {} to {} while number of steps {}'.format(self._prev_state, self._state, self._num_state_steps) + '\033[0m')
            steps_to_change = 10
            if (self._prev_state in ['REACHED', 'LIFTING']) and (self._num_state_steps < steps_to_change):
                # if we are in the grasped state, atleast continue to be in one for 20 steps
                print('continue to be in grasp {}/{}'.format(self._num_state_steps, steps_to_change))
                self._state = self._prev_state
                self._num_state_steps += 1
            else:
                self._num_state_steps = 0
        else:
            self._num_state_steps += 1

        # if self._state index in PickSkill.STATES is lower than the previous state index, then keep the previous state
        if (self._state is not None) and (self._prev_state is not None):
            if PickSkill.STATES.index(self._state) < PickSkill.STATES.index(self._prev_state):
                self._state = self._prev_state

        assert self._state in PickSkill.STATES

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        if 'obj_radius' not in kwargs.keys():
            self._config['lift_height'] = self._params[2] + 0.15
        else:
            self._config['lift_height'] = self._params[2] + kwargs['obj_radius']
        print('lift_height', self._config['lift_height'], 'earlier z', self._params[2]+0.15)
        self._num_steps_steps = 0
        self._num_state_steps = 0

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
            # pos = cur_pos.copy()
            # pos[2] = goal_pos[2]
            pos = goal_pos.copy()
        elif self._state == 'REACHED': # close the gripper
            # pos = goal_pos.copy()
            pos = cur_pos.copy()
        elif self._state == 'GRASPED': # go to the pre-lift height
            # pos = goal_pos.copy()
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTING':
            # pos = goal_pos.copy()
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            #pos = goal_pos.copy()
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
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
        params = self._params
        rc_dim = self._config['robot_controller_dim']
        rg_dim = self._config['robot_gripper_dim']
        gripper_action = params[rc_dim: rc_dim + rg_dim].copy() # we will ignore this and override it using pick state
        if (self._state == 'REACHED') or (self._state == 'GRASPED') or (self._state == 'LIFTED') or (self._state == 'LIFTING'):
            # close the gripper
            gripper_action[:] = [1.0 for _ in range(len(gripper_action))]
        else:
            # open the gripper
            gripper_action[:] = [-1.0 for _ in range(len(gripper_action))]

        if not self._config['use_gripper_params']:
            gripper_action[:] = [0.0 for _ in range(len(gripper_action))]

        return gripper_action

    def _get_reach_pos(self, info):
        params = self._params
        pos = params[:3]
        return pos

    def check_success(self, obs, **kwargs):
        assert 'object_name' in kwargs
        object_name = kwargs['object_name']
        pos = obs[f'{object_name}_pos']
        th = self._config['reach_threshold']
        #print('pos', pos, 'th', self._config['lift_height'] - th)
        success = (pos[2] >= self._config['lift_height'] - th)
        #print('success', success, pos[2], self._config['lift_height'] - th)
        return success

    def is_success(self, info):
        return self._state == 'LIFTED'

    def _get_aff_centers(self, info):
        aff_centers = info.get('reach_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)
