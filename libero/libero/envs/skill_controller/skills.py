import numpy as np
import robosuite.utils.transform_utils as trans

class BaseSkill:
    def __init__(
            self,
            skill_type,

            ### common settings ###
            global_xyz_bounds=np.array([
                [-0.30, -0.30, 0.80],
                [0.15, 0.30, 0.90]
            ]),
            delta_xyz_scale=np.array([0.15, 0.15, 0.05]),
            local_xyz_scale=np.array([0.05, 0.05, 0.05]),
            lift_height=0.95,
            reach_threshold=0.01,
            aff_threshold=0.08,
            aff_type=None,
            binary_gripper=True,
            aff_tanh_scaling=10.0,
            **config
    ):
        self._skill_type = skill_type
        self._config = dict(
            global_xyz_bounds=global_xyz_bounds,
            delta_xyz_scale=delta_xyz_scale,
            local_xyz_scale=local_xyz_scale,
            lift_height=lift_height,
            reach_threshold=reach_threshold,
            aff_threshold=aff_threshold,
            aff_type=aff_type,
            binary_gripper=binary_gripper,
            aff_tanh_scaling=aff_tanh_scaling,
            **config
        )

        for k in ['global_xyz_bounds', 'delta_xyz_scale', 'local_xyz_scale']:
            assert self._config[k] is not None
            self._config[k] = np.array(self._config[k])

        assert self._config['aff_type'] in [None, 'sparse', 'dense']

    def get_param_dim(self, base_param_dim):
        assert NotImplementedError

    def update_state(self, info):
        pass

    def reset(self, params, config_update, info):
        self._params = params
        self._state = None
        self._config.update(config_update)
        self._aff_reward, self._aff_success = \
            self._compute_aff_reward_and_success(info)

    def get_pos_ac(self, info):
        raise NotImplementedError

    def get_ori_ac(self, info):
        params = self._params
        rc_dim = self._config['robot_controller_dim']
        ori = params[3:rc_dim].copy()
        return ori

    def get_gripper_ac(self, info):
        params = self._params
        rc_dim = self._config['robot_controller_dim']
        rg_dim = self._config['robot_gripper_dim']
        gripper_action = params[rc_dim: rc_dim + rg_dim].copy()

        if self._config['binary_gripper']:
            if np.abs(gripper_action) < 0.10:
                gripper_action[:] = 0
            elif gripper_action < 0:
                gripper_action[:] = -1
            else:
                gripper_action[:] = 1

        return gripper_action

    def get_max_ac_calls(self):
        return self._config['max_ac_calls']

    def get_aff_reward(self):
        return self._aff_reward

    def get_aff_success(self):
        return self._aff_success

    def _get_unnormalized_pos(self, pos, bounds):
        pos = np.clip(pos, -1, 1)
        pos = (pos + 1) / 2
        low, high = bounds[0], bounds[1]
        return low + (high - low) * pos

    def _reached_goal_ori(self, info):
        ori, _ = self.get_ori_ac(info)
        ori = ori.copy()

        if len(ori) == 0 or (not self._config['use_ori_params']):
            return True
        robot_controller = self._config['robot_controller']
        #goal_ori = robot_controller.get_global_euler_from_ori_ac(ori)
        goal_ori = ori # assuming input is in axis-angle format
        ee_ori_mat = robot_controller.ee_ori_mat
        #cur_ori = trans.mat2euler(robot_controller.ee_ori_mat, axes="rxyz")
        cur_ori = trans.quat2axisangle(trans.mat2quat(robot_controller.ee_ori_mat))
        print("cur_ori: ", cur_ori)
        print("goal_ori: ", goal_ori)

        # check the difference between the current and goal orientation in axis-angle format
        #ee_ori_diff = np.minimum(
        #    (goal_ori - cur_ori) % (2 * np.pi),
        #    (cur_ori - goal_ori) % (2 * np.pi)
        #)
        ee_ori_diff = np.abs(goal_ori - cur_ori) % (2 * np.pi)

        raise NotImplementedError
        if ee_ori_diff[-1] <= 0.20:
            return True
        else:
            return False

    def _compute_aff_reward_and_success(self, info):
        if self._config['aff_type'] is None:
            return 1.0, True

        aff_centers = self._get_aff_centers(info)
        reach_pos = self._get_reach_pos(info)

        if aff_centers is None:
            return 1.0, True

        if len(aff_centers) == 0:
            return 0.0, False

        th = self._config['aff_threshold']
        within_th = (np.abs(aff_centers - reach_pos) <= th)
        aff_success = np.any(np.all(within_th, axis=1))

        if self._config['aff_type'] == 'dense':
            if aff_success:
                aff_reward = 1.0
            else:
                dist = np.clip(np.abs(aff_centers - reach_pos) - th, 0, None)
                min_dist = np.min(np.sum(dist, axis=1))
                aff_reward = 1.0 - np.tanh(self._config['aff_tanh_scaling'] * min_dist)
        else:
            aff_reward = float(aff_success)

        return aff_reward, aff_success

    def _get_aff_centers(self, info):
        raise NotImplementedError

    def _get_reach_pos(self, info):
        raise NotImplementedError

class ReachSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED']

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

        if reached_xyz and reached_ori:
            self._state = 'REACHED'
        else:
            if reached_xy and reached_ori:
                self._state = 'HOVERING'
            else:
                if reached_lift:
                    self._state = 'LIFTED'
                else:
                    self._state = 'INIT'

        assert self._state in ReachSkill.STATES

    def get_pos_ac(self, info):
        cur_pos = info['cur_ee_pos']
        goal_pos = self._get_reach_pos(info)
        print('cur_pos', cur_pos, 'goal_pos', goal_pos, 'state', self._state)

        is_delta = False
        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'HOVERING':
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
                ori[:] = 0.0
                is_delta = True
            else:
                is_delta = False
        else:
            ori[:] = 0.0
            is_delta = True
        return ori, is_delta

    def get_gripper_ac(self, info):
        gripper_action = super().get_gripper_ac(info)
        if not self._config['use_gripper_params']:
            gripper_action[:] = 0

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

class GripperSkill(BaseSkill):
    def __init__(
            self,
            skill_type,
            max_ac_calls=4,
            **config
    ):
        super().__init__(
            skill_type,
            max_ac_calls=max_ac_calls,
            **config
        )
        self._num_steps_steps = 0

    def get_param_dim(self, base_param_dim):
        return 0

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self._num_steps_steps = 0

    def update_state(self, info):
        self._num_steps_steps += 1

    def get_pos_ac(self, info):
        pos = info['cur_ee_pos'].copy()
        is_delta = False
        return pos, is_delta

    def get_ori_ac(self, info):
        rc_dim = self._config['robot_controller_dim']
        ori = np.zeros(rc_dim - 3)
        ori[:] = 0.0
        is_delta = True
        #raise NotImplementedError
        return ori, is_delta

    def get_gripper_ac(self, info):
        rg_dim = self._config['robot_gripper_dim']
        gripper_action = np.zeros(rg_dim)
        if self._skill_type in ['close', 'close_pos']:
            gripper_action[:] = 1
        elif self._skill_type in ['open', 'open_pos']:
            gripper_action[:] = -1
        else:
            raise ValueError

        return gripper_action

    def is_success(self, info):
        return (self._num_steps_steps == self._config['max_ac_calls'])

    def _get_aff_centers(self, info):
        return None

    def _get_reach_pos(self, info):
        return info['cur_ee_pos']
