import os
import numpy as np
import robosuite as suite
import matplotlib.cm as cm

from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs.skill_controller import SkillController
from libero.libero.envs import *


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        control_delta=True,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        controller_configs = suite.load_controller_config(default_controller=controller)
        controller_configs["control_delta"] = control_delta

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue

        return ret

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)


class SegmentationRenderEnv(OffScreenRenderEnv):
    """
    This wrapper will additionally generate the segmentation mask of objects,
    which is useful for comparing attention.
    """

    def __init__(
        self,
        camera_segmentations="instance",
        camera_heights=128,
        camera_widths=128,
        **kwargs,
    ):
        assert camera_segmentations is not None
        kwargs["camera_segmentations"] = camera_segmentations
        kwargs["camera_heights"] = camera_heights
        kwargs["camera_widths"] = camera_widths
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.segmentation_robot_id = None
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        self.segmentation_id_mapping = {}

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name == "Panda0":
                self.segmentation_robot_id = i

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
                self.segmentation_id_mapping[i] = instance_name

        self.instance_to_id = {
            v: k + 1 for k, v in self.segmentation_id_mapping.items()
        }
        return obs

    def get_segmentation_instances(self, segmentation_image):
        # get all instances' segmentation separately
        seg_img_dict = {}
        segmentation_image[segmentation_image > self.segmentation_robot_id] = (
            self.segmentation_robot_id + 1
        )
        seg_img_dict["robot"] = segmentation_image * (
            segmentation_image == self.segmentation_robot_id + 1
        )

        for seg_id, instance_name in self.segmentation_id_mapping.items():
            seg_img_dict[instance_name] = segmentation_image * (
                segmentation_image == seg_id + 1
            )
        return seg_img_dict

    def get_segmentation_of_interest(self, segmentation_image):
        # get the combined segmentation of obj of interest
        # 1 for obj_of_interest
        # -1.0 for robot
        # 0 for other things
        ret_seg = np.zeros_like(segmentation_image)
        for obj in self.obj_of_interest:
            ret_seg[segmentation_image == self.instance_to_id[obj]] = 1.0
        # ret_seg[segmentation_image == self.segmentation_robot_id+1] = -1.0
        ret_seg[segmentation_image == 0] = -1.0
        return ret_seg

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:
            # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
            rstate = np.random.RandomState(seed=2)
            inds = np.arange(256)
            rstate.shuffle(inds)
            seg_img = (
                np.array(255.0 * cm.rainbow(inds[seg_im], 10))
                .astype(np.uint8)[..., :3]
                .astype(np.uint8)
                .squeeze(-2)
            )
            print(seg_img.shape)
            cv2.imshow("Seg Image", seg_img[::-1])
            cv2.waitKey(1)
            # use @inds to map each geom ID to a color
            return seg_img


class DemoRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = True
        kwargs["has_offscreen_renderer"] = False
        kwargs["render_camera"] = "frontview"

        super().__init__(**kwargs)

    def _get_observations(self):
        return self.env._get_observations()

class SkillControllerEnv(ControlEnv):
    def __init__(self, skill_config, return_all_info=False, **kwargs):
        self._action_dim = 0
        self._skill_config = skill_config
        self._return_all_info = return_all_info
        super().__init__(**kwargs)
        self._skill_controller = SkillController(robots=self.robots, config=self._skill_config)

    def dummy_actions(self, num_actions=5):
        for _ in range(num_actions):
            self.env.step(np.zeros(self.env._action_dim))

    def reset(self):
        # returns all the obs from the robosuite environment as configured.
        obs = super().reset()
        # Note: individual skills is reset at each step.
        self._reset_skill()
        return obs

    def _reset_skill(self):
        # Reset the action space
        skill_dim = self._skill_controller.get_skill_dim() # number of skills
        self._action_dim = 0
        for robot in self.robots:
            param_dim = self._skill_controller.get_param_dim(robot.action_dim)
            self._action_dim += (param_dim + skill_dim)

    def step(self, action):
        # action: first skill_dim indicates the skill to be executed, one hot encoded
        # the rest of the action is the parameter of the skill
        # reset the skill to current_skill
        # reset all info variable for the skill including parameter of the skill
        self._skill_controller.reset(action)
        reward_sum = 0.0
        obs = self._get_observations()
        info = []
        if self._return_all_info:
            obs_list = []
            reward_list = []
            done_list = []
            action_list = []
        while True:
            action_ll = self._skill_controller.step()
            next_obs, reward, done, info = super().step(action_ll)
            #self.env.render()
            reward_sum += reward
            # update the info with the observation
            if self._return_all_info:
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                action_list.append(action_ll)
            if self._skill_controller.done():
                break
            obs = next_obs
        if self._return_all_info:
            obs = np.stack(obs_list, axis=0)
            reward_sum = np.stack(reward_list, axis=0)
            done = np.stack(done_list, axis=0)
            action_ll = np.stack(action_list, axis=0)
            info.update({'actions_ll': action_ll})
        return obs, reward_sum, done, info

    def check_skill_success(self, obs=None, **kwargs):
        if obs is None:
            obs = self._get_observations()
        return self._skill_controller.check_skill_success(obs, **kwargs)

    @property
    def action_spec(self):
        low, high = [], []

        skill_dim = self._skill_controller.get_skill_dim()

        for robot in self.robots:
            # lo, hi = robot.action_limits
            param_dim = self._skill_controller.get_param_dim(robot.action_dim)
            robot_dim = param_dim + skill_dim
            lo, hi = (-1 * np.ones(robot_dim), np.ones(robot_dim))
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])

        return low, high

    @property
    def action_skill_dim(self):
        return self._skill_controller.get_skill_dim()

    @property
    def action_dim(self):
        return self._action_dim

    def _get_observations(self):
        return self.env._get_observations()
