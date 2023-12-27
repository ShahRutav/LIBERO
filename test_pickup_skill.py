import os
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, DemoRenderEnv, SkillControllerEnv

def get_skill_config(skills=['reach']):
    '''
        Returns a dictionary of skill controller parameters.
    '''
    config_dict = {
            "aff_penalty_fac": 0.5,
            "success_penalty_fac": 1.2,
            "skills": skills,
            "base_config": {
                "lift_height": 1.0,
                "global_xyz_bounds": [[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]],
            },
            "reach_config": {
                "use_gripper_params": True,
                "use_ori_params": False,
                "max_ac_calls": 100,},
            "pick_config": {
                "use_gripper_params": True,
                "use_ori_params": False,
                "reach_threshold": 0.01,
                "max_ac_calls": 100,},
        }
    return config_dict

def get_action(env, object_name='butter_1', skill_index=0):
    skill_dim = env.action_skill_dim
    # create the one hot vector of skill dim with 1 at skill_index
    skill_vec = np.zeros(skill_dim)
    skill_vec[skill_index] = 1.0
    obs = env._get_observations()
    pos = obs[f'{object_name}_pos']
    action = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0])
    action = np.concatenate((skill_vec, action))
    return action

def main():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    task_id = 0
    total_successes = 0
    total_trials = 0
    for task_id in range(10):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
              f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

        # get skill config here
        # pass the skill config to the env wrapper. The env wrapper handles initializing the skill controller class.
        skill_config = get_skill_config(skills=['pick'])

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "use_camera_obs": False,
            "has_renderer": True,
            "has_offscreen_renderer": False,
            "render_camera": "frontview",
            "skill_config": skill_config,
            "controller": "OSC_POSITION",
            "control_delta": False,}

        env = SkillControllerEnv(**env_args)
        env.seed(0)
        env.reset()
        init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        obs_keys = env._get_observations().keys()
        pos_keys = [key for key in obs_keys if (('pos' in key) and ('robot' not in key))]
        object_names = ['_'.join(key.split('_')[:-1]) for key in pos_keys]
        for obj in object_names:
            env.reset()
            env.set_init_state(init_states[init_state_id])
            env.dummy_actions()
            action = get_action(env, object_name=obj, skill_index=0)
            obs, reward, done, info = env.step(action)
            success = env.check_skill_success(obs, object_name=obj)
            total_successes += success
            total_trials += 1
            print(f"[info] object {obj} success: {success}")
        env.close()
    print(f"[info] total successes: {total_successes}/{total_trials}")

if __name__ == "__main__":
    main()
