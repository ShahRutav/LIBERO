import os
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, DemoRenderEnv, SkillControllerEnv
from robosuite.utils.camera_utils import project_points_from_world_to_camera, get_camera_transform_matrix

OBJECTS_TO_IGNORE = ['plate', 'basket', 'frypan', 'moka']
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256

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
                "max_ac_calls": 200,},
        }
    return config_dict

def point2pixel(env, camera_name, point):
    camera_matrix = get_camera_transform_matrix(env.env.sim, camera_name, CAMERA_HEIGHT, CAMERA_WIDTH)
    pixel  = project_points_from_world_to_camera(point, camera_matrix, CAMERA_HEIGHT, CAMERA_WIDTH)
    return pixel

def opengl2pil(image):
    image = image[::-1, :, :] # flip image to convert from opengl to PIL
    image = Image.fromarray(image)
    return image

def mark_pixel(img, pixel, radius=5, color=(255, 0, 0)):
    # mark a circle with radius around the pixel
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            img.putpixel((pixel[0]+i, pixel[1]+j), color)
    return img

def plot_object_pos(env, obj):
    env.dummy_actions(num_actions=10)
    obs = env._get_observations()
    img = obs['agentview_image']
    img = opengl2pil(img)

    pos = obs[f'{obj}_pos']
    obj_id = env.env.sim.model.geom_name2id(obj+'_g0')
    obj_pos = env.env.sim.data.geom_xpos[obj_id]

    obj_sim_id = env.env.sim.model.body_name2id(obj+'_main')
    obj_sim_pos = env.env.sim.data.body_xpos[obj_sim_id]

    pixel_pos = point2pixel(env, 'agentview', pos)
    obj_pixel_pos = point2pixel(env, 'agentview', obj_pos)
    # flip x,y of pixel
    pixel_pos = (pixel_pos[1], pixel_pos[0])
    obj_pixel_pos = (obj_pixel_pos[1], obj_pixel_pos[0])
    print(pixel_pos, obj_pixel_pos)
    marked_img = mark_pixel(img, pixel_pos, radius=2, color=(255, 0, 0))
    marked_img = mark_pixel(marked_img, obj_pixel_pos, radius=2, color=(255, 255, 255))
    marked_img.save('marked_img.png')

    env.close()
    exit()

def get_action(env, object_name='butter_1', skill_index=0):
    adjustments = {'ketchup_1': 0.06, 'alphabet_soup_1': 0.02, 'tomato_sauce_1': 0.02, 'orange_juice_1': 0.06, 'milk_1': 0.05}
    skill_dim = env.action_skill_dim
    # create the one hot vector of skill dim with 1 at skill_index
    skill_vec = np.zeros(skill_dim)
    skill_vec[skill_index] = 1.0
    obs = env._get_observations()
    pos = obs[f'{object_name}_pos']

    if object_name in adjustments.keys():
        pos[2] += adjustments[object_name]
    action = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0])
    action = np.concatenate((skill_vec, action))
    return action

def eval_loop(env, obj, init_state):
    obj_id = env.env.sim.model.geom_name2id(obj+'_g0')
    obj_model_radius = env.env.sim.model.geom_rbound[obj_id]

    obs = env.reset()
    env.set_init_state(init_state)
    env.dummy_actions(num_actions=10)

    action = get_action(env, object_name=obj, skill_index=0)
    obs, reward, done, info = env.step(action)
    success = env.check_skill_success(obs, object_name=obj)
    print(f"[info] object {obj} success: {success}")

    return success

def main():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_90" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    task_id = 0
    total_successes = 0
    total_trials = 0
    for task_id in range(90):
        task = task_suite.get_task(task_id)
        task_name = task.name
        if not 'LIVING_ROOM' in task_name:
            continue
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
              f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")
        continue

        # get skill config here
        # pass the skill config to the env wrapper. The env wrapper handles initializing the skill controller class.
        skill_config = get_skill_config(skills=['pick'])

        # step over the environment
        camera_obs = False
        env_args = {
            "bddl_file_name": task_bddl_file,
            "use_camera_obs": camera_obs,
            "has_renderer": not camera_obs,
            "has_offscreen_renderer": camera_obs,
            "render_camera": "frontview",
            "skill_config": skill_config,
            "controller": "OSC_POSITION",
            "camera_heights": CAMERA_HEIGHT,
            "camera_widths": CAMERA_WIDTH,
            "control_delta": False,}

        env = SkillControllerEnv(**env_args)
        env.seed(0)
        env.reset()
        init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        obs_keys = env._get_observations().keys()
        pos_keys = [key for key in obs_keys if (('pos' in key) and ('robot' not in key))]
        object_names = ['_'.join(key.split('_')[:-1]) for key in pos_keys]


        # skill doesn't work for milk_1, orange_juice_1
        for obj in object_names:
            ignore = False
            for obj_to_ignore in OBJECTS_TO_IGNORE:
                if obj_to_ignore in obj:
                    ignore = True
                    break
            if ignore:
                continue
        # obj = 'milk_1'
        # # plot_object_pos(env, obj)

            print(f"[info] evaluating object {obj} for task {task_id}")
            success = eval_loop(env, obj, init_state=init_states[init_state_id])
            if success is not None:
                total_successes += success
                total_trials += 1
        env.close()
        # exit()
    print(f"[info] total successes: {total_successes}/{total_trials}")

if __name__ == "__main__":
    main()
