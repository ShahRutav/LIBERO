import os
import h5py

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.algos import get_algo_class, get_algo_list
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    get_task_embs,
)

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

def raw_obs_to_tensor_obs(obs, cfg, goal_emb):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)
    obs['goal_emb_dinov2-base'] = goal_emb

    data = {
        "obs": {},
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for obs_name in all_obs_keys:
        data["obs"][obs_name].append(
            ObsUtils.process_obs(
                torch.from_numpy(obs[cfg.data.obs_key_mapping[obs_name]]),
                obs_key=obs_name,
            ).float()
        )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data

def check_success(obs, lift_height, object_name):
    th = 0.01
    pos = obs[f'{object_name}_pos']
    success = (pos[2] >= lift_height - th)
    return success

@hydra.main(config_path="../configs/eval", config_name="default", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    eval_cfg = EasyDict(yaml.safe_load(yaml_config))

    experiment_dir = eval_cfg.load_dir
    benchmark_dict = benchmark.get_benchmark_dict()
    # check that it is a valid experiment
    os.path.exists(experiment_dir)

    # laod the config file inside the experiment dir
    cfg_path = os.path.join(experiment_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))

    dataset_path = os.path.join(cfg.folder, 'libero_gcs.hdf5')
    skill_dataset, shape_meta = get_dataset(
        dataset_path=dataset_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )

    pretrain_model_paths = os.listdir(os.path.join(experiment_dir, 'models'))
    # sort the model paths
    pretrain_model_paths = sorted(pretrain_model_paths, key=lambda x: int(x[:-4].split('_')[-1]))
    pretrain_model_paths = [to_absolute_path(os.path.join(experiment_dir, 'models', x)) for x in pretrain_model_paths]
    pretrain_model_paths = pretrain_model_paths[-1:]
    n_tasks = 1
    result_store = {}
    for pretrain_model_path in pretrain_model_paths:
        num_success = 0.0
        total_trials = 0.0
        # define lifelong algorithm
        algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
        algo.policy.load_state_dict(torch_load_model(pretrain_model_path)[0])

        # prepare datasets from the benchmark
        manip_datasets = []
        descriptions = []
        shape_meta = None

        dataset_path = os.path.join(cfg.folder, 'libero_gcs.hdf5')
        dataset = h5py.File(dataset_path, 'r')['data']
        for key in dataset.keys():
            bm = dataset[key]['meta_data/benchmark_name'][()]
            bm = bm.decode('utf-8')
            bm = str(bm)
            task_index = dataset[key]['meta_data/task_index'][()]
            task_index = int(task_index)
            print(bm, task_index)
            task_suite = benchmark_dict[bm]()
            task = task_suite.get_task(task_index)
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128,
                "controller": "OSC_POSITION",
                "control_delta": False,
            }
            env = OffScreenRenderEnv(**env_args)

            init_state = dataset[key]['meta_data/init_state'][()]
            env.set_init_state(init_state)
            for _ in range(5):
                zero_action = np.zeros(4)
                env.step(zero_action)

            object_name = dataset[key]['meta_data/object_name'][()]
            object_name = object_name.decode('utf-8')
            object_name = str(object_name)
            goal_emb = dataset[key]['goal_emb_dinov2-base'][()]

            obs = env.env._get_observations()
            lift_height = obs[f'{object_name}_pos'][2] + 0.15
            is_success = False
            for _ in range(120):
                data = raw_obs_to_tensor_obs(obs, cfg, goal_emb=goal_emb)
                actions = algo.policy.get_action(data)[0]
                obs, reward, done, info = env.step(actions)
                is_success = check_success(obs, lift_height, object_name)
                if is_success:
                    break
            print("success:", is_success)
            if is_success:
                num_success += 1
            total_trials += 1
            env.close()
            # print pretrained model path in blue color
            print("\033[94m" + f"pretrained model path: {pretrain_model_path}" + "\033[0m")
            # print in red color
            print("\033[91m" + f"success rate: {num_success / total_trials}" + "\033[0m")
            result_store[pretrain_model_path] = {'success_rate': num_success / total_trials, 'num_success': num_success, 'total_trials': total_trials}
    print(result_store)
    with open(os.path.join(experiment_dir, 'eval_results.json'), 'w') as f:
        json.dump(result_store, f)

if __name__ == '__main__':
    main()
