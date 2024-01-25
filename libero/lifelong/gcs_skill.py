import os

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

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
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

def create_experiment_dir(cfg):
    if cfg.experiment_dir is None:
        # create experiment dir
        rgb_modalities = cfg.data.obs.modality.rgb
        low_dim_modalities = cfg.data.obs.modality.low_dim
        # convert to string with '_'
        rgb_modalities = "_".join(rgb_modalities)
        low_dim_modalities = "_".join(low_dim_modalities)

        # join all modalities
        modalities = "_".join([rgb_modalities, low_dim_modalities])
        modalities = modalities[1:]
        dataset_name = cfg.dataset_n[:-5]
        cfg.experiment_dir = to_absolute_path(f"experiments/{dataset_name}_{modalities}_s{cfg.seed}")
    Path(cfg.experiment_dir).mkdir(parents=True, exist_ok=True)
    # check number of directories starting with run_ in the experiment dir
    run_dirs = [d for d in os.listdir(cfg.experiment_dir) if d.startswith("run_")]
    index = len(run_dirs) + 1
    cfg.experiment_dir = os.path.join(cfg.experiment_dir, f"run_{index:03d}")
    Path(cfg.experiment_dir).mkdir(parents=True, exist_ok=True)
    return cfg.experiment_dir

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    pp.pprint("Available algorithms:")
    pp.pprint(get_algo_list())

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None

    dataset_path = os.path.join(cfg.folder, cfg.dataset_n)
    skill_dataset, shape_meta = get_dataset(
        dataset_path=dataset_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    manip_datasets.append(skill_dataset)
    task_embs = [np.zeros_like((1,))]*len(manip_datasets)
    datasets = [
        SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
    ]
    n_tasks = len(datasets)

    val_dataset_path = os.path.join(cfg.folder, cfg.val_dataset_n)
    val_skill_dataset, _ = get_dataset(
        dataset_path=val_dataset_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=False,
        seq_len=cfg.data.seq_len,
    )
    val_dataset = SequenceVLDataset(val_skill_dataset, np.zeros_like((1,)))

    # prepare experiment and update the config
    cfg.experiment_dir = create_experiment_dir(cfg)
    cfg.experiment_name = cfg.experiment_dir.split("/")[-2]
    cfg.shape_meta = shape_meta

    if cfg.use_wandb:
        wandb.init(project="libero_gcs", config=cfg)
        wandb.run.name = cfg.experiment_name
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", "epoch")
        wandb.define_metric("val/*", "epoch")

    # define lifelong algorithm
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
    if cfg.pretrain_model_path != "":  # load a pretrained model if there is any
        try:
            algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path)[0])
        except:
            print(
                f"[error] cannot load pretrained model from {cfg.pretrain_model_path}"
            )
            sys.exit(0)

    print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    GFLOPs, MParams = compute_flops(algo, datasets[0], cfg)
    print(f"[info] policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)
    for i in range(n_tasks):
        print(f"[info] start training on task {i}")
        algo.train()
        t0 = time.time()
        algo.learn_one_task(
            datasets[i],
            val_dataset=val_dataset,
            task_id=i, result_summary=None,
            logger=wandb if cfg.use_wandb else None,
        )
        t1 = time.time()

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
