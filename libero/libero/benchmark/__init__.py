import abc
import os
import csv
import glob
import random
import torch
import pathlib

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str
    goal_language: str
    instructions: str


def grab_language_from_filename(x):
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        elif x.startswith("RW"):
            language = " ".join(x.split("_")[1:])
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]

p=pathlib.Path(__file__).parent.resolve()
filepath=os.path.join(p, "task_annotations_multiple_ts.csv")
instruction_mapping = {}
max_instructions=8
with open(filepath, 'r') as data:
    for line in csv.DictReader(data):
        instruction_mapping[line['task_id']] = {'instructions': [], 'goal_language': []}
        for key in line.keys():
            if key == 'task_id':
                continue
            else:
                if key.startswith('inst'):
                    inst_list = line[key].split('\n')
                    # check if a value in inst_list is not empty
                    for inst_line in inst_list:
                        if inst_line == '' or inst_line == ' ':
                            inst_list.remove(inst_line)
                    instruction_mapping[line['task_id']]['instructions'].append(inst_list)
                elif key.startswith('gl'):
                    instruction_mapping[line['task_id']]['goal_language'].append(line[key])
                else:
                    raise ValueError(f"Unknown key {key}")

libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
    "libero_100",
    "debug",
    "rw_all"
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
            goal_language=instruction_mapping[task]['goal_language'] if task in instruction_mapping.keys() else None,
            instructions=instruction_mapping[task]['instructions'] if task in instruction_mapping.keys() else None
        )

task_orders = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        if self.name == "libero_90" or self.name == "libero_100" or\
                self.name == "rw_all" or self.name == "debug":
            self.tasks = tasks
        else:
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states(self, i):
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        init_states = torch.load(init_states_path)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs

    def set_gl_embs(self, gl_embs):
        self.gl_embs = gl_embs

    def set_inst_embs(self, inst_embs):
        self.inst_embs = inst_embs

    def set_task_tokens(self, task_tokens):
        self.task_tokens = task_tokens

    def set_inst_tokens(self, inst_tokens):
        self.inst_tokens = inst_tokens

    def set_visual_task_specifications(self, vis_task_spec):
        self.vis_task_spec = vis_task_spec

    def set_ag_task_specs(self, ag_task_specs):
        self.ag_task_specs = ag_task_specs

    def set_ai_task_specs(self, ai_task_specs):
        self.ai_task_specs = ai_task_specs

    def get_task_token(self, i):
        return {k: v[i] for k,v in self.task_tokens.items()}

    def get_visual_task_specification(self, i):
        return self.vis_task_spec[i]

    def get_gl_emb(self, i):
        return self.gl_embs[i]

    def get_inst_emb(self, i):
        return self.inst_embs[i]

    def get_inst_token(self, i):
        inst_token = {}
        for k,v in self.inst_tokens.items():
            inst_token[k] = v[i]
        return inst_token

    def get_ag_task_spec(self, i):
        return self.ag_task_specs[i]

    def get_ai_task_spec(self, i):
        return self.ai_task_specs[i]

class RW_CLASS(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        self.tasks = tasks
        self.n_tasks = len(self.tasks)

@register_benchmark
class RW_ALL(RW_CLASS):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "rw_all"
        self._make_benchmark()

@register_benchmark
class DEBUG(Benchmark):
    """
    This is a debugging benchmark that includes only 1 task.
    """
    def __init__(self, task_order_index=0):
        super().__init__()
        self.name="debug"
        self._make_benchmark()

@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()
