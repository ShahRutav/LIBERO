# mjlab backend

Status: Active
Last verified: 2026-09-21

LIBERO can run its existing environments with `backend="mjlab"`. Physics steps
run through `mjlab.sim.Simulation` and MuJoCo-Warp on one GPU. Task definitions,
OSC actions, observations, cameras, BDDL predicates, and the public reset API
retain their existing interfaces. The default backend remains `"mujoco"`.

Fresh resets with `backend="mjlab"` default to collision-support placement
(`placement_height_mode="collision_bounds"`) with 4 mm clearance
(`placement_height_clearance=0.004`). Explicit arguments override these defaults.
Native MuJoCo keeps legacy placement by default. Loading recorded demonstration
states or injecting a batch reset bank does not adjust those saved states;
native-generated banks must request this policy explicitly when sampling.

Fixture geometry refresh is automatic when batch resets inject fixture poses.
The pinned MuJoCo-Warp revision below includes both the regularization floor
correction and normalized tangential Newton curvature, avoiding a separately
floored `T³` denominator in dense and sparse Hessians. These defaults do not
change the solver selection or contact dimensions. The combined qualification
had zero numerical failures in 10,000 resets, but four NaNs in 5,000 Newton
demo replays; general Newton training remains unqualified.

## Install and run

Use a Python 3.11 environment on a supported NVIDIA GPU machine. From this
repository root:

```bash
uv pip install --python "$(command -v python)" --overrides constraints-mjlab.txt -r requirements-mjlab.txt
python -m pip install -e .
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/outputs/libero-config"
python - <<'PY'
from pathlib import Path
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import torch

task = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
env = OffScreenRenderEnv(
    bddl_file_name=str(Path(get_libero_path("bddl_files")) / "libero_spatial" / f"{task}.bddl"),
    backend="mjlab",
    mjlab_device="cuda:0",
    hard_reset=False,
)
try:
    states = torch.load(
        Path(get_libero_path("init_states")) / "libero_spatial" / f"{task}.pruned_init",
        weights_only=False,
    )
    env.reset()
    obs = env.set_init_state(states[0])
    obs, reward, done, info = env.step(np.zeros(7))
    print(env.check_success(), obs["agentview_image"].shape)
finally:
    env.close()
PY
```

MuJoCo-Warp is pinned to `ShahRutav/mujoco_warp`, branch
`fix/elliptic-normalized-curvature`, commit
`9df8bdb02acdb9e2661bcfa4a7fef265952d35d2`. Install the commit, not the
moving branch name. The explicit override is needed because mjlab 1.6 still
declares MuJoCo/MuJoCo-Warp 3.11 dependencies. This newer stack includes
MuJoCo `3.12.1.dev974703000` and Warp `1.15.0`. The legacy 3.11 Newton
fallback must not be enabled without separately qualifying its private API.

The init file above is supplied by this repository. Initial setup creates the default path config
without prompting when stdin is not a terminal. Use a separate config directory
for each checkout so paths do not point to a different copy of LIBERO.

The first step compiles GPU kernels and can take several minutes. Subsequent
runs reuse Warp's kernel cache. Each environment owns one GPU world. Existing
subprocess wrappers do not become GPU-batched environments; start with one
process per GPU and specify its device explicitly.

## Replay an expert demonstration

Download the one task file from the original LIBERO dataset:

```bash
python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    filename="libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
    local_dir="datasets",
)
PY
export LIBERO_SOURCE_REVISION="$(git rev-parse HEAD)"
python -m scripts.replay_mjlab \
    --dataset datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 \
    --demo demo_0 --cameras --output outputs/mjlab-replay
```

The output directory must be new. The runner relocates recorded asset paths,
loads the recorded XML and first state, then executes every recorded action.
It does not inject reference states during replay. Stored states are compared
at the pre-action index. Cameras are checked for valid nonblack RGB output.
`report.json` records versions, input checksum, source revision, reset errors,
success predicates, physical trajectory differences, and runtime. NPZ files
contain states, object positions, end-effector positions, and agent-view frames.

The command exits nonzero unless both backends finish successfully, resets pass,
end-effector deviation is below 5 mm, and object deviation is below 10 mm.
These distance thresholds compare mjlab against native MuJoCo 3.11. They do not
claim exact equality to the historical simulator used to collect the dataset.
The mixed-unit flattened-state RMSE against recorded data is diagnostic only.
A truncated `--max-steps` run cannot pass the full-replay gate.

## Compatibility decisions

- **Physics:** the original XML timestep, Euler integrator, elliptic cone,
  solver, and other options remain in effect. mjlab's default options are not
  applied over the task model.
- **Contacts:** absent an explicit XML override, set `multiccd="disable"`.
  [MuJoCo 3.8 enabled it by default](https://mujoco.readthedocs.io/en/3.8.0/changelog.html),
  changing the expert grasp on both backends. Disabling it restored success.
- **Controllers:** robosuite 1.4 uses the removed `MjData.qM` interface. A small
  shared compatibility update uses MuJoCo 3.11's `mj_fullM(model, data, matrix)`.
  Gains, action scaling, control rate, and the OSC control law are unchanged.
- **Reset:** soft resets also clear robosuite's incremental gripper command.
  Placement failures retry at most 100 times; other errors propagate immediately.
  Call `reset()` before `set_init_state()` to clear controller history. The
  flattened state itself contains time, qpos, and qvel, not controller history.
- **Fixtures:** fixed-body pose changes made by reset samplers cause the GPU
  model to rebuild before the next integration step. Hard resets and XML resets
  create a new simulator and reconnect the existing robot interfaces.
- **Host mirror:** CPU `mj_forward` supplies Jacobians, mass matrices, contacts,
  predicates, and render geometry. CPU `mj_step` is never used by this backend.
  Controls, applied forces, activation, mocap, equality state, and warm starts
  are uploaded before GPU stepping; integrated state is copied back afterward.
  Transfers, GPU stepping, and readback share Warp's CUDA stream. GPU reset
  clears the complete mjlab state, including solver and collision buffers.

Direct runtime edits of model physics other than fixed-body poses are not
tracked automatically. Recreate the environment after those edits. The single-world compatibility interface keeps CPU-facing observations and
predicates. The separate [training batch API](#privileged-state-training-batch)
uses GPU control, observations, and supported predicates. Neither interface
provides batched rendering.

## Validation

Run all simulator tests on the GPU machine:

```bash
python -m unittest discover -s tests -v
```

The three `test_model_compat.py` cases use only the Python standard library.
`test_gpu_osc_indexed_reset.py` uses CPU tensors only. These can run without a
GPU or model compilation. Simulator integration tests require the GPU pod.
They cover three benchmark initial states, GPU-only integration, fixed fixture
updates, hard-reset references, gripper reset, and bounded error handling.

See [the dated validation result](../results/2026-09-16-mjlab-replay.md). Validation
currently covers one task and one expert demo. The backend hook is shared by
all BDDL domain classes, but the remaining task suites are not yet certified.

## Measure a 50-world batch

Run this on the GPU pod from the fork's root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LIBERO_SOURCE_REVISION="$(git rev-parse HEAD)" \
python -m scripts.benchmark_mjlab_batch \
    --dataset datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 \
    --sizes 1 50 --repeats 3 --output outputs/batch50
```

This benchmark creates one shared-model mjlab Simulation with independent data
for every world. Each world has its own original robosuite OSC controller. It
uploads controls and downloads state in bulk. All worlds start from the same
recorded initial state and receive the same expert actions; feedback control
and integration remain independent. Controller pose and goals are refreshed
from the injected initial state before either backend starts.

Two timing scopes are reported:

- `cpu_osc_and_physics`: original CPU controllers, native forward calculations,
  physics, and the bulk transfers required by the GPU bridge.
- `physics_only_torque_replay`: precomputed native torques, resident on GPU,
  with no controller or per-step host readback. This isolates physics throughput;
  open-loop torque replay does not guarantee task success on another solver.

Neither scope includes cameras, the observation manager, policy inference,
initialization, reset, or final success checks. Each configuration runs a full
warmup episode, then three timed episodes. GPU synchronization brackets timing.
The native CPU baseline processes its worlds serially with one BLAS thread.
This comparison is not a benchmark against a parallel multicore CPU runner.

The benchmark first requires native expert success and agreement between its
controller loop and ordinary LIBERO stepping from the same controller state.
It records final success counts for each measured configuration, even when
numerical differences change task outcomes. Non-finite state invalidates a run.

`batch_seconds_median` is the actual wall-clock latency of all episodes.
`amortized_episode_seconds` divides that value by the number of worlds. The
latter measures throughput cost, not how quickly one world finishes while
running in the batch. `amortized_control_step_ms` divides again by 103 actions.

This is an isolated benchmarking bridge. The regular `OffScreenRenderEnv`
interface still owns one world, and this script does not add batched rendering
or a production vector-environment API.

See the [1-versus-50-world measurements](../results/2026-09-16-mjlab-batch50.md)
for throughput, latency, and per-repeat success counts.

For larger physics-only sweeps, add `--backends mjlab
--scopes physics_only_torque_replay` and increase `--sizes` gradually. Use
`--scopes cpu_osc_and_physics` to measure the retained CPU controller separately.
The default still measures both scopes. Reports include device-wide used VRAM
after trials and the process's peak host RSS. VRAM is a snapshot, not a peak;
it includes other processes if the device is shared. Use an otherwise idle GPU.

Reserve at least 30% of VRAM and apply a 20% margin to projected memory before
trying another batch size. Stop if throughput plateaus or the next size exceeds
that budget. Confirm a candidate size in a fresh process because allocator
caches can retain memory across configurations. These checks establish a tested
operating range, not the allocation-failure limit or a camera-enabled limit.
See the [capacity sweep and operating budget](../results/2026-09-17-mjlab-capacity.md)
for measured counts on the bowl-to-plate task.

## Benchmark the GPU controller branch

The `libero-mjlab` branch includes a batched GPU implementation
of the demo's fixed delta `OSC_POSE` controller and Panda gripper. It keeps
controller goals, Jacobians, inertia, feedback, and torque calculations on the
GPU between steps. It retains robosuite's action scaling, torque limits,
position limits, orientation-goal convention, nullspace target, pseudoinverse
cutoff, and gripper accumulation at every physics substep.

Run on a free GPU in the prepared pod container, from this repository:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD" LIBERO_SOURCE_REVISION="$(git rev-parse HEAD)" \
python -m scripts.benchmark_gpu_osc \
  --dataset /work/datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 \
  --output /work/gpu-osc-benchmark --sizes 1 50 100 256 --repeats 3
```

The output directory must be new. The runner first checks native replay success
and exact agreement with ordinary LIBERO stepping. It then compares GPU
controller math against robosuite at 103 matched native trajectory states.
GPU-derived Jacobians and inertia are checked separately against native data.
Each requested batch receives a full warmup followed by synchronized timing
replicates and original LIBERO success checks. `report.json` records source
hashes, memory use, final EEF/object distances, and all trial outcomes.

The controller math uses float64, matching robosuite. MuJoCo-Warp physics remains
float32. Axis-angle rotation uses the equivalent Rodrigues formula, with small
rounding differences from robosuite's float32 quaternion-to-matrix conversion.
Torch's matrix factorizations still synchronize with the host; no simulation
state is downloaded during control. This first port does not promise a fully
asynchronous CUDA graph.

Only fixed delta `OSC_POSE` with no interpolation or orientation limits and a
Panda gripper is supported. Other configurations raise `NotImplementedError`.
The benchmark supports independent per-world controller state but replays one
shared task, initial state, and action sequence for comparison. Cameras,
observations, policy inference, reset/setup, and final predicate evaluation
remain outside the timer. `OffScreenRenderEnv` remains the existing single-world
compatibility interface. The benchmark retains its existing scope; the separate
training batch API below adds privileged observations, rewards, and indexed resets.

See the [measured GPU controller results](../results/2026-09-17-gpu-controller.md)
for success, throughput, memory headroom, and the tested batch range.


## Privileged-state training batch

Use `libero.libero.envs.mjlab_batch.LiberoBatchEnv` for independent GPU worlds
with privileged observations. Run model preparation, construction, stepping,
and conversion on the GPU pod. The calling application owns the prepared
native `ControlEnv` and must load the exact source XML before construction.
The existing `backend="mjlab"` single-world API remains unchanged.

The constructor accepts:

- `env`: prepared native task and controller metadata.
- `initial_states`: finite `[S, 1+nq+nv+na]` reset candidates.
- `num_envs`, `device`, `horizon`, and `seed`.
- Optional `initial_body_pos[S,nbody,3]` and
  `initial_body_quat[S,nbody,4]` banks, aligned with reset candidates.

Certify model compatibility before combining candidates. Matching array
sizes is insufficient. The current conversion permits explicitly identified
fixed-fixture pose differences; other XML differences require separate models
or a new validated extension. The batch expands model body-pose fields per
world and rebuilds simulation graphs before using those fields.

The training interface is:

- `reset(env_ids=None, state_ids=None)` resets selected worlds and returns all
  observations. It clears physics, controller goals, gripper accumulation,
  previous actions, and episode bookkeeping for those worlds.
- `step(actions[N,7])` returns observations, binary rewards, terminations,
  truncations, and success/invalid-state information. It does not auto-reset.
  The caller must retain terminal observations before resetting done worlds.
- `observe()` uses the same observation builder as demonstration conversion.
- `set_reset_bank(...)` replaces compatible candidates without resetting live
  worlds. Supply body-pose banks when replacing a fixture-aware bank.
- `restore_recorded_state(state, previous_action)` reconstructs a one-world
  demonstration prefix. It advances persistent controller memory, injects the
  recorded physical state, and produces the shared observation.
- `export_state()` returns physical, object, controller, and model-pose tensors.
  `close()` releases the batch; the caller closes its native environment.

Observation dimensions come from the model and ordered schema. Fields include
all qpos/qvel/act, named objects and fixtures, their world poses and velocities,
end-effector state, controller goals, gripper accumulation, and previous action.
The schema specifies orientation packing and quaternion ordering. Check its
hash when loading converted data or checkpoints.

`action_spec` records control/physics timing, substeps, input/output scaling,
gains, nullspace reference, coupling, position limits, gripper speed, and
clipping/accumulation semantics. Compare the full specification before reuse;
seven action dimensions alone do not establish compatibility.

Only the existing fixed delta OSC_POSE/Panda controller and conjunctions of
object `On` predicates are supported. Site predicates and other goal types
fail explicitly. `On` uses matching contact geometry, vertical ordering, and
the native 3 cm horizontal-center criterion. Sparse reward is one on success;
success and invalid states terminate, while horizon expiry truncates.

Run the gate inside the pinned pod container from this repository root:

```bash
python -m scripts.validate_mjlab_training_batch \
  --dataset /source/task.hdf5 \
  --output /work/outputs/batch-gate.json --steps 8
```

The gate uses heterogeneous stable starts and fixture poses. It checks matched
controller inputs, world permutations, one-versus-many stepping, partial resets,
positive/negative native predicates, observation parity, and fast prefix memory.
It records physical-field errors and bounds. `--diagnostic` measures differences
without accepting rollout equivalence and always writes `passed: false`.
Failures preserve a compact NPZ reproduction bundle and available JSON metrics.

The accepted short-rollout bounds are `1e-5` absolute for position, orientation,
and other nonvelocity fields, and `1e-3` for velocity fields. Reset isolation
and fast-prefix memory comparisons are exact. These bounds are not a claim of
deterministic long contact-rich trajectories. See the
[training-batch validation result](../results/2026-09-17-mjlab-training-batch.md)
for measured errors and the rejected mid-grasp reproducibility test.

Keep durable datasets and reports in the experiment's registered S3 locations.
Use task-owned pod staging only while needed. Remove staging after verifying
publication; retain a small named debugging bundle if needed. Do not clean
shared source caches or unrelated runs.

For a native/GPU success-label discrepancy, call the diagnostic helper
`predicate_parity(batch, diagnostics_path=..., context=...)` from the gate
module. It writes world/time, goal positions, matching native/GPU contact pairs,
distances, and state before raising. `LIBERO_PREDICATE_DIAGNOSTICS` supplies a
JSONL path when the caller cannot pass one directly.

An aggregate audit can use `raise_on_mismatch=False` to retain collision-label
differences and continue replay. This returns a dictionary with native/GPU
labels, mismatch records, and a separate logical parity result. Logical
predicate errors on identical GPU contact inputs always raise. Report collision
mismatch counts explicitly; do not label a run with mismatches as exact native
predicate parity or change reward tolerances just to silence the diagnostic.
