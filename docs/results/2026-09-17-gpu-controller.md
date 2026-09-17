# 2026-09-17 GPU controller

Status: Complete
Last verified: 2026-09-17
Experiment: [LIBERO mjlab in the Sim2Real workspace](../../../../docs/experiments/libero-mjlab.md)

## Results

At 50 worlds, moving OSC to the GPU reduces batch time from 77.868 s to
10.496 s, a 7.42× throughput improvement over the retained CPU controller.
Success is 140/150 replays (93.3%), compared with 139/150 for the retained
[CPU controller benchmark](2026-09-16-mjlab-batch50.md). The matched native
reference succeeds. These repeats do not establish a change in success probability.

Each row is the median of three complete 103-action episodes after warmup.
Amortized time divides batch duration by the number of worlds; every world
in the batch still takes the full batch duration to finish.

| Worlds | Batch seconds, median (range) | Seconds/environment, amortized | Device memory, GiB | Successes / 3 batches |
|---:|---:|---:|---:|---:|
| 1 | 8.564 (8.556–8.606) | 8.5639 | 0.66 | 3 / 3 |
| 50 | 10.496 (10.446–10.508) | 0.2099 | 1.07 | 140 / 150 |
| 100 | 11.131 (11.116–11.168) | 0.1113 | 1.57 | 275 / 300 |
| 256 | 14.063 (14.015–14.128) | 0.0549 | 2.98 | 728 / 768 |
| 512 | 19.539 (19.530–19.645) | 0.0382 | 5.35 | 1428 / 1536 |
| 1024 | 30.501 (30.484–30.537) | 0.0298 | 10.04 | 2857 / 3072 |
| 2048 | 52.197 (52.180–52.357) | 0.0255 | 19.64 | 5743 / 6144 |
| 2560 | 64.156 (64.041–64.230) | 0.0251 | 24.33 | 7177 / 7680 |

The one-world port is slower than the original CPU controller bridge: 8.564 s
versus 3.667 s. GPU factorization and launch overhead dominate this small batch.
At 100 worlds, the GPU controller takes 11.131 s versus 153.388 s for the
retained CPU controller, a 13.78× improvement. That 100-world CPU comparison
uses [one timed episode](2026-09-17-mjlab-capacity.md) after warmup; the 50-world CPU comparison uses three.

Final-position errors are not uniformly small. Across the 2048-world trials,
the maximum EEF distance from native is 16.52 mm and maximum object distance
is 146.21 mm. These maxima cover every world, including unsuccessful replays.
The report does not identify which world produced each maximum. Success
counts are the acceptance evidence; this is not a blanket trajectory-closeness
guarantee. The 50-world maxima are 4.66 mm for the EEF and 7.53 mm for objects.
These distances compare final positions, not full trajectories.

## CPU and GPU throughput

These rates include feedback control and physics, with cameras, observation
construction, policy inference, reset, and setup excluded. Throughput is summed
across worlds: `worlds * 103 / median_batch_seconds` control steps/s. Each
control step contains 25 physics steps. Rates count all steps, including those
in unsuccessful replays; success counts are reported separately above.

| Physics / controller | Worlds | Aggregate control steps/s | Aggregate physics steps/s |
|---|---:|---:|---:|
| Native CPU / CPU | 1 | 57.19 | 1,430 |
| Native CPU / CPU, serial | 50 | 52.19 | 1,305 |
| GPU / CPU | 1 | 28.09 | 702 |
| GPU / CPU | 50 | 66.14 | 1,653 |
| GPU / CPU | 100 | 67.15 | 1,679 |
| GPU / GPU | 1 | 12.03 | 301 |
| GPU / GPU | 50 | 490.68 | 12,267 |
| GPU / GPU | 100 | 925.38 | 23,134 |
| GPU / GPU | 256 | 1,874.95 | 46,874 |
| GPU / GPU | 512 | 2,699.02 | 67,475 |
| GPU / GPU | 1,024 | 3,458.03 | 86,451 |
| GPU / GPU | 2,048 | 4,041.29 | 101,032 |
| GPU / GPU | 2,560 | 4,109.98 | 102,749 |

The native CPU baseline is a serial runner with one Torch/BLAS/OpenMP thread,
not a multicore subprocess benchmark. Native CPU throughput has not been
measured at 1,024–2,560 worlds; no CPU timings are extrapolated to those sizes.
At the matched 50-world count, GPU physics plus GPU control delivers 9.40×
the native serial CPU throughput and 7.42× the GPU-physics/CPU-controller
throughput. The higher-count GPU results use greater batching and should not
be described as matched-count CPU speedups.

CPU and hybrid rates are calculated from the accepted
[1/50-world reports](2026-09-16-mjlab-batch50.md) and the
[100-world hybrid report](2026-09-17-mjlab-capacity.md). All rates use three-run
median timings except the 100-world hybrid baseline, which has one timed run.
This table adds derived rates from existing measurements; it introduces no
new simulation runs.

## Capacity decision

Use **2048 worlds per L40S** as a practical starting point for this headless
workload. It completes a batch in 52.197 s, amortized 25.49 ms per environment,
at 101,032 environment physics steps/s. Success is 5743/6144 replays (93.47%).
It uses 20,112 MiB (19.64 GiB), leaving room for workloads excluded here.

**2560 is the largest tested safe batch**, at 64.156 s and 24,914 MiB
(24.33 GiB). Success is 7177/7680 (93.45%). Throughput rises only 1.70% to
102,749 physics steps/s for 23.88% more memory. The 2560 run used physical GPU2;
all smaller sizes used GPU1, both L40S cards. A small cross-device difference
is not enough to establish an exact optimum. The useful upper range tested is
2048–2560, with 2048 offering the better memory/latency tradeoff.

Each candidate had to fit below 70% of CUDA-reported memory after adding 20%
to the linear projection from an already measured batch. The next 256-world
increment, 2816, projects to 32,887 MiB with this margin, exceeding the
31,820 MiB budget. It was not launched. No test intentionally approached an
out-of-memory failure. This is a conservative measured bound, not the GPU's
absolute hardware maximum.

For comparison, the [physics-only capacity sweep](2026-09-17-mjlab-capacity.md)
also found little additional throughput from 2048 to 2560. Camera rendering,
policy inference, observation storage, and heterogeneous tasks require their
own capacity check. The original CPU-controller bridge remains a separate
performance case.

## Scope and validation

The GPU controller branch ports the demo's fixed delta OSC_POSE controller and
Panda gripper to batched GPU tensors. Controller state and physics state stay on
GPU throughout each episode. Original LIBERO predicates evaluate the final
states after timing. The ordinary single-world environment API is unchanged.

The native reference succeeds and its controller loop matches ordinary LIBERO
stepping exactly. At 103 native trajectory states, matched-input GPU torque
error is at most 2.19e-6 Nm. Goal position error is zero, goal orientation matrix
error is 1.55e-8, and gripper control error is below 8.95e-10. GPU-derived
Jacobian error is at most 3.19e-7, inertia error 5.21e-6, and bias force error
2.13e-5 in the respective array units. These are maximum absolute elementwise
errors, not trajectory distances. The test uses original robosuite as its
reference and checks GPU dynamics extraction separately from controller math.

The port retains action clipping/scaling, torque clipping, position limits,
nullspace torque, gripper accumulation, and the NumPy pseudoinverse cutoff of
1e-15. It uses float64 controller arithmetic over float32 GPU physics. Its
Rodrigues rotation is mathematically equivalent to robosuite's quaternion
conversion, with the small rounding discrepancy measured above. Unsupported
controller modes fail explicitly.

## Provenance

- Branch: `codex/libero-gpu-controller`.
- Tested controller source: `b05ed72fd6b83aaa6aa46daa6c5c42ec198f68d5`.
- Runner source for 2048/2560: `a3853c95c2fcf7a24fb7b24aaa962e9caf0129b4`,
  adding stricter parity gates without changing controller math.
- Controller: [gpu_osc.py](../../libero/libero/envs/gpu_osc.py).
- Runner: [benchmark_gpu_osc.py](../../scripts/benchmark_gpu_osc.py).
- Procedure: [GPU controller benchmark](../runbooks/mjlab-backend.md#benchmark-the-gpu-controller-branch).
- Task: `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
- Demo: `demo_0`, 103 actions, 25 physics substeps per action.
- Dataset SHA-256: `75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3`.
- Hardware: NVIDIA L40S on the shared GPU pod. CUDA reports 45,458 MiB
  usable device memory; nvidia-smi reports 46,068 MiB. Safety budgets use the
  smaller CUDA total. Each batch uses one GPU.
- Stack: mjlab 1.6.0, MuJoCo and MuJoCo-Warp 3.11.0, robosuite 1.4.0,
  Torch 2.7.1, Warp 1.16.0, Python 3.11, bddl 1.0.1, gym 0.25.2.
- One Torch/BLAS/OpenMP thread. Each configuration has one full warmup.
- GPU synchronization brackets each complete-episode timer. No cameras,
  observation construction, policy inference, resets, or success checks are timed.

## Limitations

Every world runs the same task, initial state, and expert action sequence.
Worlds have independent controller goals and dynamics. Success differences
may reflect nondeterministic contact reductions and numerical differences; these
repeats do not estimate suite-wide generalization.

The GPU tensor port removes per-substep state readback and serial CPU control.
Torch inverse and pseudoinverse operations still synchronize with the host.
It also performs a full GPU forward pass before control. These overheads limit
single-world latency and remain optimization opportunities. This implementation
does not use CUDA graph capture for the controller.

The timing excludes cameras and policy memory. Memory reported by the runner
is device-wide usage after an episode, not an allocator-traced high-water mark.
The practical batch limit depends on task geometry, collision workload,
observation resolution, and policy size. The experiment keeps substantial VRAM
headroom and does not search for an out-of-memory failure.

## Artifacts

All four accepted processes exited with code 0. All reported simulation states
were finite. Logs contain package startup/deprecation notices and no solver
instability, overflow, or non-finite-state warnings.

- GPU1 pod root: `/data/rutavms/libero-gpu-controller-20260917/`.
- Accepted runs there: `batch1-256-v2`, `batch512-1024`, and `batch2048`.
- GPU2 pod root: `/data/rutavms/libero-gpu-controller-2560-20260917/`.
- Accepted run there: `batch2560`.
- Local report/log copies: ignored `artifacts/libero-gpu-controller-20260917/`
  in the Sim2Real workspace.
- Prepared runtime snapshot: `libero-gpu-controller:20260917`, image
  `c623b8c44878914519bf9883aa33cdbcbd78305e5428cf66796d054d82c15323`.

Each JSON report includes runner/controller source hashes, the reference
success/parity checks, three trial durations, per-world success flags, final
position-error maxima, and device-wide memory snapshots. Initial dependency
setup attempts and the one-world smoke run are excluded from the table.
