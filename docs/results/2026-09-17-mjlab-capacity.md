# 2026-09-17 mjlab capacity

Status: Complete
Last verified: 2026-09-17
Experiment: [LIBERO mjlab in the Sim2Real workspace](../../../../docs/experiments/libero-mjlab.md)

## Result

Use 2,048 worlds as a practical physics-only batch on this L40S and task.
It delivers 359,411 environment physics steps/s at 19.8 GiB used VRAM. The
largest tested batch, 2,560 worlds, also completes within the memory budget,
but adds only 1.35% throughput for 24.2% more used VRAM. Stop the sweep there.
This is a measured operating range, not an absolute hardware maximum.

With the original CPU controllers, 100 worlds barely improve throughput over
50. The [separate GPU controller branch](https://github.com/ShahRutav/LIBERO/tree/codex/libero-gpu-controller)
measures feedback-control scaling. Its timings must not be replaced by these
physics-only values.

## Scope and interpretation

LIBERO has no fixed 50-world limit. Its native `SubprocVectorEnv` creates one
worker for each supplied environment factory. Our benchmark creates one mjlab
Simulation with a configurable number of independent worlds sharing one model.
This does not extend the regular single-world environment to batched cameras.
All worlds in this experiment use the same task model and recorded initial
state. This is not a test of batching different LIBERO task geometries together.

The CPU controller bridge downloads state after every physics step, computes
forward dynamics and OSC control serially on CPU, and uploads controls before
the next GPU step. There are 25 such round trips per expert action. Both the
dependent synchronization and CPU computations limit throughput; the previous
comparison did not isolate their individual costs.

This sweep measures physics-only torque replay separately from feedback control.
It uses the same task, demo, stack, reset initialization, finite-state check, and
native-loop parity gate as the [50-world benchmark](2026-09-16-mjlab-batch50.md).
Each episode has 103 actions and 2,575 integration steps. Cameras, observations,
policy inference, reset, and initialization are outside timing. GPU timers are
synchronized. Results cover one L40S with 45,458 MiB total reported by the CUDA
memory API (46,068 MiB in `nvidia-smi`). Memory budgets use the CUDA API value.

## Physics-only measurements

One complete warmup and three measured episodes per batch size. These are
precomputed native torques, not a closed-loop GPU controller.

| Worlds | Median batch seconds | Environment physics steps/s | Used VRAM, MiB | Success counts, three repeats |
|---:|---:|---:|---:|---|
| 100 | 3.068 | 83,928 | 1,536 | 89, 81, 86 |
| 256 | 3.861 | 170,729 | 2,976 | 208, 216, 219 |
| 512 | 5.317 | 247,938 | 5,440 | 443, 429, 437 |
| 1,024 | 8.293 | 317,960 | 10,324 | 868, 876, 877 |
| 2,048 | 14.673 | 359,411 | 20,244 | 1,742, 1,724, 1,747 |
| 2,560 | 18.097 | 364,260 | 25,140 | 2,175, 2,144, 2,147 |

VRAM is a device-wide snapshot after the trials on an otherwise idle GPU, not
a measured allocation peak. Host peak RSS reached 4,632 MiB through 1,024 worlds.
The first four configurations share a process, so retained allocator caches
may contribute. Larger candidate sizes are checked in fresh processes.
The 2,560-world process peaks at 8,001 MiB host RSS. All 18 measured physics
trials have finite state and complete without GPU allocation failures.

Physics batch timing ranges in seconds: 100, 3.061–3.088; 256, 3.850–3.861;
512, 5.303–5.354; 1,024, 8.261–8.335; 2,048, 14.651–14.695;
2,560, 18.076–18.100. These are repeat ranges, not confidence intervals.

## Retained CPU controller

At 100 worlds, one measured full replay after one warmup took 153.388 seconds,
or 1.534 seconds per environment amortized. Success was 94/100. This is a single
timing replicate. Compared with the prior 50-world median of 1.557 seconds per
environment, doubling the batch barely improves throughput with CPU controllers.
It doubles the batch latency. This supports moving feedback control to GPU
rather than merely increasing world count in the CPU bridge.

## Operating budget

Reserve 30% of GPU memory and add 20% to projected usage before increasing batch
size. These are operational margins, not a guarantee for every task. Between
512 and 1,024 worlds, measured usage grows by 9.54 MiB/world. A linear projection
puts 2,048 worlds at about 20,092 MiB, or 24,110 MiB with the projection margin,
below the 31,820 MiB budget. Actual 2,048-world usage is 20,244 MiB. Scaling all
of that usage conservatively to 2,560 worlds and adding 20% gives 30,366 MiB,
still inside the budget. A guarded fresh-process 2,560-world check ran only
after the 2,048-world run completes successfully and improves throughput by at
least 3% over 1,024 worlds. The measured improvement is 13.0%.

At 2,560 worlds, measured usage is 25,140 MiB, about 55.3% of CUDA-reported
total. Projecting to 2,816 with the same 20% margin gives 33,185 MiB, exceeding
the 31,820 MiB budget. The small observed throughput gain is a second reason
not to increase further. No out-of-memory failure was deliberately attempted.

Throughput, not just allocation capacity, determines the useful batch size.
Stop when additional worlds no longer improve throughput or when the next
projection exceeds the budget. Do not deliberately provoke an out-of-memory
failure. Different task geometry, contacts, cameras, policies, and GPU controller
buffers can change the usable count.

## Provenance

- Source: `48cac80`, [benchmark runner](../../scripts/benchmark_mjlab_batch.py).
- Input SHA-256: `75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3`.
- Task: `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `demo_0`.
- Runtime: mjlab 1.6.0, MuJoCo/MuJoCo-Warp 3.11.0, robosuite 1.4.0, Torch 2.7.1.
- Thread limits: Torch, OpenBLAS, OMP, MKL each 1.
- Pod artifacts: `/data/rutavms/libero-mjlab-20260916/capacity-*-20260917`.
- Local ignored evidence: `artifacts/libero-mjlab-capacity-20260917/` in Sim2Real.
- Procedure: [batch benchmark](../runbooks/mjlab-backend.md#measure-a-50-world-batch).

The first sweep completed with the original per-world robosuite cleanup. That
cleanup was slow because `MjSim.free()` invokes `gc.collect()` for each mirror.
Source `03a97d35446c2e75765bfc3573e25cb082cd1be9` releases all mirror references and
collects once. This change is outside the timed region and is used for the
fresh-process larger-batch checks.
