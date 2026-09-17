# 2026-09-17 privileged-state training batch

Status: Complete
Last verified: 2026-09-17
Experiment: [LIBERO mjlab in the Sim2Real workspace](../../../../docs/experiments/libero-mjlab.md)

## Result

The privileged-state batch passed its GPU correctness gate for bowl-to-plate.
Four heterogeneous worlds and a separate one-world instance agreed within the
stated short-rollout bounds. Indexed reset isolation and matched-state
controller permutation were exact. Native and GPU predicates agreed on three
negative recorded states and one positive final-placement state.

This validates the environment interface for the first training task. It does
not establish policy success, PPO improvement, suite-wide predicate support,
or training throughput at the earlier controller benchmark's batch sizes.

## Provenance

- Task: `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
- Source: `demo_0` from the existing task dataset, 103 actions.
- Dataset SHA-256: `75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3`.
- GPU-tested source: `9eb3936`, branch `codex/libero-batch-training`.
- CPU-tested explicit controller-restoration dtype fix: `2852e1f`.
- GPU: physical GPU0, NVIDIA L40S, standing shared pod.
- Runtime: `libero-gpu-controller:20260917`, image
  `c623b8c44878914519bf9883aa33cdbcbd78305e5428cf66796d054d82c15323`.
- Interpreter: `/opt/conda/envs/molmospaces/bin/python`.
- Launcher: `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`; inherited pinned
  MuJoCo, MuJoCo-Warp, MJLab, Torch, and robosuite stack.
- Model: shared topology with independent fixture body-pose fields.
- Stable starts: original initial physical state, distinct first-arm-joint
  offsets of `0`, `0.002`, `0.004`, and `0.006` radians, and distinct fixture
  position offsets. These are controlled isolation inputs, not an evaluation
  distribution.
- Actions: eight heterogeneous random actions per world, seed 17, each
  component in `[-0.1, 0.1]`.
- Timing: 25 physics substeps per control action.
- Procedure: [training batch runbook](../runbooks/mjlab-backend.md#privileged-state-training-batch).
- Runner: [validate_mjlab_training_batch.py](../../scripts/validate_mjlab_training_batch.py).

The earlier [GPU-controller benchmark](2026-09-17-gpu-controller.md) supplies
controller and capacity context. It uses shared initial states and action
sequences and excludes observations, resets, and predicates from timing.
These new correctness checks do not replace those timings or extend their
capacity claim to PPO training.

## Accepted measurements

The final gate exited zero. Values below are maximum absolute array-element
errors over eight control steps. qpos and qvel combine joint coordinates with
different physical units; named object positions are reported separately.

| Comparison | qpos | qvel | Named object position | Any explicit velocity field |
|---|---:|---:|---:|---:|
| Repeated same-order rollout | 2.38e-7 | 1.94e-6 | 0 m | 8.64e-7 |
| Permuted worlds and actions | 7.29e-9 | 1.37e-5 | 0 m | 1.37e-5 |
| One world versus its batch member | 2.33e-10 | 7.45e-9 | 0 m | 7.45e-9 |

The gate applies `1e-5` absolute bounds to nonvelocity fields and `1e-3` to
velocity fields. These separate bounds accommodate measured resting-contact
velocity variation without weakening position or orientation checks.
A prior stable-start attempt with one mixed-unit `1e-5` threshold failed on
a resting object's angular velocity at `1.365e-5`; position errors remained
below a micrometer. Each field's error and bound is retained in the JSON.

Additional checks passed:

- Matched-state controller torque under permutation: exact equality.
- Repeated reset observation: exact equality.
- Partial reset: every exported tensor for untouched worlds unchanged.
- Recorded predicate snapshots: `[false, false, false, true]` on both backends.
- Offline versus online observation builder: exact equality.
- Observation size for this model: 229, derived from its schema.
- Observation schema hash:
  `231a5d3aa4fe14a4b291b7649a6d9b935780cc29b067d8fe6f991ff90cc7b3a1`.

Matched native controller checks covered the first eight expert actions:

| Quantity | Maximum absolute error |
|---|---:|
| Torque | 4.67e-7 Nm |
| Goal position | 0 m |
| Goal orientation matrix | 1.71e-9 |
| Gripper control | 8.95e-10 |
| GPU end-effector position | 1.67e-7 m |
| GPU Jacobian | 2.12e-7 |
| GPU inertia | 3.38e-6 |
| GPU bias force | 9.97e-6 |

These are matched-state controller comparisons. They do not imply identical
long native/GPU trajectories.

## Prefix reconstruction and action contract

The converter now reconstructs controller memory without simulating between
recorded states. It uses the same goal-update function as live OSC, then repeats
the gripper accumulator update for all physics substeps. It injects the next
recorded physical state afterward.

The gate compared this path with complete live control steps for eight actions.
Goal position, orientation, and gripper memory were exactly equal. The fast
path left qpos, qvel, and simulation time unchanged. Repeated accumulator
additions preserve float64 rounding; a single multiplied increment is avoided.
No conversion speedup is claimed without a timed conversion measurement.

The instance action specification includes timing, substeps, scales, gains,
nullspace reference, coupling, position limits, gripper speed, clipping,
gripper accumulation, and the retained orientation goal for zero rotation.
Its values come from the prepared native controller. The conversion freezes
this full contract rather than accepting data solely because actions have
seven components.

## Failed contact-rich equivalence attempt

The first gate sampled physical states across the expert trajectory, reset
controller memory, and applied random actions. Some states were mid-grasp.
This deliberately omitted expert-prefix controller memory, so the subsequent
release/contact motion was unsuitable for a strict replay-equivalence test.

Eight-step diagnostics found variability even when replaying worlds in the
same order from identical reset observations:

| Comparison | Bowl position error | Bowl angular-velocity error |
|---|---:|---:|
| Same-order repeat | 0.471 mm | 3.270 rad/s |
| Permuted worlds | 0.617 mm | 0.891 rad/s |

The large velocity errors are consistent with contact-sensitive divergence;
the experiment did not isolate a single numerical cause. Exact controller
permutation and exact reset-state checks provide separate indexing evidence.
The gate was changed to controlled stable starts for short dynamics checks,
while retaining contact-rich recorded states for native predicate parity.
The rejected diagnostics remain available and are not counted as a passing
long-contact reproducibility test.

One intermediate harness also failed when reducing an empty activation array.
The diagnostic now treats an empty tensor comparison as zero error. This was
a test-reporting failure, not a simulator state failure.

## CPU tests

The two pure-tensor indexed-reset tests ran in a separate container without
GPU access. The first run found that explicitly supplied float32 controller
goals could not be assigned through advanced indexing into float64 targets.
The fix converts supplied goals and gripper memory to the controller device
and dtype. It leaves the ordinary state-derived reset path unchanged.

Both tests passed after the fix, in 0.006 seconds:

- `test_partial_reset_uses_each_selected_world`.
- `test_prefix_controller_memory_can_be_restored`.

The CPU test container was `libero-ppo-cpu-reset-tests-2-20260917`. The GPU gate
predates this narrow dtype fix; no further simulation was needed for its
explicit-input tensor conversion.

## Limitations and artifacts

- One task and one source demonstration supply this gate's evidence.
- Supported success goals are conjunctions of object `On` predicates. Other
  predicates and site goals fail explicitly.
- Camera observations, actor/critic training, evaluation success rates, and
  large-batch memory are outside this validation.
- Fixed-fixture pose banks require separately certified model compatibility.
  Dimension equality alone cannot establish compatibility.
- Fast prefix parity covers the tested controller mode and action prefix.
- Policy code must handle terminal observations before explicit reset.

Task-owned pod artifacts are under
`/data/rutavms/libero-ppo-20260917/outputs/`:

- `batch-fastprefix.json`: accepted final gate, source `9eb3936`.
- `batch-final.json`: accepted gate before fast-prefix optimization.
- `batch-stable.json`: rejected single-threshold stable-state attempt.
- `batch-onestep.json`: diagnostic-only contact-rich one-step metrics.
- `batch-diagnostic.json`: rejected contact-rich eight-step metrics.
- Corresponding `.failure.npz` files retain compact failure state where emitted.

The accepted final container was `libero-ppo-batch-fastprefix-20260917`.
These paths identify working evidence, not permanent storage guarantees.
The Sim2Real experiment owns verified S3 publication and staging cleanup.

## Follow-up: shallow-contact disagreement in 50-demo replay

A later free replay of all 50 demonstrations exposed a native/GPU predicate
disagreement outside the short gate's snapshots. A diagnostic replay stopped
at world 10, elapsed control step 108. Native MuJoCo reported success and
MuJoCo-Warp did not.

The diagnostic isolated the difference to contact generation:

- Both backends had identical top and bottom body positions.
- Horizontal center distance was 0.01353995 m, safely inside the 0.03 m criterion.
- Native collision detection returned one contact between
  `akita_black_bowl_1_g40` and `plate_1_g1` at distance `-2.71365e-5` m.
- GPU collision detection returned no bowl/plate contact pair.
- An independent CPU evaluation using the downloaded GPU contact pairs and
  GPU positions agreed with the GPU predicate logic.

This is evidence of a shallow 27-micrometer native contact disagreement. It
does not identify whether narrow-phase algorithm differences or numerical
precision caused it. No predicate threshold or production reward was changed.
The earlier native/GPU agreement remains valid for the snapshots tested;
it is not a guarantee of identical collision labels at every trajectory state.

The diagnostic helper now separates two checks. Logical disagreement on
identical GPU contacts and positions always fails. Native/GPU collision-label
differences fail by default, but an aggregate replay audit may request explicit
collection with `raise_on_mismatch=False`. That mode returns both labels and
all mismatches; it must not be reported as perfect native predicate parity.
Each mismatch includes world/time, optional source context, goal geometry,
contact names and distances, physical state, and collision settings.

The diagnostic source is `3f092ad`. Its detailed contact record is
`/data/rutavms/libero-ppo-20260917/outputs/predicate-contact-diagnostic.jsonl`.
Container `libero-ppo-replay-contact-diagnostic-20260917` exited nonzero on the
mismatch as intended. Aggregate counts and replay success rates belong to the
Sim2Real experiment's full replay report, not this stopped diagnostic.
