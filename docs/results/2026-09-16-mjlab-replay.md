# 2026-09-16 mjlab replay

Status: Complete
Last verified: 2026-09-16
Experiment: [LIBERO mjlab in the Sim2Real workspace](../../../../docs/experiments/libero-mjlab.md)

## Result

The first-task conversion passes. All three fresh-process runs with the final
backend reached LIBERO's original BDDL success predicate after replaying all
103 expert actions. Both 128 × 128 RGB cameras worked. Each run also passed
initial-state roundtrip and reset-after-stepping checks.

Relative to native MuJoCo 3.11, maximum end-effector and object-position errors
were:

- `replay-reset-1`: 1.683 mm end effector; 2.366 mm objects.
- `replay-reset-2`: 1.152 mm end effector; 3.424 mm objects.
- `replay-reset-3`: 0.785 mm end effector; 1.660 mm objects.

The largest repeated-first-step error was `7.45e-9` in the flattened state.
State restoration before stepping was exact. The 103-action GPU replay took
5.19–5.44 seconds after initialization; native replay took about 2.0 seconds.
Kernel compilation and environment construction are excluded from these times.
This CPU-mirrored prototype does not improve single-environment throughput.

## Provenance and procedure

- Fork: `ShahRutav/LIBERO`, upstream `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Tested implementation: `476c465058df4e6a2bab758bc720700f33c4ab54`.
- Task: `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
- Demo: `demo_0`, 103 seven-dimensional OSC actions.
- Dataset: `yifengzhu-hf/LIBERO-datasets`, the task's `_demo.hdf5` file.
- Dataset SHA-256: `75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3`.
- Stack: mjlab 1.6.0; MuJoCo and MuJoCo-Warp 3.11.0; robosuite 1.4.0;
  PyTorch 2.7.1; Python 3.11; one L40S.
- Physics: 0.002 s timestep, 20 Hz control, 25 integration steps per action,
  Euler integrator, Newton solver, elliptic friction cone, single convex contact.
- Each replay used the recorded model XML and first state. No reference states
  were injected during the action sequence. Native and GPU runs received the
  same actions and controller configuration.

The final repeat reports were launched before setting `LIBERO_SOURCE_REVISION`,
so that report field says `not recorded`. The deployed simulator and runner
were subsequently matched to the tested commit by SHA-256:

- `mjlab_sim.py`: `6765bb77d5b6efc0e6a58537e07b729ad27b2941dff71f8d9b099b2a0af35653`.
- `replay_mjlab.py`: `a17c132af754940e1d4fa208de431878323873ae929f75477d3c6db26f92064f`.

[Reproduction commands](../runbooks/mjlab-backend.md) include source revision
capture for subsequent runs. The runner owns the executable acceptance gates.

## Investigation record

Failed attempts remain part of the evidence:

- ~~`replay-001`: both backends failed.~~ Soft reset retained robosuite's
  incremental gripper command; first-step repeat error was about 0.0368.
- ~~`replay-002`: both backends failed after the gripper reset fix.~~ The
  initial-state repeat check passed, isolating a physics-default mismatch.
- A controlled native-only comparison restored successful grasping by disabling
  `multiccd`. Explicit legacy mesh inertia alone did not fix the failure.
  The unnecessary convex-inertia diagnostic was stopped before completion.
- `replay-003` passed on both backends after restoring single-contact behavior.
- ~~`replay-final`: the GPU replay failed on repetition.~~ This exposed remaining
  state-transfer/reset issues, so the earlier passing run was not accepted alone.
- `replay-stream-1` and `replay-stream-2` passed after stream ordering was fixed;
  ~~`replay-stream-3` failed.~~ Stream ordering alone was insufficient.
- `replay-reset-1`, `replay-reset-2`, and `replay-reset-3` passed after clearing
  complete GPU state on reset, including solver and collision buffers.

The stream fix prevents a real transfer/step race, and complete GPU reset is
required independently of these outcomes. Three successful runs do not prove
that all residual numerical variability has disappeared.

## Coverage and limits

All 11 regression tests passed in 7.47 seconds on the pod. The suite verifies
supplied benchmark states, GPU-only integration,
fixture pose synchronization, hard-reset controller references, cleared gripper
commands, CUDA stream ordering, and bounded placement-error retries. The replay
checks cover XML reset, cameras, complete action replay, and the success predicate.

Only this task and demo have full trajectory validation. The common backend hook
covers all BDDL domain classes, but the other tasks, robots, controller modes,
and vectorized training are not certified by this result. Existing process-vector
wrappers do not batch physics into one GPU simulation.

[MuJoCo-Warp documents nondeterministic GPU reductions](https://mujoco.readthedocs.io/en/latest/mjwarp/).
This result supports successful replay within measured tolerances, not bitwise
identity with CPU physics or a guarantee of identical outcomes for every demo.

Raw JSON reports and one passing NPZ trajectory are in the Sim2Real workspace at
`artifacts/libero-mjlab-20260916/`. Pod originals are under
`/data/rutavms/libero-mjlab-20260916/`. The dataset was downloaded, not republished.
