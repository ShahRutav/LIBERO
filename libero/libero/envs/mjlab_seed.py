"""Safe MJLab construction helpers for reset-bank environments."""


def make_simulation_from_reset_seed(Simulation, cfg, model, initial_state, num_envs, device):
    """Build initial GPU data from a reset without changing model semantics.

    MJLab creates and forwards an MjData at model.qpos0 before the caller can
    apply a reset bank. Some LIBERO models intentionally have objects piled at
    qpos0 and therefore produce thousands of transient contacts. Seed that
    temporary data from a qualified reset, then restore both CPU and GPU model
    reference positions so passive-force semantics remain unchanged.
    """
    qpos0 = model.qpos0.copy()
    model.qpos0[:] = initial_state[1:1 + model.nq]
    try:
        engine = Simulation(num_envs, cfg, model=model, device=str(device))
    finally:
        model.qpos0[:] = qpos0
    engine.wp_model.qpos0.assign(qpos0)
    return engine
