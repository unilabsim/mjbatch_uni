from collections.abc import Callable, Sequence
from typing import Annotated

from numpy.typing import NDArray


class Batch:
    """
    N MuJoCo simulations stepped on a C++ thread pool.

    The model is copied at construction; edit it before. Calls are serialized.

    Each simulation is an mjSTATE_INTEGRATION vector plus warning counters, loaded
    into a per-thread mjData for each call, so memory scales with threads, not
    simulations. Models with sleep enabled are rejected: sleep bookkeeping lives
    outside mjtState.

    bind("state") returns the (N, nstate) integration states themselves, in
    mj_getState's mjSTATE_INTEGRATION order: opaque rows for copying, saving and
    restoring simulations, native dtype only. A row written is the simulation's
    state at its next call, with pending field writes applied on top; the field
    views are stale until then. reset discards it, as it discards a field write.

    bind(field) returns an (N, ...) array over an mjData field in MjData's layout.
    Input fields (the mjtState components and warning) are copied in before
    a physics call, element by element, where they changed since we last wrote
    them; every bound field is copied out after, and a derived field bound between
    calls is filled for a simulation by its next call. reset applies pending writes
    after mj_resetData and then runs mj_forward, so derived fields are valid. After
    step, sensordata and every derived field are one substep behind qpos and qvel,
    as with mj_step itself, unless the batch was made with forward=True, which ends
    every step with mj_forward so they are current, at the cost of one forward per
    simulation per call.

    expand(field) returns (N, ...) per-simulation values of an mjModel field, seeded
    from the model and applied before every physics call for that simulation.
    mjOption fields expand too, a scalar as (N,) and a vector as (N, size), so
    gravity, timestep, integrator and the solver settings are per simulation. Raising
    iterations or ls_iterations, or switching cone to elliptic, can make a simulation
    need more arena than the template sized the worker's mjData for; that surfaces as
    the usual trapped MuJoCo error naming it. enableflags cannot turn sleep on, for
    the reason the constructor rejects it. The first call allocates one model copy
    per thread. set_const runs mj_setConst per simulation and expands every field it
    changed, so derived constants are per simulation too; mjModel scalars it writes
    (flags, stat) are kept per simulation as well.

    Derived constants follow expanded inputs only after set_const, as with
    mj_setConst on one model. A MuJoCo error on a worker raises RuntimeError naming
    the first failing simulation; the others still ran, and the failing one keeps the
    state it had before the call, its writes still pending. The trap is a MuJoCo log
    handler installed at import; installing another handler later disables it.
    """

    def __init__(self, model: object, num_sims: int, num_threads: int = 0, forward: bool = False, cpu_ids: Sequence[int] | None = None) -> None:
        """
        num_threads=0 uses every logical CPU, clamped to num_sims. forward=True ends every step with mj_forward, so derived fields are current with the state. cpu_ids (Linux only) pins worker i to cpu_ids[i]; its length sets num_threads when that is 0 and must equal it otherwise, and passing it on another platform raises ValueError.
        """

    @property
    def num_sims(self) -> int: ...

    @property
    def num_threads(self) -> int: ...

    @property
    def nstate(self) -> int:
        """The length of a simulation's integration state."""

    def bind(self, name: str, dtype: object | None = None) -> NDArray:
        """dtype is the field's own or float32 for mjtNum fields."""

    def expand(self, name: str, dtype: object | None = None) -> NDArray: ...

    def step(self, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None, nstep: int = 1, history: NDArray | None = None, *, callback: Callable | None = None, substep_sensor_copyout: Sequence[int] | None = None) -> None:
        """
        nstep mj_step calls per simulation, on one worker. ids: sorted unique ints or a bool mask. history: an optional caller-allocated (sims, nstep, nstate) array filled with each selected simulation's state after every substep, in bind("state") order; the rows of a simulation that raises are undefined.
        callback: fn(k, state, ctrl) invoked on the calling thread before substep k. k=0 gets the state from before the call; a later k gets the state after substep k-1. The views are live (num_sims, ...) batch rows built once per call: writes to ctrl apply to the substep that follows, writes to state at the next substep, like a state write between calls. Substeps are dispatched one at a time, so bound fields other than state stay stale until the call ends. Batch calls from inside the callback raise; an exception stops the simulations at the last completed substep, recoverably.
        substep_sensor_copyout=(start, stop) opts into an Euler-only mj_step1/mj_step2 split. Before each callback, mj_step1 runs and the half-open sensordata column range is copied into a fourth live callback argument, sensor. Position- and velocity-stage sensors in that range are current with state; acceleration-stage values (including contact forces) are not available at this split point and may be stale. Writes to ctrl and bound input fields occur between step1 and step2, so they affect the current substep. If the callback raises, state remains at the last completed substep; intermediate mjData is recovered on the next call.
        """

    def forward(self, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None) -> None: ...
    def refresh_sensor_range(
        self,
        ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = ...,
        sensor_range: Sequence[int] = ...,
    ) -> None: ...
    def refresh_sensor_ranges(
        self,
        ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = ...,
        sensor_ranges: Sequence[int] = ...,
    ) -> None: ...

    def reset(self, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None, keyframe: int = -1) -> None:
        """
        mj_resetData, or mj_resetDataKeyframe when keyframe >= 0, then mj_forward.
        """

    def jac_site(self, site: int, jacp: NDArray | None = None, jacr: NDArray | None = None, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None) -> None:
        """
        mj_jacSite per selected simulation into caller-allocated (sel, 3, nv) rows; either output may be None. Runs mj_kinematics and mj_comPos only, not mj_forward, and does not refresh the bound views: the outputs are the caller-allocated rows.
        """

    def sample_hfield(self, geom: int, body: int, offsets: NDArray, out: NDArray, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None, alignment: str = 'world') -> None:
        """
        Bilinear hfield sampling per selected simulation at XY offsets around a frame body's origin, into caller-allocated (sel, npoint) rows: the world z of the sampled hfield surface (the local elevation for an unrotated geom at the origin). alignment rotates the sampling grid: "world" keeps offsets in world axes, "yaw" rotates them by the frame body's yaw about world z. Runs mj_kinematics only, not mj_forward, and does not refresh the bound views. All simulations sample the template's hfield; a per-sim geom_pos or geom_quat moves the sampling frame.
        """

    def set_const(self, ids: Annotated[NDArray, dict(shape=(None,), order='C')] | None = None) -> None: ...
