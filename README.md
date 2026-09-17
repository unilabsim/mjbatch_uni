# mjbatch-uni

[![Build](https://img.shields.io/github/actions/workflow/status/unilabsim/mjbatch_uni/ci.yml?branch=main)](https://github.com/unilabsim/mjbatch_uni/actions)
[![PyPI version](https://img.shields.io/pypi/v/mjbatch-uni)](https://pypi.org/project/mjbatch-uni/)

`mjbatch-uni` is a Python library for running thousands of MuJoCo simulations in parallel on CPU.

Features include:

* C++ thread pool execution, with the GIL released;
* Live array access to simulation state and controls across the batch, with `bind` for MjData fields;
* Per-simulation model parameters, with `expand` for MjModel fields and `set_const` to recompute derived constants.
* Same-layout compiler-coherent mesh variants, with `VariantPack.from_specs()` and `Batch.from_variant_pack()`.
* Explicit topology-affine groups, with `ModelAffineBatch` routing global ids to incompatible layouts.
* Batched queries beyond stepping: site Jacobians with `jac_site` and heightfield sampling with `sample_hfield` (world/yaw grid alignment); query ops return caller-allocated results without refreshing the bound views.
* Per-substep control from Python with `step(..., callback=...)`.
* Opt-in split substeps with fresh position/velocity sensor views for body wrench control.
* Optional per-worker CPU pinning on Linux, with `cpu_ids` binding pool worker `i` to `cpu_ids[i]`.

For example:

```python
import mujoco, numpy as np
from mjbatch import Batch

model = mujoco.MjModel.from_xml_path("scene.xml")
batch = Batch(model, num_sims=4096)  # threads default to every logical CPU
qpos, ctrl = batch.bind("qpos"), batch.bind("ctrl")
batch.expand("geom_friction")[:, :, 0] = np.random.uniform(0.4, 1.2, (4096, 1))
for _ in range(1000):
  ctrl[:] = policy(qpos)             # your controller, all 4096 at once
  batch.step()                       # step them in parallel; qpos updates in place
```

Callbacks can also consume tracked sensors at the exact state they are controlling. With
Euler, `substep_sensor_copyout=(start, stop)` splits each substep at `mj_step1` /
`mj_step2`, copies the declared sensor columns into a fourth callback view, and applies
`ctrl` plus bound inputs such as `xfrc_applied` to that substep:

```python
xfrc = batch.bind("xfrc_applied")

def control(k, state, ctrl, sensor):
  ctrl[:] = impedance(sensor)       # position/velocity sensors match state
  xfrc[:, body_id] = wrench(sensor)

batch.step(callback=control, substep_sensor_copyout=(sensor_start, sensor_stop))
```

Acceleration-stage sensors and contact forces are not available before `mj_step2` and may
be stale in that view. The default callback path and all outputs are unchanged unless the
optional range is passed.

After a normal `step()` with `forward=False`, selected position- and velocity-stage sensor
columns can be refreshed from the final integration state without a full `mj_forward`:

```python
batch.refresh_sensor_range(ids, (sensor_start, sensor_stop))
```

The refresh runs in native worker threads, copies only the requested columns into the bound
`sensordata` view, and leaves acceleration-stage sensors (including contact forces) and all
other bound fields unchanged. Disjoint ranges can be refreshed in one dispatch with
`refresh_sensor_ranges(ids, (start0, stop0, start1, stop1, ...))`.

## Model randomization and variants

`expand(field)` returns a live `(num_sims, ...)` view of a non-asset `MjModel` field.
Rows are applied before each selected simulation runs. `model_field_specs()` describes
the shape, dtype, writability, and recompute dependency of every field. Declare the
fields being written to `model_update`; like mjlab event terms, mjbatch computes the
strongest recompute level and runs one conservative stock-MuJoCo `mj_setConst` pass.

```python
from mjbatch import RecomputeLevel

assert batch.model_field_specs()["body_mass"].recompute == RecomputeLevel.SET_CONST
with batch.model_update("body_mass", "geom_friction", ids=reset_envs):
  batch.expand("body_mass")[reset_envs, body_id] *= 1.1
  batch.expand("geom_friction")[reset_envs, :, 0] = friction_samples
```

For mesh-only differences, `VariantPack.from_specs()` independently compiles each
source spec, pools and deduplicates meshes, aligns named geom slots, disables missing
optional slots, and scatters compiler-derived geometry and inertia fields.
`Batch.from_variant_pack()` applies its fixed assignment and initial recompute.

```python
from mjbatch import Batch, VariantPack

pack = VariantPack.from_specs([tool_spec_0, tool_spec_1, tool_spec_2])
batch = Batch.from_variant_pack(pack, num_sims, np.arange(num_sims) % 3)
```

Truly incompatible topologies are not silently padded. Put each one in its own `Batch`
and route fixed global assignments through `ModelAffineBatch`. State and field views
remain on each `TopologyGroup`; global `history` is rejected.

```python
from mjbatch import ModelAffineBatch

sharded = ModelAffineBatch([batch_a, batch_b], names=["one_joint", "two_joint"])
sharded.step(np.array([0, 3, 4]))
state_a, state_b = sharded["one_joint"].state, sharded["two_joint"].state
```

Reproducible cold-start, RSS, stepping, and model-field-update measurements:

```bash
uv run python benchmarks/topology_groups.py --num-sims 512 --threads 4
```

`sample_hfield` samples a shared heightfield at XY offsets around a body origin,
returning the world z of the sampled surface. `alignment="yaw"` rotates the
sampling grid by the frame body's yaw about world z, so a per-sim `geom_quat`
rotates each simulation's scan pattern.

## Examples

We showcase a range of applications built using `mjbatch`: RL, MPC, SysID, and hardware
co-design. Each example is a self-contained, performant implementation. For instance, the Go1
RL controller learns to walk in under a minute on a five-year-old M1 laptop.

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/unilabsim/mjbatch_uni/blob/main/examples/cartpole_swingup.py"><img width="400" src="https://raw.githubusercontent.com/unilabsim/mjbatch_uni/main/examples/assets/cartpole_swingup.gif" alt="cart-pole swing-up"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/unilabsim/mjbatch_uni/blob/main/examples/cartpole_mpc.py"><img width="400" src="https://raw.githubusercontent.com/unilabsim/mjbatch_uni/main/examples/assets/cartpole_mpc.gif" alt="cart-pole MPC"></a>
    </td>
  </tr>
  <tr>
    <td align="center">A two-pole cart swung upright with <a href="https://ieeexplore.ieee.org/document/6386025">iLQR</a></td>
    <td align="center">A cart-pole swing-up controller using <a href="https://arxiv.org/abs/2212.00541">predictive sampling</a></td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/unilabsim/mjbatch_uni/blob/main/examples/g1_flip.py"><img width="400" src="https://raw.githubusercontent.com/unilabsim/mjbatch_uni/main/examples/assets/g1_flip.gif" alt="G1 backflip"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/unilabsim/mjbatch_uni/blob/main/examples/go1_joystick.py"><img width="400" src="https://raw.githubusercontent.com/unilabsim/mjbatch_uni/main/examples/assets/go1_joystick.gif" alt="Go1 joystick"></a>
    </td>
  </tr>
  <tr>
    <td align="center">A G1 humanoid tracking a reference backflip with receding-horizon iLQR</td>
    <td align="center">A Go1 quadruped joystick controller trained with PPO</td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/unilabsim/mjbatch_uni/blob/main/examples/arm_throw.py"><img width="400" src="https://raw.githubusercontent.com/unilabsim/mjbatch_uni/main/examples/assets/arm_throw.gif" alt="throwing arm co-design"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/unilabsim/mjbatch_uni/blob/main/examples/rizon_inertia.py"><img width="400" src="https://raw.githubusercontent.com/unilabsim/mjbatch_uni/main/examples/assets/rizon_inertia.gif" alt="Rizon inertia identification"></a>
    </td>
  </tr>
  <tr>
    <td align="center">CEM jointly optimizes a robot arm's proportions, gears, and controls</td>
    <td align="center">Damped Gauss–Newton fits a Rizon arm's inertial parameters to synthetic motion data</td>
  </tr>
</table>

Run with `uv run examples/<file>.py`; some need `uv sync --group examples`. The ones that open
a window need a display; `--headless` runs the solver without one.

## License

Apache-2.0.
