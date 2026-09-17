# SPDX-License-Identifier: Apache-2.0

import copy
import os
import subprocess
import sys
import threading
from typing import cast

import mujoco
import numpy as np
import pytest

from mjbatch import Batch, ModelAffineBatch, ModelFieldSpec, RecomputeLevel
from mjbatch._bindings import Batch as RawBatch
from mjbatch.variants import VariantPack

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="cam" pos="1 1 1"/>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <site name="base"/>
      <body name="pole" pos="0 0 .05" gravcomp="0">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
        <site name="tip" pos="0 0 .5"/>
      </body>
    </body>
    <body name="puck" pos="0 1 .5">
      <joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>
    </body>
    <body name="mocap" mocap="true" pos="1 0 1">
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <tendon><spatial name="t"><site site="base"/><site site="tip"/></spatial></tendon>
  <equality><weld body1="mocap" body2="cart" active="false"/></equality>
  <actuator><motor joint="slide" gear="10"/><position joint="hinge" kp="1"/></actuator>
  <sensor><jointpos joint="hinge"/><framepos objtype="site" objname="tip"/></sensor>
  <keyframe><key qpos="0.5 0.2"/></keyframe>
</mujoco>
"""
N = 8
TETRAHEDRON_OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""

# Activation dynamics, a mocap weld, a keyframe, and sensors that set mjData's
# lazy-evaluation flags (accelerometer, subtreelinvel).
LOCKSTEP_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <body name="pole" pos="0 0 .05">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
        <site name="tip" pos="0 0 .5"/>
      </body>
    </body>
    <body name="ball" pos="0 1 .5">
      <freejoint/>
      <geom type="sphere" size=".05" mass=".2"/>
    </body>
    <body name="mocap" mocap="true" pos="0 1 .5">
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality><weld body1="mocap" body2="ball"/></equality>
  <actuator>
    <motor joint="slide" gear="5"/>
    <general joint="hinge" dyntype="filter" dynprm="0.02" gainprm="5"
             biastype="affine" biasprm="0 -5 0"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <accelerometer site="tip"/>
    <subtreelinvel body="cart"/>
  </sensor>
  <keyframe>
    <key qpos="0.3 0.5 0 1 .5 1 0 0 0" ctrl="0.1 0.2" act="0.2" mpos="0 1 .5"/>
  </keyframe>
</mujoco>
"""
COMPARED = ("qpos", "qvel", "act", "time", "sensordata", "site_xpos")


@pytest.fixture
def model():
  return mujoco.MjModel.from_xml_string(XML)


def lockstep(nstep, num_threads, heavy=(), use_callback=False):
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  models = [model] * N
  batch = Batch(model, N, num_threads=num_threads)
  if heavy:
    batch.expand("body_mass")[list(heavy), 1] *= 3.0
    batch.set_const(np.array(heavy))
    for i in heavy:
      models[i] = copy.copy(model)
      models[i].body_mass[1] *= 3.0
      mujoco.mj_setConst(models[i], mujoco.MjData(models[i]))
  datas = [mujoco.MjData(m) for m in models]
  pending = [[] for _ in range(N)]
  bound = {f: batch.bind(f) for f in COMPARED}
  rng = np.random.default_rng(0)

  def write(field, ids, index, value):
    batch.bind(field)[(ids, *index)] = value
    for j, i in enumerate(ids):
      pending[i].append((field, index, value[j]))

  def apply(i):
    for field, index, value in pending[i]:
      getattr(datas[i], field)[index] = value
    pending[i].clear()

  every = list(range(N))
  for call in range(50):
    write("ctrl", every, (slice(None),), rng.uniform(-1, 1, (N, model.nu)))
    if call == 10:
      write("qpos", [1, 4, 6], (slice(0, 2),), rng.uniform(-0.3, 0.3, (3, 2)))
    if call == 15:
      write("mocap_pos", every, (0,), rng.uniform(-0.2, 0.2, (N, 3)) + [0, 1, 0.5])
    if call == 20:
      write("xfrc_applied", [2, 3], (3, slice(0, 3)), rng.uniform(-1, 1, (2, 3)))
    if call == 30:
      write("eq_active", [0, 2, 4, 6], (0,), np.zeros(4, np.uint8))
    ids = [0, 2, 3, 7] if call % 4 == 3 else every
    if call == 25:
      ids = [1, 5]
      batch.reset(np.array(ids), keyframe=0)
      for i in ids:
        mujoco.mj_resetDataKeyframe(models[i], datas[i], 0)
        apply(i)
        mujoco.mj_forward(models[i], datas[i])
    else:
      if use_callback:
        ctrl_now = batch.bind("ctrl").copy()

        def apply_ctrl(k, state, ctrl, value=ctrl_now):
          ctrl[:] = value

        batch.step(None if len(ids) == N else np.array(ids), nstep=nstep, callback=apply_ctrl)
      else:
        batch.step(None if len(ids) == N else np.array(ids), nstep=nstep)
      for i in ids:
        apply(i)
        for _ in range(nstep):
          mujoco.mj_step(models[i], datas[i])
    for field, arr in bound.items():
      np.testing.assert_array_equal(arr, [getattr(d, field) for d in datas], field)


@pytest.mark.parametrize("nstep", [1, 3])
@pytest.mark.parametrize("num_threads", [1, 3, 7])
def test_lockstep_with_mj_step(nstep, num_threads):
  lockstep(nstep, num_threads)


@pytest.mark.parametrize("nstep", [1, 3])
def test_lockstep_with_expanded_mass(nstep):
  lockstep(nstep, num_threads=4, heavy=(1, 2, 5))


@pytest.mark.parametrize("num_threads", [1, 3])
def test_mesh_pool_geom_dataid_matches_reference_models(tmp_path, num_threads):
  """A canonical model can pool meshes and select one per simulation.

  The compiler-derived fields are scattered from independently compiled reference
  models, then set_const recomputes the constants that depend on them. This is the
  stock-CPU foundation for future variant-pack construction: assets stay shared,
  while geom_dataid and non-asset model fields become per-simulation rows.
  """
  obj_path = tmp_path / "tetrahedron.obj"
  obj_path.write_text(TETRAHEDRON_OBJ)

  def compile_model(assets: str) -> mujoco.MjModel:
    xml = f"""
<mujoco>
  <option timestep="0.002"/>
  <asset>{assets}</asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="body" pos=".01 .02 .8">
      <freejoint/>
      <geom name="mesh" type="mesh" mesh="mesh0" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)

  canonical = compile_model(
    f'<mesh name="mesh0" file="{obj_path}"/><mesh name="mesh1" file="{obj_path}" scale=".7 .8 1.2"/>'
  )
  references = [
    compile_model(f'<mesh name="mesh0" file="{obj_path}"/>'),
    compile_model(f'<mesh name="mesh0" file="{obj_path}" scale=".7 .8 1.2"/>'),
  ]

  batch = Batch(canonical, N, num_threads=num_threads)
  variant_ids = np.array([1, 3, 4, 7])
  mesh_geom = canonical.geom("mesh").id
  variant_fields = (
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "geom_pos",
    "geom_quat",
    "body_mass",
    "body_subtreemass",
    "body_inertia",
    "body_invweight0",
    "body_ipos",
    "body_iquat",
  )
  for field in variant_fields:
    batch.expand(field)[variant_ids] = getattr(references[1], field)

  geom_dataid = batch.expand("geom_dataid")
  geom_dataid[variant_ids, mesh_geom] = 1
  batch.set_const(variant_ids)

  expected_variants = np.zeros(N, dtype=np.int32)
  expected_variants[variant_ids] = 1
  np.testing.assert_array_equal(geom_dataid[:, mesh_geom], expected_variants)
  for field in (*variant_fields, "dof_M0", "dof_invweight0", "dof_length"):
    expected = [getattr(references[v], field) for v in expected_variants]
    np.testing.assert_array_equal(batch.expand(field), expected, field)

  state = batch.bind("state")
  batch.step(nstep=25)
  for i, reference in enumerate(references):
    data = mujoco.MjData(reference)
    for _ in range(25):
      mujoco.mj_step(reference, data)
    expected_state = np.empty(batch.nstate)
    mujoco.mj_getState(reference, data, expected_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    rows = np.flatnonzero(expected_variants == i)
    np.testing.assert_array_equal(state[rows], np.tile(expected_state, (len(rows), 1)))


def test_variant_pack_matches_independently_compiled_references(tmp_path):
  obj_path = tmp_path / "tetrahedron.obj"
  obj_path.write_text(TETRAHEDRON_OBJ)

  def make_spec(scale_a: str, scale_b: str, *, include_b: bool = True):
    meshes = f'<mesh name="a" file="{obj_path}" scale="{scale_a}"/>'
    if include_b:
      meshes += f'<mesh name="b" file="{obj_path}" scale="{scale_b}"/>'
    geoms = '<geom name="a" type="mesh" mesh="a" mass="1"/>'
    if include_b:
      geoms += '<geom name="b" type="mesh" mesh="b" mass="1"/>'
    return mujoco.MjSpec.from_string(
      f"""
<mujoco>
  <option timestep="0.002"/>
  <asset>{meshes}</asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    <body name="body" pos=".01 .02 .8">
      <freejoint name="free"/>
      {geoms}
    </body>
  </worldbody>
</mujoco>
"""
    )

  specs = [
    make_spec("1 1 1", "1 1 1"),
    make_spec(".7 .8 1.2", ".9 .9 .9"),
    make_spec(".5 .6 .7", "1 1 1", include_b=False),
  ]
  pack = VariantPack.from_specs(specs)
  assert pack.num_variants == 3
  assert pack.model.nmesh == 5
  assert pack.model.ngeom == 3
  dataids = pack.fields["geom_dataid"]
  missing = pack.model.geom("b").id
  assert np.take(dataids, [1, 2, 4, 5, 7]).min() >= 0
  assert dataids[0, 0] == -1
  assert dataids[2, missing] == -1
  assert pack.fields["geom_type"][2, missing] == mujoco.mjtGeom.mjGEOM_NONE
  assert pack.fields["geom_contype"][2, missing] == 0
  assert pack.fields["geom_conaffinity"][2, missing] == 0
  assert pack.fields["geom_size"][2, missing].shape == (3,)
  np.testing.assert_array_equal(pack.fields["geom_size"][2, missing], 0.0)

  assignment = np.arange(N) % 3
  batch = Batch.from_variant_pack(pack, N, assignment, num_threads=3)
  state = batch.bind("state")
  batch.step(nstep=25)
  for variant, spec in enumerate(specs):
    reference = spec.compile()
    data = mujoco.MjData(reference)
    for _ in range(25):
      mujoco.mj_step(reference, data)
    expected = np.empty(batch.nstate)
    mujoco.mj_getState(reference, data, expected, mujoco.mjtState.mjSTATE_INTEGRATION)
    rows = np.flatnonzero(assignment == variant)
    np.testing.assert_array_equal(state[rows], np.tile(expected, (len(rows), 1)))


def test_variant_pack_deduplicates_identical_meshes(tmp_path):
  obj_path = tmp_path / "tetrahedron.obj"
  obj_path.write_text(TETRAHEDRON_OBJ)
  spec = mujoco.MjSpec.from_string(
    f"""
<mujoco>
  <asset><mesh name="mesh" file="{obj_path}"/></asset>
  <worldbody><body><freejoint name="free"/><geom name="mesh" type="mesh" mesh="mesh"/></body></worldbody>
</mujoco>
"""
  )
  pack = VariantPack.from_specs([spec, spec])
  assert pack.num_variants == 2
  assert pack.model.nmesh == 1
  np.testing.assert_array_equal(pack.fields["geom_dataid"], np.zeros((2, 1), dtype=np.int32))
  with pytest.raises(ValueError, match="read-only"):
    pack.fields["geom_dataid"][0, 0] = -1
  if sys.platform == "linux":
    cpus = sorted(os.sched_getaffinity(0))[:1]  # pyright: ignore[reportAttributeAccessIssue]
    pinned = Batch.from_variant_pack(pack, 2, np.array([0, 1]), cpu_ids=cpus)
    assert pinned.num_threads == 1


def test_variant_pack_validates_slots_layout_and_assignment(tmp_path):
  obj_path = tmp_path / "tetrahedron.obj"
  obj_path.write_text(TETRAHEDRON_OBJ)
  spec = mujoco.MjSpec.from_string(
    f"""
<mujoco>
  <asset><mesh name="mesh" file="{obj_path}"/></asset>
  <worldbody>
    <body><freejoint name="free"/><geom name="mesh" type="mesh" mesh="mesh"/></body>
  </worldbody>
</mujoco>
"""
  )
  changed_layout = mujoco.MjSpec.from_string(
    f"""
<mujoco>
      <asset><mesh name="mesh" file="{obj_path}"/></asset>
      <worldbody>
        <site name="extra"/>
        <body>
          <freejoint name="free"/>
          <geom name="mesh" type="mesh" mesh="mesh"/>
        </body>
      </worldbody>
</mujoco>
"""
  )
  with pytest.raises(ValueError, match="changes layout field"):
    VariantPack.from_specs([spec, changed_layout])

  pack = VariantPack.from_specs([spec, spec])
  with pytest.raises(ValueError, match="assignment entries"):
    Batch.from_variant_pack(pack, N, np.full(N, 2))


def test_variant_pack_rejects_shared_parameter_changes(tmp_path):
  obj_path = tmp_path / "tetrahedron.obj"
  obj_path.write_text(TETRAHEDRON_OBJ)

  def make_spec(ctrlrange: str):
    return mujoco.MjSpec.from_string(
      f"""
<mujoco>
  <asset><mesh name="mesh" file="{obj_path}"/></asset>
  <worldbody>
    <body><freejoint name="free"/><geom name="mesh" type="mesh" mesh="mesh"/></body>
  </worldbody>
  <actuator><motor joint="free" ctrlrange="{ctrlrange}"/></actuator>
</mujoco>
"""
    )

  with pytest.raises(ValueError, match="changes shared field actuator_ctrlrange"):
    VariantPack.from_specs([make_spec("-1 1"), make_spec("-2 2")])


def test_model_affine_batch_routes_global_ids(model):
  other = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  group_batch, other_batch = Batch(model, N // 2), Batch(other, N // 2)
  control_batch, other_control = Batch(model, N // 2), Batch(other, N // 2)
  sharded = ModelAffineBatch(
    [group_batch, other_batch],
    names=["cart", "lockstep"],
  )
  assert sharded.num_sims == N
  assert len(sharded.groups) == 2
  np.testing.assert_array_equal(sharded["cart"].global_ids, np.arange(N // 2))
  np.testing.assert_array_equal(sharded["lockstep"].global_ids, np.arange(N // 2, N))
  assert sharded["cart"].nstate != sharded["lockstep"].nstate

  ids = np.array([0, 2, 5, 7])
  sharded.step(ids)
  control_batch.step(np.array([0, 2]))
  other_control.step(np.array([1, 3]))
  np.testing.assert_array_equal(sharded["cart"].state, control_batch.bind("state"))
  np.testing.assert_array_equal(sharded["lockstep"].state, other_control.bind("state"))

  sharded.reset(ids, keyframe=0)
  control_batch.reset(np.array([0, 2]), keyframe=0)
  other_control.reset(np.array([1, 3]), keyframe=0)
  np.testing.assert_array_equal(sharded["cart"].state, control_batch.bind("state"))
  np.testing.assert_array_equal(sharded["lockstep"].state, other_control.bind("state"))

  with pytest.raises(ValueError, match="collect state from each group"):
    sharded.step(history=np.empty((0, 1, 1)))


def test_model_affine_batch_model_update_and_validation(model):
  other = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batches = [Batch(model, N // 2), Batch(other, N // 2)]
  sharded = ModelAffineBatch(batches)
  ids = np.array([0, 2, 5, 7])
  with sharded.model_update("body_mass", ids=ids):
    for group in sharded.groups:
      local_ids = np.searchsorted(group.global_ids, ids[np.isin(ids, group.global_ids)])
      group.expand("body_mass")[local_ids, 1] = 2.0

  for group in sharded.groups:
    subtree = group.expand("body_subtreemass")
    local_ids = np.searchsorted(group.global_ids, ids[np.isin(ids, group.global_ids)])
    np.testing.assert_array_equal(subtree[local_ids, 1], 2.1)
    unselected = np.setdiff1d(np.arange(group.num_sims), local_ids)
    np.testing.assert_array_equal(subtree[unselected, 1], group.batch.model.body_subtreemass[1])

  with pytest.raises(ValueError, match="cover every global id"):
    ModelAffineBatch(batches, [np.arange(N // 2), np.arange(N // 2)])
  with pytest.raises(ValueError, match="names must be unique"):
    ModelAffineBatch(batches, names=["same", "same"])
  with pytest.raises(ValueError, match="ids must be sorted"):
    sharded.step(np.array([2, 0]))
  with pytest.raises(ValueError, match="keyframe out of range"):
    sharded.reset(keyframe=99)


@pytest.mark.parametrize("nstep", [1, 3])
@pytest.mark.parametrize("num_threads", [1, 3])
def test_lockstep_with_callback(nstep, num_threads):
  lockstep(nstep, num_threads, use_callback=True)


@pytest.mark.parametrize("nstep", [1, 3])
def test_lockstep_with_callback_and_expanded_mass(nstep):
  lockstep(nstep, num_threads=4, heavy=(1, 2, 5), use_callback=True)


@pytest.mark.parametrize("num_threads", [1, 4])
def test_step_callback_matches_a_python_loop(num_threads):
  # A callback-driven rollout against step(nstep=1) calls applying the same
  # per-substep controls, compared bit for bit.
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch, ref = Batch(model, N, num_threads=num_threads), Batch(model, N, num_threads=3)
  fields = ("state", "qpos", "qvel", "act", "time", "sensordata", "site_xpos")
  bound = {f: batch.bind(f) for f in fields}
  reference = {f: ref.bind(f) for f in fields}
  nstep = 6
  rng = np.random.default_rng(1)
  ctrls = rng.uniform(-1, 1, (nstep, N, model.nu))
  seen = []

  def apply_ctrl(k, state, ctrl):
    seen.append(state.copy())
    ctrl[:] = ctrls[k]

  initial = batch.bind("state").copy()
  history = np.empty((N, nstep, batch.nstate))
  batch.step(nstep=nstep, history=history, callback=apply_ctrl)
  assert len(seen) == nstep
  np.testing.assert_array_equal(seen[0], initial)  # k=0: the state before the call
  for k in range(nstep):
    ref.bind("ctrl")[:] = ctrls[k]
    ref.step()
    np.testing.assert_array_equal(history[:, k], reference["state"])
    if k + 1 < nstep:  # the state view at k + 1 reflects substep k
      np.testing.assert_array_equal(seen[k + 1], reference["state"])
  for field in fields:
    np.testing.assert_array_equal(bound[field], reference[field], field)


@pytest.mark.parametrize("num_threads", [1, 3])
def test_step_callback_error_keeps_completed_substeps(model, num_threads):
  batch = Batch(model, N, num_threads=num_threads)
  ctrl, qpos, time = batch.bind("ctrl"), batch.bind("qpos"), batch.bind("time")

  def cb(k, state, ctrl_view):
    ctrl_view[:] = 0.5
    if k == 2:
      raise ValueError("boom")

  with pytest.raises(ValueError, match="boom"):
    batch.step(nstep=6, callback=cb)
  # Two substeps completed; the batch stopped there, consistent and recoverable.
  np.testing.assert_allclose(time, 2 * 0.002)
  np.testing.assert_array_equal(ctrl, 0.5)
  data = mujoco.MjData(model)
  data.ctrl[:] = 0.5
  for _ in range(2):
    mujoco.mj_step(model, data)
  np.testing.assert_array_equal(qpos, np.tile(data.qpos, (N, 1)))
  batch.step()
  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(qpos, np.tile(data.qpos, (N, 1)))
  np.testing.assert_allclose(time, 3 * 0.002)


def test_step_callback_is_not_reentrant(model):
  batch = Batch(model, N, num_threads=2)
  time = batch.bind("time")
  errors = []

  def cb(k, state, ctrl):
    for call in (lambda: batch.step(), lambda: batch.forward(), lambda: batch.bind("qpos")):
      try:
        call()
      except RuntimeError as e:
        errors.append(str(e))

  batch.step(nstep=2, callback=cb)
  assert len(errors) == 6
  assert all("callback" in e for e in errors)
  np.testing.assert_allclose(time, 2 * 0.002)  # the outer step completed


def test_step_callback_initial_state(model):
  batch, ref = Batch(model, N), Batch(model, N)
  for b in (batch, ref):
    b.bind("ctrl")[:] = 0.3
    b.step(nstep=2)
  seen = []

  def cb(k, state, ctrl):
    seen.append(state.copy())

  initial = batch.bind("state").copy()
  batch.step(nstep=3, callback=cb)
  np.testing.assert_array_equal(seen[0], initial)  # k=0: the state before the call
  for k in (1, 2):
    ref.step()
    np.testing.assert_array_equal(seen[k], ref.bind("state"))


@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("ids", [None, [1, 4, 6]])
def test_split_substep_callback_matches_mj_step_split_reference(model, num_threads, ids):
  """The opt-in split gives fresh stage-one sensors and exact mj_step2 results."""
  batch = Batch(model, N, num_threads=num_threads)
  xfrc = batch.bind("xfrc_applied")
  selected = np.arange(N) if ids is None else np.array(ids)
  nstep = 5
  rng = np.random.default_rng(7)
  ctrls = rng.uniform(-1, 1, (nstep, N, model.nu))
  wrenches = rng.uniform(-0.5, 0.5, (nstep, N, 6))
  sensors = []
  history = np.empty((len(selected), nstep, batch.nstate))

  def apply_wrench(k, state, ctrl_view, sensor):
    ctrl_view[:] = ctrls[k]
    xfrc[:, 1, :] = wrenches[k]
    sensors.append(sensor.copy())

  batch.step(
    None if ids is None else selected,
    nstep=nstep,
    history=history,
    callback=apply_wrench,
    substep_sensor_copyout=(1, model.nsensordata),
  )

  datas = [mujoco.MjData(model) for _ in range(N)]
  for d in datas:
    mujoco.mj_forward(model, d)
  state = np.empty(batch.nstate)
  for k in range(nstep):
    for j, i in enumerate(selected):
      d = datas[i]
      mujoco.mj_step1(model, d)
      np.testing.assert_array_equal(sensors[k][i], d.sensordata[1:], f"k={k}, sim={i}")
      d.ctrl[:] = ctrls[k, i]
      d.xfrc_applied[1] = wrenches[k, i]
      mujoco.mj_step2(model, d)
      mujoco.mj_getState(model, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
      np.testing.assert_array_equal(history[j, k], state, f"k={k}, sim={i}")
  for i in selected:
    mujoco.mj_getState(model, datas[i], state, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.testing.assert_array_equal(batch.bind("state")[i], state, f"sim={i}")


@pytest.mark.parametrize("num_threads", [1, 3])
def test_split_substep_callback_error_keeps_completed_substeps(model, num_threads):
  batch = Batch(model, N, num_threads=num_threads)
  xfrc = batch.bind("xfrc_applied")
  time, qpos = batch.bind("time"), batch.bind("qpos")

  def cb(k, state, ctrl_view, sensor):
    ctrl_view[:] = 0.25
    xfrc[:, 1, 2] = 0.125
    if k == 2:
      raise ValueError("boom")

  with pytest.raises(ValueError, match="boom"):
    batch.step(nstep=6, callback=cb, substep_sensor_copyout=(1, model.nsensordata))

  np.testing.assert_allclose(time, 2 * 0.002)
  data = mujoco.MjData(model)
  data.ctrl[:] = 0.25
  data.xfrc_applied[1, 2] = 0.125
  for _ in range(2):
    mujoco.mj_step1(model, data)
    mujoco.mj_step2(model, data)
  np.testing.assert_array_equal(qpos, np.tile(data.qpos, (N, 1)))
  batch.step()
  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(qpos, np.tile(data.qpos, (N, 1)))
  np.testing.assert_allclose(time, 3 * 0.002)


def test_split_substep_callback_rejects_rk4(model):
  rk4 = mujoco.MjModel.from_xml_string(XML.replace("<option", '<option integrator="RK4"'))
  batch = Batch(rk4, N)
  with pytest.raises(ValueError, match="Euler"):
    batch.step(callback=lambda k, s, c, sensor: None, substep_sensor_copyout=(0, 1))

  mixed = Batch(model, N)
  mixed.expand("integrator")[0] = mujoco.mjtIntegrator.mjINT_RK4
  with pytest.raises(ValueError, match="sim 0.*Euler"):
    mixed.step(callback=lambda k, s, c, sensor: None, substep_sensor_copyout=(0, 1))


def test_substep_sensor_copyout_validation(model):
  batch = Batch(model, N)

  def cb(k, state, ctrl, sensor):
    pass

  with pytest.raises(ValueError, match="shape \\(2,\\)"):
    batch.step(callback=cb, substep_sensor_copyout=(1,))
  with pytest.raises(ValueError, match="0 <= start < stop"):
    batch.step(callback=cb, substep_sensor_copyout=(2, 2))
  with pytest.raises(ValueError, match="<= nsensordata"):
    batch.step(callback=cb, substep_sensor_copyout=(0, model.nsensordata + 1))
  with pytest.raises(ValueError, match="requires callback"):
    batch.step(substep_sensor_copyout=(0, 1))


@pytest.mark.parametrize("ids", [None, np.array([1, 3])])
def test_refresh_sensor_range_updates_only_selected_columns(ids):
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch = Batch(model, N, num_threads=2)
  qpos, qvel, ctrl, sensordata, warmstart = (
    batch.bind(f) for f in ("qpos", "qvel", "ctrl", "sensordata", "qacc_warmstart")
  )
  ctrl[:] = np.linspace(-0.2, 0.2, N)[:, None]
  batch.step(nstep=3)
  before = sensordata.copy()
  qpos_before, qvel_before, warmstart_before = qpos.copy(), qvel.copy(), warmstart.copy()
  selected = np.arange(N) if ids is None else ids

  batch.refresh_sensor_range(ids, (0, 1))

  np.testing.assert_array_equal(sensordata[selected, 0], qpos[selected, 1])
  np.testing.assert_array_equal(sensordata[:, 1:], before[:, 1:])
  np.testing.assert_array_equal(qpos, qpos_before)
  np.testing.assert_array_equal(qvel, qvel_before)
  np.testing.assert_array_equal(warmstart, warmstart_before)


def test_refresh_sensor_ranges_updates_disjoint_columns():
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch = Batch(model, N, num_threads=2)
  qpos, qvel, sensordata = (batch.bind(f) for f in ("qpos", "qvel", "sensordata"))
  batch.step(nstep=2)
  before = sensordata.copy()
  selected = np.array([0, 2])

  batch.refresh_sensor_ranges(selected, (0, 1, 4, 7))

  np.testing.assert_array_equal(sensordata[selected, 0], qpos[selected, 1])
  np.testing.assert_array_equal(sensordata[:, 1:4], before[:, 1:4])
  np.testing.assert_array_equal(sensordata[[1, 3, 4, 5, 6, 7]], before[[1, 3, 4, 5, 6, 7]])
  for row in selected:
    reference = mujoco.MjData(model)
    reference.qpos[:] = qpos[row]
    reference.qvel[:] = qvel[row]
    mujoco.mj_kinematics(model, reference)
    mujoco.mj_comPos(model, reference)
    mujoco.mj_comVel(model, reference)
    mujoco.mj_sensorPos(model, reference)
    mujoco.mj_sensorVel(model, reference)
    np.testing.assert_array_equal(sensordata[row, 4:7], reference.sensordata[4:7])


def test_refresh_sensor_range_validation(model):
  batch = Batch(model, N)
  with pytest.raises(ValueError, match=r"shape \(2,\)"):
    batch.refresh_sensor_range(sensor_range=(0,))
  with pytest.raises(ValueError, match="0 <= start < stop"):
    batch.refresh_sensor_range(sensor_range=(2, 2))
  with pytest.raises(ValueError, match="<= nsensordata"):
    batch.refresh_sensor_range(sensor_range=(0, model.nsensordata + 1))
  with pytest.raises(ValueError, match=r"\(start, stop\) pairs"):
    batch.refresh_sensor_ranges(sensor_ranges=(0, 1, 2))


def test_sleep_is_rejected():
  xml = LOCKSTEP_XML.replace("<option", '<option><flag sleep="enable"/></option><option')
  with pytest.raises(ValueError, match="sleep"):
    Batch(mujoco.MjModel.from_xml_string(xml), N)


def test_per_sim_gravity_matches_separate_models():
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch = Batch(model, N, num_threads=3)
  gravity = batch.expand("gravity")
  gravity[:, 2] = np.linspace(-9.81, -1.0, N)
  batch.bind("ctrl")[:, 0] = 0.5
  qpos, qvel = batch.bind("qpos"), batch.bind("qvel")
  batch.step(nstep=50)
  for i in range(N):
    m = copy.copy(model)
    m.opt.gravity[2] = gravity[i, 2]
    d = mujoco.MjData(m)
    d.ctrl[0] = 0.5
    for _ in range(50):
      mujoco.mj_step(m, d)
    np.testing.assert_array_equal(qpos[i], d.qpos)
    np.testing.assert_array_equal(qvel[i], d.qvel)


def test_per_sim_timestep(model):
  batch = Batch(model, N, num_threads=2)
  timestep = batch.expand("timestep")
  timestep[:] = 0.001 * (1 + np.arange(N))
  assert timestep.shape == (N,)
  time = batch.bind("time")
  batch.step(nstep=4)
  np.testing.assert_allclose(time, 4 * timestep)


def test_per_sim_integrator(model):
  batch = Batch(model, N)
  integrator = batch.expand("integrator")
  assert integrator.dtype == np.int32
  np.testing.assert_array_equal(integrator, model.opt.integrator)
  integrator[::2] = mujoco.mjtIntegrator.mjINT_RK4
  batch.bind("ctrl")[:, 0] = 1.0
  qvel = batch.bind("qvel")
  batch.step(nstep=20)
  np.testing.assert_array_equal(qvel[1], qvel[3])
  assert not np.array_equal(qvel[0], qvel[1])
  ref = copy.copy(model)
  ref.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
  d = mujoco.MjData(ref)
  d.ctrl[0] = 1.0
  for _ in range(20):
    mujoco.mj_step(ref, d)
  np.testing.assert_array_equal(qvel[0], d.qvel)


def test_expanded_option_seeds_from_the_template(model):
  batch, ref = Batch(model, N), Batch(model, N)
  gravity = batch.expand("gravity")
  np.testing.assert_array_equal(gravity, np.tile(model.opt.gravity, (N, 1)))
  np.testing.assert_array_equal(batch.expand("o_solref"), np.tile(model.opt.o_solref, (N, 1)))
  for b in (batch, ref):
    b.bind("ctrl")[:, 0] = 1.0
    b.step(nstep=20)
  np.testing.assert_array_equal(batch.bind("state"), ref.bind("state"))


def test_model_field_specs_describe_the_contract(model):
  specs = Batch(model, N).model_field_specs()

  assert specs["body_mass"].shape == (model.nbody,)
  assert specs["body_mass"].dtype == np.dtype(np.float64)
  assert specs["body_mass"].writable
  assert not specs["body_mass"].asset
  assert specs["body_mass"].recompute == RecomputeLevel.SET_CONST
  assert specs["body_mass"].recompute == RecomputeLevel.SET_CONST

  assert specs["body_gravcomp"].recompute == RecomputeLevel.SET_CONST
  assert specs["qpos0"].recompute == RecomputeLevel.SET_CONST
  assert specs["geom_friction"].recompute == RecomputeLevel.NONE
  assert specs["timestep"].shape == ()
  assert specs["timestep"].dtype == np.dtype(np.float64)
  assert specs["integrator"].dtype == np.dtype(np.int32)
  assert specs["gravity"].shape == (3,)

  assert specs["mesh_vert"].asset
  assert not specs["mesh_vert"].writable
  assert not specs["body_parentid"].writable
  with pytest.raises(TypeError):
    cast(dict[str, ModelFieldSpec], specs)["new_field"] = specs["body_mass"]


def test_model_update_recomputes_selected_rows_once(model, monkeypatch):
  batch = Batch(model, N, num_threads=2)
  pole = model.body("pole").id
  ids = np.array([1, 4])
  calls: list[np.ndarray | None] = []
  raw_set_const = RawBatch.set_const

  def counting_set_const(raw_batch: RawBatch, ids: np.ndarray | None = None) -> None:
    calls.append(None if ids is None else ids.copy())
    raw_set_const(raw_batch, ids)

  monkeypatch.setattr(RawBatch, "set_const", counting_set_const)
  with batch.model_update("body_mass", "geom_friction", ids=ids):
    batch.expand("body_mass")[ids, pole] = 2.0
    batch.expand("geom_friction")[ids, :, 0] = 0.7
  assert len(calls) == 1
  np.testing.assert_array_equal(calls[0], ids)

  subtree = batch.expand("body_subtreemass")
  np.testing.assert_array_equal(subtree[ids, pole], 2.0)
  np.testing.assert_array_equal(
    subtree[np.setdiff1d(np.arange(N), ids), pole],
    model.body_subtreemass[pole],
  )
  expected_mass = batch.expand("body_mass").copy()

  with batch.model_update("geom_friction", ids=ids):
    batch.expand("geom_friction")[ids, :, 1] = 0.8
  assert len(calls) == 1
  np.testing.assert_array_equal(batch.expand("body_mass"), expected_mass)


def test_model_update_is_transactional_and_fail_closed(model):
  batch = Batch(model, N)
  with pytest.raises(ValueError, match="read-only structural data"):
    batch.expand("body_parentid")
  with pytest.raises(ValueError, match="not writable"):
    with batch.model_update("body_parentid"):
      pass
  with pytest.raises(RuntimeError, match="cannot be nested"):
    with batch.model_update("body_mass"):
      with batch.model_update("body_mass"):
        pass
  with pytest.raises(RuntimeError, match="after model_update exits"):
    with batch.model_update("body_mass"):
      batch.set_const()


def test_option_is_untouched_by_set_const(model):
  # mj_setConst never writes opt, so no opt field can be flagged as one of its
  # outputs and expanded behind the caller's back.
  ref = mujoco.MjModel.from_xml_string(XML)
  before = {f: np.array(getattr(ref.opt, f), copy=True) for f in dir(ref.opt) if f[0] != "_"}
  mujoco.mj_setConst(ref, mujoco.MjData(ref))
  for f, v in before.items():
    np.testing.assert_array_equal(getattr(ref.opt, f), v, f)
  batch = Batch(model, N)
  timestep = batch.expand("timestep")
  timestep[:] = 0.001 * (1 + np.arange(N))
  batch.expand("body_mass")[:, model.body("pole").id] = 0.5
  batch.set_const()
  np.testing.assert_array_equal(timestep, 0.001 * (1 + np.arange(N)))
  time = batch.bind("time")
  batch.step()
  np.testing.assert_allclose(time, timestep)


def test_per_sim_sleep_is_rejected(model):
  batch = Batch(model, N, num_threads=2)
  enableflags = batch.expand("enableflags")
  enableflags[3] |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  batch.step(np.array([0, 1]))  # sim 3 is not running
  for call in (batch.step, batch.forward, batch.set_const):
    with pytest.raises(ValueError, match="sim 3: sleep"):
      call()
  enableflags[3] = model.opt.enableflags
  batch.step()


def test_step_history_matches_a_loop(model):
  batch, ref = Batch(model, N, num_threads=3), Batch(model, N, num_threads=3)
  for b in (batch, ref):
    b.bind("ctrl")[:, 0] = np.linspace(-1, 1, N)
  history = np.empty((N, 7, batch.nstate))
  batch.step(nstep=7, history=history)
  np.testing.assert_array_equal(history[:, -1], batch.bind("state"))
  for k in range(7):
    ref.step()
    np.testing.assert_array_equal(history[:, k], ref.bind("state"))


def test_step_history_with_ids(model):
  batch = Batch(model, N)
  batch.bind("ctrl")[:, 0] = np.linspace(-1, 1, N)
  ids = np.array([1, 4, 6])
  history = np.empty((len(ids), 5, batch.nstate))
  batch.step(ids, nstep=5, history=history)
  state, time = batch.bind("state"), batch.bind("time")
  np.testing.assert_array_equal(history[:, -1], state[ids])
  np.testing.assert_allclose(history[:, :, 0], np.tile(np.arange(1, 6) * 0.002, (3, 1)))
  assert not np.any(time[[0, 2, 3, 5, 7]])


def test_step_history_validation(model):
  batch = Batch(model, N)
  nstate = batch.nstate
  for bad in (
    np.empty((N, 3)),
    np.empty((N, 2, nstate)),
    np.empty((N - 1, 3, nstate)),
    np.empty((N, 3, nstate), np.float32),
    np.empty((N, 3, nstate + 1))[:, :, :-1],
  ):
    with pytest.raises(ValueError, match="history"):
      batch.step(nstep=3, history=bad)
  with pytest.raises(ValueError, match="history"):
    batch.step(np.array([0, 1]), nstep=3, history=np.empty((N, 3, nstate)))
  assert not np.any(batch.bind("time"))


def test_step_history_accepts_degenerate_strides(model):
  # numpy gives a size-0 array all-zero strides and a new axis a zero stride.
  batch = Batch(model, N)
  batch.step(np.zeros(N, dtype=bool), nstep=3, history=np.empty((0, 3, batch.nstate)))
  batch.step(np.array([], dtype=np.int64), nstep=2, history=np.empty((0, 2, batch.nstate)))
  history = np.empty((4, batch.nstate))
  batch.step(np.array([2]), nstep=4, history=history[None])
  np.testing.assert_array_equal(history[-1], batch.bind("state")[2])
  assert not np.any(np.delete(batch.bind("time"), 2))


def test_step_history_error_names_the_sim():
  batch = Batch(mujoco.MjModel.from_xml_string(LOCKSTEP_XML), N, num_threads=2)
  batch.expand("eq_type")[2] = 99
  history = np.zeros((N, 3, batch.nstate))
  with pytest.raises(RuntimeError, match="sim 2"):
    batch.step(nstep=3, history=history)
  assert np.any(history[0])  # the sims that ran still wrote their rows


@pytest.mark.skipif(sys.platform == "win32", reason="the resource module is Unix-only")
def test_memory_does_not_scale_with_num_sims():
  # A fresh process, so ru_maxrss growth is this batch's. One mjData per sim
  # grows it by 712 MB here; 4096 state vectors are a few MB.
  code = """
import resource, sys, mujoco
from mjbatch import Batch
xml = '<mujoco><size memory="256K"/><worldbody><body><joint/><geom size=".02"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
scale = 1 if sys.platform == "darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
batch = Batch(model, 4096)
batch.step()
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) * scale)
"""
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
  assert out.returncode == 0, out.stderr
  assert int(out.stdout) < 256 * 2**20


@pytest.mark.parametrize("num_threads", [1, 3])
def test_step_error_keeps_the_failing_sim(num_threads):
  model = mujoco.MjModel.from_xml_string(LOCKSTEP_XML)
  batch = Batch(model, N, num_threads=num_threads)
  eq_type = batch.expand("eq_type")
  eq_type[2] = 99
  ctrl, qpos = batch.bind("ctrl"), batch.bind("qpos")
  ctrl[:, 0] = np.linspace(-1, 1, N)
  with pytest.raises(RuntimeError, match="sim 2"):
    batch.step(nstep=3)
  # The others ran; sim 2 kept its state and its pending write, and the worker
  # it failed on serves later sims correctly.
  eq_type[2] = model.eq_type
  batch.step()
  one, four = reference(model, ctrl, 1), reference(model, ctrl, 4)
  for i in range(N):
    np.testing.assert_array_equal(qpos[i], (one if i == 2 else four)[i].qpos)


def reference(model, ctrl, nstep):
  datas = [mujoco.MjData(model) for _ in range(N)]
  for i, d in enumerate(datas):
    d.ctrl[:] = ctrl[i]
    for _ in range(nstep):
      mujoco.mj_step(model, d)
  return datas


@pytest.mark.parametrize("num_threads", [1, 4])
def test_step_matches_reference(model, num_threads):
  batch = Batch(model, N, num_threads=num_threads)
  ctrl = batch.bind("ctrl")
  ctrl[:, 0] = np.linspace(-1, 1, N)
  qpos, qvel, sensordata = (batch.bind(f) for f in ("qpos", "qvel", "sensordata"))
  for _ in range(25):
    batch.step()
  batch.step(nstep=25)
  for i, d in enumerate(reference(model, ctrl, 50)):
    np.testing.assert_array_equal(qpos[i], d.qpos)
    np.testing.assert_array_equal(qvel[i], d.qvel)
    np.testing.assert_array_equal(sensordata[i], d.sensordata)
  if sys.platform == "linux":
    # Pinning workers to CPUs must not change the numerics bit for bit.
    cpus = sorted(os.sched_getaffinity(0))[:num_threads]  # pyright: ignore[reportAttributeAccessIssue]
    pinned = Batch(model, N, cpu_ids=cpus)
    pinned.bind("ctrl")[:] = ctrl
    for _ in range(25):
      pinned.step()
    pinned.step(nstep=25)
    assert pinned.num_threads == len(cpus)
    np.testing.assert_array_equal(pinned.bind("state"), batch.bind("state"))


@pytest.mark.skipif(sys.platform != "linux", reason="cpu_ids pinning is Linux-only")
def test_cpu_ids_pins_workers(model):
  cpus = sorted(os.sched_getaffinity(0))[:3]  # pyright: ignore[reportAttributeAccessIssue]
  batch = Batch(model, N, cpu_ids=cpus)
  assert batch.num_threads == len(cpus)
  # An explicit num_threads equal to the length is accepted too.
  assert Batch(model, N, num_threads=len(cpus), cpu_ids=cpus).num_threads == len(cpus)
  ctrl = batch.bind("ctrl")
  ctrl[:, 0] = np.linspace(-1, 1, N)
  qpos = batch.bind("qpos")
  batch.step(nstep=5)
  for i, d in enumerate(reference(model, ctrl, 5)):
    np.testing.assert_array_equal(qpos[i], d.qpos)
  # Every worker's affinity mask is exactly its one pinned CPU, read back
  # through the per-thread view of sched_getaffinity.
  masks = set()
  for tid in os.listdir("/proc/self/task"):
    try:
      masks.add(frozenset(os.sched_getaffinity(int(tid))))  # pyright: ignore[reportAttributeAccessIssue]
    except (ProcessLookupError, PermissionError):
      continue
  assert {frozenset({cpu}) for cpu in cpus} <= masks


@pytest.mark.skipif(sys.platform != "linux", reason="cpu_ids pinning is Linux-only")
def test_cpu_ids_validation(model):
  cpus = sorted(os.sched_getaffinity(0))  # pyright: ignore[reportAttributeAccessIssue]
  outside = next(cpu for cpu in range(4096) if cpu not in cpus)
  with pytest.raises(ValueError, match="not available"):
    Batch(model, N, cpu_ids=[outside])
  with pytest.raises(ValueError, match="unique"):
    Batch(model, N, cpu_ids=[cpus[0], cpus[0]])
  with pytest.raises(ValueError, match="non-empty"):
    Batch(model, N, cpu_ids=[])
  with pytest.raises(ValueError, match=r">= 0"):
    Batch(model, N, cpu_ids=[-1])
  with pytest.raises(ValueError, match="num_threads"):
    Batch(model, N, num_threads=len(cpus) + 1, cpu_ids=cpus)


@pytest.mark.skipif(sys.platform == "linux", reason="Linux accepts cpu_ids")
def test_cpu_ids_rejected_off_linux(model):
  with pytest.raises(ValueError, match="only supported on Linux"):
    Batch(model, N, cpu_ids=[0])


def test_forward_after_step(model):
  """With forward=True the derived fields are current after a step; without, one
  substep behind, as with mj_step."""
  for forward in (False, True):
    batch = Batch(model, N, forward=forward)
    ctrl, xpos, sensordata = (batch.bind(f) for f in ("ctrl", "xpos", "sensordata"))
    ctrl[:, 0] = np.linspace(-1, 1, N)
    batch.step(nstep=10)
    behind = xpos.copy(), sensordata.copy()
    batch.forward()
    same = np.array_equal(behind[0], xpos) and np.array_equal(behind[1], sensordata)
    assert same == forward


def test_shapes_and_dtypes_match_mjdata(model):
  batch = Batch(model, N)
  d = mujoco.MjData(model)
  for name in dir(d):
    if name.startswith("_"):
      continue
    val = getattr(d, name)
    if not isinstance(val, np.ndarray) or name in ("contact", "efc_type"):
      continue
    try:
      arr = batch.bind(name)
    except ValueError:
      continue
    assert arr.shape == (N, *val.shape), name
    assert arr.dtype == val.dtype, name
  assert batch.bind("time").shape == (N,)
  batch = Batch(model, N)
  assert batch.bind("qpos", np.float32).dtype == np.float32
  with pytest.raises(ValueError):
    batch.bind("qpos")  # already bound as float32
  with pytest.raises(ValueError):
    batch.bind("eq_active", np.float32)
  with pytest.raises(ValueError):
    batch.bind("nope")
  with pytest.raises(ValueError, match=r'use expand\("geom_friction"\)'):
    batch.bind("geom_friction")
  with pytest.raises(ValueError, match=r'use bind\("qpos"\)'):
    batch.expand("qpos")
  with pytest.raises(ValueError):
    batch.expand("mesh_vert")
  with pytest.raises(ValueError):
    Batch(model, 0)


def test_float32_rows_only_written_when_changed(model):
  batch = Batch(model, N, num_threads=2)
  ctrl = batch.bind("ctrl", np.float32)
  qpos = batch.bind("qpos", np.float32)
  ctrl[:, 0] = 0.3
  for _ in range(200):
    batch.step()
  # Physics ran in mjtNum precision: untouched float32 rows never round-trip.
  ref_ctrl = np.zeros((N, model.nu))
  ref_ctrl[:, 0] = np.float32(0.3)
  d = reference(model, ref_ctrl, 200)[0]
  np.testing.assert_array_equal(qpos[0], d.qpos.astype(np.float32))
  # A written row takes effect on the next call; a derived field is read-only.
  site_xpos = batch.bind("site_xpos")
  before = site_xpos[1].copy()
  qpos[1, 1] = 0.7
  site_xpos[0] = 42.0
  batch.forward()
  assert not np.array_equal(site_xpos[1], before)
  assert not np.any(site_xpos[0] == 42.0)


def test_expand_and_set_const_are_complete(model):
  pole = model.body("pole").id
  results = []
  for num_threads in (1, 4):
    batch = Batch(model, N, num_threads=num_threads)
    mass = batch.expand("body_mass")
    mass[:, pole] = np.linspace(0.1, 1.0, N)
    batch.set_const()
    # Everything mj_setConst derived is now per sim, in native dtype.
    subtree = batch.expand("body_subtreemass")
    np.testing.assert_allclose(subtree[:, pole], mass[:, pole])
    assert batch.expand("dof_invweight0").shape == (N, model.nv)
    batch.bind("ctrl")[:, 0] = 1.0
    qvel = batch.bind("qvel")
    batch.step(nstep=20)
    results.append(qvel.copy())
  np.testing.assert_array_equal(results[0], results[1])
  ref = mujoco.MjModel.from_xml_string(XML)
  ref.body_mass[pole] = 1.0
  mujoco.mj_setConst(ref, mujoco.MjData(ref))
  d = mujoco.MjData(ref)
  d.ctrl[0] = 1.0
  for _ in range(20):
    mujoco.mj_step(ref, d)
  np.testing.assert_array_equal(results[0][-1], d.qvel)


def test_gravcomp_flag_is_per_sim(model):
  pole = model.body("pole").id
  results = []
  for num_threads in (1, 4):
    batch = Batch(model, N, num_threads=num_threads)
    gravcomp = batch.expand("body_gravcomp")
    gravcomp[::2, pole] = 1.0
    batch.set_const()
    batch.bind("qpos")[:, 1] = 0.3
    qvel = batch.bind("qvel")
    batch.step(nstep=5)
    results.append(qvel.copy())
  np.testing.assert_array_equal(results[0], results[1])
  assert np.all(results[0][::2, 1] != results[0][1::2, 1])


def test_reset_keyframe_and_expanded_qpos0(model):
  batch = Batch(model, N)
  qpos0 = batch.expand("qpos0")
  qpos0[:, 1] = np.arange(N) * 0.1
  qpos = batch.bind("qpos")
  batch.bind("ctrl")[:, 0] = 1.0
  batch.step()
  qpos[5, 0] = 0.25  # A write survives a reset of that sim and lands after it.
  xpos = batch.bind("xpos")
  batch.reset(np.array([2, 5]))
  np.testing.assert_array_equal(qpos[2], qpos0[2])
  assert qpos[5, 0] == 0.25
  np.testing.assert_array_equal(qpos[5, 1:], qpos0[5, 1:])
  assert qpos[0, 0] != 0.0
  assert xpos[2, 1, 2] != 0.0  # reset forwards
  mask = np.zeros(N, dtype=bool)
  mask[0] = True
  batch.reset(mask)
  assert qpos[0, 0] == 0.0
  batch.reset(keyframe=0)
  np.testing.assert_array_equal(qpos, np.tile(model.key_qpos[0], (N, 1)))
  for bad in (1, -2):
    with pytest.raises(ValueError):
      batch.reset(keyframe=bad)


def test_ids_validation_and_time(model):
  batch = Batch(model, N, num_threads=3)
  time = batch.bind("time")
  batch.step(np.array([1, 3]))
  np.testing.assert_array_equal(time, [0, 0.002, 0, 0.002, 0, 0, 0, 0])
  batch.step(np.array([1, 3], dtype=np.int32))
  np.testing.assert_array_equal(time, [0, 0.004, 0, 0.004, 0, 0, 0, 0])
  for bad in ([N], [1, 1], [3, 1], [1.0, 2.0]):
    with pytest.raises(ValueError):
      batch.step(np.array(bad))


def test_warning_counters(model):
  batch = Batch(model, N)
  warning = batch.bind("warning")
  batch.bind("qpos")[2] = np.nan
  batch.step()
  assert warning.shape == (N, mujoco.mjtWarning.mjNWARNING.value, 2)
  assert warning[2, mujoco.mjtWarning.mjWARN_BADQPOS, 1] == 1
  assert warning[:, :, 1].sum() == 1


@pytest.mark.parametrize("num_threads", [1, 2])
def test_mujoco_error_becomes_exception(model, num_threads):
  batch = Batch(model, N, num_threads=num_threads)
  mass = batch.expand("body_mass")
  mass[3, model.body("puck").id] = 0.0
  mass[5, model.body("pole").id] = 2.0
  with pytest.raises(RuntimeError, match="sim 3"):
    batch.set_const()
  # The other sims still ran; a later call works after fixing the input.
  assert batch.expand("body_subtreemass")[5, model.body("pole").id] == 2.0
  mass[3, model.body("puck").id] = 1.0
  batch.set_const()


def test_set_const_subset_keeps_every_sim_consistent(model):
  batch = Batch(model, N)
  pole = model.body("pole").id
  batch.expand("body_mass")[:, pole] = 5.0
  batch.set_const(np.array([0, 1]))
  np.testing.assert_array_equal(batch.expand("body_subtreemass")[:, pole], 5.0)


def test_concurrent_calls_are_serialized(model):
  batch = Batch(model, N, num_threads=4)
  time = batch.bind("time")

  other = Batch(model, N, num_threads=4)
  other_time = other.bind("time")

  def work(b, nstep):
    for _ in range(50):
      b.step(nstep=nstep)

  threads = [threading.Thread(target=work, args=a) for a in ((batch, 1), (batch, 5), (other, 2))]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  np.testing.assert_allclose(time, 300 * 0.002)
  np.testing.assert_allclose(other_time, 100 * 0.002)


def test_named_views(model):
  batch = Batch(model, N)
  hinge, tip = batch.joint("hinge"), batch.site("tip")
  assert hinge.qpos.shape == (N, 1) and hinge.qvel.shape == (N, 1)
  batch.bind("ctrl")[:, 0] = 1.0
  batch.step(nstep=10)
  np.testing.assert_array_equal(hinge.qpos[:, 0], batch.bind("qpos")[:, 1])
  np.testing.assert_array_equal(tip.xpos, batch.bind("site_xpos")[:, model.site("tip").id])
  assert batch.body("pole").xquat.shape == (N, 4)
  assert batch.body("pole").xpos.shape == (N, 3)


def test_state_rows_copy_restore_and_compose(model):
  batch = Batch(model, N)
  state, qpos, xpos = batch.bind("state"), batch.bind("qpos"), batch.bind("xpos")
  assert batch.nstate == mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
  assert state.shape == (N, batch.nstate) and state.dtype == np.float64
  batch.bind("ctrl")[:] = np.arange(N)[:, None] * 0.1
  batch.step(nstep=5)
  # Copying a row copies the physics: sim 1 becomes sim 0 in every field.
  assert not np.array_equal(qpos[0], qpos[1])
  state[1] = state[0]
  batch.forward(np.array([0, 1]))  # sim 0's derived fields lag by a substep
  np.testing.assert_array_equal(qpos[1], qpos[0])
  np.testing.assert_array_equal(xpos[1], xpos[0])
  # A state write and a field write compose, the field write winning on its overlap,
  # and the step matches mj_setState + mj_step on the same data.
  data = mujoco.MjData(model)
  saved = state[3].copy()
  batch.step(nstep=3)
  state[3] = saved
  qpos[3, 0] = 0.7
  mujoco.mj_setState(model, data, saved, mujoco.mjtState.mjSTATE_INTEGRATION)
  data.qpos[0], data.ctrl[:] = 0.7, batch.bind("ctrl")[3]
  batch.step(np.array([3]))
  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(qpos[3], data.qpos)
  np.testing.assert_array_equal(state[3, 1 : 1 + model.nq], data.qpos)
  # Round trip: restoring the rows replays the same steps bit for bit.
  before = state.copy()
  batch.step(nstep=2)
  once = state.copy()
  state[:] = before
  batch.step(nstep=2)
  np.testing.assert_array_equal(state, once)
  # reset discards a pending state write; float32 is refused; the view is the same.
  state[2] = once[5]
  batch.reset(np.array([2]))
  np.testing.assert_array_equal(qpos[2], 0.0)
  with pytest.raises(ValueError):
    batch.bind("state", np.float32)
  assert np.shares_memory(batch.bind("state"), state)


HFIELD_XML = """
<mujoco>
  <asset>
    <hfield name="hf" nrow="9" ncol="9" size="1 1 0.5 0.1"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="hf"/>
    <body name="cart" pos="0 0 0.2">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ref_hfield(model, qpos_row, geom, body, offsets):
  d = mujoco.MjData(model)
  d.qpos[:] = qpos_row
  mujoco.mj_forward(model, d)
  hfield = model.geom_dataid[geom]
  nrow, ncol = model.hfield_nrow[hfield], model.hfield_ncol[hfield]
  size = model.hfield_size[hfield]
  data = model.hfield_data[hfield * nrow * ncol : (hfield + 1) * nrow * ncol].reshape(nrow, ncol)
  gpos, gmat = d.geom_xpos[geom], d.geom_xmat[geom].reshape(3, 3)
  bpos = d.xpos[body]
  out = np.zeros(len(offsets))
  for k, (ox, oy) in enumerate(offsets):
    w = np.array([bpos[0] + ox, bpos[1] + oy, gpos[2]])
    lp = gmat.T @ (w - gpos)
    fx = np.clip((lp[0] / size[0] + 1.0) * 0.5 * (ncol - 1), 0, ncol - 1.001)
    fy = np.clip((lp[1] / size[1] + 1.0) * 0.5 * (nrow - 1), 0, nrow - 1.001)
    ix, iy = int(fx), int(fy)
    sx, sy = fx - ix, fy - iy
    h = (
      (1 - sx) * (1 - sy) * data[iy, ix]
      + sx * (1 - sy) * data[iy, min(ix + 1, ncol - 1)]
      + (1 - sx) * sy * data[min(iy + 1, nrow - 1), ix]
      + sx * sy * data[min(iy + 1, nrow - 1), min(ix + 1, ncol - 1)]
    )
    out[k] = h * size[2]
  return out


def test_jac_site_matches_serial(model):
  batch = Batch(model, N, num_threads=3)
  rng = np.random.default_rng(0)
  qpos = batch.bind("qpos")
  qpos[:] = model.qpos0 + rng.uniform(-0.2, 0.2, (N, model.nq))
  expected_qpos = qpos.copy()
  site = model.site("tip").id
  jacp, jacr = batch.jac_site("tip")
  assert jacp.shape == (N, 3, model.nv) and jacr.shape == (N, 3, model.nv)
  for i in range(N):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[i]
    mujoco.mj_kinematics(model, d)
    mujoco.mj_comPos(model, d)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, d, jp, jr, site)
    np.testing.assert_array_equal(jacp[i], jp)
    np.testing.assert_array_equal(jacr[i], jr)
  # A subset fills one row per selected sim, in selection order.
  ids = np.array([1, 3, 4])
  sub_p, sub_r = batch.jac_site("tip", ids=ids)
  np.testing.assert_array_equal(sub_p, jacp[ids])
  np.testing.assert_array_equal(sub_r, jacr[ids])
  mask = np.zeros(N, dtype=bool)
  mask[5] = True
  np.testing.assert_array_equal(batch.jac_site("tip", ids=mask)[0], jacp[5][None])
  # The call runs kinematics only: qpos is untouched.
  np.testing.assert_array_equal(qpos, expected_qpos)
  from mjbatch._bindings import Batch as RawBatch

  raw = RawBatch(model, 2)
  with pytest.raises(ValueError, match="out of range"):
    raw.jac_site(99, None, np.zeros((2, 3, model.nv)))
  with pytest.raises(ValueError, match="both"):
    raw.jac_site(0, None, None)
  with pytest.raises(ValueError, match="shape"):
    raw.jac_site(0, np.zeros((2, 3, model.nv + 1)), None)


def test_sample_hfield_matches_reference():
  model = mujoco.MjModel.from_xml_string(HFIELD_XML)
  hfield = 0
  nrow, ncol = model.hfield_nrow[hfield], model.hfield_ncol[hfield]
  model.hfield_data[: nrow * ncol] = np.linspace(0, 1, nrow * ncol)  # known ramp
  geom = model.geom("terrain").id
  body = model.body("cart").id
  batch = Batch(model, N, num_threads=3)
  rng = np.random.default_rng(1)
  qpos = batch.bind("qpos")
  qpos[:] = model.qpos0 + rng.uniform(-0.5, 0.5, (N, model.nq))
  offsets = np.array([[i * 0.06, j * 0.06] for i in (-1, 0, 1) for j in (-1, 0, 1)])
  out = batch.sample_hfield("terrain", "cart", offsets)
  assert out.shape == (N, len(offsets))
  for i in range(N):
    np.testing.assert_allclose(out[i], _ref_hfield(model, qpos[i], geom, body, offsets))
  # Points outside the grid clamp to the border height.
  far = np.array([[5.0, 5.0], [-5.0, -5.0]])
  out_far = batch.sample_hfield("terrain", "cart", far, ids=np.array([0]))
  np.testing.assert_allclose(out_far[0], _ref_hfield(model, qpos[0], geom, body, far))
  # Subset rows follow the selection.
  ids = np.array([2, 6])
  np.testing.assert_allclose(batch.sample_hfield("terrain", "cart", offsets, ids=ids), out[ids])
  from mjbatch._bindings import Batch as RawBatch

  raw = RawBatch(model, 2)
  with pytest.raises(ValueError, match="not a hfield"):
    raw.sample_hfield(1, body, offsets, np.zeros((2, len(offsets))))
  with pytest.raises(ValueError, match="out of range"):
    raw.sample_hfield(geom, 99, offsets, np.zeros((2, len(offsets))))
  with pytest.raises(ValueError, match="offsets"):
    raw.sample_hfield(geom, body, np.zeros(3), np.zeros((2, 3)))


# A hfield geom away from the origin with its own rotation, so its pose moves
# the sampling frame; the cart is a free body, so its yaw drives the grid.
HFIELD_ROT_XML = """
<mujoco>
  <asset>
    <hfield name="hf" nrow="9" ncol="9" size="1 1 0.5 0.1"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="hf" pos="0.3 -0.2 0.1"/>
    <body name="cart" pos="0 0 0.2">
      <freejoint name="root"/>
      <geom type="sphere" size=".05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _rot_hfield_model():
  model = mujoco.MjModel.from_xml_string(HFIELD_ROT_XML)
  nrow, ncol = model.hfield_nrow[0], model.hfield_ncol[0]
  model.hfield_data[: nrow * ncol] = np.linspace(0, 1, nrow * ncol)  # known ramp
  return model


def _place_cart(rng, qpos):
  """Randomize the cart's planar position, with pure-yaw rotations."""
  qpos[:, 0:2] += rng.uniform(-0.5, 0.5, (qpos.shape[0], 2))
  yaw = rng.uniform(-np.pi, np.pi, qpos.shape[0])
  qpos[:, 3:7] = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=1)


def test_sample_hfield_yaw_alignment():
  model = _rot_hfield_model()
  geom, body = model.geom("terrain").id, model.body("cart").id
  batch = Batch(model, N, num_threads=3)
  rng = np.random.default_rng(2)
  qpos = batch.bind("qpos")
  qpos[:] = model.qpos0
  _place_cart(rng, qpos)
  # A per-sim terrain orientation through expanded geom_quat moves the
  # sampling frame; the grid itself follows the cart's yaw.
  quat = batch.expand("geom_quat")
  tq = rng.normal(size=(N, 4))
  quat[:, geom] = tq / np.linalg.norm(tq, axis=1, keepdims=True)
  offsets = np.array([[i * 0.06, j * 0.06] for i in (-1, 0, 1) for j in (-1, 0, 1)])
  out = batch.sample_hfield("terrain", "cart", offsets, alignment="yaw")
  assert out.shape == (N, len(offsets))
  for i in range(N):
    model.geom_quat[geom] = quat[i, geom]
    ref = _ref_hfield_yaw(model, qpos[i], geom, body, offsets)
    np.testing.assert_allclose(out[i], ref, rtol=1e-7, atol=1e-12)
  # Subset rows follow the selection.
  ids = np.array([2, 6])
  sub = batch.sample_hfield("terrain", "cart", offsets, ids=ids, alignment="yaw")
  np.testing.assert_allclose(sub, out[ids])


def _ref_hfield_yaw(model, qpos_row, geom, body, offsets):
  d = mujoco.MjData(model)
  d.qpos[:] = qpos_row
  mujoco.mj_forward(model, d)
  hfield = model.geom_dataid[geom]
  nrow, ncol = model.hfield_nrow[hfield], model.hfield_ncol[hfield]
  size = model.hfield_size[hfield]
  data = model.hfield_data[hfield * nrow * ncol : (hfield + 1) * nrow * ncol].reshape(nrow, ncol)
  gpos, gmat = d.geom_xpos[geom], d.geom_xmat[geom].reshape(3, 3)
  bpos = d.xpos[body]
  bmat = d.xmat[body].reshape(3, 3)
  out = np.zeros(len(offsets))
  for k, (ox, oy) in enumerate(offsets):
    yaw = np.arctan2(bmat[1, 0], bmat[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.array([c * ox - s * oy, s * ox + c * oy, 0.0])
    w = bpos + r
    w[2] = gpos[2]  # the grid samples in the geom-center plane
    lp = gmat.T @ (w - gpos)
    fx = np.clip((lp[0] / size[0] + 1.0) * 0.5 * (ncol - 1), 0, ncol - 1.001)
    fy = np.clip((lp[1] / size[1] + 1.0) * 0.5 * (nrow - 1), 0, nrow - 1.001)
    ix, iy = int(fx), int(fy)
    sx, sy = fx - ix, fy - iy
    h = (
      (1 - sx) * (1 - sy) * data[iy, ix]
      + sx * (1 - sy) * data[iy, min(ix + 1, ncol - 1)]
      + (1 - sx) * sy * data[min(iy + 1, nrow - 1), ix]
      + sx * sy * data[min(iy + 1, nrow - 1), min(ix + 1, ncol - 1)]
    ) * size[2]
    out[k] = gpos[2] + (gmat @ np.array([lp[0], lp[1], h]))[2]
  return out


def test_sample_hfield_alignment_errors():
  model = _rot_hfield_model()
  geom, body = model.geom("terrain").id, model.body("cart").id
  offsets = np.zeros((3, 2))
  batch = Batch(model, 2)
  with pytest.raises(ValueError, match="alignment"):
    batch.sample_hfield("terrain", "cart", offsets, alignment="diagonal")
  from mjbatch._bindings import Batch as RawBatch

  raw = RawBatch(model, 2)
  with pytest.raises(ValueError, match="alignment"):
    raw.sample_hfield(geom, body, offsets, np.zeros((2, 3)), None, "diagonal")
  # Default is world: the same call spelled out matches the default.
  default = batch.sample_hfield("terrain", "cart", offsets)
  spelled = batch.sample_hfield("terrain", "cart", offsets, alignment="world")
  np.testing.assert_array_equal(default, spelled)


# A hfield, a site and a sensor on one model, so both query ops and a derived
# bound field are exercised together.
QUERY_XML = """
<mujoco>
  <asset>
    <hfield name="hf" nrow="9" ncol="9" size="1 1 0.5 0.1"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="hf"/>
    <body name="cart" pos="0 0 0.2">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".05" mass="1"/>
      <site name="base"/>
    </body>
  </worldbody>
  <sensor><jointpos joint="x"/><jointpos joint="y"/></sensor>
</mujoco>
"""


def test_query_ops_do_not_refresh_bound_views():
  # The query ops run mj_kinematics only and skip the bound-field copy-out:
  # their outputs are the caller-allocated rows, and the bound views stay
  # byte-identical across a query call. The worker's shared mjData holds
  # another sim's stale derived fields, which must not leak into the views.
  model = mujoco.MjModel.from_xml_string(QUERY_XML)
  nrow, ncol = model.hfield_nrow[0], model.hfield_ncol[0]
  model.hfield_data[: nrow * ncol] = np.linspace(0, 1, nrow * ncol)  # known ramp
  batch = Batch(model, N, num_threads=2)
  qpos, sensordata = batch.bind("qpos"), batch.bind("sensordata")
  rng = np.random.default_rng(5)
  qpos[:] = model.qpos0 + rng.uniform(-0.5, 0.5, (N, model.nq))
  batch.step(nstep=3)
  qpos_before, sensordata_before = qpos.copy(), sensordata.copy()
  offsets = np.array([[0.0, 0.0], [0.12, -0.06], [-0.24, 0.18]])
  jacp, jacr = batch.jac_site("base")
  out = batch.sample_hfield("terrain", "cart", offsets)
  np.testing.assert_array_equal(qpos, qpos_before)
  np.testing.assert_array_equal(sensordata, sensordata_before)
  # The query results are unchanged: checked against serial references.
  geom, body, site = (model.geom("terrain").id, model.body("cart").id, model.site("base").id)
  for i in range(N):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos_before[i]
    mujoco.mj_kinematics(model, d)
    mujoco.mj_comPos(model, d)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, d, jp, jr, site)
    np.testing.assert_array_equal(jacp[i], jp)
    np.testing.assert_array_equal(jacr[i], jr)
    np.testing.assert_allclose(out[i], _ref_hfield(model, qpos_before[i], geom, body, offsets))
