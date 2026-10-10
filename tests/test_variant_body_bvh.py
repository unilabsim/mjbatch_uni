# SPDX-License-Identifier: Apache-2.0

import mujoco
import numpy as np
import pytest

from mjbatch import Batch, RecomputeLevel
from mjbatch.variants import VariantPack


def _tool_spec(handle_half_length: float) -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(
    f"""
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="obstacle" type="box" pos="0 0 0.55" size="0.1 0.1 0.1"/>
    <body name="tool" pos="0 0 0.2">
      <freejoint name="free"/>
      <geom name="handle" type="capsule" size="0.02 {handle_half_length}"/>
    </body>
  </worldbody>
</mujoco>
"""
  )


def test_body_bvh_aabb_metadata_and_expandable_prefix():
  model = _tool_spec(0.4).compile()
  body_rows = int(model.body_bvhnum.sum())
  specs = Batch(model, 2).model_field_specs()

  assert specs["bvh_aabb"].asset
  assert not specs["bvh_aabb"].writable
  assert specs["body_bvh_aabb"].shape == (body_rows, 6)
  assert specs["body_bvh_aabb"].dtype == np.dtype(np.float64)
  assert specs["body_bvh_aabb"].writable
  assert not specs["body_bvh_aabb"].asset
  assert specs["body_bvh_aabb"].recompute == RecomputeLevel.NONE
  assert Batch(model, 2).expand("body_bvh_aabb").shape == (2, body_rows, 6)

  with pytest.raises(ValueError, match="asset data"):
    Batch(model, 2).expand("bvh_aabb")


def test_variant_pack_scatters_exact_analytic_body_bvh_aabb():
  """A long analytic handle must retain its own broad-phase AABB (#40).

  The short variant is canonical. With the old canonical-only ``bvh_aabb``, its
  0.12 m long body AABB culls the obstacle before narrow phase for the long
  variant, whose capsule actually reaches the obstacle. Exact per-variant body
  rows make the batch match each independently compiled MuJoCo model.
  """

  specs = [_tool_spec(0.1), _tool_spec(0.4)]
  references = [spec.compile() for spec in specs]
  pack = VariantPack.from_specs(specs)
  body_rows = int(pack.model.body_bvhnum.sum())
  tool = pack.model.body("tool").id
  tool_bvh = int(pack.model.body_bvhadr[tool])

  assert pack.fields["body_bvh_aabb"].shape == (2, body_rows, 6)
  np.testing.assert_array_equal(
    pack.fields["body_bvh_aabb"][:, tool_bvh],
    [reference.bvh_aabb[tool_bvh] for reference in references],
  )
  assert pack.fields["body_bvh_aabb"][0, tool_bvh, 5] == pytest.approx(0.12)
  assert pack.fields["body_bvh_aabb"][1, tool_bvh, 5] == pytest.approx(0.42)
  short_data, long_data = (mujoco.MjData(reference) for reference in references)
  mujoco.mj_forward(references[0], short_data)
  mujoco.mj_forward(references[1], long_data)
  assert short_data.ncon == 0
  assert long_data.ncon == 1

  batch = Batch.from_variant_pack(pack, 2, np.array([0, 1]), num_threads=1)
  np.testing.assert_array_equal(batch.expand("body_bvh_aabb"), pack.fields["body_bvh_aabb"])
  state = batch.bind("state")
  batch.step(nstep=5)

  for variant, reference in enumerate(references):
    data = mujoco.MjData(reference)
    for _ in range(5):
      mujoco.mj_step(reference, data)
    expected = np.empty(batch.nstate)
    mujoco.mj_getState(reference, data, expected, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.testing.assert_array_equal(state[variant], expected)


def test_empty_body_bvh_prefix_is_expandable():
  spec = mujoco.MjSpec.from_string(
    """
<mujoco>
  <worldbody>
    <body name="body">
      <freejoint name="free"/>
      <inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
  )
  pack = VariantPack.from_specs([spec, spec])
  assert pack.fields["body_bvh_aabb"].shape == (2, 0, 6)
  batch = Batch.from_variant_pack(pack, 2, np.array([0, 1]))
  assert batch.expand("body_bvh_aabb").shape == (2, 0, 6)


def test_variant_pack_fails_closed_on_body_bvh_topology_mismatch():
  def make_spec(geom_count: int) -> mujoco.MjSpec:
    geoms = "\n".join(f'<geom name="geom{i}" type="box" size="0.02 0.02 0.02"/>' for i in range(geom_count))
    return mujoco.MjSpec.from_string(
      f"""
<mujoco>
  <worldbody>
    <body name="body">
      <freejoint name="free"/>
      {geoms}
    </body>
  </worldbody>
</mujoco>
"""
    )

  with pytest.raises(ValueError, match="body broadphase BVH topology"):
    VariantPack.from_specs([make_spec(2), make_spec(1)])


def test_body_bvh_alignment_accepts_equivalent_permuted_tree():
  """Equivalent BVH child order is platform-dependent and must remain exact."""
  from mjbatch.variants import _align_body_bvh_aabb, _snapshot_variant

  def make_spec():
    return mujoco.MjSpec.from_string(
      """
<mujoco>
  <worldbody>
    <body name="body">
      <freejoint name="free"/>
      <geom name="a" type="box" pos="-0.2 0 0" size="0.05 0.05 0.05" mass="1"/>
      <geom name="b" type="box" pos="0.2 0 0" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    )

  spec = make_spec()
  model = spec.compile()
  snapshot = _snapshot_variant(spec, model)
  rows = int(model.body_bvhnum.sum())
  body = model.body("body").id
  adr = int(model.body_bvhadr[body])
  assert adr == 0 and rows == 3

  # Swap the two leaves while preserving the same unordered tree.  MuJoCo can
  # produce this ordering difference across platforms when geom extents change.
  left, right = int(model.bvh_child[adr, 0]), int(model.bvh_child[adr, 1])
  permuted = _snapshot_variant(spec, model)
  permuted.body_bvh_topology["bvh_child"][adr, 0] = right
  permuted.body_bvh_topology["bvh_child"][adr, 1] = left
  permuted.body_bvh_topology["bvh_nodeid"][adr + left] = model.bvh_nodeid[adr + right]
  permuted.body_bvh_topology["bvh_nodeid"][adr + right] = model.bvh_nodeid[adr + left]
  permuted.body_bvh_aabb[adr + left], permuted.body_bvh_aabb[adr + right] = (
    snapshot.body_bvh_aabb[adr + right].copy(),
    snapshot.body_bvh_aabb[adr + left].copy(),
  )

  aligned = _align_body_bvh_aabb([permuted], model, [np.arange(model.ngeom)])
  np.testing.assert_array_equal(aligned[0], snapshot.body_bvh_aabb)
