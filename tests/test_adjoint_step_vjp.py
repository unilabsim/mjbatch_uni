# SPDX-License-Identifier: Apache-2.0

import ctypes
import os

import mujoco
import numpy as np

from mjbatch import Batch

XML = """
<mujoco>
  <option solver="Newton" integrator="Euler" timestep=".002">
    <flag island="disable"/>
  </option>
  <worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40">
      <freejoint/>
      <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
    </body>
  </worldbody>
</mujoco>
"""

UNSUPPORTED_XML = """
<mujoco>
  <option solver="Newton" integrator="RK4"><flag island="disable"/></option>
  <worldbody>
    <body><joint type="hinge"/><geom size=".1"/></body>
    <body pos="1 0 0"><joint type="slide"/><geom size=".1"/></body>
  </worldbody>
</mujoco>
"""


def adjoint_library():
  lib = ctypes.CDLL(os.environ.get("MJBATCH_ADJOINT_LIBRARY", "libmujoco_adjoint.so.3.11.0"))
  lib.mj_makeAdjContext.argtypes = [ctypes.c_void_p]
  lib.mj_makeAdjContext.restype = ctypes.c_void_p
  lib.mj_reserveAdjContext.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
  lib.mj_deleteAdjContext.argtypes = [ctypes.c_void_p]
  lib.mjd_stepAdjoint.argtypes = [ctypes.c_void_p] * 7
  return lib


def array(values):
  return np.ctypeslib.as_ctypes(np.ascontiguousarray(values, dtype=np.float64))


def serial_step(lib, model, data, state, bar_qpos, bar_qvel):
  mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
  ctx = lib.mj_makeAdjContext(model._address)
  assert ctx
  assert lib.mj_reserveAdjContext(ctx, 0, 0, 0, 0) == 0
  grad_qpos = np.full(6, np.nan)
  grad_qvel = np.full(6, np.nan)
  status = lib.mjd_stepAdjoint(
    model._address,
    data._address,
    ctx,
    array(bar_qpos),
    array(bar_qvel),
    array(grad_qpos),
    array(grad_qvel),
    None,
  )
  lib.mj_deleteAdjContext(ctx)
  return status, grad_qpos, grad_qvel, data.qpos.copy(), data.qvel.copy(), data.time


def test_step_vjp_matches_serial_and_preserves_unselected_rows():
  lib = adjoint_library()
  model = mujoco.MjModel.from_xml_string(XML)
  batch = Batch(model, 8, num_threads=3)
  batch.reset()
  initial = batch.bind("state").copy()
  rng = np.random.default_rng(53)
  bar_qpos = rng.normal(size=(8, 6))
  bar_qvel = rng.normal(size=(8, 6))
  grad_qpos = np.full((8, 6), 123.0)
  grad_qvel = np.full((8, 6), 123.0)
  status = np.full(8, -99, dtype=np.int32)
  batch.step_vjp(bar_qpos, bar_qvel, grad_qpos, grad_qvel, None, status, np.array([1, 4, 6]))

  for i in range(8):
    if i not in (1, 4, 6):
      assert status[i] == -99
      np.testing.assert_array_equal(grad_qpos[i], np.full(6, 123.0))
      np.testing.assert_array_equal(grad_qvel[i], np.full(6, 123.0))
      np.testing.assert_array_equal(batch.bind("state")[i], initial[i])
      continue
    expected_status, expected_qpos, expected_qvel, qpos, qvel, time = serial_step(
      lib, model, mujoco.MjData(model), initial[i], bar_qpos[i], bar_qvel[i]
    )
    assert status[i] == expected_status == 0
    np.testing.assert_allclose(grad_qpos[i], expected_qpos, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad_qvel[i], expected_qvel, rtol=1e-12, atol=1e-12)
    state = batch.bind("state")[i]
    np.testing.assert_array_equal(state[1:8], qpos)
    np.testing.assert_array_equal(state[8:14], qvel)
    assert state[0] == time


def test_step_vjp_unsupported_status_does_not_commit_or_write_gradients():
  model = mujoco.MjModel.from_xml_string(UNSUPPORTED_XML)
  batch = Batch(model, 2, num_threads=2)
  batch.reset()
  initial = batch.bind("state").copy()
  nv = model.nv
  bar_qpos = np.ones((2, nv))
  bar_qvel = np.ones((2, nv))
  grad_qpos = np.full((2, nv), 123.0)
  grad_qvel = np.full((2, nv), 123.0)
  status = np.full(2, -99, dtype=np.int32)
  batch.step_vjp(bar_qpos, bar_qvel, grad_qpos, grad_qvel, None, status)
  np.testing.assert_array_equal(status, np.full(2, 2, dtype=np.int32))
  np.testing.assert_array_equal(grad_qpos, np.full((2, nv), 123.0))
  np.testing.assert_array_equal(grad_qvel, np.full((2, nv), 123.0))
  np.testing.assert_array_equal(batch.bind("state"), initial)
