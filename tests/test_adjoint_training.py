# SPDX-License-Identifier: Apache-2.0

import os
import xml.etree.ElementTree as ET

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from mjbatch import Batch
from mjbatch.adjoint import _raw_qpos_to_tangent, _tangent_to_raw_qpos, make_step_vjp
from test_adjoint_step_vjp import adjoint_library, array

jax.config.update("jax_enable_x64", True)


def load_g1():
  root = ET.parse(os.path.join(os.path.dirname(__file__), "../examples/assets/g1.xml")).getroot()
  for include in list(root.findall("include")):
    root.remove(include)
  compiler = root.find("compiler")
  if compiler is None:
    compiler = ET.SubElement(root, "compiler")
  compiler.set("discardvisual", "true")
  for parent in root.iter():
    for child in list(parent):
      if child.tag == "mesh" or (child.tag == "geom" and (child.get("type") == "mesh" or child.get("mesh"))):
        parent.remove(child)
  model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
  model.opt.iterations = 10
  assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  assert model.nv == 29 and model.nu == 23
  return model


def initial_state(model, num_sims):
  qpos = np.tile(model.qpos0, (num_sims, 1))
  qpos[:, 2] += 0.2
  return qpos, np.zeros((num_sims, model.nv))


def stock_rollout(model, qpos, qvel, ctrl, steps):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  data.ctrl[:] = ctrl
  for _ in range(steps):
    mujoco.mj_step(model, data)
  return float(np.mean(data.qvel**2))


def serial_rollout_vjp(model, qpos, qvel, ctrl, steps):
  lib = adjoint_library()
  batch = Batch(model, 1, num_threads=1)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  data.ctrl[:] = ctrl
  states = []
  positions = [data.qpos.copy()]
  for _ in range(steps):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    states.append(state)
    mujoco.mj_step(model, data)
    positions.append(data.qpos.copy())
  bar_qpos = np.zeros((1, model.nq))
  bar_qvel = 2 * data.qvel[None] / model.nv
  grad_ctrl = np.zeros(model.nu)
  for index in range(steps - 1, -1, -1):
    mujoco.mj_setState(model, data, states[index], mujoco.mjtState.mjSTATE_INTEGRATION)
    tangent = _raw_qpos_to_tangent(batch, positions[index + 1][None], bar_qpos)
    grad_qpos = np.empty(model.nv)
    grad_qvel = np.empty(model.nv)
    grad_u = np.empty(model.nu)
    ctx = lib.mj_makeAdjContext(model._address)
    assert ctx
    assert lib.mj_reserveAdjContext(ctx, 0, 0, 0, 0) == 0
    status = lib.mjd_stepAdjoint(
      model._address, data._address, ctx, array(tangent[0]), array(bar_qvel[0]),
      array(grad_qpos), array(grad_qvel), array(grad_u),
    )
    lib.mj_deleteAdjContext(ctx)
    assert status == 0
    grad_ctrl += grad_u
    bar_qpos = _tangent_to_raw_qpos(batch, positions[index][None], grad_qpos[None])
    bar_qvel = grad_qvel[None]
  return grad_ctrl


def test_g1_custom_vjp_ctrl_matches_stock_fd():
  model = load_g1()
  batch = Batch(model, 8, num_threads=8)
  batch.reset()
  qpos, qvel = initial_state(model, 8)
  rng = np.random.default_rng(12)
  ctrl = 0.01 * rng.standard_normal((8, model.nu))
  step = make_step_vjp(batch, expose_ctrl=True)

  def rollout(value):
    def scan_step(state, _):
      (next_qpos, next_qvel), _ = step(state[0], state[1], value)
      return (next_qpos, next_qvel), None

    (final_qpos, final_qvel), _ = jax.lax.scan(scan_step, (qpos, qvel), None, length=16)
    return jnp.mean(final_qvel**2)

  value_and_grad = jax.jit(jax.value_and_grad(rollout))
  value, gradient = value_and_grad(ctrl)
  assert np.isfinite(value) and np.all(np.isfinite(gradient))
  serial_gradient = serial_rollout_vjp(model, qpos[0], qvel[0], ctrl[0], 16)
  np.testing.assert_allclose(gradient[0], serial_gradient / 8, rtol=1e-10, atol=1e-10)
  for column in range(model.nu):
    epsilon = 1e-3
    losses = []
    for sign in (-1.0, 1.0):
      perturbed = ctrl[0].copy()
      perturbed[column] += sign * epsilon
      losses.append(stock_rollout(model, qpos[0], qvel[0], perturbed, 16))
    expected = (losses[1] - losses[0]) / (2 * epsilon)
    np.testing.assert_allclose(
      gradient[0, column], expected / 8, rtol=3e-4, atol=3e-5, err_msg=f"ctrl column={column}"
    )


def test_g1_4096_envs_64_step_bptt_and_training_update():
  model = load_g1()
  num_sims = 4096
  batch = Batch(model, num_sims, num_threads=32)
  batch.reset()
  qpos, qvel = initial_state(model, num_sims)
  rng = np.random.default_rng(23)
  initial_ctrl = 0.01 * rng.standard_normal((num_sims, model.nu))
  step = make_step_vjp(batch, expose_ctrl=True)

  def rollout(value):
    def scan_step(state, _):
      (next_qpos, next_qvel), _ = step(state[0], state[1], value)
      return (next_qpos, next_qvel), None

    (final_qpos, final_qvel), _ = jax.lax.scan(scan_step, (qpos, qvel), None, length=64)
    return jnp.mean(final_qvel**2)

  initial_loss, gradient = jax.jit(jax.value_and_grad(rollout))(initial_ctrl)
  assert initial_loss.shape == ()
  assert gradient.shape == initial_ctrl.shape
  assert np.isfinite(initial_loss)
  assert np.all(np.isfinite(gradient))
  assert np.linalg.norm(gradient) > 0

  epsilon = 1e-5
  losses = []
  for sign in (-1.0, 1.0):
    perturbed = initial_ctrl[0].copy()
    perturbed[0] += sign * epsilon
    losses.append(stock_rollout(model, qpos[0], qvel[0], perturbed, 64))
  expected = (losses[1] - losses[0]) / (2 * epsilon)
  np.testing.assert_allclose(gradient[0, 0], expected / num_sims, rtol=2e-3, atol=3e-5)

  learning_rate = 1e-3
  updated_ctrl = initial_ctrl - learning_rate * gradient
  updated_loss = jax.jit(rollout)(updated_ctrl)
  assert np.isfinite(updated_loss)
  assert updated_loss < initial_loss
  print(
    f"4096x64 G1 BPTT: initial_loss={initial_loss:.12g} "
    f"updated_loss={updated_loss:.12g} first_ctrl_fd={expected:.12g}"
  )
