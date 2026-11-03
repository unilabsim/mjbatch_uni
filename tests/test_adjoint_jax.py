# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from mjbatch import Batch
from mjbatch.adjoint import _raw_qpos_to_tangent, _tangent_to_raw_qpos, make_step_vjp
from test_adjoint_step_vjp import XML, adjoint_library, serial_step

jax.config.update("jax_enable_x64", True)


def loss(qpos, qvel):
  quaternion_weights = np.array([0.3, -0.2, 0.5, -0.4])
  return np.sum(qpos[..., :3]) + np.sum(qpos[..., 3:] * quaternion_weights) + np.sum(qvel)


def stock_rollout(model, qpos, qvel, steps):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  for _ in range(steps):
    mujoco.mj_step(model, data)
  return data.qpos.copy(), data.qvel.copy()


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 16, 64])
def test_jax_custom_vjp_bptt_matches_stock_finite_differences(steps):
  model = mujoco.MjModel.from_xml_string(XML)
  batch = Batch(model, 2, num_threads=2)
  batch.reset()
  initial_qpos = np.tile(model.qpos0, (2, 1))
  initial_qvel = np.tile(np.array([0.4, -0.3, 0.2, 5.0, -3.0, 2.0]), (2, 1))
  initial_qpos[1, 0] += 0.025
  initial_qvel[1] += np.array([-0.07, 0.11, 0.04, 0.15, -0.09, 0.12])
  step = make_step_vjp(batch)

  def rollout(qpos, qvel):
    def scan_step(state, _):
      (next_qpos, next_qvel), status = step(*state)
      return (next_qpos, next_qvel), status

    (final_qpos, final_qvel), statuses = jax.lax.scan(scan_step, (qpos, qvel), None, length=steps)
    return loss(final_qpos, final_qvel), statuses

  value, statuses = jax.jit(rollout)(initial_qpos, initial_qvel)
  expected = [stock_rollout(model, initial_qpos[i], initial_qvel[i], steps) for i in range(2)]
  expected_qpos = np.stack([state[0] for state in expected])
  expected_qvel = np.stack([state[1] for state in expected])
  np.testing.assert_array_equal(statuses, np.zeros((steps, 2), dtype=np.int32))
  np.testing.assert_allclose(value, loss(expected_qpos, expected_qvel), rtol=2e-12, atol=2e-11)
  gradient = jax.jit(jax.grad(lambda qpos, qvel: rollout(qpos, qvel)[0], argnums=(0, 1)))(
    initial_qpos, initial_qvel
  )
  assert gradient[0].shape == initial_qpos.shape
  assert gradient[1].shape == initial_qvel.shape

  errors = []
  relative_errors = []
  for state_index, state in enumerate((initial_qpos, initial_qvel)):
    for row in range(2):
      for column in range(state.shape[1]):
        direction = np.zeros(state.shape[1])
        direction[column] = 1.0
        differences = []
        for epsilon in (1e-5, 2e-6):
          values = []
          for sign in (-1.0, 1.0):
            perturbed = state.copy()
            perturbed[row] += sign * epsilon * direction
            qpos, qvel = (
              stock_rollout(model, perturbed[row], initial_qvel[row], steps)
              if state_index == 0
              else stock_rollout(model, initial_qpos[row], perturbed[row], steps)
            )
            values.append(loss(qpos, qvel))
          differences.append((values[1] - values[0]) / (2.0 * epsilon))
        actual = gradient[state_index][row, column]
        errors.append(abs(actual - differences[1]))
        relative_errors.append(errors[-1] / max(1.0, abs(differences[1])))
        assert abs(differences[0] - differences[1]) <= max(2e-5, 3e-4 * abs(differences[1]))
        np.testing.assert_allclose(
          actual,
          differences[1],
          rtol=2e-4,
          atol=2e-4,
          err_msg=f"state={state_index} row={row} column={column} FD={differences}",
        )
  print(f"{steps}-step BPTT FD absolute errors: {errors}")
  print(f"{steps}-step BPTT FD scaled relative errors: {relative_errors}")


def test_serial_c_abi_64_step_vjp_matches_jax_and_stock_finite_differences():
  model = mujoco.MjModel.from_xml_string(XML)
  batch = Batch(model, 1, num_threads=1)
  batch.reset()
  initial_qpos = model.qpos0.copy()
  initial_qvel = np.array([0.4, -0.3, 0.2, 5.0, -3.0, 2.0])
  data = mujoco.MjData(model)
  data.qpos[:] = initial_qpos
  data.qvel[:] = initial_qvel
  states = []
  positions = [initial_qpos.copy()]
  for _ in range(64):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    states.append(state)
    mujoco.mj_step(model, data)
    positions.append(data.qpos.copy())
  final_state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
  mujoco.mj_getState(model, data, final_state, mujoco.mjtState.mjSTATE_INTEGRATION)
  states.append(final_state)
  bar_qpos = np.array([[1.0, 1.0, 1.0, 0.3, -0.2, 0.5, -0.4]])
  bar_qvel = np.ones((1, 6))
  lib = adjoint_library()
  manual_inputs = {}
  manual_raw = {}
  for index in range(63, -1, -1):
    previous = positions[index][None]
    current = positions[index + 1][None]
    tangent = _raw_qpos_to_tangent(batch, current, bar_qpos)
    manual_inputs[index] = (tangent[0].copy(), bar_qvel[0].copy())
    manual_raw[index] = bar_qpos.copy()
    status, grad_qpos, grad_qvel, _, _, _ = serial_step(
      lib, model, mujoco.MjData(model), states[index], tangent[0], bar_qvel[0]
    )
    assert status == 0
    bar_qpos = _tangent_to_raw_qpos(batch, previous, grad_qpos[None])
    bar_qvel = grad_qvel[None]

  class CheckedBatch(Batch):
    def step_vjp(self, *args):  # pyright: ignore[reportIncompatibleMethodOverride]
      state = self.bind("state").copy()
      index = int(round(state[0, 0] / model.opt.timestep))
      np.testing.assert_array_equal(state[0], states[index])
      np.testing.assert_allclose(
        args[0].reshape(-1), manual_inputs[index][0], rtol=1e-12, atol=1e-12, err_msg=f"index={index}"
      )
      np.testing.assert_allclose(
        args[1].reshape(-1), manual_inputs[index][1], rtol=1e-12, atol=1e-12, err_msg=f"index={index}"
      )
      super().step_vjp(*args)
      expected_status, expected_qpos, expected_qvel, _, _, _ = serial_step(
        lib, model, mujoco.MjData(model), state[0], args[0].reshape(-1), args[1].reshape(-1)
      )
      assert args[5][0] == expected_status
      np.testing.assert_allclose(args[2].reshape(-1), expected_qpos, rtol=1e-12, atol=1e-12)
      np.testing.assert_allclose(args[3].reshape(-1), expected_qvel, rtol=1e-12, atol=1e-12)
      actual_raw = _tangent_to_raw_qpos(self, positions[index][None], args[2].reshape(1, -1))
      expected_raw = manual_raw[index - 1] if index else bar_qpos
      np.testing.assert_allclose(actual_raw, expected_raw, rtol=1e-12, atol=1e-12, err_msg=f"index={index}")

  checked_batch = CheckedBatch(model, 1, num_threads=1)
  checked_batch.reset()
  step = make_step_vjp(checked_batch)

  def rollout(qpos, qvel):
    def scan_step(state, _):
      (next_qpos, next_qvel), _ = step(*state)
      return (next_qpos, next_qvel), None

    (final_qpos, final_qvel), _ = jax.lax.scan(scan_step, (qpos, qvel), None, length=64)
    return loss(final_qpos, final_qvel)

  jax_gradient = jax.jit(jax.grad(rollout, argnums=(0, 1)))(initial_qpos[None], initial_qvel[None])
  np.testing.assert_allclose(jax_gradient[0], bar_qpos, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(jax_gradient[1], bar_qvel, rtol=1e-10, atol=1e-10)
  for state_index, state in enumerate((initial_qpos, initial_qvel)):
    for column in range(state.size):
      direction = np.zeros(state.size)
      direction[column] = 1.0
      epsilon = 2e-6
      values = []
      for sign in (-1.0, 1.0):
        perturbed = state + sign * epsilon * direction
        qpos, qvel = (
          stock_rollout(model, perturbed, initial_qvel, 64)
          if state_index == 0
          else stock_rollout(model, initial_qpos, perturbed, 64)
        )
        values.append(loss(qpos, qvel))
      expected = (values[1] - values[0]) / (2 * epsilon)
      np.testing.assert_allclose(
        (bar_qpos if state_index == 0 else bar_qvel)[0, column],
        expected,
        rtol=2e-4,
        atol=2e-4,
        err_msg=f"state={state_index} column={column}",
      )


def test_jax_custom_vjp_zero_velocity_quaternion_projection():
  model = mujoco.MjModel.from_xml_string(XML)
  batch = Batch(model, 2, num_threads=2)
  batch.reset()
  qpos = np.tile(model.qpos0, (2, 1))
  qpos[1, 3:] *= 1.1
  qvel = jnp.zeros((2, model.nv))
  step = make_step_vjp(batch)
  gradient = jax.jit(jax.grad(lambda position: step(position, qvel)[0][0].sum()))(qpos)
  expected = np.zeros_like(qpos)
  expected[:, :3] = 1.0
  for row in range(2):
    quat = qpos[row, 3:]
    norm = np.linalg.norm(quat)
    unit = quat / norm
    expected[row, 3:] = (np.eye(4) - np.outer(unit, unit)) @ np.ones(4) / norm
  np.testing.assert_allclose(gradient, expected, rtol=1e-10, atol=1e-10)


def test_jax_custom_vjp_rejects_unsupported_models_in_backward():
  model = mujoco.MjModel.from_xml_string(
    """
    <mujoco>
      <option integrator="RK4"><flag island="disable"/></option>
      <worldbody><body><joint type="hinge"/><geom size=".1"/></body></worldbody>
    </mujoco>
    """
  )
  batch = Batch(model, 2, num_threads=2)
  batch.reset()
  step = make_step_vjp(batch)
  qpos = jnp.zeros(batch.bind("qpos").shape)
  qvel = jnp.zeros(batch.bind("qvel").shape)
  with pytest.raises(Exception, match="mjd_stepAdjoint failed"):
    jax.jit(jax.grad(lambda velocity: step(qpos, velocity)[0][1].sum()))(qvel).block_until_ready()
