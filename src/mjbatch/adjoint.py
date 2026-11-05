# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import jax
import mujoco
import numpy as np
from jax.experimental import io_callback

_ordered_callback = partial(io_callback, ordered=True)

if TYPE_CHECKING:
  from mjbatch import Batch


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
  w, x, y, z = (quat[:, i] for i in range(4))
  return 0.5 * np.stack(
    (
      np.stack((-x, -y, -z), axis=1),
      np.stack((w, -z, y), axis=1),
      np.stack((z, w, -x), axis=1),
      np.stack((-y, x, w), axis=1),
    ),
    axis=1,
  )


def _raw_qpos_to_tangent(batch: "Batch", current: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
  result = np.empty((batch.num_sims, batch.model.nv), dtype=np.float64)
  for joint in range(batch.model.njnt):
    joint_type = batch.model.jnt_type[joint]
    padr = batch.model.jnt_qposadr[joint]
    vadr = batch.model.jnt_dofadr[joint]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
      result[:, vadr : vadr + 3] = cotangent[:, padr : padr + 3]
      padr += 3
      vadr += 3
    # NOTE: use == instead of `in (enum, enum)` — with numpy 2.5.4 + mujoco 3.17
    # pybind enums, `np.int32(0) in (mjJNT_FREE, mjJNT_BALL)` evaluates False,
    # which would leave result slots unwritten (np.empty garbage).
    if joint_type == mujoco.mjtJoint.mjJNT_FREE or joint_type == mujoco.mjtJoint.mjJNT_BALL:
      quaternion_slice = slice(padr, padr + 4)
      matrix = _quat_matrix(current[:, quaternion_slice])
      result[:, vadr : vadr + 3] = np.einsum("nij,ni->nj", matrix, cotangent[:, quaternion_slice])
    else:
      result[:, vadr] = cotangent[:, padr]
  return result


def _tangent_to_raw_qpos(batch: "Batch", qpos: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
  result = np.empty((batch.num_sims, batch.model.nq), dtype=np.float64)
  for joint in range(batch.model.njnt):
    joint_type = batch.model.jnt_type[joint]
    padr = batch.model.jnt_qposadr[joint]
    vadr = batch.model.jnt_dofadr[joint]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
      result[:, padr : padr + 3] = cotangent[:, vadr : vadr + 3]
      padr += 3
      vadr += 3
    if joint_type == mujoco.mjtJoint.mjJNT_FREE or joint_type == mujoco.mjtJoint.mjJNT_BALL:
      quat = qpos[:, padr : padr + 4]
      norm = np.linalg.norm(quat, axis=1, keepdims=True)
      unit = quat / norm
      matrix = _quat_matrix(unit)
      gradient = 4 * np.einsum("nij,nj->ni", matrix, cotangent[:, vadr : vadr + 3])
      result[:, padr : padr + 4] = (gradient - unit * np.sum(unit * gradient, axis=1, keepdims=True)) / norm
    else:
      result[:, padr] = cotangent[:, vadr]
  return result


def make_step_vjp(
  batch: "Batch", expose_ctrl: bool = False
) -> Callable[..., tuple[tuple[jax.Array, jax.Array], jax.Array]]:
  """Return a differentiable batched step backed by mjd_stepAdjoint.

  The primal callback commits the supplied state. The backward callback uses the
  committed step and the per-simulation status is returned as an auxiliary output.
  Set expose_ctrl to include the batched control input and its gradient.
  """
  if jax.default_backend() != "cpu":
    raise ValueError("mjbatch step_vjp requires the JAX CPU backend")
  qpos_view = batch.bind("qpos")
  qvel_view = batch.bind("qvel")
  ctrl_view = batch.bind("ctrl")
  state_view = batch.bind("state")
  qpos_shape = (batch.num_sims, batch.model.nq)
  qvel_shape = (batch.num_sims, batch.model.nv)
  qpos_struct = jax.ShapeDtypeStruct(qpos_shape, np.float64)
  qvel_struct = jax.ShapeDtypeStruct(qvel_shape, np.float64)
  ctrl_struct = jax.ShapeDtypeStruct((batch.num_sims, batch.model.nu), np.float64)
  state_struct = jax.ShapeDtypeStruct(state_view.shape, np.float64)
  status_struct = jax.ShapeDtypeStruct((batch.num_sims,), np.int32)

  def primal(qpos, qvel, ctrl=None):
    def callback(qpos_value, qvel_value, ctrl_value=None):
      qpos_view[:] = np.asarray(qpos_value).reshape(qpos_view.shape)
      qvel_view[:] = np.asarray(qvel_value).reshape(qvel_view.shape)
      if expose_ctrl:
        ctrl_view[:] = np.asarray(ctrl_value).reshape(ctrl_view.shape)
      state_before = state_view.copy()
      batch.step()
      status = np.zeros(batch.num_sims, dtype=np.int32)
      return (
        qpos_view.reshape(qpos_shape).copy(),
        qvel_view.reshape(qvel_shape).copy(),
        status,
        state_before,
        state_view.copy(),
      )

    if expose_ctrl:
      return _ordered_callback(
        callback, (qpos_struct, qvel_struct, status_struct, state_struct, state_struct), qpos, qvel, ctrl
      )
    return _ordered_callback(
      callback, (qpos_struct, qvel_struct, status_struct, state_struct, state_struct), qpos, qvel
    )

  @jax.custom_vjp
  def step(qpos, qvel, ctrl=None):
    qpos_next, qvel_next, status, _, _ = primal(qpos, qvel, ctrl)
    return (qpos_next, qvel_next), status

  def forward(qpos, qvel, ctrl=None):
    qpos_next, qvel_next, status, state_before, state_after = primal(qpos, qvel, ctrl)
    return ((qpos_next, qvel_next), status), (state_before, state_after)

  def backward(residual, cotangents):
    state_before, state_after = residual
    (bar_qpos, bar_qvel), _ = cotangents

    def callback(state_before_value, state_after_value, bar_qpos_value, bar_qvel_value):
      state_view[:] = state_before_value
      qpos_array = state_before_value[:, 1 : 1 + batch.model.nq]
      qpos_next_array = state_after_value[:, 1 : 1 + batch.model.nq]
      bar_qpos_tangent = _raw_qpos_to_tangent(batch, qpos_next_array, np.asarray(bar_qpos_value))
      bar_qvel_array = np.array(bar_qvel_value, dtype=np.float64, order="C", copy=True)
      grad_qpos = np.empty((batch.num_sims, batch.model.nv), dtype=np.float64)
      grad_qvel = np.empty_like(bar_qvel_array)
      grad_ctrl = np.empty((batch.num_sims, batch.model.nu), dtype=np.float64) if expose_ctrl else None
      backward_status = np.empty(batch.num_sims, dtype=np.int32)
      batch.step_vjp(
        np.ascontiguousarray(bar_qpos_tangent).reshape(qvel_view.shape),
        bar_qvel_array.reshape(qvel_view.shape),
        grad_qpos.reshape(qvel_view.shape),
        grad_qvel.reshape(qvel_view.shape),
        grad_ctrl,
        backward_status,
      )
      if np.any(backward_status):
        failed = np.flatnonzero(backward_status)
        raise RuntimeError(
          f"mjd_stepAdjoint failed for simulations {failed.tolist()}: "
          f"status {backward_status[failed].tolist()}"
        )
      if expose_ctrl:
        return _tangent_to_raw_qpos(batch, qpos_array, grad_qpos), grad_qvel, grad_ctrl
      return _tangent_to_raw_qpos(batch, qpos_array, grad_qpos), grad_qvel, None

    if expose_ctrl:
      return _ordered_callback(
        callback, (qpos_struct, qvel_struct, ctrl_struct), state_before, state_after, bar_qpos, bar_qvel
      )
    grad_qpos, grad_qvel = _ordered_callback(
      callback, (qpos_struct, qvel_struct), state_before, state_after, bar_qpos, bar_qvel
    )
    return grad_qpos, grad_qvel, None

  step.defvjp(forward, backward)  # pyright: ignore[reportFunctionMemberAccess]
  return step
