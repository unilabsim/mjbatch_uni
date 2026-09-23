# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterator, Mapping

import mujoco
import numpy as np

from mjbatch._bindings import Batch as _Batch
from mjbatch.groups import ModelAffineBatch as ModelAffineBatch
from mjbatch.groups import TopologyGroup as TopologyGroup
from mjbatch.model import ModelFieldSpec as ModelFieldSpec
from mjbatch.model import RecomputeLevel as RecomputeLevel
from mjbatch.model import build_model_fields
from mjbatch.variants import VariantPack as VariantPack
from mjbatch.variants import VariantPackBuilder as VariantPackBuilder

_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}  # by mjtJoint: free, ball, slide, hinge
_DOF_WIDTH = {0: 6, 1: 3, 2: 1, 3: 1}


@dataclass(frozen=True)
class Joint:
  qpos: np.ndarray
  qvel: np.ndarray


@dataclass(frozen=True)
class Actuator:
  ctrl: np.ndarray
  force: np.ndarray


@dataclass(frozen=True)
class Body:
  xpos: np.ndarray
  xquat: np.ndarray
  cvel: np.ndarray


@dataclass(frozen=True)
class Site:
  xpos: np.ndarray
  xmat: np.ndarray


class Batch(_Batch):
  """See mjbatch._bindings.Batch. The named accessors return live (N, ...) views of
  the corresponding bound fields, like MjData's sensor(), joint(), body() and site().

  model is the template, for names, ids and sizes. The batch copied it at construction,
  so writes to it do not reach the simulations; expand() a field to change it per sim."""

  def __init__(
    self,
    model: mujoco.MjModel,
    num_sims: int,
    num_threads: int = 0,
    forward: bool = False,
    cpu_ids: Sequence[int] | None = None,
  ) -> None:
    super().__init__(model, num_sims, num_threads, forward, cpu_ids)
    self.model = model
    self._active_model_update = False

  @classmethod
  def from_variant_pack(
    cls,
    pack: VariantPack,
    num_sims: int,
    assignment: Any,
    num_threads: int = 0,
    forward: bool = False,
    cpu_ids: Sequence[int] | None = None,
  ) -> "Batch":
    """Construct a batch from compiler-coherent, same-layout mesh variants."""
    ids = np.ascontiguousarray(assignment)
    if ids.ndim != 1 or ids.shape[0] != num_sims:
      raise ValueError("assignment must have one entry per simulation")
    if ids.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
      raise ValueError("assignment must contain int32 or int64 variant ids")
    checked = ids.astype(np.int64, copy=False)
    if checked.size and (checked.min() < 0 or checked.max() >= pack.num_variants):
      raise ValueError("assignment entries must be in variant range")

    batch = cls(
      pack.model,
      num_sims,
      num_threads=num_threads,
      forward=forward,
      cpu_ids=cpu_ids,
    )
    for name, values in pack.fields.items():
      batch.expand(name)[:] = values[ids]
    batch.set_const()
    return batch

  @cached_property
  def _model_fields(self) -> Mapping[str, ModelFieldSpec]:
    return build_model_fields(self.model)

  def _normalize_ids(self, ids: Any) -> np.ndarray | None:
    if ids is None:
      return None
    array = np.ascontiguousarray(ids)
    if array.ndim != 1:
      raise ValueError("ids must be one-dimensional")
    if array.dtype == bool:
      if array.shape[0] != self.num_sims:
        raise ValueError("a boolean ids mask must have num_sims entries")
      return array
    if array.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
      raise ValueError("ids must be int32, int64 or a bool mask")
    checked = array.astype(np.int64, copy=False)
    if (checked.size and (checked[0] < 0 or checked[-1] >= self.num_sims)) or np.any(np.diff(checked) <= 0):
      raise ValueError("ids must be sorted, unique and in range")
    return array

  def model_field_specs(self) -> Mapping[str, ModelFieldSpec]:
    """Return immutable metadata for every model field exposed by ``expand``."""
    return self._model_fields

  def expand(self, name: str, dtype: Any = None) -> np.ndarray:
    spec = self._model_fields.get(name)
    if spec is not None and not spec.writable:
      reason = "asset data" if spec.asset else "read-only structural data"
      raise ValueError(f"{name} is {reason}")
    return super().expand(name, dtype)

  @contextmanager
  def model_update(self, *fields: str, ids: Any = None) -> Iterator[None]:
    """Declare model-field writes, then perform one recompute on exit.

    This mirrors mjlab event terms: callers declare fields, and mjbatch computes the
    strongest recompute level from ``model_field_specs()``. A level above ``NONE``
    currently uses one conservative full stock-MuJoCo ``mj_setConst`` pass.
    """
    ids_array = self._normalize_ids(ids)
    specs = self._model_fields
    unknown = [field for field in fields if field not in specs]
    if unknown:
      raise ValueError(f"unknown model fields: {unknown}")
    unwritable = [field for field in fields if not specs[field].writable]
    if unwritable:
      raise ValueError(f"model fields are not writable: {unwritable}")
    if self._active_model_update:
      raise RuntimeError("model_update calls cannot be nested")
    self._active_model_update = True
    try:
      yield
    finally:
      self._active_model_update = False
      level = max((specs[field].recompute for field in fields), default=RecomputeLevel.NONE)
      if level != RecomputeLevel.NONE:
        super().set_const(ids_array)

  def set_const(self, ids: Any = None) -> None:
    if self._active_model_update:
      raise RuntimeError("call set_const after model_update exits, not inside it")
    super().set_const(ids)

  def sensor(self, name: str, dtype: Any = None) -> np.ndarray:
    s = self.model.sensor(name)
    return self.bind("sensordata", dtype)[:, s.adr[0] : s.adr[0] + s.dim[0]]

  def joint(self, name: str, dtype: Any = None) -> Joint:
    j = self.model.joint(name)
    nq, nv = _QPOS_WIDTH[int(j.type[0])], _DOF_WIDTH[int(j.type[0])]
    return Joint(
      self.bind("qpos", dtype)[:, j.qposadr[0] : j.qposadr[0] + nq],
      self.bind("qvel", dtype)[:, j.dofadr[0] : j.dofadr[0] + nv],
    )

  def actuator(self, name: str, dtype: Any = None) -> Actuator:
    i = self.model.actuator(name).id
    return Actuator(self.bind("ctrl", dtype)[:, i], self.bind("actuator_force", dtype)[:, i])

  def body(self, name: str, dtype: Any = None) -> Body:
    i = self.model.body(name).id
    return Body(
      self.bind("xpos", dtype)[:, i],
      self.bind("xquat", dtype)[:, i],
      self.bind("cvel", dtype)[:, i],
    )

  def site(self, name: str, dtype: Any = None) -> Site:
    i = self.model.site(name).id
    return Site(self.bind("site_xpos", dtype)[:, i], self.bind("site_xmat", dtype)[:, i])

  def _nsel(self, ids: Any) -> int:
    if ids is None:
      return self.num_sims
    ids = np.asarray(ids)
    return int(ids.sum()) if ids.dtype == bool else len(ids)

  def jac_site(  # pyright: ignore[reportIncompatibleMethodOverride]  # name-based allocating wrapper
    self, name: str, ids: Any = None
  ) -> tuple[np.ndarray, np.ndarray]:
    """World-frame position/rotation Jacobians of a site, per selected simulation.

    Returns (jacp, jacr) with shape (nsel, 3, nv). Runs kinematics and comPos
    only, not mj_forward, and does not refresh the bound views."""
    i = self.model.site(name).id
    n = self._nsel(ids)
    jacp = np.zeros((n, 3, self.model.nv))
    jacr = np.zeros((n, 3, self.model.nv))
    super().jac_site(i, jacp, jacr, ids)
    return jacp, jacr

  def sample_hfield(  # pyright: ignore[reportIncompatibleMethodOverride]  # name-based allocating wrapper
    self,
    geom: str,
    body: str,
    offsets: np.ndarray,
    ids: Any = None,
    alignment: str = "world",
  ) -> np.ndarray:
    """Bilinear hfield sampling at XY offsets around a body's origin.

    offsets: (npoint, 2), in the sampling grid's frame. alignment rotates the
    grid: "world" keeps offsets in world axes, "yaw" rotates them by the frame
    body's yaw about world z. Returns (nsel, npoint): the world z of the
    sampled hfield surface (the local elevation for an unrotated geom at the
    origin). All simulations sample the template's hfield; a per-sim
    geom_pos/geom_quat moves the sampling frame. Runs kinematics only, not
    mj_forward, and does not refresh the bound views."""
    g = self.model.geom(geom).id
    b = self.model.body(body).id
    offsets = np.ascontiguousarray(offsets, dtype=np.float64)
    out = np.zeros((self._nsel(ids), offsets.shape[0]))
    super().sample_hfield(g, b, offsets, out, ids, alignment)
    return out
