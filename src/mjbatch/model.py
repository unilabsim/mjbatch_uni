"""Metadata and recompute semantics for per-simulation model fields."""

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping

import mujoco
import numpy as np


class RecomputeLevel(IntEnum):
  """Whether a model-field write requires a stock-MuJoCo ``set_const`` pass."""

  NONE = 0
  SET_CONST = 1


_RECOMPUTE_BY_FIELD = {
  "body_gravcomp": RecomputeLevel.SET_CONST,
  "body_pos": RecomputeLevel.SET_CONST,
  "body_quat": RecomputeLevel.SET_CONST,
  "qpos0": RecomputeLevel.SET_CONST,
  "dof_armature": RecomputeLevel.SET_CONST,
  "tendon_armature": RecomputeLevel.SET_CONST,
  "body_mass": RecomputeLevel.SET_CONST,
  "body_ipos": RecomputeLevel.SET_CONST,
  "body_inertia": RecomputeLevel.SET_CONST,
  "body_iquat": RecomputeLevel.SET_CONST,
}

_ASSET_PREFIXES = ("mesh_", "hfield_", "tex_", "skin_", "bvh_", "oct_")
_WRITABLE_ID_SUFFIXES = ("dataid", "matid", "texid")
_READ_ONLY_MODEL_FIELDS = {"names", "names_map", "paths", "plugin"}


@dataclass(frozen=True)
class ModelFieldSpec:
  """Public metadata for one per-simulation model field."""

  name: str
  shape: tuple[int, ...]
  dtype: np.dtype[Any]
  writable: bool
  asset: bool
  recompute: RecomputeLevel


def _is_writable(name: str, dtype: np.dtype[Any]) -> bool:
  if name in _READ_ONLY_MODEL_FIELDS or name.endswith("plugin") or (name[:1].isupper() and name[1:2] == "_"):
    return False
  if dtype in (np.dtype(np.int8), np.dtype(np.int64)):
    return False
  if name.endswith(_WRITABLE_ID_SUFFIXES):
    return True
  if name.endswith(("adr", "num", "id", "sameframe", "simple", "signature")):
    return False
  return "_rowadr" not in name and "_colind" not in name and "_rownnz" not in name and "_diag" not in name


def _body_bvh_rows(model: mujoco.MjModel) -> int:
  """Return the body-first prefix length after validating MuJoCo's BVH layout."""

  rows = 0
  for body in range(model.nbody):
    adr, num = int(model.body_bvhadr[body]), int(model.body_bvhnum[body])
    if num < 0:
      raise ValueError("unexpected mjModel BVH layout: negative body_bvhnum")
    if num == 0:
      if adr != -1:
        raise ValueError("unexpected mjModel BVH layout: an empty body BVH must have address -1")
      continue
    if adr != rows:
      raise ValueError("unexpected mjModel BVH layout: body BVH rows are not contiguous and body-first")
    rows += num
  if rows > model.nbvhstatic:
    raise ValueError("unexpected mjModel BVH layout: body BVH rows exceed nbvhstatic")

  mesh_rows = rows
  for mesh in range(model.nmesh):
    adr, num = int(model.mesh_bvhadr[mesh]), int(model.mesh_bvhnum[mesh])
    if num < 0:
      raise ValueError("unexpected mjModel BVH layout: negative mesh_bvhnum")
    if num == 0:
      if adr != -1:
        raise ValueError("unexpected mjModel BVH layout: an empty mesh BVH must have address -1")
      continue
    if adr != mesh_rows:
      raise ValueError("unexpected mjModel BVH layout: mesh BVH rows are not contiguous after body rows")
    mesh_rows += num
  if mesh_rows != model.nbvhstatic:
    raise ValueError("unexpected mjModel BVH layout: nbvhstatic contains an unrecognized suffix")
  return rows


def build_model_fields(model: mujoco.MjModel) -> Mapping[str, ModelFieldSpec]:
  """Build immutable metadata for the arrays and options accepted by ``expand``."""
  specs: dict[str, ModelFieldSpec] = {}
  for name in dir(model):
    if name.startswith("_"):
      continue
    value = getattr(model, name)
    if not isinstance(value, np.ndarray):
      continue
    asset = name.startswith(_ASSET_PREFIXES)
    specs[name] = ModelFieldSpec(
      name=name,
      shape=tuple(value.shape),
      dtype=value.dtype,
      writable=not asset and _is_writable(name, value.dtype),
      asset=asset,
      recompute=_RECOMPUTE_BY_FIELD.get(name, RecomputeLevel.NONE),
    )

  if "body_bvh_aabb" in specs:
    raise ValueError("body_bvh_aabb collides with an mjModel field")
  specs["body_bvh_aabb"] = ModelFieldSpec(
    name="body_bvh_aabb",
    shape=(int(_body_bvh_rows(model)), 6),
    dtype=np.dtype(np.float64),
    writable=True,
    asset=False,
    recompute=RecomputeLevel.NONE,
  )

  integer, floating = np.dtype(np.int32), np.dtype(np.float64)
  for name in dir(model.opt):
    if name.startswith("_") or name in specs:
      continue
    value = getattr(model.opt, name)
    if isinstance(value, np.ndarray):
      shape, dtype = tuple(value.shape), value.dtype
    elif isinstance(value, bool):
      shape, dtype = (), np.dtype(np.bool_)
    elif isinstance(value, int):
      shape, dtype = (), integer
    elif isinstance(value, float):
      shape, dtype = (), floating
    else:
      continue
    specs[name] = ModelFieldSpec(
      name=name,
      shape=shape,
      dtype=dtype,
      writable=True,
      asset=False,
      recompute=RecomputeLevel.NONE,
    )
  return MappingProxyType(specs)
