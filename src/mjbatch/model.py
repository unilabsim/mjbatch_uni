"""Metadata and recompute semantics for per-simulation model fields."""

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping

import mujoco
import numpy as np

from mjbatch._bindings import _setconst_inputs


class RecomputeLevel(IntEnum):
  """Whether a model-field write requires a stock-MuJoCo ``set_const`` pass."""

  NONE = 0
  SET_CONST = 1


_SETCONST_INPUTS = frozenset(_setconst_inputs())

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
      recompute=RecomputeLevel.SET_CONST if name in _SETCONST_INPUTS else RecomputeLevel.NONE,
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
