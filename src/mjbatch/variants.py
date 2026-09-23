"""Compiler-coherent, same-layout mesh variant construction."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

_GEOM_VARIANT_FIELDS = (
  "geom_type",
  "geom_contype",
  "geom_conaffinity",
  "geom_matid",
  "geom_rgba",
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "geom_pos",
  "geom_quat",
)
_BODY_VARIANT_FIELDS = (
  "body_mass",
  "body_subtreemass",
  "body_inertia",
  "body_invweight0",
  "body_ipos",
  "body_iquat",
)
_DOF_VARIANT_FIELDS = ("dof_M0", "dof_invweight0", "dof_length")
_DIRECT_VARIANT_FIELDS = _BODY_VARIANT_FIELDS + _DOF_VARIANT_FIELDS
_VARIANT_FIELDS = _GEOM_VARIANT_FIELDS + _DIRECT_VARIANT_FIELDS
_ALLOWED_GEOM_FIELDS = frozenset(_GEOM_VARIANT_FIELDS) | {"geom_dataid"}
_ALLOWED_BODY_FIELDS = frozenset(_BODY_VARIANT_FIELDS)
_ALLOWED_DOF_FIELDS = frozenset(_DOF_VARIANT_FIELDS)
_ALLOWED_DERIVED_FIELDS = frozenset({"tendon_length0", "tendon_invweight0", "actuator_acc0"})
_SHARED_PARAMETER_PREFIXES = (
  "body_",
  "geom_",
  "jnt_",
  "dof_",
  "actuator_",
  "sensor_",
  "site_",
  "pair_",
  "eq_",
  "wrap_",
  "light_",
  "cam_",
  "tendon_",
  "qpos0",
  "qpos_spring",
)
# Compiler-derived fields, not structure: ``body_simple``/``dof_simplenum``
# are fast-path flags/counts the compiler derives from each variant's own
# inertial frames, and ``light_poscom0`` is derived from the model's center of
# mass.  They legitimately differ across same-layout variants whose allowed
# per-variant fields (``body_mass``/``body_ipos``/``body_iquat``/
# ``body_inertia``, geom pose/size) already carry the underlying physics, and
# none of them changes the dynamics computed with the canonical model's
# flags.  Treating them as shared skeleton parameters falsely rejects
# mass-varying pools (unilabsim/mjbatch_uni#33).
_IGNORED_COMPILER_FLAGS = frozenset(
  {"body_sameframe", "geom_sameframe", "body_simple", "dof_simplenum", "light_poscom0"}
)
_IGNORED_COMPILER_METADATA_PREFIXES = ("body_geom", "body_bvh", "geom_bvh")

_LAYOUT_SCALARS = (
  "nq",
  "nv",
  "na",
  "nbody",
  "njnt",
  "nsite",
  "ncam",
  "nlight",
  "nmat",
  "npair",
  "nexclude",
  "neq",
  "ntendon",
  "nwrap",
  "nsensor",
  "nnumeric",
  "ntext",
  "ntuple",
  "nmocap",
  "nplugin",
  "nuser_body",
  "nuser_jnt",
  "nuser_geom",
  "nuser_site",
  "nuser_cam",
  "nuser_sensor",
  "nuser_tendon",
  "nuser_actuator",
)
_ENTITY_KIND_COUNTS = (
  ("body", "nbody"),
  ("joint", "njnt"),
  ("site", "nsite"),
  ("cam", "ncam"),
  ("light", "nlight"),
  ("actuator", "nu"),
  ("sensor", "nsensor"),
  ("tendon", "ntendon"),
)


@dataclass(frozen=True)
class VariantPack:
  """A canonical model plus per-variant, compiler-derived model-field rows."""

  model: mujoco.MjModel
  num_variants: int
  fields: Mapping[str, np.ndarray]

  @classmethod
  def from_specs(cls, specs: Sequence[mujoco.MjSpec]) -> "VariantPack":
    """Compile variants independently and merge their meshes into one model.

    Variants must have the same named structural layout. A canonical variant may
    contain optional mesh-geom slots that are absent in another variant; missing
    slots are disabled with ``mjGEOM_NONE``, ``geom_dataid=-1``, and zero contact
    bits. Non-mesh topology and parameters must not vary.

    Construction is streamed through :class:`VariantPackBuilder`: one variant is
    compiled at a time and only small compiler-derived rows are retained, so
    peak memory stays independent of the variant count.
    """
    if not specs:
      raise ValueError("at least one variant spec is required")
    builder = VariantPackBuilder()
    for spec in specs:
      builder.add_variant(spec)
    return builder.build(specs[builder.canonical_index].copy())

  @classmethod
  def builder(cls) -> "VariantPackBuilder":
    """Return an incremental, streaming VariantPack constructor."""
    return VariantPackBuilder()


class VariantPackBuilder:
  """Incrementally accumulate same-layout variants with streaming compilation.

  ``add_variant`` compiles one spec (or adopts an existing compilation of it),
  snapshots the small per-variant rows the pack needs, and returns the model so
  the caller can validate it and then release both spec and model. Peak memory
  is bounded by the canonical model, the unique mesh pool, and one transient
  variant realization instead of scaling with the variant count.

  ``build`` must receive a spec equivalent to the one added at
  ``canonical_index`` (a fresh parse/copy is expected); it applies the pooled
  mesh catalog, compiles the canonical model, validates every snapshot against
  it, and assembles a :class:`VariantPack` identical to ``from_specs``.
  """

  def __init__(self) -> None:
    self._snapshots: list[_VariantSnapshot] = []

  @property
  def num_variants(self) -> int:
    return len(self._snapshots)

  @property
  def canonical_index(self) -> int:
    """Index of the variant with the most geoms (first wins ties)."""
    if not self._snapshots:
      raise ValueError("at least one variant spec is required")
    return max(
      range(len(self._snapshots)),
      key=lambda index: (self._snapshots[index].ngeom, -index),
    )

  def add_variant(self, spec: mujoco.MjSpec, model: mujoco.MjModel | None = None) -> mujoco.MjModel:
    """Snapshot one variant; ``model`` must be ``spec.compile()`` when given."""
    if model is None:
      model = spec.compile()
      assert model is not None
    self._snapshots.append(_snapshot_variant(spec, model))
    return model

  def build(self, canonical_spec: mujoco.MjSpec) -> VariantPack:
    """Assemble the pack around a fresh spec of the canonical variant."""
    snapshots = self._snapshots
    if not snapshots:
      raise ValueError("at least one variant spec is required")
    canonical_index = self.canonical_index

    mesh_pool: dict[bytes, str] = {}
    for mesh in canonical_spec.meshes:
      mesh_pool[_mesh_digest(canonical_spec, mesh)] = mesh.name

    mesh_names_by_variant: list[dict[str, str]] = []
    for variant_index, snapshot in enumerate(snapshots):
      mesh_names: dict[str, str] = {}
      for record in snapshot.meshes:
        if variant_index == canonical_index:
          pooled_name = record.name
        else:
          pooled_name = mesh_pool.get(record.key)
          if pooled_name is None:
            pooled_name = _copy_mesh_record(canonical_spec, record, variant_index)
            mesh_pool[record.key] = pooled_name
        if record.name in mesh_names:
          raise ValueError(f"variant has duplicate mesh name {record.name!r}")
        mesh_names[record.name] = pooled_name
      mesh_names_by_variant.append(mesh_names)

    _disable_simple_where_variants_break_inertial_frame(canonical_spec, snapshots, canonical_index)
    canonical = canonical_spec.compile()
    _validate_layout(snapshots, canonical)
    geom_maps = _validate_names_and_build_geom_maps(snapshots, canonical)
    _validate_shared_parameters(snapshots, canonical, geom_maps)
    _validate_shared_options(snapshots, canonical)
    mesh_id_maps = _build_mesh_id_maps(snapshots, canonical, mesh_names_by_variant)

    fields: dict[str, np.ndarray] = {}
    for name in _VARIANT_FIELDS:
      canonical_value = np.asarray(getattr(canonical, name))
      values = np.tile(canonical_value, (len(snapshots), *(1,) * canonical_value.ndim))
      if name in _GEOM_VARIANT_FIELDS:
        for variant, (snapshot, geom_map) in enumerate(zip(snapshots, geom_maps, strict=True)):
          values[variant][geom_map] = snapshot.fields[name]
        values = _disable_missing_geom_slots(
          values,
          name,
          geom_maps,
          canonical,
        )
      else:
        for variant, snapshot in enumerate(snapshots):
          values[variant] = snapshot.fields[name]
      fields[name] = values

    dataids = fields["geom_dataid"] = np.full((len(snapshots), canonical.ngeom), -1, dtype=np.int32)
    for variant, (snapshot, geom_map) in enumerate(zip(snapshots, geom_maps, strict=True)):
      mesh_ids = mesh_id_maps[variant]
      for reference_geom, canonical_geom in enumerate(geom_map):
        dataid = int(snapshot.fields["geom_dataid"][reference_geom])
        dataids[variant, canonical_geom] = mesh_ids.get(dataid, -1)

    for values in fields.values():
      values.flags.writeable = False

    return VariantPack(
      model=canonical,
      num_variants=len(snapshots),
      fields=MappingProxyType(fields),
    )


@dataclass
class _MeshRecord:
  """One spec mesh's identity digest and the payload needed to re-add it."""

  name: str
  key: bytes
  file: str
  content_type: str
  refpos: np.ndarray
  refquat: np.ndarray
  scale: np.ndarray
  inertia: int
  smoothnormal: bool
  needsdf: bool
  maxhullvert: int
  octree_maxdepth: int
  material: Any
  uservert: list
  usernormal: list
  usertexcoord: list
  userface: list
  userfacenormal: list
  userfacetexcoord: list


@dataclass
class _VariantSnapshot:
  """Small compiler-derived rows retained per variant by the builder."""

  ngeom: int
  nbody: int
  layout: tuple[tuple[str, int], ...]
  entity_names: dict[str, list[str]]
  fields: dict[str, np.ndarray]
  body_simple: np.ndarray
  shared: dict[str, np.ndarray]
  options: dict[str, Any]
  mesh_ids: dict[str, int]
  meshes: list[_MeshRecord]


def _snapshot_variant(spec: mujoco.MjSpec, model: mujoco.MjModel) -> _VariantSnapshot:
  fields = {name: np.asarray(getattr(model, name)).copy() for name in _VARIANT_FIELDS}
  fields["geom_dataid"] = np.asarray(model.geom_dataid).copy()
  shared: dict[str, np.ndarray] = {}
  for name in dir(model):
    if (
      name.startswith(_SHARED_PARAMETER_PREFIXES)
      and name not in _IGNORED_COMPILER_FLAGS
      and not name.startswith(_IGNORED_COMPILER_METADATA_PREFIXES)
      and name not in _ALLOWED_BODY_FIELDS
      and name not in _ALLOWED_DOF_FIELDS
      and name not in _ALLOWED_GEOM_FIELDS
      and name not in _ALLOWED_DERIVED_FIELDS
    ):
      value = getattr(model, name, None)
      if isinstance(value, np.ndarray):
        shared[name] = value.copy()
  options: dict[str, Any] = {}
  for name in dir(model.opt):
    if name.startswith("_") or name == "timestep":
      continue
    value = getattr(model.opt, name, None)
    if isinstance(value, np.ndarray):
      options[name] = value.copy()
    elif isinstance(value, (bool, int, float)):
      options[name] = value
  entity_names = {kind: _names(model, kind, getattr(model, count)) for kind, count in _ENTITY_KIND_COUNTS}
  entity_names["geom"] = _names(model, "geom", model.ngeom)
  return _VariantSnapshot(
    ngeom=int(model.ngeom),
    nbody=int(model.nbody),
    layout=_layout(model),
    entity_names=entity_names,
    fields=fields,
    body_simple=np.asarray(model.body_simple).copy(),
    shared=shared,
    options=options,
    mesh_ids={mesh.name: int(model.mesh(mesh.name).id) for mesh in spec.meshes},
    meshes=[_record_mesh(spec, mesh) for mesh in spec.meshes],
  )


def _mesh_path(spec: mujoco.MjSpec, mesh: mujoco.MjsMesh) -> Path | None:
  if not mesh.file:
    return None
  path = Path(mesh.file)
  if path.is_absolute() or path.exists():
    return path.resolve()
  candidates = (spec.modelfiledir, spec.meshdir)
  for candidate in candidates:
    if candidate:
      resolved = (Path(candidate) / path).resolve()
      if resolved.exists():
        return resolved
  return path.resolve()


def _mesh_key(spec: mujoco.MjSpec, mesh: mujoco.MjsMesh) -> tuple[Any, ...]:
  path = _mesh_path(spec, mesh)
  content = path.read_bytes() if path is not None and path.exists() else b""
  vector_keys = tuple(
    (name, np.asarray(getattr(mesh, name)).tobytes())
    for name in (
      "refpos",
      "refquat",
      "scale",
      "uservert",
      "usernormal",
      "usertexcoord",
      "userface",
      "userfacenormal",
      "userfacetexcoord",
    )
  )
  return (
    mesh.content_type,
    content,
    vector_keys,
    int(mesh.inertia),
    bool(mesh.smoothnormal),
    bool(mesh.needsdf),
    int(mesh.maxhullvert),
    int(mesh.octree_maxdepth),
    mesh.material,
  )


def _mesh_digest(spec: mujoco.MjSpec, mesh: mujoco.MjsMesh) -> bytes:
  """Content-addressed mesh identity bounded to a fixed-size digest.

  The raw key embeds whole mesh files and vertex buffers; retaining one digest
  per variant mesh keeps builder memory independent of mesh payload size.
  """
  digest = hashlib.sha256()
  for part in _mesh_key(spec, mesh):
    if isinstance(part, bytes):
      digest.update(b"\xfe")
      digest.update(part)
    elif isinstance(part, tuple):
      for name, payload in part:
        digest.update(name.encode())
        digest.update(b"\xfe")
        digest.update(payload)
    else:
      digest.update(repr(part).encode())
  return digest.digest()


def _record_mesh(spec: mujoco.MjSpec, mesh: mujoco.MjsMesh) -> _MeshRecord:
  path = _mesh_path(spec, mesh)
  return _MeshRecord(
    name=mesh.name,
    key=_mesh_digest(spec, mesh),
    file="" if path is None else str(path),
    content_type=mesh.content_type,
    refpos=np.array(mesh.refpos),
    refquat=np.array(mesh.refquat),
    scale=np.array(mesh.scale),
    inertia=int(mesh.inertia),
    smoothnormal=bool(mesh.smoothnormal),
    needsdf=bool(mesh.needsdf),
    maxhullvert=int(mesh.maxhullvert),
    octree_maxdepth=int(mesh.octree_maxdepth),
    material=mesh.material,
    uservert=list(mesh.uservert),
    usernormal=list(mesh.usernormal),
    usertexcoord=list(mesh.usertexcoord),
    userface=list(mesh.userface),
    userfacenormal=list(mesh.userfacenormal),
    userfacetexcoord=list(mesh.userfacetexcoord),
  )


def _copy_mesh_record(
  target: mujoco.MjSpec,
  record: _MeshRecord,
  variant_index: int,
) -> str:
  base_name = record.name or "mesh"
  index = variant_index
  while True:
    pooled_name = f"mjbatch_mesh_v{index}_{base_name}"
    if not any(mesh.name == pooled_name for mesh in target.meshes):
      break
    index += 1

  copied = target.add_mesh(name=pooled_name)
  copied.file = record.file
  copied.content_type = record.content_type
  copied.refpos = record.refpos
  copied.refquat = record.refquat
  copied.scale = record.scale
  copied.inertia = record.inertia
  copied.smoothnormal = record.smoothnormal
  copied.needsdf = record.needsdf
  copied.maxhullvert = record.maxhullvert
  copied.octree_maxdepth = record.octree_maxdepth
  copied.material = record.material
  copied.uservert = list(record.uservert)
  copied.usernormal = list(record.usernormal)
  copied.usertexcoord = list(record.usertexcoord)
  copied.userface = list(record.userface)
  copied.userfacenormal = list(record.userfacenormal)
  copied.userfacetexcoord = list(record.userfacetexcoord)
  return pooled_name


def _layout(model: mujoco.MjModel) -> tuple[tuple[str, int], ...]:
  return tuple((name, int(getattr(model, name))) for name in _LAYOUT_SCALARS if hasattr(model, name))


def _validate_layout(snapshots: Sequence[_VariantSnapshot], canonical: mujoco.MjModel) -> None:
  canonical_layout = dict(_layout(canonical))
  for variant, snapshot in enumerate(snapshots):
    for name, value in snapshot.layout:
      if value != canonical_layout[name]:
        raise ValueError(
          f"variant {variant} changes layout field {name}: {value} != {canonical_layout[name]}"
        )
    if snapshot.ngeom > canonical.ngeom:
      raise ValueError(f"variant {variant} has more geoms than the canonical model")


def _names(model: mujoco.MjModel, kind: str, count: int) -> list[str]:
  accessor = getattr(model, kind)
  return [str(accessor(i).name) for i in range(count)]


def _validate_names_and_build_geom_maps(
  snapshots: Sequence[_VariantSnapshot], canonical: mujoco.MjModel
) -> list[np.ndarray]:
  canonical_geoms = _names(canonical, "geom", canonical.ngeom)
  if len(set(canonical_geoms)) != len(canonical_geoms) or "" in canonical_geoms:
    raise ValueError("canonical geoms must have unique, non-empty names")

  canonical_by_kind = {
    kind: _names(canonical, kind, getattr(canonical, count)) for kind, count in _ENTITY_KIND_COUNTS
  }
  canonical_geom_ids = {name: i for i, name in enumerate(canonical_geoms)}

  maps: list[np.ndarray] = []
  for variant, snapshot in enumerate(snapshots):
    for kind, expected in canonical_by_kind.items():
      actual = snapshot.entity_names[kind]
      if actual != expected:
        raise ValueError(f"variant {variant} changes {kind} names or order")
    names = snapshot.entity_names["geom"]
    if len(set(names)) != len(names) or "" in names:
      raise ValueError(f"variant {variant} geoms must have unique, non-empty names")
    unknown = set(names) - set(canonical_geoms)
    if unknown:
      raise ValueError(f"variant {variant} has geoms absent from the canonical layout: {sorted(unknown)}")
    maps.append(np.asarray([canonical_geom_ids[name] for name in names], dtype=np.int32))
  return maps


def _validate_shared_parameters(
  snapshots: Sequence[_VariantSnapshot],
  canonical: mujoco.MjModel,
  geom_maps: Sequence[np.ndarray],
) -> None:
  """Reject shared parameter changes that a canonical executor cannot represent."""

  shared_names = tuple(
    name
    for name in dir(canonical)
    if name.startswith(_SHARED_PARAMETER_PREFIXES)
    and name not in _IGNORED_COMPILER_FLAGS
    and not name.startswith(_IGNORED_COMPILER_METADATA_PREFIXES)
    and name not in _ALLOWED_BODY_FIELDS
    and name not in _ALLOWED_DOF_FIELDS
    and name not in _ALLOWED_GEOM_FIELDS
    and name not in _ALLOWED_DERIVED_FIELDS
  )
  for variant, (snapshot, geom_map) in enumerate(zip(snapshots, geom_maps, strict=True)):
    for name in shared_names:
      expected = getattr(canonical, name, None)
      actual = snapshot.shared.get(name)
      if not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray):
        continue
      if expected.shape != actual.shape:
        continue
      canonical_values = expected[geom_map] if name.startswith("geom_") else expected
      if not np.array_equal(canonical_values, actual):
        raise ValueError(f"variant {variant} changes shared field {name}")


def _disable_simple_where_variants_break_inertial_frame(
  canonical_spec: mujoco.MjSpec,
  snapshots: Sequence[_VariantSnapshot],
  canonical_index: int,
) -> None:
  """Force the general (non-simple) code path where variant inertia requires it.

  The compiler marks a body ``simple`` when its inertial frame coincides with
  the body frame (zero ``ipos``, identity ``iquat``), and the batch executor
  refuses per-variant inertial writes that break that invariant on a simple
  canonical body ("compiled as simple but sameframe no longer holds").
  Variant pools legitimately differ in exactly those allowed variant fields,
  so the canonical model must not carry the fast-path flag on such bodies.
  ``simple=False`` is a codegen hint only; the compiled dynamics are
  numerically identical.
  """
  canonical = snapshots[canonical_index]
  # MjSpec.bodies is the flattened depth-first list, aligned with the compiled
  # model's body ids (world first in both).
  spec_bodies = list(canonical_spec.bodies)[1:]
  if len(spec_bodies) != canonical.nbody - 1:
    raise ValueError(
      f"canonical spec has {len(spec_bodies)} bodies but its compiled model has "
      f"{canonical.nbody - 1}; cannot map the simple flags"
    )
  zero = np.zeros(3)
  identity = np.array([1.0, 0.0, 0.0, 0.0])
  for spec_body, body_id in zip(spec_bodies, range(1, canonical.nbody), strict=True):
    if not canonical.body_simple[body_id]:
      continue
    for snapshot in snapshots:
      if not np.array_equal(snapshot.fields["body_ipos"][body_id], zero) or not np.array_equal(
        snapshot.fields["body_iquat"][body_id], identity
      ):
        spec_body.simple = False
        break


def _validate_shared_options(snapshots: Sequence[_VariantSnapshot], canonical: mujoco.MjModel) -> None:
  option_names = tuple(name for name in dir(canonical.opt) if not name.startswith("_") and name != "timestep")
  for variant, snapshot in enumerate(snapshots):
    for name in option_names:
      expected = getattr(canonical.opt, name, None)
      actual = snapshot.options.get(name)
      if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        same = np.array_equal(expected, actual)
      elif isinstance(expected, (bool, int, float)) and isinstance(actual, (bool, int, float)):
        same = expected == actual
      else:
        continue
      if not same:
        raise ValueError(f"variant {variant} changes option {name}")


def _build_mesh_id_maps(
  snapshots: Sequence[_VariantSnapshot],
  canonical: mujoco.MjModel,
  mesh_names_by_variant: Sequence[Mapping[str, str]],
) -> list[dict[int, int]]:
  maps: list[dict[int, int]] = []
  for snapshot, mesh_names in zip(snapshots, mesh_names_by_variant, strict=True):
    mesh_ids: dict[int, int] = {}
    for record in snapshot.meshes:
      reference_id = snapshot.mesh_ids[record.name]
      pooled_name = mesh_names[record.name]
      mesh_ids[reference_id] = canonical.mesh(pooled_name).id
    maps.append(mesh_ids)
  return maps


def _disable_missing_geom_slots(
  values: np.ndarray,
  name: str,
  geom_maps: Sequence[np.ndarray],
  canonical: mujoco.MjModel,
) -> np.ndarray:
  for variant, geom_map in enumerate(geom_maps):
    present = np.zeros(canonical.ngeom, dtype=bool)
    present[geom_map] = True
    missing = np.flatnonzero(~present)
    if missing.size == 0:
      continue
    if name == "geom_type":
      values[variant, missing] = int(mujoco.mjtGeom.mjGEOM_NONE)
    elif name in ("geom_contype", "geom_conaffinity", "geom_matid"):
      values[variant, missing] = -1 if name == "geom_matid" else 0
    elif name == "geom_rgba":
      values[variant, missing] = (0.0, 0.0, 0.0, 0.0)
    elif name == "geom_quat":
      values[variant, missing] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=values.dtype)
    else:
      values[variant, missing] = 0
  return values
