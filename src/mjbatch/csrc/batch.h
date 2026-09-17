// SPDX-License-Identifier: Apache-2.0

// Batch: N MuJoCo simulations on a thread pool, with batch buffers for data
// fields and per-sim storage for model fields. Each sim keeps only its
// mjSTATE_INTEGRATION vector and warning counters; the mjData is per worker.
#pragma once

#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#include "threadpool.h"

namespace nb = nanobind;

// Every logical CPU: MuJoCo's small dense linear algebra stalls on latency, so a
// core runs two threads at 1.3-1.5x the throughput of one (measured on a 7960X).
inline int DefaultThreadCount() {
  return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

// Resolves the worker count and builds the pool. An explicit cpu_ids pins
// worker i to cpu_ids[i] (Linux only), so the list must be non-empty, unique,
// within the process's sched_getaffinity mask, and its length fixes the worker
// count: it must equal num_threads, or set it when num_threads <= 0.
inline std::unique_ptr<ThreadPool> MakePool(int num_threads, int num_sims,
                                            std::optional<std::vector<int>> cpu_ids) {
  std::vector<int> pin;
  if (cpu_ids) {
#ifdef __linux__
    pin = std::move(*cpu_ids);
    if (pin.empty()) throw nb::value_error("cpu_ids must be non-empty");
    for (int id : pin) {
      if (id < 0) throw nb::value_error("cpu_ids entries must be >= 0");
    }
    if (std::set<int>(pin.begin(), pin.end()).size() != pin.size()) {
      throw nb::value_error("cpu_ids entries must be unique");
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
      throw std::runtime_error("sched_getaffinity failed");
    }
    std::string bad;
    for (int id : pin) {
      if (id >= CPU_SETSIZE || !CPU_ISSET(id, &allowed)) {
        if (!bad.empty()) bad += ", ";
        bad += std::to_string(id);
      }
    }
    if (!bad.empty()) {
      throw nb::value_error(("cpu_ids [" + bad + "] are not available to this process").c_str());
    }
    if (num_threads <= 0) num_threads = static_cast<int>(pin.size());
    if (num_threads != static_cast<int>(pin.size())) {
      throw nb::value_error(("cpu_ids length (" + std::to_string(pin.size()) +
                             ") must equal num_threads (" + std::to_string(num_threads) + ")")
                                .c_str());
    }
#else
    throw nb::value_error("cpu_ids pinning is only supported on Linux");
#endif
  }
  if (num_threads <= 0) num_threads = DefaultThreadCount();
  // With pins the pool is exactly len(cpu_ids) workers so worker i always pairs
  // with cpu_ids[i]; without, the pool is capped at num_sims.
  int size = pin.empty() ? std::max(1, std::min(num_threads, num_sims)) : num_threads;
  auto pool = std::make_unique<ThreadPool>(size, std::move(pin));
  if (int err = pool->PinError()) {
    throw nb::value_error(("failed to pin a worker thread to the requested cpu_ids (error " +
                           std::to_string(err) + ")")
                              .c_str());
  }
  return pool;
}

enum class Elem { Num, Float, Int, Byte, Bool, Other };

template <class T>
constexpr Elem ElemOf() {
  if constexpr (std::is_same_v<T, mjtNum>) return Elem::Num;
  if constexpr (std::is_same_v<T, float>) return Elem::Float;
  if constexpr (std::is_same_v<T, int>) return Elem::Int;
  if constexpr (std::is_same_v<T, mjtByte>) return Elem::Byte;
  if constexpr (std::is_same_v<T, mjtBool>) return Elem::Bool;
  return Elem::Other;
}

inline const char* DtypeName(Elem e) {
  switch (e) {
    case Elem::Num:
      return sizeof(mjtNum) == 8 ? "float64" : "float32";
    case Elem::Float:
      return "float32";
    case Elem::Int:
      return "int32";
    case Elem::Byte:
      return "uint8";
    case Elem::Bool:
      return "bool";
    default:
      return "";
  }
}

// numpy's rule: an extent-1 axis may carry any stride and a size-0 array is contiguous.
inline bool IsCContig(const nb::ndarray<>& a) {
  if (a.size() == 0) return true;
  int64_t expect = 1;
  for (size_t i = a.ndim(); i-- > 0;) {
    if (a.shape(i) != 1 && a.stride(i) != expect) return false;
    expect *= static_cast<int64_t>(a.shape(i));
  }
  return true;
}

// One array field of mjModel or mjData, sized for a specific model.
struct FieldInfo {
  const char* name;
  void* (*get)(void* obj);
  int nr, nc;
  int ndim;  // per sim: 0 for scalars, 1 when nc is the literal 1, else 2.
  size_t elem_size;
  Elem elem;
  bool input;  // mjData: persists across calls (an mjtState component or warning); else derived.
  bool asset;  // mjModel: large constant data mj_setConst never writes.
  size_t bytes() const { return static_cast<size_t>(nr) * nc * elem_size; }
};

using FieldTable = std::unordered_map<std::string, FieldInfo>;

inline bool IsInput(std::string_view name) {
  for (std::string_view s : {"time", "qpos", "qvel", "act", "history", "qacc_warmstart", "ctrl",
                             "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
                             "userdata", "plugin_state", "warning"}) {
    if (name == s) return true;
  }
  return false;
}

inline bool IsAsset(std::string_view name) {
  for (std::string_view p : {"mesh_", "hfield_", "tex_", "skin_", "bvh_", "oct_"}) {
    if (name.substr(0, p.size()) == p) return true;
  }
  return false;
}

inline FieldTable DataFields(const mjModel* m) {
  FieldTable t;
#undef MJ_M
#define MJ_M(n) (m->n)
#define X(type, name, nr, nc)                                                          \
  t[#name] = FieldInfo{#name,                                                          \
                       [](void* o) -> void* { return static_cast<mjData*>(o)->name; }, \
                       static_cast<int>(MJ_M(nr)),                                     \
                       static_cast<int>(nc),                                           \
                       std::string_view(#nc) == "1" ? 1 : 2,                           \
                       sizeof(type),                                                   \
                       ElemOf<type>(),                                                 \
                       IsInput(#name),                                                 \
                       false};
  MJDATA_POINTERS
#undef X
#undef MJ_M
#define MJ_M(n) n
  t["time"] = FieldInfo{"time",    [](void* o) -> void* { return &static_cast<mjData*>(o)->time; },
                        1,         1,
                        0,         sizeof(mjtNum),
                        Elem::Num, true,
                        false};
  // mjWarningStat is two ints: (lastinfo, number) per warning type.
  t["warning"] =
      FieldInfo{"warning",  [](void* o) -> void* { return static_cast<mjData*>(o)->warning; },
                mjNWARNING, 2,
                2,          sizeof(int),
                Elem::Int,  true,
                false};
  return t;
}

inline FieldTable ModelFields(const mjModel* m) {
  FieldTable t;
#undef MJ_M
#define MJ_M(n) (m->n)
#define X(type, name, nr, nc)                                                           \
  t[#name] = FieldInfo{#name,                                                           \
                       [](void* o) -> void* { return static_cast<mjModel*>(o)->name; }, \
                       static_cast<int>(MJ_M(nr)),                                      \
                       static_cast<int>(nc),                                            \
                       std::string_view(#nc) == "1" ? 1 : 2,                            \
                       sizeof(type),                                                    \
                       ElemOf<type>(),                                                  \
                       false,                                                           \
                       IsAsset(#name)};
  MJMODEL_POINTERS
#undef X
#undef MJ_M
#define MJ_M(n) n
  // mjOption alongside the arrays, its scalars shaped like mjData's time. The two
  // namespaces are disjoint today; a collision would silently shadow an array field.
#define XOPT(type, name, nr, ndim, getter)                                                         \
  if (!t.emplace(#name, FieldInfo{#name, getter, nr, 1, ndim, sizeof(type), ElemOf<type>(), false, \
                                  false})                                                          \
           .second) {                                                                              \
    throw std::runtime_error("mjOption." #name " collides with an mjModel field");                 \
  }
#define X(type, name, size) \
  XOPT(type, name, 1, 0, [](void* o) -> void* { return &static_cast<mjModel*>(o)->opt.name; })
#define XVEC(type, name, size) \
  XOPT(type, name, size, 1, [](void* o) -> void* { return static_cast<mjModel*>(o)->opt.name; })
  MJOPTION_FIELDS
#undef XVEC
#undef X
#undef XOPT
  return t;
}

// A batch buffer over one field: (N, nr[, nc]) rows, native dtype or float32.
// Against a float32 libmujoco the float32 path is a plain memcpy.
struct Slot {
  const FieldInfo* info;
  bool f32;
  size_t row;  // bytes per sim
  std::unique_ptr<uint8_t[]> buf;
  std::unique_ptr<uint8_t[]> mirror;  // buf as last written by us; input data fields only
};

inline void ToBuf(uint8_t* dst, const void* src, const Slot& s) {
  size_t n = static_cast<size_t>(s.info->nr) * s.info->nc;
  if (s.f32 && sizeof(mjtNum) != sizeof(float)) {
    auto* d = reinterpret_cast<float*>(dst);
    auto* x = static_cast<const mjtNum*>(src);
    for (size_t k = 0; k < n; ++k) d[k] = static_cast<float>(x[k]);
  } else {
    std::memcpy(dst, src, s.info->bytes());
  }
}

inline void FromBuf(void* dst, const uint8_t* src, const Slot& s) {
  size_t n = static_cast<size_t>(s.info->nr) * s.info->nc;
  if (s.f32 && sizeof(mjtNum) != sizeof(float)) {
    auto* d = static_cast<mjtNum*>(dst);
    auto* x = reinterpret_cast<const float*>(src);
    for (size_t k = 0; k < n; ++k) d[k] = static_cast<mjtNum>(x[k]);
  } else {
    std::memcpy(dst, src, s.info->bytes());
  }
}

// Copy the elements of buf that differ from mirror into dst, so only what the
// caller wrote lands and the rest of the row keeps whatever physics left there.
inline void CopyChanged(void* dst, const uint8_t* buf, const uint8_t* mirror, const Slot& s) {
  size_t n = static_cast<size_t>(s.info->nr) * s.info->nc;
  size_t e = s.f32 ? sizeof(float) : s.info->elem_size;
  for (size_t k = 0; k < n; ++k) {
    if (std::memcmp(buf + k * e, mirror + k * e, e) == 0) continue;
    if (s.f32 && sizeof(mjtNum) != sizeof(float)) {
      static_cast<mjtNum*>(dst)[k] = reinterpret_cast<const float*>(buf)[k];
    } else {
      std::memcpy(static_cast<uint8_t*>(dst) + k * e, buf + k * e, e);
    }
  }
}

// mjModel scalars mj_setConst writes; kept per sim like expanded fields.
struct Scalars {
  mjtSize ngravcomp;
  mjtBool flg_gravcomp, flg_surfacevel, flg_adhesion;
  mjStatistic stat;
  static Scalars Of(const mjModel* m) {
    return {m->ngravcomp, m->flg_gravcomp, m->flg_surfacevel, m->flg_adhesion, m->stat};
  }
  void Apply(mjModel* m) const {
    m->ngravcomp = ngravcomp;
    m->flg_gravcomp = flg_gravcomp;
    m->flg_surfacevel = flg_surfacevel;
    m->flg_adhesion = flg_adhesion;
    m->stat = stat;
  }
};

// SampleHfield grid alignment: offsets stay in world axes (World) or rotate
// with the frame body's yaw about world z (Yaw).  This is the mujoco_uni
// BatchEnvPool.sample_hfield_height convention, which this op replaces.
enum class HfieldAlign { World, Yaw };

// Per-call context for the query ops (JacSite, SampleHfield): outputs are
// caller-allocated arrays, one row per selected sim, like step's history.
struct QueryCtx {
  int site = -1;              // JacSite: site id
  mjtNum* jacp = nullptr;     // (sel, 3, nv) rows, either may be null
  mjtNum* jacr = nullptr;
  int geom = -1, body = -1;   // SampleHfield: hfield geom id, frame body id
  const mjtNum* offsets = nullptr;  // (npoint, 2) grid-frame XY around the body origin
  int npoint = 0;
  HfieldAlign align = HfieldAlign::World;  // SampleHfield: sampling grid rotation
  mjtNum* out = nullptr;      // (sel, npoint) rows
  Slot* sensor = nullptr;     // RefreshSensor: bound sensordata slot
  const std::vector<int>* sensor_ranges = nullptr;  // flattened (start, stop) pairs
};

// One substep of a callback-driven step (Op::Substep): the calling thread
// dispatches substeps one at a time and runs the Python callback in between,
// so a sim's state round-trips through its states_ row every substep.
struct CallbackCtx {
  int k = 0, nstep = 1;
  mjtNum* hist = nullptr;   // (sel, nstep, nstate) row for this sim
  uint8_t* done = nullptr;  // set when this selection row runs its copy-out tail
  bool tail_only = false;   // exception cleanup: no mj_step, copy out where it stopped
  Slot* sensor = nullptr;   // Substep1: bound sensordata slot receiving selected columns
  int sensor_start = 0;
  int sensor_width = 0;
};

// Bilinear sampling of one hfield geom at XY offsets from a frame body's origin.
// The offsets form a sampling grid rotated per ctx.align: World keeps them in
// world axes, Yaw rotates them by the frame body's yaw about world z (extracted
// from xmat). Sample points are taken in the plane through the geom center,
// then transformed into the geom's local frame (its pose comes from the sim's
// mjData, so per-sim geom_pos/geom_quat apply); the hfield grid itself is the
// model's, shared by all sims. The grid mapping and clamping match MuJoCo's
// hfield contact convention for the XY extent [-size[0], size[0]] x
// [-size[1], size[1]]. The output is the world z of the sampled surface point
// (for an unrotated geom at the origin, the local elevation interp * size[2]).

inline void SampleHfield(const mjModel* m, const mjData* d, const QueryCtx& ctx) {
  int hfield = m->geom_dataid[ctx.geom];
  int nrow = m->hfield_nrow[hfield], ncol = m->hfield_ncol[hfield];
  const mjtNum* hsize = m->hfield_size + static_cast<size_t>(4) * hfield;
  const float* data = m->hfield_data + static_cast<size_t>(hfield) * nrow * ncol;
  const mjtNum* gpos = d->geom_xpos + 3 * ctx.geom;
  const mjtNum* gmat = d->geom_xmat + 9 * ctx.geom;
  const mjtNum* bpos = d->xpos + 3 * ctx.body;
  const mjtNum* bmat = d->xmat + 9 * ctx.body;
  mjtNum cyaw = 1.0, syaw = 0.0;  // frame body yaw about world z, for HfieldAlign::Yaw
  if (ctx.align == HfieldAlign::Yaw) {
    // xmat is row-major world = R * body; the body x-axis in world is column 0.
    cyaw = mju_cos(mju_atan2(bmat[3], bmat[0]));
    syaw = mju_sin(mju_atan2(bmat[3], bmat[0]));
  }
  for (int k = 0; k < ctx.npoint; ++k) {
    mjtNum ox = ctx.offsets[2 * k], oy = ctx.offsets[2 * k + 1];
    mjtNum rx = ox, ry = oy;
    if (ctx.align == HfieldAlign::Yaw) {
      rx = cyaw * ox - syaw * oy;
      ry = syaw * ox + cyaw * oy;
    }
    // Both alignments sample in the geom-center plane.
    mjtNum wp[3] = {bpos[0] + rx, bpos[1] + ry, gpos[2]};
    mjtNum rel[3], lp[3];
    mju_sub3(rel, wp, gpos);
    mju_mulMatTVec(lp, gmat, rel, 3, 3);
    mjtNum fx = (lp[0] / hsize[0] + 1.0) * 0.5 * (ncol - 1);
    mjtNum fy = (lp[1] / hsize[1] + 1.0) * 0.5 * (nrow - 1);
    fx = std::min(std::max(fx, 0.0), ncol - 1.001);
    fy = std::min(std::max(fy, 0.0), nrow - 1.001);
    int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
    mjtNum sx = fx - ix, sy = fy - iy;
    int ix1 = std::min(ix + 1, ncol - 1), iy1 = std::min(iy + 1, nrow - 1);
    mjtNum h = (1 - sx) * (1 - sy) * data[iy * ncol + ix] +
               sx * (1 - sy) * data[iy * ncol + ix1] +
               (1 - sx) * sy * data[iy1 * ncol + ix] +
               sx * sy * data[iy1 * ncol + ix1];
    h *= hsize[2];
    // World z of the surface point above the sample: gpos + gmat @ (lp0, lp1, h).
    ctx.out[k] = gpos[2] + gmat[6] * lp[0] + gmat[7] * lp[1] + gmat[8] * h;
  }
}

// mju_error trap for worker threads: throw back to the worker's guard;
// everything else goes to the handler that was active before.
struct MjError : std::exception {
  std::string message;
  explicit MjError(std::string text) : message(std::move(text)) {}
  const char* what() const noexcept override { return message.c_str(); }
};
inline mjfLogHandler prev_log_handler = nullptr;
inline thread_local bool tls_guarded = false;
inline void LogTrap(const mjLogMessage* msg) {
  if (msg->level == mjLOG_ERROR && tls_guarded) {
#ifdef _WIN32
    std::fprintf(stderr, "MJDIAG trap\n");
    std::fflush(stderr);
#endif
    throw MjError(msg->subject);
  }
  prev_log_handler(msg);
}
inline void InstallLogTrap() { prev_log_handler = mju_setLogHandler(LogTrap); }

// The Batch whose step callback is running on this thread: a reentrant call
// from inside it raises instead of deadlocking on mu_.
class Batch;
inline thread_local const Batch* tls_callback = nullptr;

class Batch {
 public:
  enum class Op {
    Step,
    Substep,   // callback-step recovery tail
    Substep1,  // split callback: mj_step1 and selected sensor copy-out
    Substep2,  // split callback: apply writes and mj_step2
    Forward,
    Reset,
    RefreshSensor,
    SetConst,
    JacSite,
    SampleHfield
  };
  using Ids = nb::ndarray<nb::ndim<1>, nb::c_contig>;

  Batch(nb::object model, int num_sims, int num_threads, bool forward,
        std::optional<std::vector<int>> cpu_ids)
      : num_sims_(num_sims), forward_(forward) {
    if (num_sims < 1) throw nb::value_error("num_sims must be >= 1");
    auto addr = nb::cast<uintptr_t>(model.attr("_address"));
    template_ = mj_copyModel(nullptr, reinterpret_cast<const mjModel*>(addr));
    if (template_->opt.enableflags & mjENBL_SLEEP) throw nb::value_error("sleep is not supported");
    data_fields_ = DataFields(template_);
    model_fields_ = ModelFields(template_);
    for (auto& [name, f] : model_fields_) {
      if (!f.asset) restorable_.push_back(&f);
    }
    scalars_.assign(num_sims, Scalars::Of(template_));
    pool_ = MakePool(num_threads, num_sims, std::move(cpu_ids));
    for (int t = 0; t < pool_->size(); ++t) data_.push_back(mj_makeData(template_));
    nstate_ = mj_stateSize(template_, mjSTATE_INTEGRATION);
    states_.resize(static_cast<size_t>(num_sims) * nstate_);
    warnings_.resize(static_cast<size_t>(num_sims) * mjNWARNING);
    for (int i = 0; i < num_sims; ++i) {
      mj_getState(template_, data_[0], State(i), mjSTATE_INTEGRATION);
    }
  }

  ~Batch() {
    pool_.reset();
    for (mjData* d : data_) mj_deleteData(d);
    for (mjModel* m : models_) mj_deleteModel(m);
    mj_deleteModel(template_);
  }
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  int num_sims() const { return num_sims_; }
  int num_threads() const { return pool_->size(); }
  int nstate() const { return nstate_; }

  nb::ndarray<nb::numpy> bind(const std::string& name, std::optional<nb::object> dtype) {
    ReentryGuard();
    std::lock_guard<std::mutex> lock(mu_);
    if (name == "state") {
      // The rows are the per-sim mjSTATE_INTEGRATION vectors themselves, so a write to
      // one is authoritative at the sim's next call; field writes land on top of it.
      if (dtype && !dtype->is_none()) throw nb::value_error("state cannot be float32");
      return StateView();
    }
    const FieldInfo& f = Field(data_fields_, name);
    bool f32 = ParseDtype(f, dtype);
    for (auto& s : bound_) {
      if (s->info == &f) return Matching(*s, f32);
    }
    return View(BoundOrAdd(f, f32));
  }

  nb::ndarray<nb::numpy> expand(const std::string& name, std::optional<nb::object> dtype) {
    ReentryGuard();
    std::lock_guard<std::mutex> lock(mu_);
    const FieldInfo& f = Field(model_fields_, name);
    bool f32 = ParseDtype(f, dtype);
    for (auto& s : expanded_) {
      if (s->info == &f) return Matching(*s, f32);
    }
    return View(Expand(f, f32));
  }

  void step(std::optional<Ids> ids, int nstep, std::optional<nb::ndarray<>> history,
            std::optional<nb::callable> callback,
            std::optional<std::vector<int>> substep_sensor_copyout) {
    ReentryGuard();
    if (nstep < 1) throw nb::value_error("nstep must be >= 1");
    int sensor_start = 0, sensor_stop = 0;
    if (substep_sensor_copyout) {
      if (substep_sensor_copyout->size() != 2) {
        throw nb::value_error("substep_sensor_copyout must have shape (2,)");
      }
      sensor_start = (*substep_sensor_copyout)[0];
      sensor_stop = (*substep_sensor_copyout)[1];
      if (sensor_start < 0 || sensor_start >= sensor_stop ||
          sensor_stop > template_->nsensordata) {
        throw nb::value_error(
            "substep_sensor_copyout entries must satisfy 0 <= start < stop <= nsensordata");
      }
      if (!callback) throw nb::value_error("substep_sensor_copyout requires callback");
    }
    auto sel = Parse(ids);
    mjtNum* hist = nullptr;
    if (history) {
      hist = HistoryPtr(*history, sel ? static_cast<int>(sel->size()) : num_sims_, nstep);
    }
    if (callback) {
      RunCallback(std::move(sel), nstep, hist, *callback, sensor_start,
                  sensor_stop - sensor_start);
    } else {
      Run(Op::Step, std::move(sel), nstep, hist);
    }
  }
  void forward(std::optional<Ids> ids) {
    ReentryGuard();
    Run(Op::Forward, Parse(ids), 0);
  }

  void RefreshSensorRanges(std::optional<Ids> ids, const std::vector<int>& sensor_ranges) {
    ReentryGuard();
    for (size_t i = 0; i < sensor_ranges.size(); i += 2) {
      int start = sensor_ranges[i], stop = sensor_ranges[i + 1];
      if (start < 0 || start >= stop || stop > template_->nsensordata) {
        throw nb::value_error("sensor range entries must satisfy 0 <= start < stop <= nsensordata");
      }
    }
    auto sel = Parse(ids);
    Slot* sensor;
    {
      std::lock_guard<std::mutex> lock(mu_);
      sensor = &BoundOrAdd(Field(data_fields_, "sensordata"), false);
    }
    QueryCtx ctx;
    ctx.sensor = sensor;
    ctx.sensor_ranges = &sensor_ranges;
    Run(Op::RefreshSensor, std::move(sel), 0, nullptr, &ctx);
  }

  void refresh_sensor_range(std::optional<Ids> ids, std::optional<std::vector<int>> sensor_range) {
    if (!sensor_range || sensor_range->size() != 2) {
      throw nb::value_error("sensor_range must have shape (2,)");
    }
    RefreshSensorRanges(std::move(ids), *sensor_range);
  }

  void refresh_sensor_ranges(std::optional<Ids> ids,
                             std::optional<std::vector<int>> sensor_ranges) {
    if (!sensor_ranges || sensor_ranges->empty() || sensor_ranges->size() % 2 != 0) {
      throw nb::value_error("sensor_ranges must contain (start, stop) pairs");
    }
    RefreshSensorRanges(std::move(ids), *sensor_ranges);
  }

  void reset(std::optional<Ids> ids, int keyframe) {
    ReentryGuard();
    if (keyframe < -1 || keyframe >= template_->nkey) {
      throw nb::value_error("keyframe out of range");
    }
    Run(Op::Reset, Parse(ids), keyframe);
  }

  // mj_setConst per sim from its expanded fields. Every field it changes
  // becomes expanded, so afterwards each sim steps with a complete set of its
  // own derived constants. A newly expanded field is computed for every sim.
  void set_const(std::optional<Ids> ids) {
    ReentryGuard();
    auto sel = Parse(ids);
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    if (models_.empty()) return;
    error_.clear();
    while (true) {
      RunLocked(Op::SetConst, sel, 0);
      if (changed_.empty()) break;
      for (const FieldInfo* f : changed_) Expand(*f, false);
      changed_.clear();
      sel.reset();
    }
    if (!error_.empty()) throw std::runtime_error(error_);
  }

  // mj_jacSite per selected sim into caller-allocated (sel, 3, nv) rows; either
  // output may be omitted. Runs kinematics and comPos only, not mj_forward, and
  // does not refresh the bound views: the outputs are the caller-allocated rows.
  void jac_site(int site, std::optional<nb::ndarray<>> jacp,
                std::optional<nb::ndarray<>> jacr, std::optional<Ids> ids) {
    ReentryGuard();
    if (site < 0 || site >= template_->nsite) throw nb::value_error("site out of range");
    auto sel = Parse(ids);
    int nsel = sel ? static_cast<int>(sel->size()) : num_sims_;
    QueryCtx ctx;
    ctx.site = site;
    ctx.jacp = jacp ? OutPtr(*jacp, nsel, 3, template_->nv, "jacp") : nullptr;
    ctx.jacr = jacr ? OutPtr(*jacr, nsel, 3, template_->nv, "jacr") : nullptr;
    if (!ctx.jacp && !ctx.jacr) {
      throw nb::value_error("jacp and jacr cannot both be None");
    }
    Run(Op::JacSite, std::move(sel), 0, nullptr, &ctx);
  }

  // Bilinear hfield sampling per selected sim at XY offsets around a frame
  // body's origin, into caller-allocated (sel, npoint) rows: the world z of
  // the sampled hfield surface. Runs kinematics only, not mj_forward, and does
  // not refresh the bound views. All sims sample the template's hfield; a
  // per-sim geom_pos/geom_quat moves the sampling frame. alignment rotates the
  // sampling grid: "world" keeps offsets in world axes, "yaw" rotates them by
  // the frame body's yaw about world z.
  void sample_hfield(int geom, int body, const nb::ndarray<>& offsets, nb::ndarray<> out,
                     std::optional<Ids> ids, const std::string& alignment) {
    ReentryGuard();
    if (geom < 0 || geom >= template_->ngeom) throw nb::value_error("geom out of range");
    if (template_->geom_type[geom] != mjGEOM_HFIELD) {
      throw nb::value_error("geom is not a hfield");
    }
    if (body < 0 || body >= template_->nbody) throw nb::value_error("body out of range");
    HfieldAlign align;
    if (alignment == "world") {
      align = HfieldAlign::World;
    } else if (alignment == "yaw") {
      align = HfieldAlign::Yaw;
    } else {
      throw nb::value_error("alignment must be \"world\" or \"yaw\"");
    }
    auto sel = Parse(ids);
    int nsel = sel ? static_cast<int>(sel->size()) : num_sims_;
    QueryCtx ctx;
    ctx.geom = geom;
    ctx.body = body;
    ctx.offsets = OffsetsPtr(offsets, ctx.npoint);
    ctx.align = align;
    ctx.out = OutPtr(out, nsel, 1, ctx.npoint, "out");
    Run(Op::SampleHfield, std::move(sel), 0, nullptr, &ctx);
  }

 private:
  const FieldInfo& Field(const FieldTable& table, const std::string& name) {
    auto it = table.find(name);
    if (it == table.end()) {
      std::string msg = "unknown field " + name;
      if (&table == &data_fields_ && model_fields_.count(name)) {
        msg = name + " is an mjModel field; use expand(\"" + name + "\")";
      } else if (&table == &model_fields_ && data_fields_.count(name)) {
        msg = name + " is an mjData field; use bind(\"" + name + "\")";
      }
      throw nb::value_error(msg.c_str());
    }
    if (it->second.elem == Elem::Other) {
      throw nb::value_error(("unsupported element type in " + name).c_str());
    }
    return it->second;
  }

  // True for float32 over an mjtNum field; the native dtype is always allowed.
  bool ParseDtype(const FieldInfo& f, std::optional<nb::object>& dtype) {
    if (!dtype || dtype->is_none()) return false;
    nb::object np = nb::module_::import_("numpy");
    std::string name = nb::cast<std::string>(np.attr("dtype")(*dtype).attr("name"));
    if (name == DtypeName(f.elem)) return false;
    if (name == "float32" && f.elem == Elem::Num) return true;
    throw nb::value_error((std::string(f.name) + " cannot be " + name).c_str());
  }

  nb::ndarray<nb::numpy> Matching(Slot& s, bool f32) {
    if (s.f32 != f32)
      throw nb::value_error(
          (std::string(s.info->name) + " is already bound with another dtype").c_str());
    return View(s);
  }

  std::unique_ptr<Slot> MakeSlot(const FieldInfo& f, bool f32, bool mirror) {
    auto s = std::make_unique<Slot>();
    s->info = &f;
    s->f32 = f32;
    s->row = static_cast<size_t>(f.nr) * f.nc * (f32 ? sizeof(float) : f.elem_size);
    s->buf = std::make_unique<uint8_t[]>(s->row * num_sims_);
    if (mirror) s->mirror = std::make_unique<uint8_t[]>(s->row * num_sims_);
    return s;
  }

  Slot& Expand(const FieldInfo& f, bool f32) {
    if (f.asset) throw nb::value_error((std::string(f.name) + " is asset data").c_str());
    auto s = MakeSlot(f, f32, false);
    for (int i = 0; i < num_sims_; ++i) ToBuf(s->buf.get() + i * s->row, f.get(template_), *s);
    expanded_.push_back(std::move(s));
    expanded_set_.insert(&f);
    if (std::string_view(f.name) == "enableflags") enableflags_ = expanded_.back().get();
    if (models_.empty()) {
      for (int t = 0; t < pool_->size(); ++t) models_.push_back(mj_copyModel(nullptr, template_));
    }
    return *expanded_.back();
  }

  // Batch calls serialize on mu_, which the callback-driven step holds while the
  // Python callback runs; a call from inside it would deadlock, so raise instead.
  void ReentryGuard() const {
    if (tls_callback == this) {
      throw std::runtime_error("mjbatch: Batch calls from inside a step callback are not allowed");
    }
  }

  nb::ndarray<nb::numpy> StateView() {
    size_t shape[2] = {static_cast<size_t>(num_sims_), static_cast<size_t>(nstate_)};
    return nb::ndarray<nb::numpy>(states_.data(), 2, shape, nb::find(this), nullptr,
                                  nb::dtype<mjtNum>());
  }

  // The slot for an mjData field, created and filled like bind() fills it when the
  // field is not bound yet; an existing slot keeps its dtype.
  Slot& BoundOrAdd(const FieldInfo& f, bool f32) {
    for (auto& s : bound_) {
      if (s->info == &f) return *s;
    }
    bound_.push_back(MakeSlot(f, f32, f.input));
    // A derived field stays zero until the next call fills it.
    if (f.input) {
      for (int i = 0; i < num_sims_; ++i) {
        mj_setState(template_, data_[0], State(i), mjSTATE_INTEGRATION);
        std::memcpy(data_[0]->warning, Warning(i), sizeof(data_[0]->warning));
        CopyOut(*bound_.back(), data_[0], i);
      }
    }
    return *bound_.back();
  }

  mjtNum* HistoryPtr(const nb::ndarray<>& a, int nsel, int nstep) {
    if (a.ndim() != 3 || static_cast<int>(a.shape(0)) != nsel ||
        static_cast<int>(a.shape(1)) != nstep || static_cast<int>(a.shape(2)) != nstate_) {
      throw nb::value_error("history must have shape (sims, nstep, nstate)");
    }
    if (a.dtype() != nb::dtype<mjtNum>()) {
      throw nb::value_error((std::string("history must be ") + DtypeName(Elem::Num)).c_str());
    }
    if (a.device_type() != nb::device::cpu::value || !IsCContig(a)) {
      throw nb::value_error("history must be a C-contiguous CPU array");
    }
    return static_cast<mjtNum*>(a.data());
  }

  // Validate a caller-allocated (rows, r, c) mjtNum output array; when r == 1 a
  // 2D (rows, c) array is accepted too.
  mjtNum* OutPtr(const nb::ndarray<>& a, int64_t rows, int64_t r, int64_t c,
                 const char* name) {
    bool ok3 = a.ndim() == 3 && static_cast<int64_t>(a.shape(0)) == rows &&
               static_cast<int64_t>(a.shape(1)) == r && static_cast<int64_t>(a.shape(2)) == c;
    bool ok2 = r == 1 && a.ndim() == 2 && static_cast<int64_t>(a.shape(0)) == rows &&
               static_cast<int64_t>(a.shape(1)) == c;
    if (!ok3 && !ok2) {
      std::string msg = std::string(name) + " must have shape (sel, " + std::to_string(r) +
                        ", " + std::to_string(c) + ")";
      throw nb::value_error(msg.c_str());
    }
    if (a.dtype() != nb::dtype<mjtNum>()) {
      throw nb::value_error((std::string(name) + " must be " + DtypeName(Elem::Num)).c_str());
    }
    int64_t row = r * c;
    int64_t s2 = a.stride(a.ndim() - 1), s1 = a.ndim() == 3 ? a.stride(1) : c,
            s0 = a.stride(0);
    if (a.device_type() != nb::device::cpu::value || s2 != 1 || s1 != c || s0 != row) {
      throw nb::value_error((std::string(name) + " must be a C-contiguous CPU array").c_str());
    }
    return static_cast<mjtNum*>(a.data());
  }

  // Validate the (npoint, 2) offsets array of sample_hfield.
  const mjtNum* OffsetsPtr(const nb::ndarray<>& a, int& npoint) {
    if (a.ndim() != 2 || a.shape(1) != 2 || a.shape(0) < 1) {
      throw nb::value_error("offsets must have shape (npoint, 2)");
    }
    if (a.dtype() != nb::dtype<mjtNum>()) {
      throw nb::value_error((std::string("offsets must be ") + DtypeName(Elem::Num)).c_str());
    }
    if (a.device_type() != nb::device::cpu::value || a.stride(1) != 1 || a.stride(0) != 2) {
      throw nb::value_error("offsets must be a C-contiguous CPU array");
    }
    npoint = static_cast<int>(a.shape(0));
    return static_cast<const mjtNum*>(a.data());
  }

  nb::ndarray<nb::numpy> View(const Slot& s) {
    const FieldInfo& f = *s.info;
    size_t shape[3] = {static_cast<size_t>(num_sims_), static_cast<size_t>(f.nr),
                       static_cast<size_t>(f.nc)};
    nb::dlpack::dtype dt;
    if (s.f32 || f.elem == Elem::Float) {
      dt = nb::dtype<float>();
    } else if (f.elem == Elem::Num) {
      dt = nb::dtype<mjtNum>();
    } else if (f.elem == Elem::Int) {
      dt = nb::dtype<int>();
    } else if (f.elem == Elem::Bool) {
      dt = nb::dtype<bool>();
    } else {
      dt = nb::dtype<uint8_t>();
    }
    return nb::ndarray<nb::numpy>(s.buf.get(), f.ndim + 1, shape, nb::find(this), nullptr, dt);
  }

  void CopyOut(Slot& s, mjData* d, int i) {
    uint8_t* row = s.buf.get() + i * s.row;
    ToBuf(row, s.info->get(d), s);
    if (s.mirror) std::memcpy(s.mirror.get() + i * s.row, row, s.row);
  }

  // Copy only the callback's declared sensordata columns into its bound view.
  // Unlike CopyOut, this does not touch a mirror or columns outside the range.
  void CopySensorRange(Slot& s, mjData* d, int i, int start, int width) {
    uint8_t* row = s.buf.get() + i * s.row;
    const mjtNum* src = d->sensordata + start;
    if (s.f32 && sizeof(mjtNum) != sizeof(float)) {
      auto* dst = reinterpret_cast<float*>(row) + start;
      for (int k = 0; k < width; ++k) dst[k] = static_cast<float>(src[k]);
    } else {
      std::memcpy(row + static_cast<size_t>(start) * sizeof(mjtNum), src,
                  static_cast<size_t>(width) * sizeof(mjtNum));
    }
  }

  void Restore(mjModel* m) {
    for (const FieldInfo* f : restorable_) std::memcpy(f->get(m), f->get(template_), f->bytes());
    m->npolygonmax = template_->npolygonmax;
    m->nmeshdegmax = template_->nmeshdegmax;
    Scalars::Of(template_).Apply(m);
  }

  mjtNum* State(int i) { return states_.data() + static_cast<size_t>(i) * nstate_; }
  mjWarningStat* Warning(int i) { return warnings_.data() + static_cast<size_t>(i) * mjNWARNING; }

  void RunSim(int t, int i, Op op, int arg, mjtNum* hist, const QueryCtx* ctx,
              const CallbackCtx* cctx) {
    mjModel* m = models_.empty() ? template_ : models_[t];
    mjData* d = data_[t];
    if (op == Op::SetConst) Restore(m);
    for (auto& s : expanded_) FromBuf(s->info->get(m), s->buf.get() + i * s->row, *s);
    if (op == Op::SetConst) {
      mj_setConst(m, d);
      for (auto& s : expanded_) ToBuf(s->buf.get() + i * s->row, s->info->get(m), *s);
      scalars_[i] = Scalars::Of(m);
      for (const FieldInfo* f : restorable_) {
        if (!expanded_set_.count(f) && std::memcmp(f->get(m), f->get(template_), f->bytes())) {
          std::lock_guard<std::mutex> lock(changed_mu_);
          changed_.insert(f);
        }
      }
      Restore(m);
      return;
    }
    if (!models_.empty()) scalars_[i].Apply(m);
    if (op == Op::Reset) {
      if (arg >= 0) {
        mj_resetDataKeyframe(m, d, arg);
      } else {
        mj_resetData(m, d);
      }
    } else {
      mj_setState(m, d, State(i), mjSTATE_INTEGRATION);
      std::memcpy(d->warning, Warning(i), sizeof(d->warning));
    }
    for (auto& s : bound_) {
      if (!s->mirror) continue;
      const uint8_t* row = s->buf.get() + i * s->row;
      if (std::memcmp(row, s->mirror.get() + i * s->row, s->row) != 0) {
        CopyChanged(s->info->get(d), row, s->mirror.get() + i * s->row, *s);
      }
    }
    switch (op) {
      case Op::Step: {
        for (int k = 0; k < arg; ++k) {
          mj_step(m, d);
          if (hist) {
            mj_getState(m, d, hist + static_cast<size_t>(k) * nstate_, mjSTATE_INTEGRATION);
          }
        }
        if (forward_) mj_forward(m, d);  // derived fields current with the new state
        break;
      }
      case Op::Substep: {
        // One substep of a callback-driven step. The mirror diff above applied the
        // callback's writes; sync the mirrors so the next substep diffs against what
        // is in mjData now.
        for (auto& s : bound_) {
          if (s->mirror) {
            std::memcpy(s->mirror.get() + i * s->row, s->buf.get() + i * s->row, s->row);
          }
        }
        bool tail = cctx->tail_only;
        if (!cctx->tail_only) {
          mj_step(m, d);
          if (cctx->hist) {
            mj_getState(m, d, cctx->hist + static_cast<size_t>(cctx->k) * nstate_,
                        mjSTATE_INTEGRATION);
          }
          mj_getState(m, d, State(i), mjSTATE_INTEGRATION);
          std::memcpy(Warning(i), d->warning, sizeof(d->warning));
          if (cctx->k + 1 == cctx->nstep) tail = true;
        }
        if (!tail) return;  // mid-rollout: the copy-out tail waits for the last substep
        cctx->done[0] = 1;
        // A tail after an exception always forwards, so derived fields are current
        // with the state the sim stopped at rather than one substep behind.
        if (forward_ || cctx->tail_only) mj_forward(m, d);
        if (cctx->tail_only) std::memcpy(Warning(i), d->warning, sizeof(d->warning));
        for (auto& s : bound_) CopyOut(*s, d, i);
        return;
      }
      case Op::Substep1: {
        // Position- and velocity-stage sensors are current with State(i) here.
        // Input mirrors stay stale because Substep2 applies the callback writes.
        mj_step1(m, d);
        CopySensorRange(*cctx->sensor, d, i, cctx->sensor_start, cctx->sensor_width);
        return;
      }
      case Op::Substep2: {
        // A worker's mjData holds the last phase-one sim it visited, not a
        // per-sim snapshot. Stage one is recomputed after State(i) and the
        // callback writes have been loaded; the sensor view copied above is
        // still the fresh result at x_k. Sync mirrors only after mj_step2.
        mj_step1(m, d);
        for (auto& s : bound_) {
          if (s->mirror) {
            std::memcpy(s->mirror.get() + i * s->row, s->buf.get() + i * s->row, s->row);
          }
        }
        mj_step2(m, d);
        if (cctx->hist) {
          mj_getState(m, d, cctx->hist + static_cast<size_t>(cctx->k) * nstate_,
                      mjSTATE_INTEGRATION);
        }
        mj_getState(m, d, State(i), mjSTATE_INTEGRATION);
        std::memcpy(Warning(i), d->warning, sizeof(d->warning));
        if (cctx->k + 1 != cctx->nstep) return;
        cctx->done[0] = 1;
        if (forward_) mj_forward(m, d);
        for (auto& s : bound_) CopyOut(*s, d, i);
        return;
      }
      case Op::Forward:
      case Op::Reset:
        mj_forward(m, d);
        break;
      case Op::RefreshSensor:
        mj_kinematics(m, d);
        mj_comPos(m, d);
        mj_comVel(m, d);
        mj_sensorPos(m, d);
        mj_sensorVel(m, d);
        for (int j = 0; j < static_cast<int>(ctx->sensor_ranges->size()); j += 2) {
          CopySensorRange(*ctx->sensor, d, i, (*ctx->sensor_ranges)[j],
                          (*ctx->sensor_ranges)[j + 1] - (*ctx->sensor_ranges)[j]);
        }
        return;
      // The query ops run kinematics only: they do not change the integration
      // state, so the copy-out tail below is skipped. mjData is shared by every
      // sim a worker serves, and copying out would leak another sim's stale
      // derived fields into the bound views; the outputs are the
      // caller-allocated rows.
      case Op::JacSite:
        mj_kinematics(m, d);
        mj_comPos(m, d);
        mj_jacSite(m, d, ctx->jacp, ctx->jacr, ctx->site);
        return;
      case Op::SampleHfield:
        mj_kinematics(m, d);
        SampleHfield(m, d, *ctx);
        return;
      case Op::SetConst:
        break;
    }
    mj_getState(m, d, State(i), mjSTATE_INTEGRATION);
    std::memcpy(Warning(i), d->warning, sizeof(d->warning));
    for (auto& s : bound_) CopyOut(*s, d, i);
  }

  std::optional<std::vector<int>> Parse(const std::optional<Ids>& ids) {
    if (!ids) return std::nullopt;
    if (ids->dtype() == nb::dtype<bool>()) {
      if (static_cast<int>(ids->shape(0)) != num_sims_) {
        throw nb::value_error("a boolean ids mask must have num_sims entries");
      }
      std::vector<int> out;
      for (int i = 0; i < num_sims_; ++i) {
        if (static_cast<const bool*>(ids->data())[i]) out.push_back(i);
      }
      return out;
    }
    std::vector<int> out(ids->shape(0));
    for (size_t j = 0; j < out.size(); ++j) {
      if (ids->dtype() == nb::dtype<int64_t>()) {
        out[j] = static_cast<int>(static_cast<const int64_t*>(ids->data())[j]);
      } else if (ids->dtype() == nb::dtype<int32_t>()) {
        out[j] = static_cast<const int32_t*>(ids->data())[j];
      } else {
        throw nb::value_error("ids must be int32, int64 or a bool mask");
      }
      if (out[j] < 0 || out[j] >= num_sims_ || (j > 0 && out[j] <= out[j - 1])) {
        throw nb::value_error("ids must be sorted, unique and in range");
      }
    }
    return out;
  }

  void Guarded(int t, int i, Op op, int arg, mjtNum* hist, const QueryCtx* ctx,
               const CallbackCtx* cctx) {
    struct TrapGuard {
      bool previous;
      TrapGuard() : previous(std::exchange(tls_guarded, true)) {}
      ~TrapGuard() { tls_guarded = previous; }
    } trap_guard;
    try {
      RunSim(t, i, op, arg, hist, ctx, cctx);
    } catch (const MjError& e) {
#ifdef _WIN32
      std::fprintf(stderr, "MJDIAG catch sim %d\n", i);
      std::fflush(stderr);
#endif
      // The sim's state was not written back; the worker's mjData, left
      // mid-call with its stack and arena in use, serves other sims next.
      if (op == Op::SetConst) Restore(models_[t]);
      mj_resetData(template_, data_[t]);
#ifdef _WIN32
      std::fprintf(stderr, "MJDIAG reset sim %d\n", i);
      std::fflush(stderr);
#endif
      std::lock_guard<std::mutex> lock(changed_mu_);
      if (error_.empty()) error_ = "sim " + std::to_string(i) + ": " + e.what();
    }
  }

  void Run(Op op, std::optional<std::vector<int>> sel, int arg, mjtNum* hist = nullptr,
           const QueryCtx* ctx = nullptr) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    error_.clear();
    RunLocked(op, sel, arg, hist, ctx);
    if (!error_.empty()) {
#ifdef _WIN32
      std::fprintf(stderr, "MJDIAG run error\n");
      std::fflush(stderr);
#endif
      throw std::runtime_error(error_);
    }
  }

  // A per-sim enableflags must not switch on what the constructor refused. Raised
  // here rather than on a worker, where the setjmp path only yields a RuntimeError.
  void CheckSleep(const int* p, int n) {
    if (!enableflags_) return;
    for (int j = 0; j < n; ++j) {
      int i = p ? p[j] : j;
      int flags = *reinterpret_cast<const int*>(enableflags_->buf.get() + i * enableflags_->row);
      if (flags & mjENBL_SLEEP) {
        throw nb::value_error(("sim " + std::to_string(i) + ": sleep is not supported").c_str());
      }
    }
  }

  // The step1/step2 split is Euler-only. Expanded integrators are checked per
  // selected simulation so a mixed batch fails closed before touching state.
  void CheckEuler(const int* p, int n) {
    const Slot* integrator = nullptr;
    for (auto& s : expanded_) {
      if (std::string_view(s->info->name) == "integrator") {
        integrator = s.get();
        break;
      }
    }
    for (int j = 0; j < n; ++j) {
      int i = p ? p[j] : j;
      int value = template_->opt.integrator;
      if (integrator) {
        std::memcpy(&value, integrator->buf.get() + i * integrator->row, sizeof(value));
      }
      if (value != mjINT_EULER) {
        throw nb::value_error(("sim " + std::to_string(i) +
                               ": substep_sensor_copyout requires the Euler integrator")
                                  .c_str());
      }
    }
  }

  void RunLocked(Op op, const std::optional<std::vector<int>>& sel, int arg,
                 mjtNum* hist = nullptr, const QueryCtx* ctx = nullptr) {
    const int* p = sel ? sel->data() : nullptr;
    const int n = sel ? static_cast<int>(sel->size()) : num_sims_;
    CheckSleep(p, n);
    auto fn = [this, op, arg, p, hist, ctx](int t, int j) {
      QueryCtx row_ctx;
      const QueryCtx* c = nullptr;
      if (ctx) {
        row_ctx = *ctx;
        if (row_ctx.jacp) row_ctx.jacp += static_cast<size_t>(j) * 3 * template_->nv;
        if (row_ctx.jacr) row_ctx.jacr += static_cast<size_t>(j) * 3 * template_->nv;
        if (row_ctx.out) row_ctx.out += static_cast<size_t>(j) * row_ctx.npoint;
        c = &row_ctx;
      }
      Guarded(t, p ? p[j] : j, op, arg,
              hist ? hist + static_cast<size_t>(j) * arg * nstate_ : nullptr, c, nullptr);
    };
    if (pool_->size() == 1) {
      for (int j = 0; j < n; ++j) fn(0, j);
    } else {
      pool_->Run(n, fn);
    }
  }

  // One substep for every selected sim that has not run its copy-out tail yet.
  void SubstepLocked(const int* p, int n, int k, int nstep, mjtNum* hist, uint8_t* done,
                     bool tail_only, Op op = Op::Substep, Slot* sensor = nullptr,
                     int sensor_start = 0, int sensor_width = 0) {
    CallbackCtx base;
    base.k = k;
    base.nstep = nstep;
    base.tail_only = tail_only;
    base.sensor = sensor;
    base.sensor_start = sensor_start;
    base.sensor_width = sensor_width;
    auto fn = [&](int t, int j) {
      if (done[j]) return;
      CallbackCtx row = base;
      if (hist) row.hist = hist + static_cast<size_t>(j) * nstep * nstate_;
      row.done = done + j;
      Guarded(t, p ? p[j] : j, op, k, nullptr, nullptr, &row);
    };
    if (pool_->size() == 1) {
      for (int j = 0; j < n; ++j) fn(0, j);
    } else {
      pool_->Run(n, fn);
    }
  }

  // A step with a Python control callback: the calling thread invokes the
  // callback before every substep, with the GIL held and every worker idle so
  // the views it passes are stable, then dispatches the substep to the pool.
  void RunCallback(std::optional<std::vector<int>> sel, int nstep, mjtNum* hist,
                   nb::callable callback, int sensor_start, int sensor_width) {
    {
      // Lock mu_ without the GIL: the callback needs the GIL while mu_ is held,
      // so blocking on mu_ with it could deadlock two callback-driven steps.
      nb::gil_scoped_release release;
      mu_.lock();
    }
    std::unique_lock<std::mutex> lock(mu_, std::adopt_lock);
    Slot& ctrl = BoundOrAdd(Field(data_fields_, "ctrl"), false);
    Slot* sensor = nullptr;
    nb::object sensor_view;
    if (sensor_width) sensor = &BoundOrAdd(Field(data_fields_, "sensordata"), false);
    // The views are built once per call, so the callback sees the same live
    // arrays every substep and the loop allocates nothing.
    nb::object state = nb::cast(StateView());
    nb::object ctrl_view = nb::cast(View(ctrl));
    if (sensor) {
      nb::object full = nb::cast(View(*sensor));
      sensor_view = full.attr("__getitem__")(
          nb::make_tuple(nb::slice(nb::none(), nb::none(), nb::none()),
                         nb::slice(sensor_start, sensor_start + sensor_width)));
    }
    const int* p = sel ? sel->data() : nullptr;
    const int n = sel ? static_cast<int>(sel->size()) : num_sims_;
    CheckSleep(p, n);
    if (sensor_width) CheckEuler(p, n);
    error_.clear();
    std::vector<uint8_t> done(n, 0);
    int dispatched = 0, active = n;
    std::exception_ptr pending;
    for (int k = 0; k < nstep && active; ++k) {
      if (sensor_width) {
        {
          nb::gil_scoped_release release;
          SubstepLocked(p, n, k, nstep, hist, done.data(), false, Op::Substep1, sensor,
                        sensor_start, sensor_width);
        }
        ++dispatched;
        if (!error_.empty()) break;
      }
      tls_callback = this;
      try {
        if (sensor_width) {
          callback(k, state, ctrl_view, sensor_view);
        } else {
          callback(k, state, ctrl_view);
        }
      } catch (...) {
        pending = std::current_exception();
      }
      tls_callback = nullptr;
      if (pending) break;
      {
        nb::gil_scoped_release release;
        SubstepLocked(p, n, k, nstep, hist, done.data(), false,
                      sensor_width ? Op::Substep2 : Op::Substep, sensor, sensor_start,
                      sensor_width);
      }
      ++dispatched;
      if (!error_.empty()) break;
      active = 0;
      for (int j = 0; j < n; ++j) active += !done[j];
    }
    if ((pending || !error_.empty()) && dispatched) {
      // Sims stopped at their last completed substep; copy out where they are.
      nb::gil_scoped_release release;
      SubstepLocked(p, n, 0, nstep, hist, done.data(), true);
    }
    if (pending) std::rethrow_exception(pending);
    if (!error_.empty()) throw std::runtime_error(error_);
  }

  int num_sims_;
  bool forward_;  // step ends with mj_forward
  mjModel* template_ = nullptr;
  FieldTable data_fields_;
  FieldTable model_fields_;
  std::vector<const FieldInfo*> restorable_;
  std::vector<mjData*> data_;     // per worker
  std::vector<mjModel*> models_;  // per worker, once anything is expanded
  int nstate_;
  std::vector<mjtNum> states_;           // per sim, mjSTATE_INTEGRATION
  std::vector<mjWarningStat> warnings_;  // per sim
  std::vector<Scalars> scalars_;         // per sim
  std::vector<std::unique_ptr<Slot>> bound_;
  std::vector<std::unique_ptr<Slot>> expanded_;
  std::set<const FieldInfo*> expanded_set_;
  const Slot* enableflags_ = nullptr;   // scanned for mjENBL_SLEEP before every call
  std::set<const FieldInfo*> changed_;  // set_const outputs not yet expanded
  std::unique_ptr<ThreadPool> pool_;
  std::mutex mu_;          // serializes calls; held with the GIL released in Run
  std::mutex changed_mu_;  // changed_ and error_ from workers
  std::string error_;
};
