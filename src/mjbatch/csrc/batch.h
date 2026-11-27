// SPDX-License-Identifier: Apache-2.0

// Batch: N MuJoCo simulations on the upstream mjb_* executor, with fork-specific
// dtype adapters, Python callbacks, CPU pinning and query operations.
#pragma once

#include <mujoco/mjbatch.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>
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

namespace nb = nanobind;

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

inline bool IsCContig(const nb::ndarray<>& a) {
  if (a.size() == 0) return true;
  int64_t expect = 1;
  for (size_t i = a.ndim(); i-- > 0;) {
    if (a.shape(i) != 1 && a.stride(i) != expect) return false;
    expect *= static_cast<int64_t>(a.shape(i));
  }
  return true;
}

struct FieldInfo {
  const char* name;
  void* (*get)(void* obj);
  int nr, nc;
  int ndim;
  size_t elem_size;
  Elem elem;
  bool input;
  bool asset;
  void (*set)(void* obj, void* ptr) = nullptr;  // mjModel array fields: point the field at ptr
  size_t bytes() const { return static_cast<size_t>(nr) * nc * elem_size; }
};

using FieldTable = std::unordered_map<std::string, FieldInfo>;

inline bool IsInput(std::string_view name) {
  for (std::string_view s : {"time", "qpos", "qvel", "act", "history", "plastic", "qacc_warmstart",
                             "ctrl", "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos",
                             "mocap_quat", "userdata", "plugin_state", "warning"}) {
    if (name == s) return true;
  }
  return false;
}

inline int StateBit(std::string_view name) {
  static constexpr std::pair<std::string_view, int> bits[] = {
      {"time", mjSTATE_TIME},
      {"qpos", mjSTATE_QPOS},
      {"qvel", mjSTATE_QVEL},
      {"act", mjSTATE_ACT},
      {"history", mjSTATE_HISTORY},
      {"plastic", mjSTATE_PLASTIC},
      {"qacc_warmstart", mjSTATE_WARMSTART},
      {"ctrl", mjSTATE_CTRL},
      {"qfrc_applied", mjSTATE_QFRC_APPLIED},
      {"xfrc_applied", mjSTATE_XFRC_APPLIED},
      {"eq_active", mjSTATE_EQ_ACTIVE},
      {"mocap_pos", mjSTATE_MOCAP_POS},
      {"mocap_quat", mjSTATE_MOCAP_QUAT},
      {"userdata", mjSTATE_USERDATA},
      {"plugin_state", mjSTATE_PLUGIN}};
  for (auto [n, bit] : bits) {
    if (name == n) return bit;
  }
  return 0;
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
#define X(type, name, nr, nc)                                                    \
  t[#name] =                                                                     \
      FieldInfo{#name,                                                           \
                [](void* o) -> void* { return static_cast<mjModel*>(o)->name; }, \
                static_cast<int>(MJ_M(nr)),                                      \
                static_cast<int>(nc),                                            \
                std::string_view(#nc) == "1" ? 1 : 2,                            \
                sizeof(type),                                                    \
                ElemOf<type>(),                                                  \
                false,                                                           \
                IsAsset(#name),                                                  \
                [](void* o, void* p) { static_cast<mjModel*>(o)->name = static_cast<type*>(p); }};
  MJMODEL_POINTERS
#undef X
#undef MJ_M
#define MJ_M(n) n
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

inline bool IsOption(std::string_view name) {
#define X(type, field, size) \
  if (name == #field) return true;
#define XVEC(type, field, size) X(type, field, size)
  MJOPTION_FIELDS
#undef XVEC
#undef X
  return false;
}

struct Slot {
  const FieldInfo* info;
  bool f32;
  bool model = false;
  size_t row;
  size_t native_row;
  uint8_t* raw = nullptr;
  std::unique_ptr<uint8_t[]> buf;
  std::unique_ptr<uint8_t[]> mirror;
  int offset = -1;
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

enum class HfieldAlign { World, Yaw };

struct QueryCtx {
  int geom = -1, body = -1;
  const mjtNum* offsets = nullptr;
  int npoint = 0;
  HfieldAlign align = HfieldAlign::World;
  mjtNum* out = nullptr;
  const int* selection = nullptr;
  Slot* sensor = nullptr;
  const std::vector<int>* sensor_ranges = nullptr;
};

inline void SampleHfield(const mjModel* m, const mjData* d, const QueryCtx& ctx) {
  int hfield = m->geom_dataid[ctx.geom];
  int nrow = m->hfield_nrow[hfield], ncol = m->hfield_ncol[hfield];
  const mjtNum* hsize = m->hfield_size + static_cast<size_t>(4) * hfield;
  const float* data = m->hfield_data + static_cast<size_t>(hfield) * nrow * ncol;
  const mjtNum* gpos = d->geom_xpos + 3 * ctx.geom;
  const mjtNum* gmat = d->geom_xmat + 9 * ctx.geom;
  const mjtNum* bpos = d->xpos + 3 * ctx.body;
  const mjtNum* bmat = d->xmat + 9 * ctx.body;
  mjtNum cyaw = 1.0, syaw = 0.0;
  if (ctx.align == HfieldAlign::Yaw) {
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
    mjtNum h = (1 - sx) * (1 - sy) * data[iy * ncol + ix] + sx * (1 - sy) * data[iy * ncol + ix1] +
               (1 - sx) * sy * data[iy1 * ncol + ix] + sx * sy * data[iy1 * ncol + ix1];
    h *= hsize[2];
    ctx.out[k] = gpos[2] + gmat[6] * lp[0] + gmat[7] * lp[1] + gmat[8] * h;
  }
}

class Batch;
inline thread_local const Batch* tls_callback = nullptr;

// A worker's own copies of the mjModel arrays mj_setConst writes, and a model pointing at them.
struct SetConstScratch {
  const void* owner = nullptr;  // the Batch the buffer was sized for
  size_t bytes = 0;
  std::vector<uint8_t> buf;
  mjModel model;
};
inline thread_local SetConstScratch tls_setconst;

#ifdef __linux__
struct PinCtx {
  std::atomic<int> arrived{0};
  std::atomic<int> next{1};
  std::atomic<int> error{0};
  int expected;
  uint64_t generation;
  std::thread::id caller;
  const std::vector<int>* cpus;
};

inline std::atomic<uint64_t> pin_generation{0};

void PinWorker(const mjModel*, mjData*, int, void* arg) {
  auto* ctx = static_cast<PinCtx*>(arg);
  thread_local uint64_t seen = 0;
  if (seen == ctx->generation) return;
  seen = ctx->generation;
  if (std::this_thread::get_id() != ctx->caller) {
    int worker = ctx->next.fetch_add(1);
    if (worker < ctx->expected) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET((*ctx->cpus)[worker], &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        int expected = 0;
        ctx->error.compare_exchange_strong(expected, errno);
      }
    }
  }
  ctx->arrived.fetch_add(1);
  while (ctx->arrived.load() < ctx->expected) std::this_thread::yield();
}
#endif

class Batch {
 public:
  using Ids = nb::ndarray<nb::ndim<1>, nb::c_contig>;

  Batch(nb::object model, int num_sims, int num_threads, bool forward,
        std::optional<std::vector<int>> cpu_ids)
      : num_sims_(num_sims), forward_(forward) {
    if (num_sims < 1) throw nb::value_error("num_sims must be >= 1");
    auto addr = nb::cast<uintptr_t>(model.attr("_address"));
    template_ = mj_copyModel(nullptr, reinterpret_cast<const mjModel*>(addr));
    if (template_->opt.enableflags & mjENBL_SLEEP) {
      mj_deleteModel(template_);
      throw nb::value_error("sleep is not supported");
    }

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

    mjBatchConfig config;
    mjb_defaultConfig(&config);
    config.nthread = num_threads;
    config.persistent = false;
    char error[1024] = {};
    batch_ = mjb_makeBatch(template_, num_sims_, &config, error, sizeof(error));
    if (!batch_) {
      std::string message = error;
      mj_deleteModel(template_);
      throw nb::value_error(message.c_str());
    }
    data_fields_ = DataFields(template_);
    model_fields_ = ModelFields(template_);
    int nbvh = 0;  // bounding volumes of bodies, which mj_setConst refits, before the meshes'
    for (int i = 0; i < template_->nbody; ++i) {
      if (template_->body_bvhadr[i] >= 0) {
        nbvh = std::max(nbvh, template_->body_bvhadr[i] + template_->body_bvhnum[i]);
      }
    }
    size_t offset = 0;
    for (auto& [name, f] : model_fields_) {
      // an empty field never differs from the template, so mj_setConst never expands it
      size_t bytes =
          name == "bvh_aabb" ? 6 * static_cast<size_t>(nbvh) * sizeof(mjtNum) : f.bytes();
      if (!bytes || !SetConstOutput(name)) continue;
      out_fields_.push_back({&f, bytes, (offset + 7) & ~size_t(7)});
      offset = out_fields_.back().offset + bytes;
    }
    out_bytes_ = offset;
    for (int g = 0; g < template_->ngeom; ++g) {
      // mj_setConst derives the bound of every geom but these, whose bound is their mesh's
      if (template_->geom_type[g] == mjGEOM_MESH || template_->geom_type[g] == mjGEOM_SDF) {
        mesh_geoms_.push_back(g);
      }
    }
    nstate_ = mjb_nstate(batch_);
    states_ = mjb_state(batch_);
    written_.resize(static_cast<size_t>(num_sims_) * nstate_);
    std::memcpy(written_.data(), states_, written_.size() * sizeof(mjtNum));
    num_threads_ = pin.empty() ? mjb_nthread(batch_) : static_cast<int>(pin.size());
#ifdef __linux__
    if (!pin.empty()) PinWorkers(pin);
#endif
  }

  ~Batch() {
    mjb_deleteBatch(batch_);
    mj_deleteModel(template_);
  }
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  int num_sims() const { return num_sims_; }
  int num_threads() const { return num_threads_; }
  int nstate() const { return nstate_; }

  nb::ndarray<nb::numpy> bind(const std::string& name, std::optional<nb::object> dtype) {
    ReentryGuard();
    std::lock_guard<std::mutex> lock(mu_);
    if (name == "state") {
      if (dtype && !dtype->is_none()) throw nb::value_error("state cannot be float32");
      return StateView();
    }
    const FieldInfo& f = Field(data_fields_, name);
    return View(BindLocked(f, ParseDtype(f, dtype)));
  }

  nb::ndarray<nb::numpy> expand(const std::string& name, std::optional<nb::object> dtype) {
    ReentryGuard();
    std::lock_guard<std::mutex> lock(mu_);
    const FieldInfo& f = Field(model_fields_, name);
    return View(Expand(f, ParseDtype(f, dtype)));
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
      if (sensor_start < 0 || sensor_start >= sensor_stop || sensor_stop > template_->nsensordata) {
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
      RunCallback(std::move(sel), nstep, hist, *callback, sensor_start, sensor_stop - sensor_start);
    } else if (hist) {
      RunHistory(std::move(sel), nstep, hist);
    } else {
      RunSimple(std::move(sel), true, nstep);
    }
  }

  void forward(std::optional<Ids> ids) { RunSimple(Parse(ids), false, 0); }

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
    ApplyCtx ctx;
    ctx.op = ApplyOp::RefreshSensor;
    ctx.sensor = sensor;
    ctx.sensor_ranges = &sensor_ranges;
    RunApply(std::move(sel), ctx);
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

  // mj_resetData, or mj_resetDataKeyframe, per selected sim, with what was written to the sim's
  // state since its last call kept on top, then mj_forward unless the caller runs its own.
  void reset(std::optional<Ids> ids, int keyframe, bool forward) {
    ReentryGuard();
    if (keyframe < -1 || keyframe >= template_->nkey) {
      throw nb::value_error("keyframe out of range");
    }
    auto sel = Parse(ids);
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    auto selected = Selected(sel);
    CheckSleep(selected);
    PrepareExpanded(selected);

    // What the caller last left in each selected row, which the reset keeps wherever it differs
    // from what the sim's last call wrote. Only the selection is copied: a call with ids costs
    // in proportion to their number, not to the size of the batch.
    std::vector<mjtNum> before;
    before.reserve(static_cast<size_t>(selected.size()) * nstate_);
    for (int i : selected) before.insert(before.end(), State(i), State(i) + nstate_);
    std::vector<int> keys;
    const int* key = nullptr;
    if (keyframe >= 0) {
      keys.assign(selected.size(), keyframe);
      key = keys.data();
    }
    mjb_reset(batch_, selected.data(), static_cast<int>(selected.size()), key, mjSTAGE_NONE);
    std::vector<uint8_t> failed(num_sims_, 0);
    std::string error;
    RecordFailures(selected, failed, error);
    auto active = WithoutFailed(selected, failed);
    size_t position = 0;  // of a sim in selected, which active follows in order
    for (int i : active) {
      while (selected[position] != i) ++position;
      mjtNum* row = State(i);
      mjtNum* last = Written(i);
      const mjtNum* old = before.data() + position * nstate_;
      for (int k = 0; k < nstate_; ++k) {
        if (std::memcmp(old + k, last + k, sizeof(mjtNum)) != 0) row[k] = old[k];
      }
    }
    Prepare(active);
    if (forward && !active.empty()) {
      mjb_forward(batch_, active.data(), static_cast<int>(active.size()), mjSTAGE_ACC);
      RecordFailures(active, failed, error);
    }
    Finish(selected, failed);
    if (!error.empty()) throw std::runtime_error(error);
  }

  // mj_setConst per selected sim from its expanded fields, on the upstream executor, which runs
  // only the groups of constants that read an expanded field. Every field mj_setConst changes
  // becomes expanded, so afterwards each sim steps with a complete set of its own derived
  // constants. A newly expanded one starts from the template's value in every row, so the sims
  // outside the selection are computed too, once per such field, to keep them all consistent.
  void set_const(std::optional<Ids> ids) {
    ReentryGuard();
    auto sel = Parse(ids);
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    if (expanded_.empty()) return;
    auto selected = Selected(sel);
    if (selected.empty()) return;
    CheckSleep(selected);
    // The executor loads the rows of an expanded field only where mj_setConst reads it first,
    // which the bounds of mesh geoms are not, although dof_length is computed from them: keep
    // them across its pass, then compute once more with every expanded row loaded, as on a
    // model of the simulation's own.
    bool keep_bounds = rbound_ && !mesh_geoms_.empty();
    std::vector<uint8_t> failed(num_sims_, 0);
    std::string error;
    bool again;
    do {
      again = false;  // a field became expanded, with the template's value outside the selection
      PrepareExpanded(selected);
      if (keep_bounds) CopyBounds(selected, true);
      mjb_setConst(batch_, selected.data(), static_cast<int>(selected.size()));
      RecordFailures(selected, failed, error);
      for (OutField& f : out_fields_) {
        int size = 0, elemsize = 0;
        if (f.expanded) continue;
        const uint8_t* row =
            static_cast<const uint8_t*>(mjb_expanded(batch_, f.info->name, &size, &elemsize));
        if (row && static_cast<size_t>(size) * elemsize == f.bytes) {
          f.row = row;
          f.expanded = true;
          again = sel.has_value();
        }
      }
      if (keep_bounds) {
        CopyBounds(selected, false);
        SetConstCtx ctx{this};
        ctx.changed.resize(out_fields_.size());
        mjb_apply(batch_, selected.data(), static_cast<int>(selected.size()), SetConstApply, &ctx,
                  0);
        RecordFailures(selected, failed, error);
        for (size_t k = 0; k < ctx.changed.size(); ++k) {
          if (!ctx.changed[k]) continue;
          OutField& f = out_fields_[k];
          int size = 0, elemsize = 0;
          f.row = static_cast<const uint8_t*>(mjb_expand(batch_, f.info->name, &size, &elemsize));
          f.expanded = f.row != nullptr;
          again = true;
        }
      }
      if (!again) break;
      selected = Awake();
    } while (!selected.empty());
    for (auto& s : expanded_) {
      if (s->f32 && SetConstOutput(s->info->name)) {
        for (int i : selected) SyncOut(*s, i);
      }
    }
    if (!error.empty()) throw std::runtime_error(error);
  }

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
    ApplyCtx ctx;
    ctx.op = ApplyOp::Hfield;
    ctx.query.geom = geom;
    ctx.query.body = body;
    ctx.query.offsets = OffsetsPtr(offsets, ctx.query.npoint);
    ctx.query.align = align;
    ctx.query.out = OutPtr(out, nsel, 1, ctx.query.npoint, "out");
    auto selected = Selected(sel);
    auto rows = SelectionRows(selected);
    ctx.query.selection = rows.data();
    RunApply(std::move(sel), ctx);
  }

  void rays(nb::ndarray<> pnt, nb::ndarray<> vec, nb::ndarray<> dist,
            std::optional<nb::ndarray<>> geomid, std::optional<nb::ndarray<>> normal,
            std::optional<nb::ndarray<const uint8_t, nb::shape<mjNGROUP>, nb::c_contig>> geomgroup,
            bool flg_static,
            std::optional<nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig>> bodyexclude,
            std::optional<Ids> ids) {
    if (pnt.ndim() != 3) throw nb::value_error("pnt must have shape (num_sims, nray, 3)");
    const size_t n = num_sims_, nray = pnt.shape(1);
    const Elem real = RealOf(pnt);
    const bool f32 = real == Elem::Float;
    const void* origins = Array(pnt, "pnt", real, {n, nray, 3});
    const void* directions = Array(vec, "vec", real, {n, nray, 3});
    void* dists = Array(dist, "dist", real, {n, nray});
    int* geomids =
        geomid ? static_cast<int*>(Array(*geomid, "geomid", Elem::Int, {n, nray})) : nullptr;
    void* normals = normal ? Array(*normal, "normal", real, {n, nray, 3}) : nullptr;
    const mjtByte* group = geomgroup ? geomgroup->data() : nullptr;
    if (bodyexclude && bodyexclude->shape(0) != nray) {
      throw nb::value_error("bodyexclude must have nray entries");
    }
    const int* exclude = bodyexclude ? bodyexclude->data() : nullptr;
    Query(ids, [=](const mjModel* m, mjData* d, int i) {
      mj_kinematics(m, d);
      if (m->nflex) mj_flex(m, d);
      for (size_t k = 0; k < nray; ++k) {
        const size_t at = i * nray + k;
        mjtNum origin[3], direction[3], hit_normal[3];
        for (int c = 0; c < 3; ++c) {
          origin[c] = Get(origins, f32, 3 * at + c);
          direction[c] = Get(directions, f32, 3 * at + c);
        }
        int hit = -1;
        const mjtNum x = mj_ray(m, d, origin, direction, group, flg_static,
                                exclude ? exclude[k] : -1, &hit, normals ? hit_normal : nullptr);
        Put(dists, f32, at, x);
        if (geomids) geomids[at] = hit;
        for (int c = 0; normals && c < 3; ++c)
          Put(normals, f32, 3 * at + c, x < 0 ? 0 : hit_normal[c]);
      }
    });
  }

  void jac(std::optional<nb::ndarray<>> jacp, std::optional<nb::ndarray<>> jacr,
           nb::ndarray<> point, nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig> body,
           std::optional<Ids> ids) {
    const size_t n = num_sims_, row = 3 * static_cast<size_t>(template_->nv);
    const Elem real = RealOf(point);
    const bool f32 = real == Elem::Float;
    const void* points = Array(point, "point", real, {n, 3});
    void* positional = jacp ? Array(*jacp, "jacp", real, {n, 3, row / 3}) : nullptr;
    void* rotational = jacr ? Array(*jacr, "jacr", real, {n, 3, row / 3}) : nullptr;
    if (body.shape(0) != n) throw nb::value_error("body must have num_sims entries");
    const int* bodies = body.data();
    for (size_t i = 0; i < n; ++i) {
      if (bodies[i] < 0 || bodies[i] >= template_->nbody) {
        throw nb::value_error("body ids must be in [0, nbody)");
      }
    }
    Query(ids, [=](const mjModel* m, mjData* d, int i) {
      mj_kinematics(m, d);
      mj_comPos(m, d);
      const mjtNum p[3] = {Get(points, f32, 3 * i), Get(points, f32, 3 * i + 1),
                           Get(points, f32, 3 * i + 2)};
      if (!f32) {
        auto at = [=](void* a) { return a ? static_cast<mjtNum*>(a) + i * row : nullptr; };
        mj_jac(m, d, at(positional), at(rotational), p, bodies[i]);
        return;
      }
      mj_markStack(d);
      mjtNum* jp = positional ? mj_stackAllocNum(d, row) : nullptr;
      mjtNum* jr = rotational ? mj_stackAllocNum(d, row) : nullptr;
      mj_jac(m, d, jp, jr, p, bodies[i]);
      for (size_t k = 0; k < row; ++k) {
        if (jp) Put(positional, true, i * row + k, jp[k]);
        if (jr) Put(rotational, true, i * row + k, jr[k]);
      }
      mj_freeStack(d);
    });
  }

 private:
  using QueryFn = std::function<void(const mjModel*, mjData*, int)>;
  enum class ApplyOp { Substep1, Substep2, RefreshSensor, Hfield, Query };

  // an mjModel array mj_setConst writes, which can differ from the template's
  struct OutField {
    const FieldInfo* info;
    size_t bytes;   // per simulation; bvh_aabb: the bodies' bounding volumes only
    size_t offset;  // in a worker's scratch buffer
    bool expanded = false;
    const uint8_t* row = nullptr;  // its per-simulation rows, once expanded
  };
  struct SetConstCtx {
    Batch* self;
    std::mutex mu;
    std::vector<uint8_t> changed;  // per OutField: a sim's value differs from the template's
  };

  struct ApplyCtx {
    Batch* self = nullptr;
    ApplyOp op;
    QueryCtx query;
    Slot* sensor = nullptr;
    const std::vector<int>* sensor_ranges = nullptr;
    int sensor_start = 0;
    int sensor_width = 0;
    const QueryFn* fn = nullptr;
  };

  static bool SetConstOutput(std::string_view name) {
#define X(field, fixed, kin0, inert0, spring) \
  if (name == #field && ((fixed | kin0 | inert0 | spring) & mjSC_OUT)) return true;
    MJSETCONST_ARRAYS
#undef X
    return false;
  }

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

  bool ParseDtype(const FieldInfo& f, std::optional<nb::object>& dtype) {
    if (!dtype || dtype->is_none()) return false;
    nb::object np = nb::module_::import_("numpy");
    std::string name = nb::cast<std::string>(np.attr("dtype")(*dtype).attr("name"));
    if (name == DtypeName(f.elem)) return false;
    if (name == "float32" && f.elem == Elem::Num) return true;
    throw nb::value_error((std::string(f.name) + " cannot be " + name).c_str());
  }

  std::unique_ptr<Slot> MakeSlot(const FieldInfo& f, bool f32, bool model = false) {
    auto s = std::make_unique<Slot>();
    s->info = &f;
    s->f32 = f32;
    s->model = model;
    s->row = static_cast<size_t>(f.nr) * f.nc * (f32 ? sizeof(float) : f.elem_size);
    s->native_row = f.bytes();
    return s;
  }

  void AllocateConverted(Slot& s, bool mirror) {
    s.buf = std::make_unique<uint8_t[]>(s.row * num_sims_);
    if (mirror) s.mirror = std::make_unique<uint8_t[]>(s.row * num_sims_);
  }

  Slot& BindLocked(const FieldInfo& f, bool f32) {
    for (auto& s : bound_) {
      if (s->info == &f) return Matching(*s, f32);
    }
    auto s = MakeSlot(f, f32);
    int bit = StateBit(f.name);
    if (bit && f.elem == Elem::Num && !f32) {
      s->offset = mj_stateSize(template_, mjSTATE_INTEGRATION & (bit - 1));
      if (mj_stateSize(template_, bit) != f.nr * f.nc) {
        throw std::runtime_error(std::string(f.name) + " does not match its state component");
      }
    } else if (bit) {
      s->offset = mj_stateSize(template_, mjSTATE_INTEGRATION & (bit - 1));
      AllocateConverted(*s, true);
    } else if (std::string_view(f.name) == "warning") {
      s->raw = reinterpret_cast<uint8_t*>(mjb_warning(batch_));
      AllocateConverted(*s, true);
    } else {
      int size = 0, elemsize = 0;
      s->raw = static_cast<uint8_t*>(mjb_output(batch_, f.name, &size, &elemsize));
      if (!s->raw || size != f.nr * f.nc || elemsize != static_cast<int>(f.elem_size)) {
        throw nb::value_error((std::string(f.name) + " cannot be bound").c_str());
      }
      if (f32) AllocateConverted(*s, false);
    }
    bound_.push_back(std::move(s));
    Slot& out = *bound_.back();
    if (out.buf) {
      for (int i = 0; i < num_sims_; ++i) SyncOut(out, i);
    }
    return out;
  }

  Slot& BoundOrAdd(const FieldInfo& f, bool f32) { return BindLocked(f, f32); }

  Slot& Expand(const FieldInfo& f, bool f32) {
    if (f.asset) throw nb::value_error((std::string(f.name) + " is asset data").c_str());
    for (auto& s : expanded_) {
      if (s->info == &f) return Matching(*s, f32);
    }
    auto s = MakeSlot(f, f32, true);
    std::string name = IsOption(f.name) ? "opt." + std::string(f.name) : f.name;
    int size = 0, elemsize = 0;
    s->raw = static_cast<uint8_t*>(mjb_expand(batch_, name.c_str(), &size, &elemsize));
    if (!s->raw || size != f.nr * f.nc || elemsize != static_cast<int>(f.elem_size)) {
      throw nb::value_error((std::string(f.name) + " cannot be expanded").c_str());
    }
    if (f32) AllocateConverted(*s, false);
    expanded_.push_back(std::move(s));
    Slot& out = *expanded_.back();
    for (OutField& o : out_fields_) {
      if (o.info == &f) {
        o.expanded = true;
        o.row = out.raw;
      }
    }
    if (std::string_view(f.name) == "geom_rbound") rbound_ = &out;
    if (out.f32) {
      for (int i = 0; i < num_sims_; ++i) SyncOut(out, i);
    }
    if (std::string_view(f.name) == "enableflags") enableflags_ = &out;
    return out;
  }

  Slot& Matching(Slot& s, bool f32) {
    if (s.f32 != f32) {
      throw nb::value_error(
          (std::string(s.info->name) + " is already bound with another dtype").c_str());
    }
    return s;
  }

  void ReentryGuard() const {
    if (tls_callback == this) {
      throw std::runtime_error("mjbatch: Batch calls from inside a step callback are not allowed");
    }
  }

  mjtNum* State(int i) { return states_ + static_cast<size_t>(i) * nstate_; }
  mjtNum* Written(int i) { return written_.data() + static_cast<size_t>(i) * nstate_; }

  nb::ndarray<nb::numpy> StateView() {
    size_t shape[2] = {static_cast<size_t>(num_sims_), static_cast<size_t>(nstate_)};
    return nb::ndarray<nb::numpy>(states_, 2, shape, nb::find(this), nullptr, nb::dtype<mjtNum>());
  }

  nb::ndarray<nb::numpy> View(const Slot& s) {
    const FieldInfo& f = *s.info;
    size_t shape[3] = {static_cast<size_t>(num_sims_), static_cast<size_t>(f.nr),
                       static_cast<size_t>(f.nc)};
    if (s.offset >= 0 && !s.buf) {
      int64_t strides[3] = {nstate_, f.ndim == 2 ? f.nc : 1, 1};
      return nb::ndarray<nb::numpy>(states_ + s.offset, f.ndim + 1, shape, nb::find(this), strides,
                                    nb::dtype<mjtNum>());
    }
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
    void* data = s.buf ? s.buf.get() : s.raw;
    return nb::ndarray<nb::numpy>(data, f.ndim + 1, shape, nb::find(this), nullptr, dt);
  }

  static mjtNum ReadExposed(const Slot& s, size_t k) {
    if (s.f32) return reinterpret_cast<const float*>(s.buf.get())[k];
    switch (s.info->elem) {
      case Elem::Num:
        return reinterpret_cast<const mjtNum*>(s.buf.get())[k];
      case Elem::Float:
        return reinterpret_cast<const float*>(s.buf.get())[k];
      case Elem::Int:
        return reinterpret_cast<const int*>(s.buf.get())[k];
      case Elem::Byte:
        return reinterpret_cast<const uint8_t*>(s.buf.get())[k];
      case Elem::Bool:
        return reinterpret_cast<const bool*>(s.buf.get())[k];
      default:
        return 0;
    }
  }

  static void WriteExposed(Slot& s, size_t k, mjtNum value) {
    if (s.f32) {
      reinterpret_cast<float*>(s.buf.get())[k] = static_cast<float>(value);
      return;
    }
    switch (s.info->elem) {
      case Elem::Num:
        reinterpret_cast<mjtNum*>(s.buf.get())[k] = value;
        break;
      case Elem::Float:
        reinterpret_cast<float*>(s.buf.get())[k] = static_cast<float>(value);
        break;
      case Elem::Int:
        reinterpret_cast<int*>(s.buf.get())[k] = static_cast<int>(value);
        break;
      case Elem::Byte:
        reinterpret_cast<uint8_t*>(s.buf.get())[k] = static_cast<uint8_t>(value);
        break;
      case Elem::Bool:
        reinterpret_cast<bool*>(s.buf.get())[k] = value != 0;
        break;
      default:
        break;
    }
  }

  void SyncIn(Slot& s, int i) {
    uint8_t* exposed = s.buf.get() + i * s.row;
    if (s.model) {
      if (s.f32) FromBuf(s.raw + i * s.native_row, exposed, s);
      return;
    }
    if (!s.mirror) return;
    uint8_t* mirror = s.mirror.get() + i * s.row;
    size_t count = static_cast<size_t>(s.info->nr) * s.info->nc;
    size_t elem = s.f32 ? sizeof(float) : s.info->elem_size;
    if (s.offset < 0) {
      uint8_t* raw = s.raw + i * s.native_row;
      for (size_t k = 0; k < count; ++k) {
        if (std::memcmp(exposed + k * elem, mirror + k * elem, elem) != 0) {
          std::memcpy(raw + k * elem, exposed + k * elem, elem);
        }
      }
      return;
    }
    mjtNum* state = State(i) + s.offset;
    for (size_t k = 0; k < count; ++k) {
      if (std::memcmp(exposed + k * elem, mirror + k * elem, elem) != 0) {
        state[k] = ReadExposed(s, static_cast<size_t>(i) * count + k);
      }
    }
  }

  void SyncOut(Slot& s, int i) {
    uint8_t* exposed = s.buf.get() + i * s.row;
    if (s.model) {
      ToBuf(exposed, s.raw + i * s.native_row, s);
      return;
    }
    if (s.offset >= 0) {
      size_t count = static_cast<size_t>(s.info->nr) * s.info->nc;
      const mjtNum* state = State(i) + s.offset;
      for (size_t k = 0; k < count; ++k) {
        WriteExposed(s, static_cast<size_t>(i) * count + k, state[k]);
      }
    } else if (s.f32) {
      ToBuf(exposed, s.raw + i * s.native_row, s);
    } else if (s.mirror) {
      std::memcpy(exposed, s.raw + i * s.native_row, s.native_row);
    }
    if (s.mirror) std::memcpy(s.mirror.get() + i * s.row, exposed, s.row);
  }

  void Prepare(int i) {
    for (auto& s : expanded_) SyncIn(*s, i);
    for (auto& s : bound_) SyncIn(*s, i);
  }
  void Prepare(const std::vector<int>& ids) {
    for (int i : ids) Prepare(i);
  }
  void PrepareExpanded(const std::vector<int>& ids) {
    for (int i : ids) {
      for (auto& s : expanded_) SyncIn(*s, i);
    }
  }

  void Finish(int i) {
    for (auto& s : bound_) {
      if (s->buf) SyncOut(*s, i);
    }
    std::memcpy(Written(i), State(i), nstate_ * sizeof(mjtNum));
  }
  void Finish(const std::vector<int>& ids, const std::vector<uint8_t>& failed) {
    for (int i : ids) {
      if (!failed[i]) Finish(i);
    }
  }

  std::string RecordFailures(const std::vector<int>& ids, std::vector<uint8_t>& failed,
                             std::string& error) {
    const int* status = mjb_status(batch_);
    for (int i : ids) {
      if (!status[i] || failed[i]) continue;
      failed[i] = 1;
      if (error.empty()) error = "sim " + std::to_string(i) + ": " + mjb_error(batch_, i);
    }
    return error;
  }

  void ThrowFailures(const std::vector<int>& ids) {
    std::vector<uint8_t> failed(num_sims_, 0);
    std::string error;
    RecordFailures(ids, failed, error);
    if (!error.empty()) throw std::runtime_error(error);
  }

  void FinishAndThrow(const std::vector<int>& ids) {
    std::vector<uint8_t> failed(num_sims_, 0);
    std::string error;
    RecordFailures(ids, failed, error);
    Finish(ids, failed);
    if (!error.empty()) throw std::runtime_error(error);
  }

  static std::vector<int> WithoutFailed(const std::vector<int>& ids,
                                        const std::vector<uint8_t>& failed) {
    std::vector<int> active;
    active.reserve(ids.size());
    for (int i : ids) {
      if (!failed[i]) active.push_back(i);
    }
    return active;
  }

  std::vector<int> Selected(const std::optional<std::vector<int>>& sel) const {
    if (sel) return *sel;
    std::vector<int> ids(num_sims_);
    for (int i = 0; i < num_sims_; ++i) ids[i] = i;
    return ids;
  }

  std::vector<int> SelectionRows(const std::vector<int>& selected) const {
    std::vector<int> rows(num_sims_, -1);
    for (size_t j = 0; j < selected.size(); ++j) rows[selected[j]] = static_cast<int>(j);
    return rows;
  }

  void RunSimple(std::optional<std::vector<int>> sel, bool step, int nstep) {
    ReentryGuard();
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    auto selected = Selected(sel);
    CheckSleep(selected);
    Prepare(selected);
    if (step) {
      mjb_step(batch_, selected.data(), static_cast<int>(selected.size()), nstep,
               forward_ ? mjSTAGE_ACC : mjSTAGE_NONE);
    } else {
      mjb_forward(batch_, selected.data(), static_cast<int>(selected.size()), mjSTAGE_ACC);
    }
    FinishAndThrow(selected);
  }

  void RunHistory(std::optional<std::vector<int>> sel, int nstep, mjtNum* hist) {
    ReentryGuard();
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    auto selected = Selected(sel);
    CheckSleep(selected);
    Prepare(selected);
    auto active = selected;
    auto rows = SelectionRows(selected);
    std::vector<uint8_t> failed(num_sims_, 0);
    std::string error;
    for (int k = 0; k < nstep && !active.empty(); ++k) {
      mjb_step(batch_, active.data(), static_cast<int>(active.size()), 1,
               forward_ && k + 1 == nstep ? mjSTAGE_ACC : mjSTAGE_NONE);
      RecordFailures(active, failed, error);
      for (int i : active) {
        if (!failed[i]) {
          std::memcpy(hist + (static_cast<size_t>(rows[i]) * nstep + k) * nstate_, State(i),
                      nstate_ * sizeof(mjtNum));
        }
      }
      active = WithoutFailed(active, failed);
    }
    Finish(selected, failed);
    if (!error.empty()) throw std::runtime_error(error);
  }

  // geom_rbound of the mesh geoms, for the sims about to run: saved before the executor's pass
  // and put back after it, that pass having overwritten their rows with the template's
  void CopyBounds(const std::vector<int>& ids, bool save) {
    bounds_.resize(ids.size() * mesh_geoms_.size());
    size_t k = 0;
    for (int i : ids) {
      mjtNum* row = reinterpret_cast<mjtNum*>(rbound_->raw + i * rbound_->native_row);
      for (int g : mesh_geoms_) {
        if (save) {
          bounds_[k] = row[g];
        } else {
          row[g] = bounds_[k];
        }
        ++k;
      }
    }
  }

  // the sims that may run: a sleeping one is rejected by CheckSleep when selected
  std::vector<int> Awake() const {
    std::vector<int> ids;
    for (int i = 0; i < num_sims_; ++i) {
      int flags = template_->opt.enableflags;
      if (enableflags_) {
        std::memcpy(&flags, enableflags_->raw + i * enableflags_->native_row, sizeof(flags));
      }
      if (!(flags & mjENBL_SLEEP)) ids.push_back(i);
    }
    return ids;
  }

  // mjb_apply task of the pass after the executor's: mj_setConst on a copy of the thread's model
  // whose written arrays are the thread's own, so that it is left as the executor loaded it
  static void SetConstApply(const mjModel* m, mjData* d, int i, void* arg) {
    auto& ctx = *static_cast<SetConstCtx*>(arg);
    Batch* self = ctx.self;
    SetConstScratch& s = tls_setconst;
    if (s.owner != self || s.bytes < self->out_bytes_) {
      // an exception must not leave this function for the pool's thread; this error is trapped
      // like a MuJoCo one, and names the simulation
      try {
        s.buf.assign(self->out_bytes_, 0);
      } catch (const std::bad_alloc&) {
        mju_error("mjbatch: out of memory for set_const");
      }
      s.bytes = self->out_bytes_;
      s.owner = self;
    }
    mjModel* work = &s.model;
    *work = *m;
    for (const OutField& f : self->out_fields_) {
      uint8_t* own = s.buf.data() + f.offset;
      std::memcpy(own, f.info->get(work), f.bytes);
      f.info->set(work, own);
    }
    mj_setConst(work, d);
    bool changed = false;
    for (size_t k = 0; k < self->out_fields_.size(); ++k) {
      const OutField& f = self->out_fields_[k];
      const uint8_t* value = s.buf.data() + f.offset;
      if (f.row) {
        std::memcpy(const_cast<uint8_t*>(f.row) + static_cast<size_t>(i) * f.bytes, value, f.bytes);
      } else {
        changed = changed || std::memcmp(value, f.info->get(self->template_), f.bytes);
      }
    }
    if (!changed) return;
    std::lock_guard<std::mutex> lock(ctx.mu);
    for (size_t k = 0; k < self->out_fields_.size(); ++k) {
      const OutField& f = self->out_fields_[k];
      if (!f.row && std::memcmp(s.buf.data() + f.offset, f.info->get(self->template_), f.bytes)) {
        ctx.changed[k] = 1;
      }
    }
  }

  static void ApplyFunc(const mjModel* m, mjData* d, int i, void* arg) {
    auto& ctx = *static_cast<ApplyCtx*>(arg);
    Batch* self = ctx.self;
    switch (ctx.op) {
      case ApplyOp::Substep1:
        mj_step1(m, d);
        self->CopySensorRange(*ctx.sensor, d, i, ctx.sensor_start, ctx.sensor_width);
        return;
      case ApplyOp::Substep2:
        mj_step1(m, d);
        mj_step2(m, d);
        self->CopyOutputs(d, i);
        return;
      case ApplyOp::RefreshSensor:
        mj_kinematics(m, d);
        mj_comPos(m, d);
        mj_comVel(m, d);
        mj_sensorPos(m, d);
        mj_sensorVel(m, d);
        for (size_t j = 0; j < ctx.sensor_ranges->size(); j += 2) {
          self->CopySensorRange(*ctx.sensor, d, i, (*ctx.sensor_ranges)[j],
                                (*ctx.sensor_ranges)[j + 1] - (*ctx.sensor_ranges)[j]);
        }
        return;
      case ApplyOp::Hfield: {
        mj_kinematics(m, d);
        QueryCtx row = ctx.query;
        row.out += static_cast<size_t>(row.selection[i]) * row.npoint;
        SampleHfield(m, d, row);
        return;
      }
      case ApplyOp::Query:
        (*ctx.fn)(m, d, i);
        return;
    }
  }

  int ApplyLocked(const std::vector<int>& ids, ApplyCtx& ctx, bool save = false) {
    ctx.self = this;
    return mjb_apply(batch_, ids.data(), static_cast<int>(ids.size()), ApplyFunc, &ctx, save);
  }

  void RunApply(std::optional<std::vector<int>> sel, ApplyCtx& ctx) {
    ReentryGuard();
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    auto selected = Selected(sel);
    CheckSleep(selected);
    Prepare(selected);
    ApplyLocked(selected, ctx);
    ThrowFailures(selected);
  }

  void Query(const std::optional<Ids>& ids, const QueryFn& fn) {
    auto sel = Parse(ids);
    ApplyCtx ctx;
    ctx.op = ApplyOp::Query;
    ctx.fn = &fn;
    RunApply(std::move(sel), ctx);
  }

  void CopySensorRange(Slot& s, mjData* d, int i, int start, int width) {
    const mjtNum* src = d->sensordata + start;
    mjtNum* raw = reinterpret_cast<mjtNum*>(s.raw + i * s.native_row) + start;
    std::memcpy(raw, src, static_cast<size_t>(width) * sizeof(mjtNum));
    if (s.buf) {
      if (s.f32 && sizeof(mjtNum) != sizeof(float)) {
        auto* dst = reinterpret_cast<float*>(s.buf.get() + i * s.row) + start;
        for (int k = 0; k < width; ++k) dst[k] = static_cast<float>(src[k]);
      } else {
        std::memcpy(s.buf.get() + i * s.row + static_cast<size_t>(start) * sizeof(mjtNum), src,
                    static_cast<size_t>(width) * sizeof(mjtNum));
      }
    }
  }

  void CopyOutputs(mjData* d, int i) {
    for (auto& s : bound_) {
      if (s->offset < 0 && s->raw && std::string_view(s->info->name) != "warning") {
        std::memcpy(s->raw + i * s->native_row, s->info->get(d), s->native_row);
      }
    }
  }

  void RunCallback(std::optional<std::vector<int>> sel, int nstep, mjtNum* hist,
                   nb::callable callback, int sensor_start, int sensor_width) {
    {
      nb::gil_scoped_release release;
      mu_.lock();
    }
    std::unique_lock<std::mutex> lock(mu_, std::adopt_lock);
    Slot& ctrl = BoundOrAdd(Field(data_fields_, "ctrl"), false);
    Slot* sensor = nullptr;
    nb::object sensor_view;
    if (sensor_width) sensor = &BoundOrAdd(Field(data_fields_, "sensordata"), false);
    nb::object state = nb::cast(StateView());
    nb::object ctrl_view = nb::cast(View(ctrl));
    if (sensor) {
      nb::object full = nb::cast(View(*sensor));
      sensor_view = full.attr("__getitem__")(
          nb::make_tuple(nb::slice(nb::none(), nb::none(), nb::none()),
                         nb::slice(sensor_start, sensor_start + sensor_width)));
    }
    auto selected = Selected(sel);
    CheckSleep(selected);
    if (sensor_width) CheckEuler(selected);
    auto active = selected;
    auto rows = SelectionRows(selected);
    std::vector<uint8_t> failed(num_sims_, 0);
    std::string error;
    std::exception_ptr pending;
    int dispatched = 0;
    for (int k = 0; k < nstep && !active.empty(); ++k) {
      if (sensor_width) {
        Prepare(active);
        ApplyCtx ctx;
        ctx.op = ApplyOp::Substep1;
        ctx.sensor = sensor;
        ctx.sensor_start = sensor_start;
        ctx.sensor_width = sensor_width;
        {
          nb::gil_scoped_release release;
          ApplyLocked(active, ctx);
        }
        ++dispatched;
        RecordFailures(active, failed, error);
        active = WithoutFailed(active, failed);
        if (!error.empty() || active.empty()) break;
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
      Prepare(active);
      {
        nb::gil_scoped_release release;
        if (sensor_width) {
          ApplyCtx ctx;
          ctx.op = ApplyOp::Substep2;
          ApplyLocked(active, ctx, true);
        } else {
          mjb_step(batch_, active.data(), static_cast<int>(active.size()), 1,
                   forward_ && k + 1 == nstep ? mjSTAGE_ACC : mjSTAGE_NONE);
        }
      }
      ++dispatched;
      RecordFailures(active, failed, error);
      if (hist) {
        for (int i : active) {
          if (!failed[i]) {
            std::memcpy(hist + (static_cast<size_t>(rows[i]) * nstep + k) * nstate_, State(i),
                        nstate_ * sizeof(mjtNum));
          }
        }
      }
      active = WithoutFailed(active, failed);
      if (!error.empty()) break;
    }
    if ((pending || !error.empty()) && dispatched && !active.empty()) {
      Prepare(active);
      {
        nb::gil_scoped_release release;
        mjb_forward(batch_, active.data(), static_cast<int>(active.size()), mjSTAGE_ACC);
      }
      RecordFailures(active, failed, error);
    }
    Finish(selected, failed);
    if (pending) std::rethrow_exception(pending);
    if (!error.empty()) throw std::runtime_error(error);
  }

  void CheckSleep(const std::vector<int>& ids) {
    if (!enableflags_) return;
    for (int i : ids) {
      int flags;
      std::memcpy(&flags, enableflags_->raw + i * enableflags_->native_row, sizeof(flags));
      if (flags & mjENBL_SLEEP) {
        throw nb::value_error(("sim " + std::to_string(i) + ": sleep is not supported").c_str());
      }
    }
  }

  void CheckEuler(const std::vector<int>& ids) {
    const Slot* integrator = nullptr;
    for (auto& s : expanded_) {
      if (std::string_view(s->info->name) == "integrator") {
        integrator = s.get();
        break;
      }
    }
    for (int i : ids) {
      int value = template_->opt.integrator;
      if (integrator) {
        std::memcpy(&value, integrator->raw + i * integrator->native_row, sizeof(value));
      }
      if (value != mjINT_EULER) {
        throw nb::value_error(
            ("sim " + std::to_string(i) + ": substep_sensor_copyout requires the Euler integrator")
                .c_str());
      }
    }
  }

#ifdef __linux__
  void PinWorkers(const std::vector<int>& cpus) {
    PinCtx ctx;
    ctx.expected = mjb_nthread(batch_);
    ctx.generation = ++pin_generation;
    ctx.caller = std::this_thread::get_id();
    ctx.cpus = &cpus;
    {
      nb::gil_scoped_release release;
      mjb_apply(batch_, nullptr, num_sims_, PinWorker, &ctx, 0);
    }
    if (ctx.error) {
      throw nb::value_error(("failed to pin a worker thread to the requested cpu_ids (error " +
                             std::to_string(ctx.error.load()) + ")")
                                .c_str());
    }
  }
#endif

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
        out[j] = static_cast<int>(static_cast<const int32_t*>(ids->data())[j]);
      } else {
        throw nb::value_error("ids must be int32, int64 or a bool mask");
      }
      if (out[j] < 0 || out[j] >= num_sims_ || (j > 0 && out[j] <= out[j - 1])) {
        throw nb::value_error("ids must be sorted, unique and in range");
      }
    }
    return out;
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

  mjtNum* OutPtr(const nb::ndarray<>& a, int64_t rows, int64_t r, int64_t c, const char* name) {
    bool ok3 = a.ndim() == 3 && static_cast<int64_t>(a.shape(0)) == rows &&
               static_cast<int64_t>(a.shape(1)) == r && static_cast<int64_t>(a.shape(2)) == c;
    bool ok2 = r == 1 && a.ndim() == 2 && static_cast<int64_t>(a.shape(0)) == rows &&
               static_cast<int64_t>(a.shape(1)) == c;
    if (!ok3 && !ok2) {
      std::string msg = std::string(name) + " must have shape (sel, " + std::to_string(r) + ", " +
                        std::to_string(c) + ")";
      throw nb::value_error(msg.c_str());
    }
    if (a.dtype() != nb::dtype<mjtNum>()) {
      throw nb::value_error((std::string(name) + " must be " + DtypeName(Elem::Num)).c_str());
    }
    int64_t row = r * c;
    int64_t s2 = a.stride(a.ndim() - 1), s1 = a.ndim() == 3 ? a.stride(1) : c, s0 = a.stride(0);
    if (a.device_type() != nb::device::cpu::value || s2 != 1 || s1 != c || s0 != row) {
      throw nb::value_error((std::string(name) + " must be a C-contiguous CPU array").c_str());
    }
    return static_cast<mjtNum*>(a.data());
  }

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

  static Elem RealOf(const nb::ndarray<>& a) {
    return a.dtype() == nb::dtype<float>() && sizeof(mjtNum) != sizeof(float) ? Elem::Float
                                                                              : Elem::Num;
  }

  static void* Array(const nb::ndarray<>& a, const char* name, Elem elem,
                     std::initializer_list<size_t> shape) {
    const auto dtype = elem == Elem::Int     ? nb::dtype<int>()
                       : elem == Elem::Float ? nb::dtype<float>()
                                             : nb::dtype<mjtNum>();
    bool ok = a.ndim() == shape.size() && a.dtype() == dtype &&
              a.device_type() == nb::device::cpu::value && IsCContig(a);
    std::string want;
    size_t k = 0;
    for (size_t extent : shape) {
      ok = ok && a.shape(k++) == extent;
      want += (want.empty() ? "" : ", ") + std::to_string(extent);
    }
    if (!ok) {
      throw nb::value_error((std::string(name) + " must be a C-contiguous CPU " + DtypeName(elem) +
                             " array of shape (" + want + ")")
                                .c_str());
    }
    return a.data();
  }

  static mjtNum Get(const void* a, bool f32, size_t k) {
    return f32 ? static_cast<const float*>(a)[k] : static_cast<const mjtNum*>(a)[k];
  }

  static void Put(void* a, bool f32, size_t k, mjtNum v) {
    if (f32) {
      static_cast<float*>(a)[k] = static_cast<float>(v);
    } else {
      static_cast<mjtNum*>(a)[k] = v;
    }
  }

  int num_sims_;
  int num_threads_;
  bool forward_;
  mjModel* template_ = nullptr;
  mjBatch* batch_ = nullptr;
  FieldTable data_fields_;
  FieldTable model_fields_;
  mjtNum* states_ = nullptr;
  int nstate_;
  std::vector<mjtNum> written_;
  std::vector<std::unique_ptr<Slot>> bound_;
  std::vector<std::unique_ptr<Slot>> expanded_;
  const Slot* enableflags_ = nullptr;
  const Slot* rbound_ = nullptr;  // geom_rbound, once the caller expands it
  std::vector<int> mesh_geoms_;   // whose bounds mj_setConst leaves to their mesh
  std::vector<mjtNum> bounds_;    // CopyBounds: theirs, for the sims of the current pass
  std::vector<OutField> out_fields_;
  size_t out_bytes_ = 0;
  std::mutex mu_;
};
