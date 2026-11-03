// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mujoco/mujoco.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

struct StepVjpCtx {
  const mjtNum* bar_qpos;
  const mjtNum* bar_qvel;
  mjtNum* grad_qpos;
  mjtNum* grad_qvel;
  mjtNum* grad_ctrl;
  int* status;
};

inline nb::dict AdjointInfo() {
  nb::dict info;
#ifdef MJBATCH_HAS_ADJOINT_HEADER
  int version = mjd_adjointVersion();
  info["available"] = version == mjADJOINT_ABI_VERSION;
  info["abi_version"] = version;
  info["capabilities"] = mjd_adjointCapabilities();
  info["expected_abi_version"] = mjADJOINT_ABI_VERSION;
  info["step_vjp_capability"] = mjADJOINT_CAP_STEP_VJP;
  info["backend"] = "adjoint";
#else
  info["available"] = false;
  info["abi_version"] = 0;
  info["capabilities"] = 0;
  info["expected_abi_version"] = 0;
  info["step_vjp_capability"] = 1;
  info["backend"] = "stock";
#endif
  return info;
}
