# SPDX-License-Identifier: Apache-2.0

"""Measure batched forward stepping and gate the future adjoint comparison.

Run with a custom MuJoCo package built from the adjoint branch:
  python benchmarks/adjoint_scaling.py scene.xml --threads 1,8,16,32 --num-sims 256

The adjoint ratio remains NOT_RUN until STEP_VJP is advertised. --require-step-vjp
turns that condition into a hard failure for R3 acceptance runs.
"""

import argparse
import json
import os
import time

import mujoco
import numpy as np

from mjbatch import Batch
from mjbatch._bindings import adjoint_info


def measure_forward(model: mujoco.MjModel, num_sims: int, threads: int, steps: int, keyframe: int) -> dict:
  batch = Batch(model, num_sims, num_threads=threads)
  batch.reset(keyframe=keyframe)
  batch.bind("qpos")[:, 2] += 2.0
  rng = np.random.default_rng(0)
  batch.bind("ctrl")[:] = 0.3 * rng.standard_normal((num_sims, model.nu))
  for _ in range(20):
    batch.step()
  start = time.perf_counter()
  for _ in range(steps):
    batch.step()
  elapsed = time.perf_counter() - start
  sim_steps = num_sims * steps
  return {
    "requested_threads": threads,
    "actual_threads": batch.num_threads,
    "elapsed_seconds": elapsed,
    "sim_steps_per_second": sim_steps / elapsed,
    "per_sim_step_latency_us": 1e6 * elapsed / sim_steps,
  }


def measure_step_vjp(model: mujoco.MjModel, num_sims: int, threads: int, steps: int, keyframe: int) -> dict:
  batch = Batch(model, num_sims, num_threads=threads)
  batch.reset(keyframe=keyframe)
  batch.bind("qpos")[:, 2] += 2.0
  rng = np.random.default_rng(1)
  batch.bind("ctrl")[:] = 0.3 * rng.standard_normal((num_sims, model.nu))
  bar_qpos = rng.standard_normal((num_sims, model.nv))
  bar_qvel = rng.standard_normal((num_sims, model.nv))
  grad_qpos = np.empty_like(bar_qpos)
  grad_qvel = np.empty_like(bar_qvel)
  grad_ctrl = np.empty((num_sims, model.nu))
  status = np.empty(num_sims, dtype=np.int32)
  for _ in range(20):
    batch.step_vjp(bar_qpos, bar_qvel, grad_qpos, grad_qvel, grad_ctrl, status)
  if np.any(status):
    raise RuntimeError(f"step_vjp failed during warmup: {status.tolist()}")
  start = time.perf_counter()
  for _ in range(steps):
    batch.step_vjp(bar_qpos, bar_qvel, grad_qpos, grad_qvel, grad_ctrl, status)
  elapsed = time.perf_counter() - start
  if np.any(status):
    raise RuntimeError(f"step_vjp failed during measurement: {status.tolist()}")
  sim_steps = num_sims * steps
  return {
    "adjoint_elapsed_seconds": elapsed,
    "adjoint_sim_steps_per_second": sim_steps / elapsed,
    "adjoint_per_sim_step_latency_us": 1e6 * elapsed / sim_steps,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("--num-sims", type=int, default=256)
  parser.add_argument("--threads", default="1,8,16,32")
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--keyframe", type=int, default=-1)
  parser.add_argument("--output")
  parser.add_argument("--require-step-vjp", action="store_true")
  args = parser.parse_args()
  if args.num_sims < 1 or args.steps < 1:
    parser.error("--num-sims and --steps must be positive")
  threads = [int(value) for value in args.threads.split(",")]
  if not threads or any(value < 1 for value in threads):
    parser.error("--threads must contain positive integers")
  info = adjoint_info()
  if not info["available"]:
    parser.error("adjoint ABI unavailable; build against the custom MuJoCo package")
  model = mujoco.MjModel.from_xml_path(args.model)
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
  step_vjp = bool(info["capabilities"] & info["step_vjp_capability"])
  results = []
  for count in threads:
    row = measure_forward(model, args.num_sims, count, args.steps, args.keyframe)
    if step_vjp:
      adjoint = measure_step_vjp(model, args.num_sims, count, args.steps, args.keyframe)
      row.update(adjoint)
      row["adjoint_over_forward"] = adjoint["adjoint_per_sim_step_latency_us"] / row["per_sim_step_latency_us"]
      row["adjoint_status"] = "MEASURED"
    else:
      row["adjoint_over_forward"] = None
      row["adjoint_status"] = "NOT_RUN"
    results.append(row)
  if args.require_step_vjp and not step_vjp:
    parser.error("STEP_VJP capability is not advertised; R3 was not measured")
  report = {
    "model": os.path.basename(args.model),
    "num_sims": args.num_sims,
    "steps": args.steps,
    "abi": info,
    "step_vjp_status": "AVAILABLE" if step_vjp else "UNSUPPORTED",
    "results": results,
  }
  encoded = json.dumps(report, indent=2)
  print(encoded)
  if args.output:
    with open(args.output, "w", encoding="utf-8") as stream:
      stream.write(encoded + "\n")


if __name__ == "__main__":
  main()
