# SPDX-License-Identifier: Apache-2.0

"""Per-call cost of set_const(ids), the recompute of a reset that randomized model fields.

uv run python benchmarks/set_const.py
uv run python benchmarks/set_const.py path/to/scene.xml --num-sims 4096 --threads 0

Without a model, a built-in chain of 31 bodies is used, so no assets are needed. For each
input below, rows of it are written and set_const is called with their ids. The time is the
mean of the calls after the first, which is the only one to expand a derived field and then
computes it for every simulation. geom_friction is read by no constant, so little is left of
its calls; body_mass and body_ipos are read by all of them, as in a reset that randomizes
friction and the center of mass.

--check fails the run on a steady-state call that faults a page: set_const used to rebuild a
batch of every simulation on each call, which cost seconds and gigabytes instead.
"""

import argparse
import os
import sys
import time

import mujoco
import numpy as np

from mjbatch import Batch

try:
  import resource
except ImportError:  # Windows
  resource = None

INPUTS = ("geom_friction", "body_mass", "body_ipos")


def chain_xml(links=29):
  body = (
    '<body name="b{i}" pos="0 0 -0.1"><joint type="hinge" axis="{a}"/>'
    '<geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.03"/>'
    '<geom type="sphere" pos="0 0.03 -0.05" size="0.03"/>'
  )
  opening = [body.format(i=i, a=["1 0 0", "0 1 0", "0 0 1"][i % 3]) for i in range(links)]
  return (
    '<mujoco><worldbody><geom name="floor" type="plane" size="0 0 0.05"/>'
    '<body name="root" pos="0 0 1"><freejoint/>'
    '<geom type="capsule" fromto="0 0 -0.2 0 0 0.2" size="0.08"/>'
    + "".join(opening)
    + "</body>" * links
    + "</body></worldbody></mujoco>"
  )


def step_ms(batch):
  for _ in range(5):
    batch.step()
  t0 = time.perf_counter()
  for _ in range(20):
    batch.step()
  return (time.perf_counter() - t0) / 20 * 1e3


def usage():
  """Minor page faults so far, and peak RSS in bytes (ru_maxrss is in bytes on macOS only)."""
  if resource is None:
    return 0, 0.0
  r = resource.getrusage(resource.RUSAGE_SELF)
  return r.ru_minflt, r.ru_maxrss * (1.0 if sys.platform == "darwin" else 1024.0)


def write(view, ids, name, k):
  # every value differs from the template's, so the first call is the one that expands a field
  if name == "geom_friction":
    view[ids, :, 0] = 0.5 + 0.01 * k
  elif name == "body_mass":
    view[ids, 1] = 1.0 + 0.01 * k
  else:
    view[ids, 1] = [0.01 * (k + 1), 0, 0]


def measure(model, name, num_sims, threads, rows_list, calls):
  """Per row count, the first call's time, then the mean time and faults of the steady ones."""
  batch = Batch(model, num_sims, num_threads=threads)
  view = batch.expand(name)
  step_before = step_ms(batch)
  _, rss0 = usage()
  rows = []
  for count in rows_list:
    ids = np.arange(count, dtype=np.int32)
    walls, faults = [], []
    first = 0.0
    for k in range(calls + 1):
      write(view, ids, name, k)
      faults0, _ = usage()
      t0 = time.perf_counter()
      batch.set_const(ids)
      wall = time.perf_counter() - t0
      count_faults, rss = usage()
      if k:
        walls.append(wall)
        faults.append(count_faults - faults0)
      else:
        first = wall
    rows.append((count, first, sum(walls) / len(walls), sum(faults) / len(faults)))
  return batch.num_threads, step_before, rows, (rss - rss0) / 2**30, step_ms(batch)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("model", nargs="?", help="a scene XML; default: a built-in chain")
  p.add_argument("--num-sims", type=int, default=4096)
  p.add_argument("--threads", type=int, default=0, help="0: one per logical CPU")
  p.add_argument("--rows", help="row counts, default: 1,64,1024 and every simulation")
  p.add_argument("--calls", type=int, default=5, help="measured calls per row count")
  p.add_argument("--check", action="store_true", help="fail on a steady-state page fault")
  a = p.parse_args()
  rows_list = [int(r) for r in a.rows.split(",")] if a.rows else [1, 64, 1024, a.num_sims]
  model = mujoco.MjModel.from_xml_path(a.model) if a.model else mujoco.MjModel.from_xml_string(chain_xml())
  print(
    f"{os.path.basename(a.model) if a.model else 'chain'}: {model.nbody} bodies / "
    f"{model.ngeom} geoms / {model.nv} dof  num_sims={a.num_sims}"
  )
  worst = 0
  for name in INPUTS:
    threads, step_before, rows, rss, step_after = measure(
      model, name, a.num_sims, a.threads, rows_list, a.calls
    )
    print(f"  {name}  threads={threads}  step() {step_before:.2f} -> {step_after:.2f} ms")
    print(f"    {'rows':>6}  {'first call':>10}  {'per call':>10}  {'faults/call':>11}")
    for count, first, wall, faults in rows:
      print(f"    {count:6d}  {first * 1e3:8.3f} ms  {wall * 1e3:8.3f} ms  {faults:11.0f}")
    print(f"    peak RSS growth {rss:.2f} GB", flush=True)
    worst = max(worst, *(faults for _, _, _, faults in rows))
  if a.check and worst >= 100:
    sys.exit(f"set_const faulted {worst:.0f} pages per steady-state call")


if __name__ == "__main__":
  main()
