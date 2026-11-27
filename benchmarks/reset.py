# SPDX-License-Identifier: Apache-2.0

"""Per-call cost of reset(ids) against the size of the batch.

uv run python benchmarks/reset.py
uv run python benchmarks/reset.py --num-sims 4096 16384 --rows 1,64,1024

Without a model, the built-in chain of set_const.py is used, so no assets are needed. For each
batch size, rows are reset by the default reset(), which runs mj_forward, and by
reset(forward=False), for a caller that writes state afterwards and runs forward itself. The
time is the mean of the calls after the first, and page faults are counted per call: reset(ids)
used to copy the state of every simulation to find the pending writes of the selected rows, so
its cost grew with the batch rather than with the number of rows.

--check fails the run if a call for fewer than every simulation faults a page.
"""

import argparse
import os
import sys
import time

import mujoco
import numpy as np

from mjbatch import Batch
from set_const import chain_xml

try:
  import resource
except ImportError:  # Windows
  resource = None


def usage():
  """Minor page faults so far, and peak RSS in bytes (ru_maxrss is in bytes on macOS only)."""
  if resource is None:
    return 0, 0.0
  r = resource.getrusage(resource.RUSAGE_SELF)
  return r.ru_minflt, r.ru_maxrss * (1.0 if sys.platform == "darwin" else 1024.0)


def measure(model, num_sims, threads, rows, calls):
  """The mean time and page faults of a reset of `rows` rows, with and without the forward."""
  batch = Batch(model, num_sims, num_threads=threads)
  ctrl = batch.bind("ctrl")
  batch.bind("qpos")[:, 0] = np.linspace(0.5, 1.5, num_sims)  # a pending write on every row
  batch.step()
  ids = np.arange(rows, dtype=np.int32)
  result = []
  for forward in (True, False):
    faults, walls = [], []
    for k in range(calls + 1):
      ctrl[ids] = 0.1 * k  # a pending write on the rows about to be reset
      faults0, _ = usage()
      t0 = time.perf_counter()
      batch.reset(ids, forward=forward)
      walls.append(time.perf_counter() - t0)
      faults.append(usage()[0] - faults0)
    result.append((sum(walls[1:]) / calls * 1e3, sum(faults[1:]) / calls))
  t0 = time.perf_counter()
  batch.reset()
  every = (time.perf_counter() - t0) * 1e3
  return batch.num_threads, result, every


def main():
  p = argparse.ArgumentParser()
  p.add_argument("model", nargs="?", help="a scene XML; default: a built-in chain")
  p.add_argument("--num-sims", type=int, nargs="+", default=[4096])
  p.add_argument("--threads", type=int, default=0, help="0: one per logical CPU")
  p.add_argument("--rows", help="row counts, default: 1,64,1024")
  p.add_argument("--calls", type=int, default=100, help="measured calls per row count")
  p.add_argument("--check", action="store_true", help="fail on a page fault of a partial reset")
  a = p.parse_args()
  rows_list = [int(r) for r in a.rows.split(",")] if a.rows else [1, 64, 1024]
  model = mujoco.MjModel.from_xml_path(a.model) if a.model else mujoco.MjModel.from_xml_string(chain_xml())
  print(
    f"{os.path.basename(a.model) if a.model else 'chain'}: {model.nbody} bodies / "
    f"{model.ngeom} geoms / {model.nv} dof"
  )
  worst = 0
  for num_sims in a.num_sims:
    print(f"  num_sims={num_sims}")
    print(f"    {'rows':>6}  {'reset()':>10}  {'faults':>7}  {'forward=False':>13}  {'faults':>7}")
    for rows in rows_list + [num_sims]:
      threads, ((with_ms, with_faults), (without_ms, without_faults)), every = measure(
        model, num_sims, a.threads, rows, a.calls if rows < num_sims else 5
      )
      print(
        f"    {rows:6d}  {with_ms:8.3f} ms  {with_faults:7.0f}  {without_ms:10.3f} ms  {without_faults:7.0f}",
        flush=True,
      )
      if rows < num_sims:
        worst = max(worst, with_faults, without_faults)
    print(f"    threads={threads}  reset() of every simulation together: {every:.1f} ms", flush=True)
  if a.check and worst >= 100:
    sys.exit(f"reset faulted {worst:.0f} pages on a call for fewer than every simulation")


if __name__ == "__main__":
  main()
