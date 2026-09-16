"""Phase 2D extended: push scale beyond N=10^4 with 6GB limit."""
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import EMPTY, Query, compose_all, make_memory, orthogonal_concepts, retrieve

DIM = 16
THETA = 0.85
SCALE_NS = [10000, 15000, 20000, 25000, 30000, 40000, 50000]
MEM_LIMIT_BYTES = 6 * 1024**3


def gen_sequence(rng, concepts, n):
    mems = []
    for i in range(n):
        k = int(rng.integers(1, 4))
        idx = rng.integers(0, len(concepts), size=k)
        nodes = {}
        for j, ci in enumerate(idx):
            nodes[f"n{j}"] = (
                concepts[ci].copy(),
                float(i) + float(rng.uniform(0.0, 0.5)),
                float(rng.uniform(0.2, 1.0)),
            )
        edges = [(f"n{j}", f"n{j + 1}", "rel") for j in range(k - 1)]
        mems.append(make_memory(f"m{i}", nodes, edges))
    return mems


def _scale_worker(n, queue):
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
    rng = np.random.default_rng(42)
    concepts = orthogonal_concepts(DIM, DIM, rng)
    mems = gen_sequence(rng, concepts, n)
    t0 = time.time()
    M = compose_all(mems)
    t_compose = time.time() - t0
    query = Query(vector=concepts[0], top_k=1, hops=0)
    t0 = time.time()
    retrieve(M, query, THETA)
    t_retrieve = time.time() - t0
    queue.put({"n_nodes": len(M.nodes), "compose_time": t_compose, "retrieve_time": t_retrieve})


def main():
    out = Path(__file__).resolve().parent / "results_phase2d_extended.json"
    results = {}
    for n in SCALE_NS:
        print(f"N={n}: ", end="", flush=True)
        queue = mp.Queue()
        p = mp.Process(target=_scale_worker, args=(n, queue))
        p.start()
        p.join(timeout=300)
        if p.is_alive():
            p.terminate()
            p.join()
            results[n] = {"timeout": True}
            print(f"TIMEOUT (>300s)", flush=True)
        elif not queue.empty():
            results[n] = queue.get()
            r = results[n]
            print(f"nodes={r['n_nodes']}, compose={r['compose_time']:.3f}s, retrieve={r['retrieve_time']:.3f}s", flush=True)
        else:
            results[n] = {"oom": True}
            print(f"OOM (>6GB)", flush=True)
        del queue, p
        gc.collect()
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}", flush=True)


if __name__ == "__main__":
    main()
