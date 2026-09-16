"""Phase 2D DIM=384: no artificial memory limit, only physical RAM."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import Query, compose_all, make_memory, retrieve

DIM = 384
THETA = 0.85
SCALE_NS = [50000, 75000, 100000]


def gen_sequence(rng, n):
    mems = []
    for i in range(n):
        k = int(rng.integers(1, 4))
        nodes = {}
        for j in range(k):
            vec = rng.standard_normal(DIM)
            vec /= np.linalg.norm(vec)
            nodes[f"n{j}"] = (vec, float(i) + float(rng.uniform(0.0, 0.5)), float(rng.uniform(0.2, 1.0)))
        edges = [(f"n{j}", f"n{j + 1}", "rel") for j in range(k - 1)]
        mems.append(make_memory(f"m{i}", nodes, edges))
    return mems


def _scale_worker(n, queue):
    rng = np.random.default_rng(42)
    mems = gen_sequence(rng, n)
    t0 = time.time()
    M = compose_all(mems)
    t_compose = time.time() - t0
    qvec = rng.standard_normal(DIM)
    qvec /= np.linalg.norm(qvec)
    query = Query(vector=qvec, top_k=1, hops=0)
    t0 = time.time()
    retrieve(M, query, THETA)
    t_retrieve = time.time() - t0
    queue.put({"n_nodes": len(M.nodes), "compose_time": t_compose, "retrieve_time": t_retrieve})


def main():
    out = Path(__file__).resolve().parent / "results_phase2d_dim384_unlimited.json"
    results = {}
    for n in SCALE_NS:
        print(f"N={n}: ", end="", flush=True)
        queue = mp.Queue()
        p = mp.Process(target=_scale_worker, args=(n, queue))
        p.start()
        p.join(timeout=900)
        if p.is_alive():
            p.terminate()
            p.join()
            results[n] = {"timeout": True}
            print(f"TIMEOUT (>900s)", flush=True)
        elif not queue.empty():
            results[n] = queue.get()
            r = results[n]
            print(f"nodes={r['n_nodes']}, compose={r['compose_time']:.3f}s, retrieve={r['retrieve_time']:.3f}s", flush=True)
        else:
            results[n] = {"oom": True}
            print(f"OOM (physical RAM)", flush=True)
        out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}", flush=True)


if __name__ == "__main__":
    main()
