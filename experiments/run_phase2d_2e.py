"""Phase 2D + 2E only — isolated to avoid OOM."""
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

from memory_algebra import EMPTY, Query, compose, compose_all, make_memory, orthogonal_concepts, retrieve

DIM = 16
THETA = 0.85
SCALE_NS = [100, 500, 1000, 5000, 10000]


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


MEM_LIMIT_BYTES = 4 * 1024**3


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


def phase2d_scale():
    results = {}
    for n in SCALE_NS:
        queue = mp.Queue()
        p = mp.Process(target=_scale_worker, args=(n, queue))
        p.start()
        p.join()
        if not queue.empty():
            results[n] = queue.get()
            r = results[n]
            print(f"  N={n}: nodes={r['n_nodes']}, compose={r['compose_time']:.3f}s, retrieve={r['retrieve_time']:.3f}s", flush=True)
        else:
            results[n] = {"n_nodes": None, "compose_time": None, "retrieve_time": None, "oom": True}
            print(f"  N={n}: OOM (exceeded {MEM_LIMIT_BYTES // 1024**3}GB limit)", flush=True)
        del queue, p
        gc.collect()
    return results


def phase2e_qa():
    rng = np.random.default_rng(42)
    concepts = orthogonal_concepts(DIM, DIM, rng)
    brasilia, brasil, lula, argentina = concepts[0], concepts[1], concepts[2], concepts[3]
    facts = [
        ("Brasilia_capital_Brasil", brasilia, brasil, 1.0),
        ("Lula_presidente_Brasil", lula, brasil, 2.0),
        ("Brasil_fronteira_Argentina", brasil, argentina, 3.0),
    ]
    M = EMPTY
    for name, subj, obj, t in facts:
        m = make_memory(name, {"subj": (subj, t, 1.0), "obj": (obj, t, 1.0)}, [("subj", "obj", "rel")])
        M = compose(M, m)
    q = Query(vector=lula, time=2.5, time_scale=2.0, top_k=3, hops=2)
    r = retrieve(M, q, 0.85)
    nodes = list(r.memory.nodes)
    has_brasil = any("Brasil" in n for n in nodes)
    has_argentina = any("Argentina" in n for n in nodes)
    return {
        "n_retrieved": len(nodes),
        "has_brasil": has_brasil,
        "has_argentina": has_argentina,
        "multi_hop_ok": has_brasil and has_argentina,
    }


def main():
    out = Path(__file__).resolve().parent / "results_phase2.json"
    data = json.loads(out.read_text()) if out.exists() else {}

    print("Phase 2D: Scale benchmark (isolated)...", flush=True)
    data["phase2d_scale"] = phase2d_scale()

    print("Phase 2E: QA multi-hop...", flush=True)
    data["phase2e_qa"] = phase2e_qa()
    print(f"  Retrieved {data['phase2e_qa']['n_retrieved']} nodes, multi_hop_ok={data['phase2e_qa']['multi_hop_ok']}", flush=True)

    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nResults merged into {out}", flush=True)


if __name__ == "__main__":
    main()
