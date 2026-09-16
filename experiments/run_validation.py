from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import (
    Query,
    compose,
    cosine,
    lens_get,
    lens_put,
    make_memory,
    orthogonal_concepts,
    quotient,
    retrieve,
    semantic_distance,
    signature,
)
from memory_algebra.baselines import LSTMCell
from memory_algebra.builder import perturb
from memory_algebra.core import EMPTY

DIM = 16
THETA = 0.85
SEED = 7


def noisy(v: np.ndarray, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    w = np.asarray(v, dtype=float) + scale * rng.standard_normal(len(v))
    return w / np.linalg.norm(w)


def fold_left(mems):
    M = EMPTY
    for m in mems:
        M = compose(M, m)
    return M


def fold_right(mems):
    M = EMPTY
    for m in reversed(mems):
        M = compose(m, M)
    return M


def fold_parallel(mems, block: int = 10):
    blocks = [mems[i : i + block] for i in range(0, len(mems), block)]
    return fold_left([fold_left(b) for b in blocks])


def gen_memories(rng, concepts, n):
    mems = []
    for i in range(n):
        k = int(rng.integers(1, 4))
        idx = rng.integers(0, len(concepts), size=k)
        nodes = {}
        for j, ci in enumerate(idx):
            nodes[f"n{j}"] = (
                noisy(concepts[ci], rng),
                float(i) + float(rng.uniform(0.0, 0.5)),
                float(rng.uniform(0.2, 1.0)),
            )
        edges = [(f"n{j}", f"n{j + 1}", "rel") for j in range(k - 1)]
        mems.append(make_memory(f"m{i}", nodes, edges))
    return mems


def protocolo_3_1(rng, concepts) -> dict:
    mems = gen_memories(rng, concepts, 100)
    left = fold_left(mems)
    right = fold_right(mems)
    parallel = fold_parallel(mems)
    max_d_lr = 0.0
    max_d_lp = 0.0
    for m in mems:
        n0 = sorted(m.nodes)[0]
        q = Query(vector=m.vectors[n0], time=m.times[n0], time_scale=5.0, top_k=1, hops=0)
        rl = retrieve(left, q, THETA).memory
        rr = retrieve(right, q, THETA).memory
        rp = retrieve(parallel, q, THETA).memory
        max_d_lr = max(max_d_lr, semantic_distance(rl, rr))
        max_d_lp = max(max_d_lp, semantic_distance(rl, rp))
    return {
        "n_memorias": len(mems),
        "assinaturas_iguais": signature(left) == signature(right) == signature(parallel),
        "max_distancia_esquerda_direita": max_d_lr,
        "max_distancia_esquerda_paralelo": max_d_lp,
        "sucesso": signature(left) == signature(parallel) and max_d_lr < 1e-9 and max_d_lp < 1e-9,
    }


def protocolo_3_2(rng, concepts) -> dict:
    n = 50
    v1 = concepts[0]
    m1 = make_memory("m0", {"a": (v1, 0.0, 1.0)})
    memories = [m1]
    pool = concepts[1:]
    for i in range(1, n):
        c1 = pool[i % len(pool)]
        c2 = pool[(i + 3) % len(pool)]
        memories.append(
            make_memory(
                f"m{i}",
                {
                    "x": (noisy(c1, rng), float(i), float(rng.uniform(0.3, 0.8))),
                    "y": (noisy(c2, rng), float(i), float(rng.uniform(0.3, 0.8))),
                },
                [("x", "y", "rel")],
            )
        )
    q1 = Query(v1, time=0.0, time_scale=100.0, top_k=1, hops=0)
    M = EMPTY
    vec_sum = np.zeros_like(v1)
    buffer: list = []
    lstm = LSTMCell(dim=len(v1), hidden=len(v1), seed=11)
    curve_algebra, curve_soma, curve_fifo, curve_lstm = [], [], [], []
    for t, m in enumerate(memories):
        M = compose(M, m)
        buffer.append(m)
        if len(buffer) > 10:
            buffer.pop(0)
        first = sorted(m.nodes)[0]
        x = np.asarray(m.vectors[first], dtype=float) / np.linalg.norm(m.vectors[first])
        vec_sum = vec_sum + x
        h = lstm.step(x)
        curve_algebra.append(semantic_distance(m1, retrieve(M, q1, THETA).memory))
        curve_soma.append(1.0 - cosine(vec_sum, v1))
        curve_lstm.append(1.0 - cosine(h, v1))
        fifo_nodes = set().union(*(b.nodes for b in buffer)) if buffer else set()
        curve_fifo.append(0.0 if m1.nodes <= fifo_nodes else 1.0)
    return {
        "n_composicoes": n,
        "algebra_final": curve_algebra[-1],
        "algebra_max": max(curve_algebra),
        "soma_vetorial_final": curve_soma[-1],
        "lstm_final": curve_lstm[-1],
        "fifo_final": curve_fifo[-1],
        "curva_algebra": [round(x, 6) for x in curve_algebra[::5]],
        "curva_soma_vetorial": [round(x, 6) for x in curve_soma[::5]],
        "curva_lstm": [round(x, 6) for x in curve_lstm[::5]],
        "curva_fifo": [round(x, 6) for x in curve_fifo[::5]],
        "sucesso": (
            max(curve_algebra) < 1e-9
            and curve_soma[-1] > 0.5
            and curve_fifo[-1] == 1.0
            and curve_lstm[-1] > 0.2
        ),
    }


def protocolo_3_3(rng, concepts) -> dict:
    e_joao, e_med, e_eng = concepts[0], concepts[1], concepts[2]
    prof = (e_med + e_eng) / np.linalg.norm(e_med + e_eng)
    mA = make_memory(
        "mA",
        {"joao": (e_joao, 1.0, 1.0), "medico": (e_med, 1.0, 1.0)},
        [("joao", "medico", "profissao")],
    )
    mB = make_memory(
        "mB",
        {"joao": (perturb(e_joao, rng, 0.05), 2.0, 1.0), "engenheiro": (e_eng, 2.0, 1.0)},
        [("joao", "engenheiro", "profissao")],
    )
    M = compose(mA, mB)
    quot = quotient(M, THETA)
    fundidas = any("mA:joao" in c and "mB:joao" in c for c in quot.clusters)
    historico_preservado = "mA:medico" in M.nodes and "mB:engenheiro" in M.nodes
    qvec = (e_joao + prof) / np.linalg.norm(e_joao + prof)
    r_now = retrieve(M, Query(qvec, time=2.0, time_scale=0.5, top_k=4, hops=0), THETA)
    r_past = retrieve(M, Query(qvec, time=1.0, time_scale=0.5, top_k=4, hops=0), THETA)

    def pos(r, node):
        for i, c in enumerate(r.focus):
            if node in c:
                return i
        return None

    atual_correto = pos(r_now, "mB:engenheiro") < pos(r_now, "mA:medico")
    passado_correto = pos(r_past, "mA:medico") < pos(r_past, "mB:engenheiro")
    return {
        "entidades_fundidas": fundidas,
        "historico_preservado": historico_preservado,
        "consulta_atual_correta": atual_correto,
        "consulta_passado_correta": passado_correto,
        "sucesso": fundidas and historico_preservado and atual_correto and passado_correto,
    }


def protocolo_3_4(rng, concepts) -> dict:
    eA, eB, eC, eD = concepts[0], concepts[1], concepts[2], concepts[3]
    m1 = make_memory("m1", {"a": (eA, 0.0, 1.0), "b": (eB, 0.0, 1.0)}, [("a", "b", "rel")])
    m2 = make_memory(
        "m2",
        {"b": (perturb(eB, rng, 0.02), 0.0, 1.0), "c": (eC, 0.0, 1.0)},
        [("b", "c", "rel")],
    )
    m3 = make_memory(
        "m3",
        {"c": (perturb(eC, rng, 0.02), 0.0, 1.0), "d": (eD, 0.0, 1.0)},
        [("c", "d", "rel")],
    )
    M = fold_left([m1, m2, m3])
    quot = quotient(M, THETA)
    g = nx.Graph()
    g.add_nodes_from(quot.clusters)
    for cu, cv, _ in quot.edges:
        g.add_edge(cu, cv)
    cl_a = quot.member_cluster["m1:a"]
    cl_d = quot.member_cluster["m3:d"]
    caminho = nx.has_path(g, cl_a, cl_d)
    aresta_direta = g.has_edge(cl_a, cl_d)
    r = retrieve(M, Query(eA, top_k=1, hops=3), THETA)
    alvo_recuperado = any(cosine(r.memory.vectors[n], eD) > 0.95 for n in r.memory.nodes)
    return {
        "caminho_transitivo_existe": caminho,
        "aresta_direta_AD_existe": aresta_direta,
        "alvo_recuperado": alvo_recuperado,
        "sucesso": caminho and not aresta_direta and alvo_recuperado,
    }


def leis_lentes(concepts) -> dict:
    mA = make_memory(
        "mA",
        {"x": (concepts[0], 0.0, 1.0), "y": (concepts[1], 0.0, 1.0)},
        [("x", "y", "e")],
    )
    mB = make_memory("mB", {"z": (concepts[2], 1.0, 1.0)})
    M = compose(mA, mB)
    delta = make_memory("delta", {"w": (concepts[3], 5.0, 1.0)})
    M2 = lens_put(M, Query(concepts[0], top_k=1, hops=1), delta, THETA)
    r = lens_get(M2, Query(concepts[3], time=5.0, time_scale=0.5, top_k=1, hops=0), THETA)
    put_get = semantic_distance(delta, r.memory)
    q = Query(concepts[2], top_k=1, hops=0)
    view = lens_get(M, q, THETA).memory
    M3 = lens_put(M, q, view, THETA)
    get_put = signature(M3) == signature(M)
    return {
        "distancia_put_get": put_get,
        "get_put_idempotente": get_put,
        "sucesso": put_get < 1e-9 and get_put,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    concepts = orthogonal_concepts(DIM, DIM, rng)
    resultados = {
        "3_1_associatividade": protocolo_3_1(rng, concepts),
        "3_2_estabilidade": protocolo_3_2(rng, concepts),
        "3_3_conflitos": protocolo_3_3(rng, concepts),
        "3_4_expressividade": protocolo_3_4(rng, concepts),
        "leis_lentes": leis_lentes(concepts),
    }
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print("=" * 68)
    print("VALIDACAO EMPIRICA - ALGEBRA DA MEMORIA (V, G, T, P)")
    print("=" * 68)
    for nome, res in resultados.items():
        status = "OK" if res["sucesso"] else "FALHOU"
        print(f"[{status}] {nome}")
        for k, v in res.items():
            if k in ("sucesso", "curva_algebra", "curva_soma_vetorial", "curva_fifo"):
                continue
            print(f"    {k}: {v}")
    print(f"\nresultados completos: {out}")


if __name__ == "__main__":
    main()
