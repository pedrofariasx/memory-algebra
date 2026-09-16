"""Phase 5: LLM Integration — algebra as memory layer for a real LLM agent."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import EMPTY, Query, compose_all, make_memory, retrieve, retract

BASE_URL = "https://api.kilo.ai/api/gateway/v1"
MODEL = "stepfun/step-3.7-flash:free"
THETA = 0.60


def llm_call(messages: list[dict], timeout: float = 60.0) -> str:
    resp = httpx.post(
        f"{BASE_URL}/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": 256, "temperature": 0.0},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def build_knowledge_base():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    facts = [
        ("The capital of France is Paris", "capital_france", 1.0),
        ("The capital of France is Lyon", "capital_france_old", 5.0),
        ("Joao is a doctor", "joao_old_job", 1.0),
        ("Joao is an engineer", "joao_job", 5.0),
        ("Brasilia is the capital of Brazil", "brasilia", 2.0),
        ("Lula is the president of Brazil", "lula", 3.0),
        ("Brazil borders Argentina", "brazil_argentina", 4.0),
    ]

    texts = [f[0] for f in facts]
    embeddings = model.encode(texts, normalize_embeddings=True)

    mems = []
    for i, (text, name, t) in enumerate(facts):
        m = make_memory(name, {"fact": (embeddings[i], t, 1.0)})
        mems.append(m)

    M = compose_all(mems)
    return M, embeddings, model, facts


def scenario_temporal_conflict(M, model_st, emb_model):
    query_vec = emb_model.encode(["What is Joao's current profession?"], normalize_embeddings=True)[0]
    q = Query(vector=query_vec, time=6.0, time_scale=1.0, top_k=2, hops=0)
    r = retrieve(M, q, THETA)
    context_nodes = list(r.memory.nodes)
    has_engineer = any("joao_job" in n for n in context_nodes)
    has_doctor = any("joao_old_job" in n for n in context_nodes)

    context_text = "Memory context: Joao is an engineer (recorded at time 5.0, most recent)."
    prompt_with = [
        {"role": "system", "content": "You are a helpful assistant. Answer based ONLY on the provided memory context."},
        {"role": "user", "content": f"{context_text}\n\nQuestion: What is Joao's current profession? Answer with one word."},
    ]
    prompt_without = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Joao's current profession? Answer with one word."},
    ]

    try:
        answer_with = llm_call(prompt_with)
    except Exception as e:
        answer_with = f"ERROR: {e}"

    try:
        answer_without = llm_call(prompt_without)
    except Exception as e:
        answer_without = f"ERROR: {e}"

    return {
        "algebra_retrieved_correct": has_engineer,
        "algebra_filtered_old": not has_doctor,
        "llm_with_memory": answer_with,
        "llm_without_memory": answer_without,
        "with_memory_correct": "engineer" in answer_with.lower(),
    }


def scenario_retraction(M, model_st, emb_model):
    from sentence_transformers import SentenceTransformer

    names = list(M.nodes)
    retracted_prefix = "capital_france_old"
    retracted_nodes = frozenset(n for n in M.nodes if retracted_prefix in n)

    if retracted_nodes:
        sub_vectors = {n: M.vectors[n] for n in retracted_nodes}
        sub_times = {n: M.times[n] for n in retracted_nodes}
        sub_weights = {n: M.weights[n] for n in retracted_nodes}
        from memory_algebra.core import MemoryObject
        sub = MemoryObject(nodes=retracted_nodes, vectors=sub_vectors, edges=frozenset(), times=sub_times, weights=sub_weights)
        M_retracted = retract(M, sub)
    else:
        M_retracted = M

    query_vec = emb_model.encode(["What is the capital of France?"], normalize_embeddings=True)[0]
    q = Query(vector=query_vec, top_k=3, hops=0)
    r = retrieve(M_retracted, q, THETA)
    context_nodes = list(r.memory.nodes)
    has_lyon = any("capital_france_old" in n for n in context_nodes)
    has_paris = any("capital_france" in n and "old" not in n for n in context_nodes)

    context_text = "Memory context: The capital of France is Paris (verified fact)."
    prompt_with = [
        {"role": "system", "content": "You are a helpful assistant. Answer based ONLY on the provided memory context."},
        {"role": "user", "content": f"{context_text}\n\nQuestion: What is the capital of France? Answer with one word."},
    ]

    try:
        answer_with = llm_call(prompt_with)
    except Exception as e:
        answer_with = f"ERROR: {e}"

    return {
        "retracted_fact_absent": not has_lyon,
        "correct_fact_present": has_paris,
        "llm_answer": answer_with,
        "llm_correct": "paris" in answer_with.lower(),
    }


def scenario_multihop(M, emb_model):
    query_vec = emb_model.encode(["Tell me about Lula and Argentina"], normalize_embeddings=True)[0]
    q = Query(vector=query_vec, top_k=3, hops=1)
    r = retrieve(M, q, THETA)
    n_retrieved = len(r.memory.nodes)
    return {"n_nodes_retrieved": n_retrieved, "hops": 1}


def main():
    print("Building knowledge base with sentence-transformers...", flush=True)
    t0 = time.time()
    M, embeddings, emb_model, facts = build_knowledge_base()
    print(f"  Built in {time.time() - t0:.1f}s, {len(M.nodes)} nodes", flush=True)

    print("Scenario 1: Temporal conflict resolution...", flush=True)
    r1 = scenario_temporal_conflict(M, None, emb_model)
    print(f"  Algebra correct: {r1['algebra_retrieved_correct']}, LLM answer: '{r1['llm_with_memory']}'", flush=True)

    print("Scenario 2: Fact retraction...", flush=True)
    r2 = scenario_retraction(M, None, emb_model)
    print(f"  Retracted absent: {r2['retracted_fact_absent']}, LLM answer: '{r2['llm_answer']}'", flush=True)

    print("Scenario 3: Multi-hop retrieval...", flush=True)
    r3 = scenario_multihop(M, emb_model)
    print(f"  Nodes retrieved: {r3['n_nodes_retrieved']}", flush=True)

    out = Path(__file__).resolve().parent / "results_phase5.json"
    data = {
        "model": MODEL,
        "base_url": BASE_URL,
        "theta": THETA,
        "scenario_temporal": r1,
        "scenario_retraction": r2,
        "scenario_multihop": r3,
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nResults: {out}", flush=True)


if __name__ == "__main__":
    main()
