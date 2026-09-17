"""Phase 5b: ReAct agent — LLM decides when to call algebraic memory tools."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import EMPTY, Query, compose, make_memory, retrieve, retract

BASE_URL = "https://api.kilo.ai/api/gateway/v1"
MODEL = "stepfun/step-3.7-flash:free"
THETA = 0.85
MAX_STEPS = 6


def llm_call(messages: list[dict], timeout: float = 90.0) -> str:
    resp = httpx.post(
        f"{BASE_URL}/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": 512, "temperature": 0.0},
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"].get("content")
    return content.strip() if content else ""


class MemoryBackend:
    def __init__(self, st_model, theta: float = THETA):
        self.st = st_model
        self.theta = theta
        self.M = EMPTY
        self.node_texts: dict[str, str] = {}
        self.fact_mems: dict[str, tuple[str, object]] = {}
        self.clock = 0.0

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray(self.st.encode([text], normalize_embeddings=True)[0], dtype=float)

    def store(self, fact: str) -> str:
        self.clock += 1.0
        prefix = f"f{len(self.node_texts)}"
        m = make_memory(prefix, {"fact": (self._embed(fact), self.clock, 1.0)})
        self.node_texts[f"{prefix}:fact"] = fact
        self.fact_mems[prefix] = (fact, m)
        self.M = compose(self.M, m)
        return f"Stored: {fact}"

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.M.nodes:
            return "Memory is empty."
        q = Query(vector=self._embed(query), time=self.clock, time_scale=2.0, top_k=top_k, hops=0)
        r = retrieve(self.M, q, self.theta)
        lines = []
        seen: set[str] = set()
        for cluster, score in sorted(r.scores.items(), key=lambda kv: -kv[1])[:top_k]:
            for node in cluster:
                if node in r.memory.nodes and node not in seen:
                    seen.add(node)
                    lines.append(f"[score={score:.2f}, t={self.M.times.get(node, 0.0):.0f}] {self.node_texts.get(node, node)}")
        return "\n".join(lines) if lines else "No relevant facts found."

    def retract_fact(self, query: str) -> str:
        if not self.M.nodes:
            return "Memory is empty."
        q = Query(vector=self._embed(query), top_k=1, hops=0)
        r = retrieve(self.M, q, self.theta)
        removed = []
        for prefix, (fact, m) in list(self.fact_mems.items()):
            if m.nodes & r.memory.nodes:
                self.M = retract(self.M, m)
                removed.append(fact)
                del self.fact_mems[prefix]
        return f"Retracted: {removed}" if removed else "No matching fact to retract."


REACT_TEMPLATE = """You are an agent with access to a factual memory. Answer the question using ONLY information retrieved from memory via the tools below.

Tools:
memory_search[query] - search memory for facts relevant to the query
memory_store[fact] - store a new fact in memory
memory_retract[query] - remove from memory the fact most relevant to the query

Use exactly this format:
Thought: <your reasoning>
Action: memory_search[some query]
Observation: <result, provided to you>
... (Thought/Action/Observation can repeat)
Thought: I now know the answer
Final Answer: <your answer>

IMPORTANT: Write exactly ONE Action per response and STOP immediately after the Action line. Never write the Observation yourself; it will be provided to you.

Question: {question}
"""

ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]+)\s*\[([^\]]*)\]")
FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def react_loop(backend: MemoryBackend, question: str) -> dict:
    trajectory = REACT_TEMPLATE.format(question=question)
    steps = 0
    tool_calls = []
    final = None
    while steps < MAX_STEPS:
        steps += 1
        out = llm_call([{"role": "user", "content": trajectory}])
        m_final = FINAL_RE.search(out)
        m_action = ACTION_RE.search(out)
        if m_final and (not m_action or m_final.start() < m_action.start()):
            final = m_final.group(1).strip()
            break
        if m_action:
            name, arg = m_action.group(1).lower(), m_action.group(2).strip()
            out = out[: m_action.end()]
            if name == "memory_search":
                obs = backend.search(arg)
            elif name == "memory_store":
                obs = backend.store(arg)
            elif name == "memory_retract":
                obs = backend.retract_fact(arg)
            else:
                obs = f"Unknown tool: {name}"
            tool_calls.append((name, arg, obs))
            trajectory += f"\n{out}\nObservation: {obs}\n"
            continue
        trajectory += f"\n{out}\nObservation: Respond with an Action or a Final Answer using the specified format.\n"
    if final is None:
        final = out if steps else ""
    return {"final_answer": final, "steps": steps, "tool_calls": tool_calls}


SCENARIOS = [
    {
        "name": "private_fact_recall",
        "setup": ["The access code for Vault-7 is ZEBRA-42"],
        "retract": [],
        "question": "What is the access code for Vault-7?",
        "check": lambda a: "zebra-42" in a.lower(),
    },
    {
        "name": "temporal_conflict",
        "setup": [
            "Alice Chen was appointed to lead the Orion initiative",
            "Marcus Webb now heads the Orion program",
        ],
        "retract": [],
        "question": "Who currently leads the Orion program?",
        "check": lambda a: "webb" in a.lower() and "chen" not in a.lower(),
    },
    {
        "name": "retraction",
        "setup": ["The Orion launch is scheduled for March 15"],
        "retract": ["Orion launch scheduled"],
        "question": "When is the Orion launch scheduled?",
        "check": lambda a: "march" not in a.lower(),
    },
    {
        "name": "multi_hop",
        "setup": [
            "The server room keycard is stored in drawer B-12",
            "Drawer B-12 is located on floor 3 of the Helix building",
        ],
        "retract": [],
        "question": "In which building and floor can I find the server room keycard?",
        "check": lambda a: "helix" in a.lower() or "floor 3" in a.lower() or "third floor" in a.lower(),
    },
]


def run_scenario(scenario: dict, st_model) -> dict:
    backend = MemoryBackend(st_model)
    for fact in scenario["setup"]:
        backend.store(fact)
    for q in scenario["retract"]:
        backend.retract_fact(q)

    react_result = react_loop(backend, scenario["question"])
    react_correct = bool(scenario["check"](react_result["final_answer"]))

    baseline_out = llm_call(
        [
            {
                "role": "user",
                "content": f"Answer concisely. If you do not know, say you do not know. Question: {scenario['question']}",
            }
        ]
    )
    baseline_correct = bool(scenario["check"](baseline_out))

    return {
        "name": scenario["name"],
        "question": scenario["question"],
        "react": {
            "answer": react_result["final_answer"],
            "correct": react_correct,
            "steps": react_result["steps"],
            "n_tool_calls": len(react_result["tool_calls"]),
            "tool_calls": [{"tool": t, "arg": a, "observation": o} for t, a, o in react_result["tool_calls"]],
        },
        "baseline": {"answer": baseline_out, "correct": baseline_correct},
    }


def main():
    import argparse

    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default="results_phase5_react.json")
    args = parser.parse_args()
    MODEL = args.model
    print(f"Using model: {MODEL}", flush=True)

    from sentence_transformers import SentenceTransformer

    print("Loading sentence-transformers model...", flush=True)
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = []
    for scenario in SCENARIOS:
        print(f"Scenario: {scenario['name']}...", flush=True)
        t0 = time.time()
        try:
            r = run_scenario(scenario, st_model)
            print(
                f"  react_correct={r['react']['correct']} (tools={r['react']['n_tool_calls']}, steps={r['react']['steps']}), "
                f"baseline_correct={r['baseline']['correct']}, {time.time() - t0:.1f}s",
                flush=True,
            )
            results.append(r)
        except httpx.HTTPStatusError as e:
            print(f"  ERROR: HTTP {e.response.status_code} — model unavailable, skipping", flush=True)
            raise
        except Exception as e:
            print(f"  ERROR: {e}, skipping scenario", flush=True)
            results.append({"name": scenario["name"], "error": str(e)})

    n = len(results)
    summary = {
        "model": MODEL,
        "theta": THETA,
        "react_accuracy": sum(r["react"]["correct"] for r in results) / n,
        "baseline_accuracy": sum(r["baseline"]["correct"] for r in results) / n,
        "avg_tool_calls": sum(r["react"]["n_tool_calls"] for r in results) / n,
        "scenarios": results,
    }

    out = Path(__file__).resolve().parent / args.out
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreact_accuracy={summary['react_accuracy']:.0%}, baseline_accuracy={summary['baseline_accuracy']:.0%}")
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
