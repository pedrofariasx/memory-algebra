"""Phase 5b multi-model comparison."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MODELS = [
    "dots-studio/dots-3-note-preview:free",
    "inclusionai/ling-3.0-flash-vl:free",
    "nex-agi/nex-n2.5-pro:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
]


def main():
    agent = Path(__file__).resolve().parent / "run_phase5_react_agent.py"
    combined = {}
    for model in MODELS:
        slug = model.replace("/", "_").replace(":", "_")
        out_name = f"results_phase5_react_{slug}.json"
        out_path = agent.parent / out_name
        print(f"=== {model} ===", flush=True)
        r = subprocess.run([sys.executable, str(agent), "--model", model, "--out", out_name])
        if r.returncode == 0 and out_path.exists():
            combined[model] = json.loads(out_path.read_text())
        else:
            combined[model] = {"error": f"exit={r.returncode}"}
        print(f"  exit={r.returncode}", flush=True)

    summary = {
        m: {
            "react_accuracy": d.get("react_accuracy"),
            "baseline_accuracy": d.get("baseline_accuracy"),
            "avg_tool_calls": d.get("avg_tool_calls"),
        }
        for m, d in combined.items()
    }
    out = agent.parent / "results_phase5_react_models.json"
    out.write_text(json.dumps({"summary": summary, "per_model": combined}, indent=2, ensure_ascii=False))
    print(f"\nResults: {out}", flush=True)
    for m, s in summary.items():
        print(f"  {m}: react={s['react_accuracy']}, baseline={s['baseline_accuracy']}", flush=True)


if __name__ == "__main__":
    main()
