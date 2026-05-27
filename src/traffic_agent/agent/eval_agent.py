from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from traffic_agent.agent.planner import RuleBasedPlanner


def evaluate_agent_cases(cases_path: str | Path, output: str | Path) -> dict[str, float]:
    """Evaluate planner intent/tool/refusal behavior on YAML cases."""
    cases = yaml.safe_load(Path(cases_path).read_text(encoding="utf-8"))["cases"]
    planner = RuleBasedPlanner()
    intent_hits = 0
    tool_hits = 0
    refusal_hits = 0
    trace_saved_rate = 0.0
    for case in cases:
        plan = planner.plan(case["query"])
        intent_hits += int(plan.intent == case["expected_intent"])
        tool_hits += int(set(plan.selected_tools) == set(case["expected_tools"]))
        refusal_hits += int(bool(plan.refusal_reason) == bool(case["should_refuse"]))
    summary = {
        "intent_accuracy": intent_hits / len(cases),
        "tool_selection_accuracy": tool_hits / len(cases),
        "refusal_accuracy": refusal_hits / len(cases),
        "trace_saved_rate": trace_saved_rate,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic agent planner cases.")
    parser.add_argument("--cases", default="experiments/agent_eval_cases.yaml")
    parser.add_argument("--output", default="experiments/results/agent_eval_summary.json")
    args = parser.parse_args()
    print(json.dumps(evaluate_agent_cases(args.cases, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
