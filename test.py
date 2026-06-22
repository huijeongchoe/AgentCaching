"""
Observation: Demonstrating context duplication across LangGraph agents
on the DeepResearch Bench dataset.

Loads research tasks from DeepResearch Bench
(https://github.com/Ayanami0730/deep_research_bench), then runs 4 research
agents in a LangGraph workflow.  Each agent answers overlapping subsets of
the benchmark queries using web search and retrieval tools.  Every tool call
is logged as (agent_id, tool_name, args, result, timestamp).

Usage:
    python test.py [--num-queries 12] [--lang en]
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from agent import build_graph
from datasets.deep_research_bench import load_tasks
from logger import RUN_ID, LOG_FILE
from system_prompt import INITIAL_USER_MESSAGE


def main():
    parser = argparse.ArgumentParser(
        description="Observe tool-call duplication on DeepResearch Bench"
    )
    parser.add_argument("--num-queries", type=int, default=12,
                        help="Number of DRB tasks to sample")
    parser.add_argument("--lang", type=str, default="en",
                        choices=["en", "zh", None],
                        help="Filter by language (en/zh), or None for all")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env or environment before running.")
        return

    print(f"Run ID: {RUN_ID}")
    print(f"Log file: {LOG_FILE}")

    # 1. Load DeepResearch Bench tasks
    all_tasks = load_tasks(lang=args.lang)
    if not all_tasks:
        print("No tasks found. Check language filter.")
        return

    # 2. Sample tasks
    random.seed(42)
    sampled = random.sample(all_tasks, min(args.num_queries, len(all_tasks)))

    print(f"\nSampled {len(sampled)} tasks:")
    for i, t in enumerate(sampled):
        print(f"  [{i}] (id={t['id']}, {t['topic']}) {t['prompt'][:80]}...")

    # 3. Build and run the multi-agent graph
    print("\nBuilding 4-agent LangGraph workflow...")
    graph = build_graph(sampled)

    initial_state = {
        "messages": [{
            "role": "user",
            "content": INITIAL_USER_MESSAGE,
        }]
    }

    print("Running workflow...")
    start = time.time()
    graph.invoke(initial_state)
    elapsed = time.time() - start

    # 4. Summary
    with open(LOG_FILE) as f:
        call_count = sum(1 for _ in f)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total tool calls logged: {call_count}")
    print(f"Log: {LOG_FILE}")
    print(f"\nAnalyze: python analyze_duplication.py {LOG_FILE}")


if __name__ == "__main__":
    main()
