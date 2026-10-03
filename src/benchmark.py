"""Reproducible offline comparison on the supplied Vietnamese datasets."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    return sum(term.casefold() in answer.casefold() for term in expected) / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline proxy: factual coverage and a short, nonempty response."""
    if not answer.strip() or "chưa có đủ thông tin" in answer.casefold():
        return 0.0
    coverage = recall_points(answer, expected)
    brevity = 1.0 if len(answer) <= 300 else 0.5
    return 0.8 * coverage + 0.2 * brevity


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    thread_ids: list[str] = []
    user_ids = {item["user_id"] for item in conversations}
    initial_bytes = sum(agent.memory_file_size(uid) for uid in user_ids) if isinstance(agent, AdvancedAgent) else 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    for conversation in conversations:
        uid = conversation["user_id"]
        thread = f"{conversation['id']}-dialogue"
        thread_ids.append(thread)
        for turn in conversation["turns"]:
            agent.reply(uid, thread, turn)
        for index, item in enumerate(conversation["recall_questions"]):
            fresh_thread = f"{conversation['id']}-recall-{index}"
            thread_ids.append(fresh_thread)
            answer = agent.reply(uid, fresh_thread, item["question"])["answer"]
            recall_scores.append(recall_points(answer, item["expected_contains"]))
            quality_scores.append(heuristic_quality(answer, item["expected_contains"]))
    final_bytes = sum(agent.memory_file_size(uid) for uid in user_ids) if isinstance(agent, AdvancedAgent) else 0
    return BenchmarkRow(agent_name,
                        sum(agent.token_usage(t) for t in thread_ids),
                        sum(agent.prompt_token_usage(t) for t in thread_ids),
                        sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
                        sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                        final_bytes - initial_bytes,
                        sum(agent.compaction_count(t) for t in thread_ids))


def format_rows(rows: list[BenchmarkRow]) -> str:
    header = "| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |"
    divider = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, divider]
    for row in rows:
        lines.append(f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | "
                     f"{row.recall_score:.1%} | {row.response_quality:.1%} | "
                     f"{row.memory_growth_bytes} | {row.compactions} |")
    return "\n".join(lines)


def main() -> None:
    base = load_config(Path(__file__).resolve().parent.parent)
    for title, filename in [("Standard Benchmark", "conversations.json"),
                            ("Long-Context Stress Benchmark", "advanced_long_context.json")]:
        conversations = load_conversations(base.data_dir / filename)
        with tempfile.TemporaryDirectory(prefix="day17-benchmark-", dir=base.state_dir) as directory:
            config = replace(base, state_dir=Path(directory))
            rows = [run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config),
                    run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config)]
        print(f"\n## {title}\n")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
