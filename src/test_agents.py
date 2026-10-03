"""Behavioral checks for persistence, correction and compact prompt savings."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, extract_profile_updates


def make_config(tmp_path: Path):
    return replace(load_config(tmp_path), compact_threshold_tokens=90, compact_keep_messages=2)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("alice") == ""
    path = store.write_text("alice", "# User\n- location: Huế\n")
    assert path.exists() and store.file_size("alice") > 0
    assert store.edit_text("alice", "Huế", "Đà Nẵng")
    assert "Đà Nẵng" in store.read_text("alice")
    assert not store.edit_text("alice", "Huế", "Hà Nội")
    store.upsert_fact("alice", "profession", "MLOps engineer")
    assert store.facts("alice")["profession"] == "MLOps engineer"
    assert store.path_for("../alice").is_relative_to(store.root_dir)


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(8):
        agent.reply("u", "long", f"Lượt {index}: " + "bối cảnh kỹ thuật " * 25)
    context = agent.compact_memory.context("long")
    assert agent.compaction_count("long") >= 2
    assert len(context["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    fact = "Mình tên là DũngCT. Mình ở Đà Nẵng và đang làm backend engineer."
    advanced.reply("u", "old", fact)
    baseline.reply("u", "old", fact)
    correction = "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer. Giờ mình đang ở Huế."
    advanced.reply("u", "old", correction)
    baseline.reply("u", "old", correction)
    question = "Tên mình là gì, hiện tại mình làm nghề gì và đang ở đâu?"
    answer = advanced.reply("u", "new", question)["answer"]
    assert all(term in answer for term in ["DũngCT", "MLOps engineer", "Huế"])
    assert "backend engineer" not in answer
    assert "chưa có đủ thông tin" in baseline.reply("u", "new", question)["answer"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for index in range(14):
        turn = f"Thông tin dài lượt {index}. " + "Artemis III và X-59 cho ví dụ về trade-off. " * 20
        advanced.reply("u", "thread", turn)
        baseline.reply("u", "thread", turn)
    assert advanced.compaction_count("thread") > 0
    assert advanced.prompt_token_usage("thread") < baseline.prompt_token_usage("thread")


def test_noisy_facts_are_not_persisted(tmp_path: Path) -> None:
    assert extract_profile_updates("Hà Nội chỉ là nơi mình vừa bay ra họp, không phải nơi ở hiện tại.") == {}
    assert "profession" not in extract_profile_updates("Mình đùa là hay chuyển sang product manager.")
    assert extract_profile_updates("Bạn có biết DũngCT không?") == {}


def test_profile_survives_agent_restart(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    first = AdvancedAgent(config, force_offline=True)
    first.reply("same-user", "one", "Mình tên là DũngCT Stress, hiện ở Huế.")
    second = AdvancedAgent(config, force_offline=True)
    answer = second.reply("same-user", "two", "Mình tên gì và hiện đang ở đâu?")["answer"]
    assert "DũngCT Stress" in answer and "Huế" in answer
