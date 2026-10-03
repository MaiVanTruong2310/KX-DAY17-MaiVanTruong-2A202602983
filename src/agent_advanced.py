"""Agent B: bounded thread context plus a cross-session User.md."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (CompactMemoryManager, UserProfileStore, answer_from_facts,
                          estimate_tokens, extract_profile_updates, is_recall_request)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(self.config.compact_threshold_tokens,
                                                    self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        self._remember(user_id, thread_id, message)
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.compact_memory.context(thread_id)
        system = "Trả lời ngắn gọn bằng tiếng Việt. Hồ sơ người dùng:\n" + self.profile_store.read_text(user_id)
        if context["summary"]:
            system += "\nTóm tắt hội thoại cũ: " + str(context["summary"])
        prompt = [{"role": "system", "content": system}] + list(context["messages"])
        answer = str(self.langchain_agent.invoke(prompt).content)
        return self._finish(thread_id, message, answer)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, thread_id: str, message: str) -> None:
        for key, value in extract_profile_updates(message).items():
            if key == "response_style" and "3 bullet" in self.profile_store.facts(user_id).get(key, "") and "3 bullet" not in value:
                continue
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)

    def _finish(self, thread_id: str, message: str, answer: str) -> dict[str, Any]:
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + estimate_tokens(message) + estimate_tokens(answer)
        return {"answer": answer, "thread_id": thread_id, "token_usage": self.token_usage(thread_id),
                "prompt_tokens_processed": self.prompt_token_usage(thread_id),
                "compactions": self.compaction_count(thread_id)}

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._remember(user_id, thread_id, message)
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        return self._finish(thread_id, message, answer)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        return (estimate_tokens(self.profile_store.read_text(user_id)) +
                estimate_tokens(str(context["summary"])) +
                sum(estimate_tokens(item["content"]) for item in context["messages"]))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        if not is_recall_request(message):
            return "Mình đã ghi nhận thông tin cần nhớ."
        return answer_from_facts(message, self.profile_store.facts(user_id))

    def _maybe_build_langchain_agent(self):
        if self.force_offline or os.getenv("LIVE_MODE") != "1":
            return None
        return build_chat_model(self.config.model)
