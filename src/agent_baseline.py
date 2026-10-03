"""Agent A: full history within a thread, no persistent profile."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import answer_from_facts, estimate_tokens, extract_profile_updates, is_recall_request
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt = [{"role": "system", "content": "Trả lời ngắn gọn bằng tiếng Việt. Chỉ dùng thông tin có trong thread."}] + session.messages
        session.prompt_tokens_processed += sum(estimate_tokens(item["content"]) for item in prompt)
        answer = str(self.langchain_agent.invoke(prompt).content)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(message) + estimate_tokens(answer)
        return {"answer": answer, "thread_id": thread_id, "token_usage": session.token_usage,
                "prompt_tokens_processed": session.prompt_tokens_processed}

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += sum(estimate_tokens(item["content"]) for item in session.messages)
        facts = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        answer = answer_from_facts(message, facts) if is_recall_request(message) else "Mình đã ghi nhận trong cuộc trò chuyện này."
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(message) + estimate_tokens(answer)
        return {"answer": answer, "thread_id": thread_id, "token_usage": session.token_usage,
                "prompt_tokens_processed": session.prompt_tokens_processed}

    def _maybe_build_langchain_agent(self):
        if self.force_offline or os.getenv("LIVE_MODE") != "1":
            return None
        return build_chat_model(self.config.model)
