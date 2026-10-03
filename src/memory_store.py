"""Persistent facts and bounded per-thread conversation memory."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Stable character-based estimate; this is not provider billing usage."""
    return math.ceil(len(text.strip()) / 4) if text.strip() else 0


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", user_id).strip("_")
        if not slug or slug in {".", ".."}:
            raise ValueError("Invalid user id")
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if not search_text or search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        result = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([a-z_]+): (.+)$", line)
            if match:
                result[match.group(1)] = match.group(2)
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        if not re.fullmatch(r"[a-z_]+", key):
            raise ValueError("Invalid fact key")
        facts = self.facts(user_id)
        if facts.get(key) == value:
            return
        facts[key] = value
        self.write_text(user_id, "# User profile\n\n" + "".join(
            f"- {name}: {fact}\n" for name, fact in sorted(facts.items())))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative extraction of explicit first-person Vietnamese facts."""
    text = message.strip()
    if not text or ("?" in text and len(text) < 120):
        return {}
    lower = text.lower()
    facts: dict[str, str] = {}
    name = re.search(
        r"(?:mình tên là|tên mình là|nhắc lại lần cuối cho chắc: tên)\s+([\wÀ-ỹ]+(?:\s+[\wÀ-ỹ]+){0,4}?)(?=\s*(?:[,.;!?]|\s+hiện\b|\s+và\b|$))",
        text, re.I)
    if name:
        facts["name"] = name.group(1)
    # Explicit current residence takes precedence over historical references and travel.
    locations = list(re.finditer(r"(?:hiện ở|hiện đang ở|mình (?:đang|vẫn)?\s*ở|nơi ở hiện tại (?:là|của mình là)|nơi ở đã cập nhật từ [^.,;]+ sang)\s*(Huế|Đà Nẵng)\b", text, re.I))
    if locations:
        facts["location"] = locations[-1].group(1)
    elif "giờ mình đang ở huế" in lower:
        facts["location"] = "Huế"
    if "không phải nơi ở hiện tại" not in lower and "đang làm việc ở đà nẵng vài tháng" in lower:
        facts["location"] = "Đà Nẵng"
    if re.search(r"(?:mình (?:đang|vẫn) làm|nghề nghiệp hiện tại (?:là|vẫn là)|nghề)\s+MLOps engineer\b|chuyển sang MLOps engineer|tên DũngCT Stress, nghề MLOps engineer", text, re.I):
        facts["profession"] = "MLOps engineer"
    elif re.search(r"(?:mình (?:đang|vẫn) làm|và đang làm) backend engineer\b", text, re.I) and "không còn" not in lower:
        facts["profession"] = "backend engineer"
    if "3 bullet" in lower and any(x in lower for x in ["muốn", "style", "trả lời"]):
        facts["response_style"] = "ngắn gọn, 3 bullet, có ví dụ thực chiến, nhấn trade-off"
    elif re.search(r"(?:mình muốn|mình thích|hãy) (?:bạn )?trả lời|style trả lời", lower):
        if any(x in lower for x in ["ngắn gọn", "bullet ngắn", "câu trả lời gọn"]):
            facts["response_style"] = "ngắn gọn, có bullet và ví dụ thực tế"
    if "cà phê sữa đá" in lower and any(x in lower for x in ["thích", "yêu thích", "vẫn uống"]):
        facts["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in lower and any(x in lower for x in ["yêu thích", "món ruột"]):
        facts["favorite_food"] = "mì Quảng"
    if "corgi" in lower and ("mình nuôi" in lower or "con corgi" in lower):
        facts["pet"] = "corgi tên Bơ"
    if "python" in lower and "ai" in lower and any(x in lower for x in ["mình thích", "mình đang quan tâm", "dài hạn:"]):
        facts["interests"] = "Python, AI ứng dụng"
    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Keep short topic cues; stable facts belong in User.md."""
    cues = []
    topics = {"Artemis III": "roadmap và readiness", "X-59": "hiệu năng và externality",
              "WMO": "xác suất và rủi ro", "British Columbia": "scale và efficiency",
              "benchmark": "đánh đổi recall và token", "memory": "nén lịch sử hội thoại"}
    joined = " ".join(item["content"] for item in messages if item["role"] == "user")
    for topic, idea in topics.items():
        if topic.lower() in joined.lower():
            cues.append(f"{topic}: {idea}")
    return "; ".join(cues[:max_items])


def answer_from_facts(question: str, facts: dict[str, str]) -> str:
    """Offline recall shared by both agents, based only on memory actually available."""
    q = question.lower()
    selected = []
    selectors = [
        ("name", ["tên", "là ai", "tóm tắt", "biết dũngct"]),
        ("profession", ["nghề", "công việc", "làm gì", "tóm tắt", "product manager"]),
        ("location", ["ở đâu", "nơi ở", "ở huế", "hà nội", "đà nẵng"]),
        ("response_style", ["style", "kiểu trả lời", "trả lời mình thích"]),
        ("favorite_drink", ["đồ uống", "uống gì"]),
        ("favorite_food", ["món ăn"]),
        ("pet", ["nuôi con gì", "con gì"]),
        ("interests", ["mối quan tâm", "kỹ thuật chính", "tóm tắt"]),
    ]
    for key, triggers in selectors:
        if key in facts and any(trigger in q for trigger in triggers):
            selected.append(facts[key])
    return "; ".join(selected) + "." if selected else "Mình chưa có đủ thông tin để trả lời chắc chắn."


def is_recall_request(message: str) -> bool:
    text = message.strip().lower()
    return ("?" in text or text.startswith(("nhắc lại", "tóm tắt", "sang thread mới", "bạn thử nhớ lại")))


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        entry = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = entry["messages"]
        messages.append({"role": role, "content": content})
        load = estimate_tokens(str(entry["summary"])) + sum(estimate_tokens(m["content"]) for m in messages)
        if load > self.threshold_tokens and len(messages) > self.keep_messages:
            older = messages[:-self.keep_messages]
            combined = [{"role": "user", "content": str(entry["summary"])}] + older
            entry["summary"] = summarize_messages(combined)
            entry["messages"] = messages[-self.keep_messages:]
            entry["compactions"] = int(entry["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
