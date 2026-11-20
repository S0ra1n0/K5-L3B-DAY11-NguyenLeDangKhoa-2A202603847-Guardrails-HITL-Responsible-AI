"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline).

    Một request = một entry, ghép từ ``record_input`` (lúc nhận) và
    ``record_output`` (lúc trả lời), nối với nhau bằng ``request_id``.
    """

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}       # request_id -> perf_counter lúc bắt đầu
        self._pending: dict[str, dict] = {}     # request_id -> phần input của entry

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or f"user:{user_id}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp. Returns the request_id used."""
        request_id = request_id or uuid.uuid4().hex[:12]
        key = self._key(user_id, request_id)
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        key = self._key(user_id, request_id)
        start = self._open.pop(key, None)
        entry = self._pending.pop(key, None) or {
            # record_output không có record_input tương ứng → vẫn ghi, đánh dấu thiếu
            "request_id": request_id,
            "user_id": user_id,
            "input": None,
            "started_at": None,
        }
        entry.update(
            {
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "finished_at": utc_now_iso(),
                "latency_ms": (
                    round((time.perf_counter() - start) * 1000, 1)
                    if start is not None
                    else None
                ),
            }
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
