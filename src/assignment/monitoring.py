"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    # Số request bị chặn theo từng lớp (rate_limiter / input_guardrail / ...)
    blocked_by_layer: dict[str, int] = field(default_factory=dict)

    def record(self, *, blocked: bool, layer: str | None = None) -> None:
        """Cập nhật bộ đếm sau mỗi request (pipeline gọi hàm này)."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
            key = layer or "unknown"
            self.blocked_by_layer[key] = self.blocked_by_layer.get(key, 0) + 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1

    def record_judge(self, *, safe: bool) -> None:
        """Cập nhật bộ đếm LLM-as-Judge (optional)."""
        self.judge_checks += 1
        if not safe:
            self.judge_fails += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates, return Alerts for every threshold currently exceeded.

        Tính lại từ đầu mỗi lần gọi (không cộng dồn) → gọi nhiều lần không
        sinh alert trùng.
        """
        snap = self.snapshot()
        alerts: list[Alert] = []

        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            alerts.append(Alert(
                metric="block_rate",
                value=round(snap["block_rate"], 3),
                threshold=self.block_rate_threshold,
                message=(
                    f"Block rate {snap['block_rate']:.0%} > "
                    f"{self.block_rate_threshold:.0%}: có thể đang bị tấn công "
                    "hoặc guardrail chặn nhầm — cần người xem audit log."
                ),
            ))
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            alerts.append(Alert(
                metric="rate_limit_hits",
                value=self.rate_limit_hits,
                threshold=self.rate_limit_hit_threshold,
                message=(
                    f"{self.rate_limit_hits} lần chạm rate limit "
                    f"(ngưỡng {self.rate_limit_hit_threshold}): nghi spam / cost attack."
                ),
            ))
        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            alerts.append(Alert(
                metric="judge_fail_rate",
                value=round(snap["judge_fail_rate"], 3),
                threshold=self.judge_fail_rate_threshold,
                message=(
                    f"Judge fail rate {snap['judge_fail_rate']:.0%} > "
                    f"{self.judge_fail_rate_threshold:.0%}: model đang trả lời không an toàn."
                ),
            ))

        self.alerts = alerts
        return alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        data = self.snapshot()
        data["blocked_by_layer"] = dict(self.blocked_by_layer)
        data["thresholds"] = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
