"""Host-side resource failure classification for ordinary sandbox entries."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.resource_guard import memory_limit_exceeded
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import EntryFailurePolicy
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, EntryOutcome, EntryRecord, StageRecord


class MemoryFailurePolicy(EntryFailurePolicy):
    """Classify OOM from the trusted supervisor's report on failed shell entries.

    Works with both ShellEntry and DetachedShellEntry via their ``failure``
    configuration. This consumes evidence; the workload launcher must first
    use AgentMemoryGuard to enforce the limit and write a fresh report for each
    invocation. Never use a report path writable by the workload.

    Args:
        report_path (str): Absolute path of the supervisor-owned report.
        diagnostics (Any): Optional existing entry diagnostics configuration.
        diagnostic_error_policy (str): Existing entry diagnostics failure policy.
    """

    def __init__(
        self,
        report_path: str,
        *,
        diagnostics: Any = None,
        diagnostic_error_policy: str = "preserve_entry_error",
    ) -> None:
        if not report_path.startswith("/"):
            raise ValueError("Memory report path must be absolute")
        super().__init__(diagnostics=diagnostics, diagnostic_error_policy=diagnostic_error_policy)
        self.report_path = report_path

    async def handle(
        self,
        client: Any,
        item: AgentRolloutItem,
        record: StageRecord,
        entry: EntryRecord,
        outcome: EntryOutcome,
    ) -> EntryOutcome:
        outcome = await super().handle(client, item, record, entry, outcome)
        try:
            evidence = json.loads(await asyncio.wait_for(client.download_file(self.report_path), timeout=5.0))
            if not isinstance(evidence, dict):
                raise ValueError("Expected a resource report object")
        except Exception as exc:
            # Missing evidence must preserve the actual monitor/exit failure.
            entry.diagnostic_errors.append({"path": self.report_path, "error": type(exc).__name__})
            return outcome
        item.artifacts.setdefault("sandbox_resources", {})[entry.id] = evidence
        rc = outcome.result.return_code
        if (
            outcome.reason not in {"timeout", "pid_monitor_error", "sandbox_unavailable", "rc_missing"}
            and rc in {137, -9}
            and memory_limit_exceeded(evidence)
        ):
            outcome.reason = "memory_limit"
            outcome.result.error = "Workload killed by sandbox memory limit"
            outcome.details = {**(outcome.details or {}), "resource_report": self.report_path}
        return outcome
