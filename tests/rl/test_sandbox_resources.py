from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.resource_guard import (
    MIB,
    AgentMemoryGuard,
    agent_memory_budget,
    memory_limit_exceeded,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.resource_policy import MemoryFailurePolicy
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import EntryOutcome, EntryRecord, StageRecord, StageResult


def cgroup(root: Path) -> None:
    for name, value in {
        "memory.max": str(2048 * MIB),
        "cgroup.controllers": "memory cpu",
        "cgroup.subtree_control": "",
    }.items():
        (root / name).write_text(value)
    child = root / "generic-agent"
    child.mkdir(mode=0o700)
    (child / "cgroup.procs").write_text("")
    (child / "memory.events").write_text("oom 0\noom_kill 0\n")


class TestMemoryGuard:
    def test_configurable_budget_and_cgroup_are_independent_of_recipe(self, tmp_path: Path) -> None:
        cgroup(tmp_path)
        report = tmp_path / "task-resources.json"
        report.write_text('{"return_code":137,"event_delta":{"oom_kill":1}}')
        guard = AgentMemoryGuard(
            tmp_path, cgroup_name="generic-agent", control_reserve_bytes=768 * MIB, report_path=report
        )
        assert not report.exists()
        assert (guard.limit, guard.reserve) == (1280 * MIB, 768 * MIB)
        assert (guard.path / "memory.max").read_text() == str(1280 * MIB)
        assert (guard.path / "memory.oom.group").read_text() == "0"
        guard.enter_child()
        assert (guard.path / "cgroup.procs").read_text().isdigit()
        (guard.path / "memory.events").write_text("oom 1\noom_kill 1\n")
        evidence = guard.report(return_code=137)
        assert memory_limit_exceeded(evidence)
        assert json.loads(report.read_text()) == evidence
        assert report.stat().st_mode & 0o077 == 0

    @pytest.mark.parametrize("reserve", [True, 0, -1, 2048 * MIB, 1900 * MIB])
    def test_invalid_reserve_cannot_remove_control_or_workload_budget(self, reserve: int) -> None:
        with pytest.raises(ValueError):
            agent_memory_budget(2048 * MIB, reserve)

    @pytest.mark.parametrize("name", ["", "..", "/bad", "../bad", "nested/group"])
    def test_child_name_cannot_escape_delegated_root(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError):
            AgentMemoryGuard(tmp_path, cgroup_name=name)

    def test_missing_controller_fails_before_launch(self, tmp_path: Path) -> None:
        cgroup(tmp_path)
        (tmp_path / "cgroup.controllers").write_text("cpu")
        with pytest.raises(RuntimeError, match="delegate"):
            AgentMemoryGuard(tmp_path, cgroup_name="generic-agent")

    def test_live_domain_cannot_be_reconfigured(self, tmp_path: Path) -> None:
        cgroup(tmp_path)
        (tmp_path / "generic-agent/cgroup.procs").write_text("123")
        with pytest.raises(RuntimeError, match="live processes"):
            AgentMemoryGuard(tmp_path, cgroup_name="generic-agent")

    @pytest.mark.parametrize(
        "changes",
        [
            {"return_code": 0},
            {"return_code": 124},
            {"timed_out": True},
            {"cancelled": True},
            {"event_delta": {"oom_kill": 0}},
            {"event_delta": {"oom_kill": True}},
            {"event_delta": {"oom_kill": "1"}},
            {"timed_out": None},
        ],
    )
    def test_unconfirmed_or_other_termination_is_not_oom(self, changes: dict) -> None:
        evidence = {"return_code": 137, "timed_out": False, "cancelled": False, "event_delta": {"oom_kill": 1}}
        evidence.update(changes)
        assert not memory_limit_exceeded(evidence)


class TestMemoryFailurePolicy:
    @pytest.mark.asyncio
    async def test_generic_entry_classifies_confirmed_oom(self) -> None:
        evidence = {"return_code": 137, "timed_out": False, "cancelled": False, "event_delta": {"oom_kill": 1}}
        client = SimpleNamespace(download_file=AsyncMock(return_value=json.dumps(evidence).encode()))
        item = SimpleNamespace(artifacts={})
        entry = EntryRecord(id="generic", name="agent", cmd="run", mode="sync")
        outcome = EntryOutcome(source="sync_exec", result=StageResult(return_code=137))
        policy = MemoryFailurePolicy("/opt/trusted/report.json")
        result = await policy.handle(client, item, StageRecord(), entry, outcome)
        assert result.reason == "memory_limit"
        assert item.artifacts["sandbox_resources"]["generic"] == evidence

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reported_rc", [None, [], {}, True, "137"])
    async def test_malformed_exit_evidence_does_not_replace_original_failure(self, reported_rc: object) -> None:
        evidence = {"return_code": reported_rc, "timed_out": False, "cancelled": False, "event_delta": {"oom_kill": 1}}
        client = SimpleNamespace(download_file=AsyncMock(return_value=json.dumps(evidence).encode()))
        entry = EntryRecord(id="generic", name="agent", cmd="run", mode="sync")
        outcome = EntryOutcome(source="sync_exec", reason="nonzero_exit", result=StageResult(return_code=137))
        result = await MemoryFailurePolicy("/opt/trusted/report.json").handle(
            client, SimpleNamespace(artifacts={}), StageRecord(), entry, outcome
        )
        assert result.reason == "nonzero_exit"

    @pytest.mark.asyncio
    async def test_missing_report_preserves_monitor_failure(self) -> None:
        client = SimpleNamespace(download_file=AsyncMock(side_effect=RuntimeError("secret URL")))
        entry = EntryRecord(id="generic", name="agent", cmd="run", mode="sync")
        outcome = EntryOutcome(source="detach_exec", reason="pid_monitor_error", result=StageResult(return_code=None))
        result = await MemoryFailurePolicy("/opt/trusted/report.json").handle(
            client, SimpleNamespace(artifacts={}), StageRecord(), entry, outcome
        )
        assert result.reason == "pid_monitor_error"
        assert entry.diagnostic_errors[0]["error"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_monitor_or_timeout_keeps_its_cause_even_with_stale_oom_report(self) -> None:
        evidence = {"return_code": 137, "timed_out": False, "cancelled": False, "event_delta": {"oom_kill": 1}}
        client = SimpleNamespace(download_file=AsyncMock(return_value=json.dumps(evidence).encode()))
        for reason in ("timeout", "pid_monitor_error", "sandbox_unavailable", "rc_missing"):
            entry = EntryRecord(id="generic", name="agent", cmd="run", mode="sync")
            outcome = EntryOutcome(source="detach_exec", reason=reason, result=StageResult(return_code=137))
            result = await MemoryFailurePolicy("/opt/trusted/report.json").handle(
                client, SimpleNamespace(artifacts={}), StageRecord(), entry, outcome
            )
            assert result.reason == reason
