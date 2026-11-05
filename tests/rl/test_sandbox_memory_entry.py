"""Memory configuration installs the shared supervisor for ordinary entries."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.memory import SandboxMemory
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import DetachedShellEntry, EntryCapture, ShellEntry
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, StageRecord


class TestMemoryEntries:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("detached", [False, True])
    async def test_config_installs_supervisor_and_collects_oom_without_task_launcher(self, detached: bool) -> None:
        uploads = {}
        commands = []
        evidence = {"return_code": 137, "timed_out": False, "cancelled": False, "event_delta": {"oom_kill": 1}}

        async def upload(path: str, content: bytes) -> dict:
            uploads[path] = content
            return {"ok": True}

        async def execute(command: str, *args: object, **kwargs: object) -> dict:
            commands.append(command)
            if "mkdtemp" in command:
                return {"return_code": 0, "stdout": "/opt/xtuner-memory/entry-unique\n"}
            return {"return_code": 0, "pid": 123} if detached else {"return_code": 137}

        async def download(path: str) -> bytes:
            if path.endswith("report.json"):
                return json.dumps(evidence).encode()
            return b"137" if path.endswith(".rc") else b""

        client = SimpleNamespace(
            execute=execute, upload_bytes=upload, download_file=download, is_pid_running=AsyncMock(return_value=False)
        )
        kwargs = {"memory": {"cwd": "/tmp", "control_reserve_bytes": 1024}, "env": {"TASK_VALUE": "foo"}}
        entry = (
            DetachedShellEntry("exit 137", capture=EntryCapture(), **kwargs)
            if detached
            else ShellEntry("exit 137", **kwargs)
        )
        item = AgentRolloutItem(id="generic", data_source="test", instruction="test", task_root=Path("/tmp"), uid=1)
        record = StageRecord()
        outcome = await entry.run(client, item, record)
        config = json.loads(uploads["/opt/xtuner-memory/entry-unique/config.json"])
        assert config["env"]["TASK_VALUE"] == "foo"
        assert set(config["env"]) == {"PATH", "HOME", "TASK_VALUE"}
        assert config["uid"] == config["gid"] == 1000
        assert config["command"] == ["/bin/bash", "-c", "exit 137"]
        assert "supervisor.py" in commands[-1]
        assert outcome.reason == "memory_limit"
        assert item.artifacts["sandbox_resources"][record.entries[0].id] == evidence

    @pytest.mark.asyncio
    async def test_failed_setup_never_launches_command(self) -> None:
        client = SimpleNamespace(execute=AsyncMock(return_value={"return_code": 1}), upload_bytes=AsyncMock())
        with pytest.raises(RuntimeError, match="trusted"):
            await SandboxMemory().prepare(client, command="touch /tmp/started", env={}, timeout=1, entry_id="id")
        assert client.execute.await_count == 1
        client.upload_bytes.assert_not_awaited()

    def test_invalid_identity_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SandboxMemory(uid=0)

    @pytest.mark.asyncio
    async def test_guard_timeout_is_distinct_from_unexplained_exit_124(self) -> None:
        from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import EntryOutcome, EntryRecord, StageResult

        for timed_out in (False, True):
            evidence = {"return_code": 124, "timed_out": timed_out, "cancelled": False, "event_delta": {"oom_kill": 0}}
            client = SimpleNamespace(download_file=AsyncMock(return_value=json.dumps(evidence).encode()))
            item = SimpleNamespace(artifacts={})
            entry = EntryRecord(id="entry", name="work", cmd="run", mode="sync")
            outcome = EntryOutcome(source="sync_exec", reason="entry_failed", result=StageResult(return_code=124))
            result = await SandboxMemory().classify(client, item, StageRecord(), entry, outcome, "/opt/report")
            assert result.reason == ("timeout" if timed_out else "entry_failed")
