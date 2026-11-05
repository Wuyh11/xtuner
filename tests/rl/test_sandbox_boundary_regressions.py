"""Regressions for bounded completion and cancellation after an earlier failure."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.finalization import CleanupError
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.judger import Judger
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.runner import Runner
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import (
    DetachedShellEntry,
    EntryCapture,
    SandboxPool,
    SandboxStage,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, SandboxSpec


class TestBoundaryRegressions:
    @pytest.mark.asyncio
    async def test_deadline_during_cleanup_replaces_terminal_reason_but_keeps_original_error(self) -> None:
        async def close() -> None:
            await asyncio.sleep(0.15)

        async def fail(*args: object) -> None:
            raise RuntimeError("original error")

        client = SimpleNamespace(health_check=AsyncMock(return_value={"ok": True}), aclose=close)
        provider = SimpleNamespace(create=AsyncMock(return_value=(client, "owned")), delete=AsyncMock())
        pool = SandboxPool(
            provider=provider,
            specs={"main": SandboxSpec(image="image")},
            creates_per_sec=None,
            cleanup_timeout_sec=1,
        )
        runner = Runner(
            pool=pool,
            infer=SandboxStage(pre=[fail]),
            validate=Judger("score", SandboxStage()),
            timeout_sec=0.05,
        )
        item = AgentRolloutItem(id="review", data_source="test", instruction="test", task_root=Path("/tmp"), uid=1)
        await runner.run(item)
        assert item.error.category == "trial_timeout"
        assert item.artifacts["error_before_timeout"]["category"] == "prehook"
        assert item.artifacts["sandbox_run_timing"]["deadline_expired"]
        assert item.artifacts["sandbox_cleanup"]["ok"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", ["running", "query_error", "query_timeout"])
    async def test_unverified_deletions_survive_failed_pass_until_confirmed(self, failure: str) -> None:
        queried = []
        state = failure

        async def create(image_tag: str, **kwargs: Any) -> tuple[Any, str]:
            return SimpleNamespace(health_check=AsyncMock(return_value={"ok": True}), aclose=AsyncMock()), image_tag

        async def query(env_id: str) -> dict[str, Any]:
            queried.append(env_id)
            if env_id == "b" and state == "query_error":
                raise RuntimeError("Lifecycle service unavailable")
            if env_id == "b" and state == "query_timeout":
                await asyncio.sleep(10)
            return {"env": {"status": "running" if env_id == "b" and state != "deleted" else "deleted"}}

        provider = SimpleNamespace(create=create, delete=AsyncMock(), get=query)
        pool = SandboxPool(
            provider=provider,
            specs={
                "main": SandboxSpec(image="c", dependencies={"a": SandboxSpec(image="a"), "b": SandboxSpec(image="b")})
            },
            creates_per_sec=None,
            cleanup_timeout_sec=0.3,
            verify_deletion=True,
        )
        await pool.get("main")
        with pytest.raises(CleanupError):
            await pool.release_all()
        assert queried == ["a", "b"]
        assert pool.cleanup["unverified_env_ids"] == ["b", "c"]
        assert not pool.cleanup["ok"] and not pool.cleanup["resources_deleted"]

        state = "running"
        queried.clear()
        with pytest.raises(CleanupError):
            await pool.release_all()
        assert queried == ["b"]
        assert pool.cleanup["unverified_env_ids"] == ["b", "c"]
        assert not pool.cleanup["ok"] and not pool.cleanup["resources_deleted"]

        state = "deleted"
        queried.clear()
        await pool.release_all()
        assert queried == ["b", "c"]
        assert pool.cleanup["ok"] and pool.cleanup["resources_deleted"]
        assert pool.cleanup["unverified_env_ids"] == []
        queried.clear()
        await pool.release_all()
        assert queried == []

    @pytest.mark.asyncio
    async def test_cancel_during_cleanup_replaces_terminal_reason_but_keeps_original_error(self) -> None:
        closing, finish = asyncio.Event(), asyncio.Event()

        class Client:
            async def health_check(self) -> dict:
                return {"ok": True}

            async def aclose(self) -> None:
                closing.set()
                await finish.wait()

        provider = SimpleNamespace(create=AsyncMock(return_value=(Client(), "owned")), delete=AsyncMock())

        async def fail(*args: object) -> None:
            raise RuntimeError("original error")

        pool = SandboxPool(provider=provider, specs={"main": SandboxSpec(image="image")}, creates_per_sec=None)
        runner = Runner(pool=pool, infer=SandboxStage(pre=[fail]), validate=Judger("score", SandboxStage()))
        item = AgentRolloutItem(id="review", data_source="test", instruction="test", task_root=Path("/tmp"), uid=1)
        task = asyncio.create_task(runner.run(item))
        await asyncio.wait_for(closing.wait(), 2)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert item.error.category == "rollout_cancelled"
        assert item.artifacts["error_before_cancellation"]["category"] == "prehook"
        assert item.artifacts["sandbox_cleanup"]["ok"]

    @pytest.mark.asyncio
    async def test_recovered_process_exit_does_not_restart_rc_or_health_budget(self) -> None:
        entry = DetachedShellEntry(
            "true",
            capture=EntryCapture(),
            timeout=1,
            pid_check_retry_interval_sec=0.001,
            pid_check_recovery_timeout_sec=0.1,
            rc_wait_timeout_sec=0.05,
        )
        reads = 0
        stopped = asyncio.Event()

        async def download(path: str) -> bytes:
            nonlocal reads
            reads += 1
            if reads == 1:
                raise httpx.ReadTimeout("transient")
            try:
                await asyncio.sleep(10)
                return b"0"
            finally:
                stopped.set()

        client = SimpleNamespace(
            is_pid_running=AsyncMock(side_effect=[httpx.ReadTimeout("transient"), False]),
            get_lifecycle_state=AsyncMock(return_value="ready"),
            download_file=download,
            health_check=AsyncMock(return_value={"ok": True}),
        )
        outcome = await asyncio.wait_for(
            entry._wait_with_pid_poll(
                client, SimpleNamespace(id="review"), SimpleNamespace(name="entry", rc_file="/tmp/rc"), 123
            ),
            0.5,
        )
        assert outcome.reason == "rc_missing"
        assert stopped.is_set()
        client.health_check.assert_not_awaited()
