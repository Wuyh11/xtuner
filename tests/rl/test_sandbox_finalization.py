from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.deadline import RunDeadline
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.finalization import CleanupError
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.judger import Judger
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.runner import Runner
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import SandboxPool, SandboxStage
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, RolloutStatus, SandboxSpec


class Provider:
    def __init__(self) -> None:
        self.owned_env_ids = {"another-pool"}
        self.events: list[str] = []
        self.stall_first = False
        self.close_gate: asyncio.Event | None = None
        self.closed = False

    async def create(self, image_tag: str, **kwargs: Any) -> tuple[Any, str]:
        self.owned_env_ids.add(image_tag)
        provider = self

        class Client:
            async def health_check(self) -> dict[str, bool]:
                return {"ok": True}

            async def aclose(self) -> None:
                if provider.close_gate is not None:
                    await provider.close_gate.wait()
                provider.events.append("close:" + image_tag)

        return Client(), image_tag

    async def delete(self, env_id: str) -> None:
        self.events.append("attempt:" + env_id)
        if env_id == "main" and self.stall_first:
            self.stall_first = False
            await asyncio.sleep(10)
        self.owned_env_ids.discard(env_id)
        self.events.append("delete:" + env_id)

    async def get(self, env_id: str) -> dict[str, Any]:
        return {"env": {"status": "running" if env_id in self.owned_env_ids else "deleted"}}

    async def aclose(self) -> None:
        self.closed = True


def make_pool(provider: Provider, **kwargs: Any) -> SandboxPool:
    return SandboxPool(
        provider=provider,
        specs={"main": SandboxSpec(image="main", dependencies={"dependency": SandboxSpec(image="dependency")})},
        cleanup_timeout_sec=0.5,
        creates_per_sec=None,
        **kwargs,
    )


def make_item() -> AgentRolloutItem:
    return AgentRolloutItem(id="generic", data_source="test", instruction="test", task_root=Path("/tmp"), uid=1)


class TestSharedPool:
    @pytest.mark.asyncio
    async def test_shared_provider_keeps_other_pools_resources_and_connection(self) -> None:
        provider = Provider()
        pool = make_pool(provider)
        await pool.get("main")
        await pool.release_all()
        assert provider.owned_env_ids == {"another-pool"}
        assert not provider.closed
        assert pool.cleanup["ok"]
        assert pool.cleanup["provider_closed"] is False
        assert provider.events.index("delete:dependency") < provider.events.index("close:main")
        before = list(provider.events)
        await pool.release_all()
        assert provider.events == before

    @pytest.mark.asyncio
    async def test_stalled_primary_leaves_time_for_dependency_and_retry(self) -> None:
        provider = Provider()
        provider.stall_first = True
        pool = make_pool(provider, verify_deletion=True)
        await pool.get("main")
        started = time.monotonic()
        await pool.release_all()
        assert time.monotonic() - started < 0.8
        assert provider.events.count("attempt:main") == 2
        assert provider.events.index("delete:dependency") < provider.events.index("delete:main")
        assert provider.events.index("delete:main") < provider.events.index("close:main")
        assert pool.cleanup["ok"]
        assert len(pool.cleanup["final_states"]) == 2

    @pytest.mark.asyncio
    async def test_dedicated_provider_cleans_orphaned_create_and_closes(self) -> None:
        provider = Provider()
        provider.owned_env_ids = {"failed-create-orphan"}
        pool = make_pool(provider, close_provider=True, verify_deletion=True)
        await pool.get("main")
        await pool.release_all()
        assert not provider.owned_env_ids
        assert provider.closed
        assert len(pool.cleanup["final_states"]) == 3

    @pytest.mark.asyncio
    async def test_close_timeout_can_be_retried_without_false_success(self) -> None:
        provider = Provider()
        provider.close_gate = asyncio.Event()
        pool = make_pool(provider)
        await pool.get("main")
        with pytest.raises(CleanupError):
            await pool.release_all()
        assert pool.cleanup["resources_deleted"]
        assert not pool.cleanup["clients_closed"]
        assert not pool.cleanup["ok"]
        provider.close_gate.set()
        await pool.release_all()
        assert pool.cleanup["ok"] and pool.cleanup["clients_closed"]
        assert provider.events.count("close:main") == 1


class TestSharedRunner:
    @pytest.mark.asyncio
    async def test_swallowed_hook_cancellation_cannot_start_next_hook_or_judge(self) -> None:
        entered = asyncio.Event()
        events: list[str] = []

        async def work(client: Any, item: Any, record: Any) -> None:
            entered.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass

        async def after(client: Any, item: Any, record: Any) -> None:
            events.append("after")

        provider = Provider()
        pool = make_pool(provider)
        runner = Runner(pool=pool, infer=SandboxStage(pre=[work, after]), validate=Judger("score", SandboxStage()))
        item = make_item()
        task = asyncio.create_task(runner.run(item))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == [] and not item.judgers
        assert item.error.category == "rollout_cancelled"
        assert item.artifacts["sandbox_cleanup"]["ok"]

    @pytest.mark.asyncio
    async def test_total_deadline_is_distinct_from_inner_timeout(self) -> None:
        async def slow(client: Any, item: Any, record: Any) -> None:
            await asyncio.sleep(10)

        async def inner_timeout(client: Any, item: Any, record: Any) -> None:
            raise TimeoutError("inner")

        for hook, expired in [(slow, True), (inner_timeout, False)]:
            pool = make_pool(Provider())
            runner = Runner(
                pool=pool, infer=SandboxStage(pre=[hook]), validate=Judger("score", SandboxStage()), timeout_sec=0.1
            )
            item = await runner.run(make_item())
            assert item.artifacts["sandbox_run_timing"]["deadline_expired"] is expired
            assert (item.error.category == "trial_timeout") is expired
            assert item.artifacts["sandbox_cleanup"]["ok"]

    @pytest.mark.asyncio
    async def test_cleanup_timeout_is_not_trial_timeout(self) -> None:
        provider = Provider()
        provider.close_gate = asyncio.Event()
        pool = make_pool(provider)

        async def score(client: Any, item: Any, record: Any) -> None:
            record.score = 0.0

        runner = Runner(
            pool=pool, infer=SandboxStage(), validate=Judger("score", SandboxStage(post=[score])), timeout_sec=5
        )
        # A scored rollout whose cleanup fails is returned as failed, not raised.
        item = await runner.run(make_item())
        assert item.status == RolloutStatus.FAILED
        assert item.error.category == "cleanup_timeout"
        assert not item.artifacts["sandbox_cleanup"]["ok"]
        assert not item.artifacts["sandbox_run_timing"]["deadline_expired"]
        assert item.artifacts["sandbox_cleanup"]["resources_deleted"]
        provider.close_gate.set()
        await pool.release_all()


class TestRunDeadline:
    def test_explicit_deadlines_reject_runtimes_without_cancellation_counters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(asyncio, "Task", object)
        RunDeadline(None)
        with pytest.raises(RuntimeError, match="Python 3.11"):
            RunDeadline(time.monotonic() + 1)

    @pytest.mark.asyncio
    async def test_external_cancel_during_cleanup_crossing_deadline_stays_cancelled(self) -> None:
        entered = asyncio.Event()
        deadline = RunDeadline(time.monotonic() + 0.1)

        async def run() -> None:
            async with deadline:
                entered.set()
                try:
                    await asyncio.sleep(10)
                finally:
                    await asyncio.sleep(0.15)

        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not deadline.expired()

    @pytest.mark.asyncio
    async def test_external_cancel_after_deadline_preserves_external_cause(self) -> None:
        cleaning = asyncio.Event()
        deadline = RunDeadline(time.monotonic() + 0.05)

        async def run() -> None:
            async with deadline:
                try:
                    await asyncio.sleep(10)
                finally:
                    cleaning.set()
                    await asyncio.sleep(10)

        task = asyncio.create_task(run())
        await cleaning.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert deadline.expired()
