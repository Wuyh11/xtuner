"""Run deadlines and cancellation checkpoints shared by sandbox tasks."""

from __future__ import annotations

import asyncio
import math
import time
from types import TracebackType


class CancellationGuard:
    """Detect cancellation suppressed by hooks on Python versions with counters."""

    def __init__(self) -> None:
        self.task = asyncio.current_task()
        self.initial = self._count()

    def requested(self) -> bool:
        return self._count() > self.initial

    def check(self) -> None:
        if self.requested():
            raise asyncio.CancelledError()

    def _count(self) -> int:
        # Task.cancelling was added in Python 3.11. Ordinary cancellation still
        # propagates on 3.10; only swallowed-cancellation detection is unavailable.
        return getattr(self.task, "cancelling", lambda: 0)()


class RunDeadline:
    """Bound a run without misclassifying inner timeouts or external cancellation.

    Args:
        deadline (float | None): Absolute time.monotonic deadline, or None.

    Explicit deadlines require Python 3.11 cancellation counters so an external
    cancellation during cleanup can never become a run timeout. An unset
    deadline remains compatible with Python 3.10.
    """

    def __init__(self, deadline: float | None) -> None:
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        if deadline is not None and not hasattr(asyncio.Task, "cancelling"):
            raise RuntimeError("Configured sandbox run deadlines require Python 3.11 or newer")
        self.deadline = deadline
        self._expired = False
        self._timer: asyncio.TimerHandle | None = None
        self._guard: CancellationGuard | None = None
        self._marker = object()

    def expired(self) -> bool:
        return self._expired

    async def __aenter__(self) -> RunDeadline:
        self._guard = CancellationGuard()
        if self.deadline is not None:
            self._timer = asyncio.get_running_loop().call_later(
                max(0.0, self.deadline - time.monotonic()), self._cancel
            )
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._timer is not None:
            self._timer.cancel()
        guard = self._guard
        assert guard is not None and guard.task is not None
        if self._expired:
            uncancel = getattr(guard.task, "uncancel", None)
            own_cancel = (
                uncancel() <= guard.initial
                if uncancel is not None
                else (isinstance(exc, asyncio.CancelledError) and exc.args == (self._marker,))
            )
            if own_cancel and (exc is None or isinstance(exc, asyncio.CancelledError)):
                raise TimeoutError("Sandbox run deadline expired") from None
        if exc is None:
            guard.check()

    def _cancel(self) -> None:
        guard = self._guard
        assert guard is not None and guard.task is not None
        # An earlier external cancellation must keep its original cause, even
        # when bounded cleanup crosses the run deadline.
        if not guard.requested():
            self._expired = True
            guard.task.cancel(self._marker)
