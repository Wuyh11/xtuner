"""Keep agent OOM inside its own cgroup, outside the sandbox control services."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping


CGROUP_ROOT = Path("/sys/fs/cgroup")
REPORT_PATH = Path("/opt/xtuner/resource_guard.json")
MIB = 1024**2


def agent_memory_budget(total: int, control_reserve_bytes: int | None = None) -> tuple[int, int]:
    """Split the existing platform limit without increasing its allocation.

    Args:
        total (int): Finite sandbox memory limit in bytes.
        control_reserve_bytes (int | None): Explicit headroom, or a default of
            512 MiB to 1 GiB based on sandbox size.

    Returns:
        tuple[int, int]: Workload limit and control-service reserve in bytes.
    """
    if type(total) is not int or total <= 0:
        raise ValueError("A finite sandbox memory limit is required")
    reserve = control_reserve_bytes
    if reserve is None:
        reserve = max(512 * MIB, min(1024 * MIB, total // 4))
    if type(reserve) is not int or reserve <= 0:
        raise ValueError("A positive control-service memory reserve is required")
    if total - reserve < 256 * MIB:
        raise ValueError("Sandbox memory is too small to reserve control-service memory")
    return total - reserve, reserve


def _events(path: Path) -> dict[str, int]:
    return {key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())}


class AgentMemoryGuard:
    """Memory isolation for a workload launched by a trusted sandbox supervisor.

    Construct in the supervisor, call ``enter_child`` in the forked workload
    before dropping privileges, and call ``report`` in the supervisor after
    the workload exits. Descendants inherit the same limit. The control
    services and supervisor must stay outside this child cgroup. All concurrent
    agent/tool processes in one sandbox must share this guard's aggregate limit;
    allocating the whole remaining budget to several independent groups would
    not preserve control-service headroom.

    This is a resource boundary, not a privilege sandbox: the launcher must
    prevent untrusted workloads from moving themselves or editing the report.
    Unsupported controllers/permissions fail before workload launch.

    Args:
        root (str | Path): Delegated cgroup v2 root with a finite memory limit.
        cgroup_name (str): Single child name for this sandbox's workload domain.
        control_reserve_bytes (int | None): Memory reserved within the sandbox's
            existing limit. None reserves 512 MiB to 1 GiB based on its size.
        report_path (str | Path): Report in a directory writable only by the supervisor.
    """

    def __init__(
        self,
        root: str | Path = CGROUP_ROOT,
        *,
        cgroup_name: str = "sandbox-agent",
        control_reserve_bytes: int | None = None,
        report_path: str | Path = REPORT_PATH,
    ) -> None:
        if not cgroup_name or cgroup_name in {".", ".."} or "/" in cgroup_name or "\\" in cgroup_name:
            raise ValueError("cgroup_name must be one directory component")
        self.report_path = Path(report_path)
        if os.geteuid() != 0:
            raise PermissionError("Agent memory setup requires the trusted manager")
        root = Path(root)
        total = (root / "memory.max").read_text().strip()
        if total == "max":
            raise RuntimeError("A finite platform memory limit is required for the agent guard")
        self.limit, self.reserve = agent_memory_budget(int(total), control_reserve_bytes)
        self.total = int(total)
        self.path = root / cgroup_name
        if "memory" not in (root / "cgroup.controllers").read_text().split():
            raise RuntimeError("The sandbox does not delegate the cgroup v2 memory controller")
        (root / "cgroup.subtree_control").write_text("+memory")
        if self.path.is_symlink():
            raise RuntimeError("Agent cgroup must not be a symlink")
        self.path.mkdir(mode=0o700, exist_ok=True)
        info = self.path.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError("Agent cgroup must be owned and writable only by the manager")
        if (self.path / "cgroup.procs").read_text().strip():
            raise RuntimeError("Agent cgroup still contains live processes")
        (self.path / "memory.max").write_text(str(self.limit))
        (self.path / "memory.swap.max").write_text("0")
        # Let a large tool process be killed individually so the agent may continue.
        # A child-domain OOM cannot select the parent control-service cgroups.
        (self.path / "memory.oom.group").write_text("0")
        expected = {"memory.max": str(self.limit), "memory.swap.max": "0", "memory.oom.group": "0"}
        if any((self.path / name).read_text().strip() != value for name, value in expected.items()):
            raise RuntimeError("Agent memory guard settings did not take effect")
        self.before = _events(self.path / "memory.events")
        # A supervisor killed before report() must not leave evidence from a
        # previous invocation that could misclassify an unrelated SIGKILL.
        self.report_path.unlink(missing_ok=True)

    def enter_child(self) -> None:
        """Called before dropping capabilities and identity in the forked child."""
        (self.path / "cgroup.procs").write_text(str(os.getpid()))

    def report(
        self,
        *,
        return_code: int | None = None,
        timed_out: bool = False,
        cancelled: bool = False,
        path: str | Path | None = None,
        termination_error: str | None = None,
    ) -> dict[str, Any]:
        """Atomically write supervisor-observed events after workload termination.

        Returns:
            dict[str, Any]: Memory limits, event deltas and the termination cause.
        """
        after = _events(self.path / "memory.events")
        evidence = {
            "cgroup": str(self.path),
            "sandbox_memory_bytes": self.total,
            "agent_memory_bytes": self.limit,
            "control_reserve_bytes": self.reserve,
            "memory_oom_group": 0,
            "events_before": self.before,
            "events_after": after,
            "event_delta": {key: value - self.before.get(key, 0) for key, value in after.items()},
            "return_code": return_code,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "termination_error": termination_error,
        }
        target = Path(path) if path is not None else self.report_path
        # The report lives in the root-owned runtime directory, not the agent's logs.
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(evidence, stream)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return evidence


def memory_limit_exceeded(evidence: Mapping[str, Any]) -> bool:
    """Classify a SIGKILL only when trusted cgroup events confirm an OOM kill.

    Pass only a report collected from the trusted supervisor. A tool OOM
    followed by workload success, a timeout, and external cancellation are
    distinct outcomes and do not classify as workload memory failures.

    Args:
        evidence (Mapping[str, Any]): Trusted supervisor report for this workload.

    Returns:
        bool: Whether an OOM kill explains this workload's failed exit.
    """
    delta = evidence.get("event_delta")
    kills = delta.get("oom_kill") if isinstance(delta, Mapping) else None
    return (
        type(evidence.get("return_code")) is int
        and evidence["return_code"] in {137, -9}
        and evidence.get("timed_out") is False
        and evidence.get("cancelled") is False
        and type(kills) is int
        and kills > 0
    )


def restrict_workload(uid: int, gid: int) -> None:
    """Drop identity and capabilities before executing an untrusted workload."""
    if uid < 1000 or gid < 1000:
        raise ValueError("Workload must use an ordinary UID and GID")
    libc = ctypes.CDLL(None, use_errno=True)

    def prctl(option: int, value: int) -> None:
        if libc.prctl(option, value, 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), "Cannot restrict workload privileges")

    for cap in range(int(Path("/proc/sys/kernel/cap_last_cap").read_text()) + 1):
        present = libc.prctl(23, cap, 0, 0, 0)
        if present < 0:
            raise OSError(ctypes.get_errno(), "Cannot inspect workload capabilities")
        if present:
            prctl(24, cap)
    prctl(47, 4)
    prctl(38, 1)
    os.setgroups([])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]

    header, data = Header(0x20080522, 0), (Data * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "Cannot clear workload capabilities")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if (
        any(int(status[key], 16) for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"))
        or status["NoNewPrivs"].strip() != "1"
    ):
        raise RuntimeError("Workload privileges were not restricted")


def stop_workload(guard: AgentMemoryGuard, timeout: float = 10) -> None:
    """Reap descendants in the workload cgroup, including detached sessions."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        pids = [int(pid) for pid in (guard.path / "cgroup.procs").read_text().split()]
        if not pids:
            return
        sig = signal.SIGTERM if time.monotonic() - started < 1 else signal.SIGKILL
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
        time.sleep(0.05)
    raise RuntimeError("Workload cgroup termination was not confirmed")


def run_guarded(
    command: list[str],
    *,
    guard: AgentMemoryGuard,
    timeout_seconds: float,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    uid: int = 1000,
    gid: int = 1000,
    prepare_child: Callable[[], None] | None = None,
    cleanup: Callable[[], Any] | None = None,
) -> int:
    """Run a command under the memory guard and publish its trusted exit report.

    The supervisor remains outside the workload domain. An optional task
    callback may impose stronger identity restrictions; otherwise privileges
    are dropped here. Only the explicit environment is passed to the workload.

    Returns:
        int: Shell-compatible exit code, including 124 for the runtime deadline.
    """
    if not command or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("A command and finite positive timeout are required")
    child = None
    interrupted = False
    timed_out = False
    result = None

    def restrict() -> None:
        guard.enter_child()
        if prepare_child is not None:
            prepare_child()
        else:
            restrict_workload(uid, gid)

    def cancel(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=env if env is not None else {},
            close_fds=True,
            start_new_session=True,
            preexec_fn=restrict,
        )
        deadline = time.monotonic() + timeout_seconds
        while child.poll() is None and not interrupted and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        result = child.poll()
        if interrupted:
            result = 143
        elif result is None:
            timed_out, result = True, 124
        elif result < 0:
            result = 128 - result
        return result
    finally:
        # Each teardown step runs even if an earlier one fails: a process stuck
        # in the cgroup must not skip the task's own cleanup, the reap, or the
        # report that preserves the observed exit code.
        primary_exception = sys.exc_info()[1]
        steps: list[Callable[[], Any]] = [lambda: stop_workload(guard)]
        if cleanup is not None:
            steps.append(cleanup)
        if child is not None:
            steps.append(lambda: child.wait(timeout=3))
        errors: list[Exception] = []
        for step in steps:
            try:
                step()
            except Exception as exc:
                errors.append(exc)
        try:
            guard.report(
                return_code=result,
                timed_out=timed_out,
                cancelled=interrupted,
                termination_error=type(errors[0]).__name__ if errors else None,
            )
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if errors and primary_exception is None:
            raise errors[0]


def main() -> int:
    """Standalone supervisor installed by the public entry memory configuration.

    Returns:
        int: Guarded workload exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    guard = AgentMemoryGuard(
        cgroup_name=config["cgroup_name"],
        report_path=config["report_path"],
        control_reserve_bytes=config["control_reserve_bytes"],
    )
    return run_guarded(
        config["command"],
        guard=guard,
        timeout_seconds=config["timeout_seconds"],
        cwd=config["cwd"],
        env=config["env"],
        uid=config["uid"],
        gid=config["gid"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
