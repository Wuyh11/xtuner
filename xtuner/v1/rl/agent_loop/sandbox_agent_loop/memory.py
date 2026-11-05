"""Public opt-in memory isolation for synchronous and detached shell entries."""

from __future__ import annotations

import json
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from xtuner.v1.rl.agent_loop.sandbox_agent_loop import resource_guard


class SandboxMemory:
    """Install and configure the trusted standalone workload supervisor.

    Args:
        python (str): Existing sandbox Python interpreter (3.10 or newer).
        uid (int): Ordinary workload user, default 1000.
        gid (int): Ordinary workload group, default 1000.
        cwd (str): Workload directory, which must be accessible to uid.
        runtime_dir (str): Trusted root-owned runtime directory; never a workload directory.
        control_reserve_bytes (int | None): Headroom inside the existing sandbox limit.

    Example: ``ShellEntry("python train.py", memory={"cwd": "/workspace",
    "python": "/opt/venv/bin/python", "control_reserve_bytes": 1073741824})``.
    The same configuration works on DetachedShellEntry; report upload/collection
    and memory failure classification are automatic.

    The sandbox must delegate cgroup v2 memory control. Unsupported images fail
    before launching the workload. All guarded entries use one aggregate cgroup. Run entries serially within
    each sandbox; simultaneous independent supervisors are unsupported.
    """

    def __init__(
        self,
        *,
        python: str = "python3",
        uid: int = 1000,
        gid: int = 1000,
        cwd: str = "/workspace",
        runtime_dir: str = "/opt/xtuner-memory",
        control_reserve_bytes: int | None = None,
    ) -> None:
        if type(uid) is not int or type(gid) is not int or uid < 1000 or gid < 1000:
            raise ValueError("Memory-isolated workloads require ordinary UID and GID")
        path = PurePosixPath(runtime_dir)
        if not path.is_absolute() or ".." in path.parts or not PurePosixPath(cwd).is_absolute():
            raise ValueError("Memory runtime and workload directories must be absolute")
        self.python, self.uid, self.gid, self.cwd = python, uid, gid, cwd
        self.runtime_dir, self.control_reserve_bytes = str(path), control_reserve_bytes

    async def prepare(
        self, client: Any, *, command: str, env: dict[str, str], timeout: float, entry_id: str
    ) -> tuple[str, str]:
        """Upload the supervisor and return its command and unique report path.

        Returns:
            tuple[str, str]: Trusted command and the supervisor's report path.
        """
        # mkdtemp under a checked manager-owned parent prevents report reuse and
        # follows no workload-controlled directories or symlinks.
        code = (
            """import os,stat,tempfile
from pathlib import Path
root=Path(%r)
root.mkdir(mode=0o700,parents=True,exist_ok=True)
for p in (root,*root.parents):
    s=p.lstat()
    if stat.S_ISLNK(s.st_mode) or s.st_uid != 0 or s.st_mode & 0o022:
        raise RuntimeError('Memory runtime must have manager-owned, non-writable ancestors')
print(tempfile.mkdtemp(prefix='entry-',dir=root))
"""
            % self.runtime_dir
        )
        result = await client.execute(shlex.join([self.python, "-I", "-c", code]), cwd="/", timeout_sec=10)
        if result.get("return_code") != 0:
            raise RuntimeError("Cannot prepare trusted memory runtime")
        directory = result.get("stdout", "").strip()
        path = PurePosixPath(directory)
        if path.parent != PurePosixPath(self.runtime_dir) or not path.name.startswith("entry-"):
            raise RuntimeError("Invalid memory runtime directory")
        report = directory + "/report.json"
        config = {
            "command": ["/bin/bash", "-c", command],
            "cwd": self.cwd,
            "env": {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": self.cwd, **env},
            "uid": self.uid,
            "gid": self.gid,
            "timeout_seconds": timeout,
            "control_reserve_bytes": self.control_reserve_bytes,
            "cgroup_name": "sandbox-agent",
            "report_path": report,
            "entry_id": entry_id,
        }
        for name, content in (
            ("supervisor.py", Path(resource_guard.__file__).read_bytes()),
            ("config.json", json.dumps(config).encode()),
        ):
            uploaded = await client.upload_bytes(directory + "/" + name, content)
            if uploaded.get("ok") is False:
                raise RuntimeError("Memory supervisor upload failed")
        return shlex.join(
            [self.python, "-I", directory + "/supervisor.py", "--config", directory + "/config.json"]
        ), report

    async def classify(self, client: Any, item: Any, record: Any, entry: Any, outcome: Any, report_path: str) -> Any:
        """Collect the fresh supervisor report and classify failed exits."""
        from xtuner.v1.rl.agent_loop.sandbox_agent_loop.resource_policy import MemoryFailurePolicy

        outcome = await MemoryFailurePolicy(report_path).handle(client, item, record, entry, outcome)
        evidence = item.artifacts.get("sandbox_resources", {}).get(entry.id, {})
        if (
            outcome.reason not in {"pid_monitor_error", "sandbox_unavailable", "rc_missing"}
            and outcome.result.return_code == 124
            and evidence.get("return_code") == 124
            and evidence.get("timed_out") is True
        ):
            outcome.reason = "timeout"
            outcome.result.error = "Workload runtime deadline expired"
        return outcome
