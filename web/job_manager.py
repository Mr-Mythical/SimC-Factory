"""In-memory job manager with subprocess lifecycle and SSE log streaming."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from web.config import PROJECT_ROOT, PYTHON_EXE

MAX_LOG_LINES = 10_000
JOB_STATE_DIR = os.path.join(PROJECT_ROOT, "local", "web_jobs")
JOB_STATE_FILE = os.path.join(JOB_STATE_DIR, "jobs.json")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class Job:
    id: str
    name: str
    preset_id: str
    script: str
    args: list[str]
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    process: subprocess.Popen | None = None
    pid: int | None = None
    log_path: str | None = None
    recovered: bool = False
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    exit_code: int | None = None
    log_lines: list[str] = field(default_factory=list)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _loop: asyncio.AbstractEventLoop | None = None
    on_complete: Callable[["Job"], None] | None = None
    _monitor_thread_started: bool = False


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def _ensure_state_dir() -> None:
    os.makedirs(JOB_STATE_DIR, exist_ok=True)


def _serialize_job(job: Job) -> dict:
    return {
        "id": job.id,
        "name": job.name,
        "preset_id": job.preset_id,
        "script": job.script,
        "args": job.args,
        "status": job.status,
        "pid": job.pid,
        "log_path": job.log_path,
        "recovered": job.recovered,
        "started_at": job.started_at.isoformat(),
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "exit_code": job.exit_code,
    }


def _save_jobs_state() -> None:
    _ensure_state_dir()
    with _lock:
        payload = [_serialize_job(job) for job in _jobs.values()]
    with open(JOB_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_log_line(job: Job, line: str) -> None:
    with _lock:
        if len(job.log_lines) < MAX_LOG_LINES:
            job.log_lines.append(line)
        elif len(job.log_lines) == MAX_LOG_LINES:
            job.log_lines.append("... [log truncated at 10000 lines] ...")

    loop = job._loop
    if loop:
        for q in list(job._subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, line)
            except Exception:
                pass


def _close_subscribers(job: Job) -> None:
    loop = job._loop
    if loop:
        for q in list(job._subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, None)
            except Exception:
                pass


def _mark_job_finished(job: Job, exit_code: int | None) -> None:
    if job.status != "running":
        if job.finished_at is None:
            job.finished_at = _utcnow()
        if job.exit_code is None:
            job.exit_code = exit_code
        _save_jobs_state()
        _close_subscribers(job)
        return

    job.exit_code = exit_code
    job.finished_at = _utcnow()
    if exit_code == 0:
        job.status = "completed"
    elif exit_code is None:
        job.status = "completed"
    else:
        job.status = "failed"

    _save_jobs_state()

    if job.on_complete:
        try:
            job.on_complete(job)
        except Exception:
            pass

    _close_subscribers(job)


def _is_pid_running_windows(pid: int) -> tuple[bool, int | None]:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False, None

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False, None

    try:
        exit_code = wintypes.DWORD()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        if not ok:
            return False, None
        if exit_code.value == STILL_ACTIVE:
            return True, None
        return False, int(exit_code.value)
    finally:
        kernel32.CloseHandle(handle)


def _is_pid_running(pid: int) -> tuple[bool, int | None]:
    if sys.platform == "win32":
        return _is_pid_running_windows(pid)

    try:
        os.kill(pid, 0)
        return True, None
    except OSError:
        return False, None


def _job_runtime_status(job: Job) -> tuple[bool, int | None]:
    proc = job.process
    if proc is not None:
        code = proc.poll()
        return (code is None, code)

    if job.pid is None:
        return False, job.exit_code

    return _is_pid_running(job.pid)


def _load_recent_log_lines(log_path: str | None, max_lines: int = MAX_LOG_LINES) -> list[str]:
    if not log_path or not os.path.exists(log_path):
        return []

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return list(deque((line.rstrip("\r\n") for line in f), maxlen=max_lines))
    except OSError:
        return []


def _start_monitor_thread(job: Job, start_offset: int = 0) -> None:
    with _lock:
        if job._monitor_thread_started:
            return
        job._monitor_thread_started = True

    t = threading.Thread(target=_monitor_job_output, args=(job, start_offset), daemon=True)
    t.start()


def _monitor_job_output(job: Job, start_offset: int) -> None:
    """Tail the job log file and update runtime status in a background thread."""
    log_path = job.log_path
    if not log_path:
        _mark_job_finished(job, job.exit_code)
        return

    while not os.path.exists(log_path):
        running, exit_code = _job_runtime_status(job)
        if not running:
            _mark_job_finished(job, exit_code)
            return
        time.sleep(0.1)

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            f.seek(start_offset)
            while True:
                line = f.readline()
                if line:
                    _append_log_line(job, line.rstrip("\r\n"))
                    continue

                running, exit_code = _job_runtime_status(job)
                if not running:
                    remainder = f.read()
                    if remainder:
                        for remainder_line in remainder.splitlines():
                            _append_log_line(job, remainder_line)
                    _mark_job_finished(job, exit_code)
                    return

                time.sleep(0.2)
    except Exception as exc:
        _append_log_line(job, f"[job monitor] {exc}")
        running, exit_code = _job_runtime_status(job)
        _mark_job_finished(job, exit_code if not running else exit_code)


def recover_jobs() -> None:
    _ensure_state_dir()
    if not os.path.exists(JOB_STATE_FILE):
        return

    try:
        with open(JOB_STATE_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    loaded_jobs: dict[str, Job] = {}
    for raw in payload:
        try:
            started_at = datetime.fromisoformat(raw["started_at"])
            finished_at = datetime.fromisoformat(raw["finished_at"]) if raw.get("finished_at") else None
            job = Job(
                id=raw["id"],
                name=raw["name"],
                preset_id=raw.get("preset_id", ""),
                script=raw.get("script", ""),
                args=list(raw.get("args", [])),
                status=raw.get("status", "completed"),
                pid=raw.get("pid"),
                log_path=raw.get("log_path"),
                recovered=True,
                started_at=started_at,
                finished_at=finished_at,
                exit_code=raw.get("exit_code"),
                log_lines=_load_recent_log_lines(raw.get("log_path")),
            )
        except Exception:
            continue

        if job.status == "running":
            running, exit_code = _job_runtime_status(job)
            if running:
                start_offset = 0
                if job.log_path and os.path.exists(job.log_path):
                    try:
                        start_offset = os.path.getsize(job.log_path)
                    except OSError:
                        start_offset = 0
                _start_monitor_thread(job, start_offset=start_offset)
            else:
                job.exit_code = exit_code
                job.finished_at = job.finished_at or _utcnow()
                job.status = "completed" if exit_code in (None, 0) else "failed"

        loaded_jobs[job.id] = job

    with _lock:
        _jobs.clear()
        _jobs.update(loaded_jobs)

    _save_jobs_state()


def launch(script_path: str, args: list[str], name: str, preset_id: str = "",
           loop: asyncio.AbstractEventLoop | None = None,
           raw_cmd: list[str] | None = None,
           cwd: str | None = None,
           on_complete: Callable[["Job"], None] | None = None) -> Job:
    """Spawn a subprocess and start streaming its output.

    If *raw_cmd* is provided it is used as-is; otherwise the command is
    built as ``[PYTHON_EXE, script_path] + args``.
    """
    job_id = uuid.uuid4().hex[:12]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    from web import credentials
    env.update(credentials.get_aws_env())

    cmd = raw_cmd if raw_cmd else [PYTHON_EXE, script_path] + args
    _ensure_state_dir()
    log_path = os.path.join(JOB_STATE_DIR, f"{job_id}.log")

    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    with open(log_path, "ab") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=cwd or PROJECT_ROOT,
            env=env,
            creationflags=creation_flags,
        )

    job = Job(
        id=job_id,
        name=name,
        preset_id=preset_id,
        script=script_path,
        args=args,
        process=proc,
        pid=proc.pid,
        log_path=log_path,
        _loop=loop,
        on_complete=on_complete,
    )

    with _lock:
        _jobs[job_id] = job

    _save_jobs_state()
    _start_monitor_thread(job, start_offset=0)

    return job


def cancel(job_id: str) -> Job | None:
    """Cancel a running job."""
    with _lock:
        job = _jobs.get(job_id)
    if not job or job.status != "running":
        return job

    job.status = "cancelled"
    proc = job.process

    if proc is not None:
        if sys.platform == "win32":
            try:
                os.kill(proc.pid, signal.CTRL_C_EVENT)
            except OSError:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
        else:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
    else:
        if job.pid is not None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(job.pid), "/T", "/F"], check=False)
            else:
                try:
                    os.kill(job.pid, signal.SIGTERM)
                except OSError:
                    pass

    job.finished_at = _utcnow()
    _save_jobs_state()
    _close_subscribers(job)

    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[Job]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)


async def subscribe(job_id: str) -> AsyncGenerator[str | None, None]:
    """Async generator that yields log lines for SSE streaming."""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return

    if job._loop is None:
        job._loop = asyncio.get_running_loop()

    # Replay existing lines
    with _lock:
        existing = list(job.log_lines)
    for line in existing:
        yield line

    # If already finished, no need to subscribe
    if job.status != "running":
        return

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    job._subscribers.append(queue)
    try:
        while True:
            line = await queue.get()
            if line is None:
                break
            yield line
    finally:
        try:
            job._subscribers.remove(queue)
        except ValueError:
            pass


def launch_with_ecr_rebuild(
    script_path: str,
    args: list[str],
    name: str,
    preset_id: str = "",
    loop: asyncio.AbstractEventLoop | None = None,
    raw_cmd: list[str] | None = None,
    cwd: str | None = None,
) -> Job:
    """Launch an ECR image rebuild, then chain the actual job on completion.

    Returns the ECR rebuild job immediately.  The simulation job is spawned
    automatically once the rebuild finishes successfully.
    """
    from web.config import PRESETS

    ecr_preset = PRESETS.get("rebuild_ecr_image")
    if not ecr_preset:
        # Fallback — launch the sim job directly if preset is missing
        return launch(script_path=script_path, args=args, name=name,
                      preset_id=preset_id, loop=loop, raw_cmd=raw_cmd, cwd=cwd)

    def _on_ecr_done(ecr_job: "Job") -> None:
        if ecr_job.status != "completed":
            return  # ECR build failed — don't chain the sim job
        launch(script_path=script_path, args=args, name=name,
               preset_id=preset_id, loop=loop, raw_cmd=raw_cmd, cwd=cwd)

    subcmd_parts = ecr_preset.subcommand.split() if ecr_preset.subcommand else []
    ecr_raw_cmd = [ecr_preset.script] + subcmd_parts
    return launch(
        script_path=ecr_preset.script,
        args=[],
        name=f"ECR Rebuild -> {name}",
        preset_id="rebuild_ecr_image",
        loop=loop,
        raw_cmd=ecr_raw_cmd,
        cwd=ecr_preset.cwd,
        on_complete=_on_ecr_done,
    )


def shutdown_all() -> None:
    """Persist current job state on server shutdown without cancelling jobs."""
    _save_jobs_state()
