#!/usr/bin/env python3
"""SimulationCraft local orchestrator for AWS Batch array jobs.

Design goals
------------
- Keep the local machine as the orchestrator and merger.
- Keep cloud workers lightweight: no Python, no pandas/numpy, no boto3.
- Use S3 only as temporary transport for chunk inputs/outputs.
- Submit one AWS Batch array job per spec.
- Download worker outputs back to the local machine and delete temporary S3 data.

How it works
------------
1. The local orchestrator builds deterministic chunks for each spec.
2. For every chunk, it generates .simc input files locally and zips them.
3. It uploads those zips to S3 and creates a small manifest JSON with the S3
    object keys for each chunk's input/output object.
4. It submits one AWS Batch array job for the spec.
5. Each array child reads AWS_BATCH_JOB_ARRAY_INDEX, uses its AWS Batch job
    role to download only its own chunk zip from S3, runs SimulationCraft,
    zips the JSON outputs, and uploads them back to S3.
6. The local orchestrator waits for the array parent to finish, downloads the
   output zips, extracts them into the local shard folders, merges the shard
   CSVs, and deletes the temporary S3 objects.

Quick Start
-----------
The default behavior is to auto-select incomplete specs (saved rows in
training_data/<spec>.csv are lower than --samples), queue them, and run with
concurrency controlled by --max-active-profiles.

With current defaults, the shortest command is:
    ./local/.venv/Scripts/python.exe local/sim_orchestrator_batch.py

This uses default values for:
- --worker-parallel 4
- --s3-bucket simc-batch-bucket
- --batch-job-queue simc-batch-array-queue
- --batch-job-definition simc-batch-array-worker
- --aws-region eu-north-1

From the repository root (Windows Git Bash):
    ./local/.venv/Scripts/python.exe local/sim_orchestrator_batch.py \
         --samples 100 \
         --max-active-profiles 2

From the local folder:
  ./.venv/Scripts/python.exe sim_orchestrator_batch.py \
         --samples 100 \
         --max-active-profiles 2

Common Start Modes
------------------
1. Auto resume/missing only (recommended)
    - Do not pass --specs.
    - Script queues only specs with saved rows below --samples.

2. Limit the number of profiles this run
    - Add --profile-limit N (for example, --profile-limit 3).

3. Keep multiple profiles in-flight
    - Use --max-active-profiles N (for example, 2 or 3).

4. Force explicit profiles
    - Use --specs MID1_Mage_Frost,MID1_Mage_Fire
    - Or --specs all
    - Or --specs existing (profiles that already have training_data CSVs)

5. Reset one run's local saved state before submitting
    - Add --clean-spec-output
    - This deletes training_data/<spec>.csv and distributed_shards/<spec> for
      each queued spec before processing.

Parameter Guide
---------------
Core (all have defaults)
- --s3-bucket: S3 bucket used for temporary manifests and chunk zips.
- --batch-job-queue: AWS Batch queue name/ARN used for array jobs.
- --batch-job-definition: AWS Batch job definition for the worker container.
- --aws-region: AWS region for Batch/S3 clients.

Selection / Resume
- --samples: Target saved row count per spec CSV (default: 100).
- --specs: Optional manual spec override. Omit to auto-resume missing work.
- --profile-limit: Optional max number of profiles to queue this run.
- --start-index: Optional explicit starting sim index; defaults to saved row
  count for that spec.

Concurrency / Throughput
- --max-active-profiles: How many spec array jobs are in-flight together.
- --chunk-size: Sims per Batch array child chunk.
- --worker-parallel: Parallel sims inside each worker container.
- --poll-seconds: Parent job polling interval.

Simulation Controls
- --iterations: SimulationCraft iterations per generated sim profile.
- --seed: RNG seed for deterministic stat generation.

Adaptive Sample Boosting
- --boost-from-model: Path to nn_metadata.json to enable per-spec sample
  targets based on model prediction error. Specs with high MAE get more
  samples; specs at or below median MAE keep the baseline.
- --boost-extra-budget: Max extra samples the worst spec gets per boost
  pass (default 200). Other above-median specs get a proportional share.
  There is no upper limit on accumulated samples per spec.

Storage / AWS
- --s3-prefix: Prefix for temporary S3 objects.
- --job-name-prefix: Prefix for submitted Batch job names.

Metadata / Profile Management
- --download-profiles: Force refresh all spec profile files from GitHub, then exit.
- --skip-profile-refresh: Skip per-spec profile refresh; use local cache only.
  Useful when a prior pipeline step already called --download-profiles.
- --check-staleness: Run drift detection (re-sim samples) and print stale profiles, then exit.
- --regenerate-stale: Run drift detection, then re-simulate stale profiles.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import glob
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from botocore.config import Config

import boto3
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
PLAYER_LEVEL = 90  # WoW character level used in generated sim profiles.

# SimulationCraft iterations per generated .simc profile.
DEFAULT_ITERATIONS = 5000
# Target number of saved stat/DPS samples per spec in training_data/<spec>.csv.
DEFAULT_SAMPLES_COUNT = 500
# RNG seed for deterministic stat override generation across runs.
DEFAULT_GEAR_RANDOM_SEED = 1337
# Number of sim profiles bundled into a single AWS Batch array child job.
DEFAULT_CHUNK_SIZE = 25
# Lifetime of pre-signed S3 URLs in seconds (kept for CLI compatibility; no longer actively used).
DEFAULT_PRESIGN_SECONDS = 24 * 60 * 60
# Seconds between polls when waiting for an AWS Batch array parent job to finish.
DEFAULT_POLL_SECONDS = 10
# Number of SimulationCraft processes run in parallel inside each worker container.
DEFAULT_WORKER_PARALLEL = 4
# Maximum number of spec array jobs submitted and in-flight at the same time.
DEFAULT_MAX_ACTIVE_PROFILES = 4
DEFAULT_SIMC_DOCKER_IMAGE = "simulationcraftorg/simc:latest"
DEFAULT_STALENESS_SAMPLE_SIZE = 10
DEFAULT_STALENESS_SAMPLE_ITERATIONS = 5000
DEFAULT_STALENESS_DRIFT_THRESHOLD_PCT = 2.0

SPEC_PROFILES_DIR = "./spec_profiles"
GITHUB_PROFILES_BASE_URL = (
    "https://raw.githubusercontent.com/simulationcraft/simc/midnight/profiles/MID1/"
)
GITHUB_GENERATORS_BASE_URL = (
    "https://raw.githubusercontent.com/simulationcraft/simc/midnight/profiles/generators/MID1/"
)
GITHUB_CONTENTS_PROFILES_MID1_URL = "https://api.github.com/repos/simulationcraft/simc/contents/profiles/MID1?ref=midnight"
GITHUB_API_COMMITS_URL = "https://api.github.com/repos/simulationcraft/simc/commits"
PROFILE_METADATA_FILE = "./profile_metadata.json"
DEFAULT_TRAINING_DATA_DIR = "./training_data"
DEFAULT_CHUNK_STATE_DIR = "./distributed_shards"
DEFAULT_EVALUATION_PLOTS_DIR = "./local/nn_website_model/evaluation_plots"
DRIFT_HISTORY_LOG_FILE = "./local/nn_website_model/drift_history.jsonl"
GENERATOR_MISMATCH_LOG_FILE = "./local/generator_mismatch_log.jsonl"
_AVAILABLE_SPEC_PROFILES_CACHE: list[str] | None = None
_PROFILE_CONTENT_CACHE: dict[str, str] | None = None
_PATH_LAST_COMMIT_CACHE: dict[str, datetime.datetime | None] = {}
_GENERATOR_FILE_CACHE: dict[str, str] = {}
_PROFILE_COMPATIBILITY_CACHE: dict[str, tuple[bool, str | None]] = {}

DEFAULT_PRIMARY_BASELINE = 1100
DEFAULT_SECONDARY_BASELINE = 800

# Stat-swing bounds (relative fractions of each spec's gear-summary baseline).
PRIMARY_SWING_DOWN_PCT = 0.30
PRIMARY_SWING_UP_PCT = 0.13
SECONDARY_SWING_DOWN_PCT = 0.23
SECONDARY_SWING_UP_PCT = 0.07
SECONDARY_BUDGET_MINIMUM = 500  # floor to prevent degenerate allocations
SECONDARY_STATS = ["crit", "haste", "mastery", "versatility"]

CANCEL_REQUESTED = threading.Event()
ACTIVE_PROCESSES: set[subprocess.Popen] = set()
ACTIVE_PROCESSES_LOCK = threading.Lock()


def _track_process(proc: subprocess.Popen) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.add(proc)


def _untrack_process(proc: subprocess.Popen) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.discard(proc)


def _terminate_active_processes() -> None:
    with ACTIVE_PROCESSES_LOCK:
        procs = list(ACTIVE_PROCESSES)
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def _handle_sigint(signum: int, frame: Any) -> None:
    if CANCEL_REQUESTED.is_set():
        return
    CANCEL_REQUESTED.set()
    print("\nCancellation requested. Stopping active work…")
    _terminate_active_processes()


signal.signal(signal.SIGINT, _handle_sigint)


@dataclasses.dataclass
class SpecConfig:
    profile_name: str
    class_keyword: str
    character_name: str
    spec: str
    race: str
    primary_stat: str
    stat_baseline: dict[str, int]
    profile_content: str


@dataclasses.dataclass
class ChunkPlan:
    chunk_id: int
    start_index: int
    sample_count: int
    profile_names: list[str]
    local_chunk_dir: str
    input_zip_path: str
    input_key: str
    output_key: str
    shard_csv_path: str


@dataclasses.dataclass
class ActiveSpecRun:
    spec_name: str
    spec_config: SpecConfig
    start_index: int
    sample_count: int
    run_prefix: str
    chunk_plans: list[ChunkPlan]
    manifest_key: str
    parent_job_id: str
    last_status: str | None = None


@dataclasses.dataclass
class FinalizeOutcome:
    merged: pd.DataFrame | None
    should_retry: bool
    reason: str | None = None


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def _run_command(cmd: list[str], capture_output: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
    )
    _track_process(proc)
    try:
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    finally:
        _untrack_process(proc)


def _profile_cache_path(profile_name: str) -> str:
    # Only strip characters that are illegal in Windows filenames.
    # Apostrophes (e.g. San'layn) are valid and must be preserved so the
    # cached filename matches the original GitHub profile name.
    safe = re.sub(r'[<>:"/\\|?*]', "_", profile_name)
    return os.path.join(SPEC_PROFILES_DIR, f"{safe}.simc")


def _is_invalid_cached_profile(profile_name: str, profile_content: str) -> bool:
    # A valid runnable profile should include a class declaration. Some stale
    # cache files can degrade to save-only stubs and are not executable.
    has_character_decl = re.search(
        r'^\s*\w+\s*=\s*"?MID1_[^"\s]+"?\s*$',
        profile_content,
        re.MULTILINE,
    ) is not None
    if not has_character_decl:
        return True

    non_comment_lines = [
        line.strip()
        for line in profile_content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    has_save = any(re.match(r'^\s*save\s*=', line, re.IGNORECASE) for line in non_comment_lines)
    if has_save and len(non_comment_lines) <= 2:
        return True
    return False


def _fetch_profile_file_names() -> list[str]:
    """Fetch the list of pre-built MID1 profile files from GitHub."""
    req = urllib.request.Request(GITHUB_CONTENTS_PROFILES_MID1_URL)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "SimulationCraft-Orchestrator")
    with urllib.request.urlopen(req, timeout=30) as resp:
        listing = json.loads(resp.read().decode("utf-8"))
    return sorted(
        item["name"]
        for item in listing
        if isinstance(item, dict)
        and item.get("type") == "file"
        and str(item.get("name", "")).startswith("MID1_")
        and str(item.get("name", "")).endswith(".simc")
    )


def _fetch_all_profiles() -> dict[str, str]:
    """Download all pre-built MID1 profiles from the repo."""
    profile_map: dict[str, str] = {}
    file_names = _fetch_profile_file_names()
    if not file_names:
        raise RuntimeError("No MID1_*.simc profile files found in profiles/MID1/")
    for file_name in file_names:
        url = GITHUB_PROFILES_BASE_URL + file_name
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        profile_name = Path(file_name).stem
        profile_map[profile_name] = content
    if not profile_map:
        raise RuntimeError("Could not fetch any MID1 profiles from profiles/MID1/")
    return profile_map


def _load_remote_profile_map(force_refresh: bool = False) -> dict[str, str]:
    global _PROFILE_CONTENT_CACHE
    if _PROFILE_CONTENT_CACHE is not None and not force_refresh:
        return dict(_PROFILE_CONTENT_CACHE)

    profile_map = _fetch_all_profiles()
    _PROFILE_CONTENT_CACHE = profile_map
    return dict(profile_map)


def _fetch_single_profile(profile_name: str) -> str:
    """Download a single profile by name via raw GitHub URL (no API rate limit)."""
    url = GITHUB_PROFILES_BASE_URL + profile_name + ".simc"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "SimulationCraft-Orchestrator")
    with urllib.request.urlopen(req, timeout=30) as resp:
        generated_content = resp.read().decode("utf-8")
    return generated_content


def _profile_commit_shas(path_in_repo: str, limit: int = 20) -> list[str]:
    query = urllib.parse.urlencode({
        "sha": "midnight",
        "path": path_in_repo,
        "per_page": min(max(limit, 1), 100),
    })
    url = f"{GITHUB_API_COMMITS_URL}?{query}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "SimulationCraft-Orchestrator")
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        return []
    shas: list[str] = []
    for entry in payload:
        sha = entry.get("sha") if isinstance(entry, dict) else None
        if isinstance(sha, str) and sha:
            shas.append(sha)
    return shas


def _fetch_profile_at_commit(profile_name: str, commit_sha: str) -> str:
    url = f"https://raw.githubusercontent.com/simulationcraft/simc/{commit_sha}/profiles/MID1/{profile_name}.simc"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "SimulationCraft-Orchestrator")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _validate_profile_with_local_simc(profile_content: str) -> tuple[bool, str | None]:
    """Validate profile parses with local SimC Docker image.

    Returns (is_valid, error_preview).
    """
    with tempfile.TemporaryDirectory(prefix="simc_profile_validate_") as temp_work_dir:
        work_dir = os.fspath(temp_work_dir)
        if isinstance(work_dir, bytes):
            work_dir = os.fsdecode(work_dir)
        profile_path = os.path.join(work_dir, "validate.simc")
        with open(profile_path, "w", encoding="utf-8") as f:
            f.write(profile_content)

        env = dict(os.environ)
        env["MSYS_NO_PATHCONV"] = "1"
        env["MSYS2_ARG_CONV_EXCL"] = "*"
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "/app/SimulationCraft/simc",
                "-v",
                f"{os.path.abspath(work_dir)}:/work",
                "-w",
                "/work",
                DEFAULT_SIMC_DOCKER_IMAGE,
                "validate.simc",
                "iterations=1",
                "fight_style=Patchwerk",
                "max_time=1",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=env,
        )
        if proc.returncode == 0:
            return True, None
        err_text = (proc.stderr or proc.stdout or "").strip().splitlines()
        preview = err_text[-1] if err_text else f"exit={proc.returncode}"
        return False, preview


def _validate_profile_with_cache(profile_content: str) -> tuple[bool, str | None]:
    key = hashlib.sha256(profile_content.encode("utf-8")).hexdigest()
    cached = _PROFILE_COMPATIBILITY_CACHE.get(key)
    if cached is not None:
        return cached
    result = _validate_profile_with_local_simc(profile_content)
    _PROFILE_COMPATIBILITY_CACHE[key] = result
    return result


def _find_compatible_profile_revision(profile_name: str, max_commits: int = 20) -> str | None:
    """Walk prior profile commits and return first content that parses with local SimC."""
    path_in_repo = f"profiles/MID1/{profile_name}.simc"
    try:
        shas = _profile_commit_shas(path_in_repo, limit=max_commits)
    except Exception:
        return None

    # First SHA is the current tip commit for this file; skip it.
    for sha in shas[1:]:
        try:
            candidate = _fetch_profile_at_commit(profile_name, sha)
        except Exception:
            continue
        is_valid, _ = _validate_profile_with_local_simc(candidate)
        if is_valid:
            return candidate
    return None


def _github_latest_commit_timestamp(path_in_repo: str) -> datetime.datetime | None:
    """Return latest commit timestamp for a path on simc/midnight (UTC)."""
    cached = _PATH_LAST_COMMIT_CACHE.get(path_in_repo, ...)
    if cached is not ...:
        return cached

    query = urllib.parse.urlencode({
        "sha": "midnight",
        "path": path_in_repo,
        "per_page": 1,
    })
    url = f"{GITHUB_API_COMMITS_URL}?{query}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "SimulationCraft-Orchestrator")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, list) or not payload:
            _PATH_LAST_COMMIT_CACHE[path_in_repo] = None
            return None
        date_str = payload[0].get("commit", {}).get("committer", {}).get("date")
        if not isinstance(date_str, str):
            _PATH_LAST_COMMIT_CACHE[path_in_repo] = None
            return None
        ts = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        _PATH_LAST_COMMIT_CACHE[path_in_repo] = ts
        return ts
    except Exception:
        _PATH_LAST_COMMIT_CACHE[path_in_repo] = None
        return None


def _generator_file_name_for_profile_content(profile_content: str) -> str | None:
    m = re.search(r'^\s*(\w+)\s*=\s*"?MID1_[^"\s]+"?\s*$', profile_content, re.MULTILINE)
    if not m:
        return None
    class_keyword = m.group(1).lower()
    class_map = {
        "deathknight": "Death_Knight",
        "demonhunter": "Demon_Hunter",
        "druid": "Druid",
        "evoker": "Evoker",
        "hunter": "Hunter",
        "mage": "Mage",
        "monk": "Monk",
        "paladin": "Paladin",
        "priest": "Priest",
        "rogue": "Rogue",
        "shaman": "Shaman",
        "warlock": "Warlock",
        "warrior": "Warrior",
    }
    class_name = class_map.get(class_keyword)
    if not class_name:
        return None
    return f"MID1_Generate_{class_name}.simc"


def _fetch_generator_file(generator_file_name: str) -> str:
    cached = _GENERATOR_FILE_CACHE.get(generator_file_name)
    if cached is not None:
        return cached

    url = GITHUB_GENERATORS_BASE_URL + generator_file_name
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "SimulationCraft-Orchestrator")
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8")
    _GENERATOR_FILE_CACHE[generator_file_name] = content
    return content


def _extract_generator_block(generator_content: str, profile_name: str) -> str | None:
    lines = generator_content.splitlines()
    current_block: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\s*\w+\s*=\s*"?MID1_[^"\s]+"?\s*$', line):
            current_block = [line]
            continue
        if current_block:
            current_block.append(line)
            if stripped.lower() == f"save={profile_name}.simc".lower():
                return "\n".join(current_block) + "\n"
    return None


def _extract_assignment_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key = stripped.split("=", 1)[0].strip()
    if not key:
        return None
    return key


def _merge_generator_values_into_profile(existing_profile: str, generator_block: str) -> str:
    """Replace value lines in existing profile using generator block assignments.

    Keeps non-value sections (notably APL) from the existing profile intact.
    """
    replacement_map: dict[str, str] = {}
    replacement_order: list[str] = []
    for raw_line in generator_block.splitlines():
        key = _extract_assignment_key(raw_line)
        if key is None:
            continue
        if key.lower() == "save":
            continue
        replacement_map[key] = raw_line.rstrip("\r\n")
        replacement_order.append(key)

    existing_lines = existing_profile.splitlines()
    seen_keys: set[str] = set()
    merged_lines: list[str] = []

    for line in existing_lines:
        key = _extract_assignment_key(line)
        if key in replacement_map:
            merged_lines.append(replacement_map[key])
            seen_keys.add(key)
        else:
            merged_lines.append(line)

    missing_keys = [k for k in replacement_order if k not in seen_keys]
    if missing_keys:
        insert_at = 0
        for idx, line in enumerate(merged_lines):
            if line.strip().startswith("actions"):
                insert_at = idx
                break
        else:
            insert_at = len(merged_lines)

        for offset, key in enumerate(missing_keys):
            merged_lines.insert(insert_at + offset, replacement_map[key])

    return "\n".join(merged_lines) + "\n"


def _merge_generator_values(profile_name: str, generated_content: str, require_generator_newer: bool = True) -> str:
    """Refresh value lines from class generator.

    If require_generator_newer is True, only apply when generator commit is newer.
    If False, apply whenever a matching generator block exists.
    """
    generator_file = _generator_file_name_for_profile_content(generated_content)
    if not generator_file:
        return generated_content

    generated_path = f"profiles/MID1/{profile_name}.simc"
    generator_path = f"profiles/generators/MID1/{generator_file}"
    generated_ts = _github_latest_commit_timestamp(generated_path)
    generator_ts = _github_latest_commit_timestamp(generator_path)

    if require_generator_newer:
        if generated_ts is None or generator_ts is None or generator_ts <= generated_ts:
            return generated_content

    try:
        generator_content = _fetch_generator_file(generator_file)
        generator_block = _extract_generator_block(generator_content, profile_name)
        if not generator_block:
            return generated_content

        before_map: dict[str, str] = {}
        after_source_map: dict[str, str] = {}
        for line in generated_content.splitlines():
            key = _extract_assignment_key(line)
            if key:
                before_map[key] = line.strip()
        for line in generator_block.splitlines():
            key = _extract_assignment_key(line)
            if key and key.lower() != "save":
                after_source_map[key] = line.strip()

        changed_keys = sorted(
            key for key, new_line in after_source_map.items()
            if before_map.get(key) != new_line
        )
        merged = _merge_generator_values_into_profile(generated_content, generator_block)
        if changed_keys:
            _append_jsonl(
                GENERATOR_MISMATCH_LOG_FILE,
                {
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "profile_name": profile_name,
                    "generator_file": generator_file,
                    "generated_profile_commit_timestamp": generated_ts.isoformat() if generated_ts else None,
                    "generator_commit_timestamp": generator_ts.isoformat() if generator_ts else None,
                    "changed_keys": changed_keys,
                    "changed_key_count": len(changed_keys),
                },
            )
        if merged != generated_content:
            print(
                f"  Info: {profile_name} generator is newer than generated profile; "
                "refreshed value lines from generator."
            )
        return merged
    except Exception:
        return generated_content


def _merge_generator_values_if_newer(profile_name: str, generated_content: str) -> str:
    return _merge_generator_values(profile_name, generated_content, require_generator_newer=True)


def load_spec_profile(profile_name: str, force_download: bool = False) -> str:
    cache_path = _profile_cache_path(profile_name)
    cached_content: str | None = None
    if not force_download and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached_content = f.read()
        if not _is_invalid_cached_profile(profile_name, cached_content):
            cached_ok, cached_err = _validate_profile_with_cache(cached_content)
            if cached_ok:
                return cached_content
            print(f"Warning: Cached profile {profile_name} is incompatible with local SimC: {cached_err}")
            print(f"  Refreshing {profile_name} from upstream sources...")
        print(f"Warning: Cached profile {profile_name} is invalid; refreshing from GitHub")

    os.makedirs(SPEC_PROFILES_DIR, exist_ok=True)
    try:
        content = _fetch_single_profile(profile_name)
    except Exception:
        # Fall back to bulk fetch if single download fails
        profile_map = _load_remote_profile_map(force_refresh=force_download)
        if profile_name not in profile_map:
            raise RuntimeError(f"Profile {profile_name} not found in MID1 profiles")
        content = profile_map[profile_name]

    is_valid, err_preview = _validate_profile_with_cache(content)
    if not is_valid:
        print(f"Warning: Downloaded profile {profile_name} is incompatible with local SimC: {err_preview}")
        compatible = _find_compatible_profile_revision(profile_name, max_commits=25)
        if compatible is not None:
            print(f"  Using latest compatible revision for {profile_name} from prior commit history")
            content = compatible
        elif cached_content and not _is_invalid_cached_profile(profile_name, cached_content):
            print(f"  Falling back to existing cached profile for {profile_name}")
            content = cached_content
        else:
            _append_jsonl(
                GENERATOR_MISMATCH_LOG_FILE,
                {
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "profile_name": profile_name,
                    "event": "no-compatible-profile-revision",
                    "simc_image": DEFAULT_SIMC_DOCKER_IMAGE,
                    "error_preview": err_preview,
                },
            )
            raise RuntimeError(
                f"No compatible profile revision found for {profile_name} with SimC image "
                f"{DEFAULT_SIMC_DOCKER_IMAGE}; update/build a newer SimC binary or pin "
                "profiles to a compatible branch."
            )

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def get_available_spec_profiles(force_refresh: bool = False) -> list[str]:
    global _AVAILABLE_SPEC_PROFILES_CACHE
    if _AVAILABLE_SPEC_PROFILES_CACHE is not None and not force_refresh:
        return list(_AVAILABLE_SPEC_PROFILES_CACHE)

    # Prefer local cached profiles to avoid GitHub API calls.
    # Only hit GitHub when force_refresh is True or no local cache exists.
    if not force_refresh:
        local_cached = sorted(
            p.stem for p in Path(SPEC_PROFILES_DIR).glob("MID1_*.simc") if p.is_file()
        )
        if local_cached:
            _AVAILABLE_SPEC_PROFILES_CACHE = local_cached
            return list(local_cached)

    try:
        names = sorted(_load_remote_profile_map(force_refresh=force_refresh).keys())
        if not names:
            raise RuntimeError("No MID1 profiles found on GitHub")
        _AVAILABLE_SPEC_PROFILES_CACHE = names
        return list(names)
    except Exception as exc:
        local_cached = sorted(
            p.stem for p in Path(SPEC_PROFILES_DIR).glob("MID1_*.simc") if p.is_file()
        )
        if local_cached:
            print(f"Warning: Failed to fetch profile list from GitHub, using local cache: {exc}")
            _AVAILABLE_SPEC_PROFILES_CACHE = local_cached
            return list(local_cached)
        raise RuntimeError(
            f"Could not fetch profile list from GitHub and no local cached profiles were found: {exc}"
        ) from exc


def download_all_profiles(force: bool = False) -> None:
    all_profiles = get_available_spec_profiles(force_refresh=True)
    print(f"Downloading {len(all_profiles)} spec profiles…")
    for i, name in enumerate(all_profiles, 1):
        cache_path = _profile_cache_path(name)
        if not force and os.path.exists(cache_path):
            print(f"  [{i}/{len(all_profiles)}] {name} (cached)")
            continue
        try:
            load_spec_profile(name, force_download=True)
            print(f"  [{i}/{len(all_profiles)}] {name} (downloaded)")
        except RuntimeError as exc:
            print(f"  [{i}/{len(all_profiles)}] {name} FAILED: {exc}")
    print("Profile download complete.")


def download_profiles_for_specs(spec_names: list[str], force: bool = True) -> None:
    """Download only the profiles needed for the given spec names."""
    print(f"Refreshing {len(spec_names)} spec profile(s) from GitHub…")
    for i, name in enumerate(spec_names, 1):
        try:
            load_spec_profile(name, force_download=force)
            print(f"  [{i}/{len(spec_names)}] {name} (downloaded)")
        except Exception as exc:
            print(f"  [{i}/{len(spec_names)}] {name} FAILED: {exc}")
    print("Profile refresh complete.")


def parse_spec_config(profile_name: str, profile_content: str) -> SpecConfig:
    char_match = re.search(r'^\s*(\w+)\s*=\s*"?([^"\s]+)"?\s*$', profile_content, re.MULTILINE)
    if not char_match:
        raise ValueError(f"Could not find character declaration in {profile_name}")
    class_keyword = char_match.group(1)
    character_name = char_match.group(2)

    spec_match = re.search(r'^spec=(\S+)', profile_content, re.MULTILINE)
    spec = spec_match.group(1) if spec_match else "unknown"

    race_match = re.search(r'^race=(\S+)', profile_content, re.MULTILINE)
    race = race_match.group(1) if race_match else "human"

    primary_stat = None
    stat_baseline: dict[str, int] = {}
    primary_candidates: dict[str, int] = {}

    def _resolve_primary_stat() -> str:
        if class_keyword.lower() in {"mage", "priest", "warlock", "evoker"}:
            return "intellect"
        if class_keyword.lower() in {"rogue", "hunter", "demonhunter"}:
            return "agility"
        return "strength"

    for stat in ("intellect", "strength", "agility"):
        m = re.search(rf'^\s*#?\s*gear_{stat}=(\d+)', profile_content, re.MULTILINE)
        if m:
            primary_candidates[stat] = int(m.group(1))
    if primary_candidates:
        primary_stat = max(primary_candidates, key=lambda s: primary_candidates[s])
        stat_baseline[primary_stat] = primary_candidates[primary_stat]
    if primary_stat is None:
        primary_stat = _resolve_primary_stat()
        stat_baseline[primary_stat] = DEFAULT_PRIMARY_BASELINE

    for key, gear_key in [
        ("crit", "crit_rating"),
        ("haste", "haste_rating"),
        ("mastery", "mastery_rating"),
        ("versatility", "versatility_rating"),
    ]:
        m = re.search(rf'^\s*#?\s*gear_{gear_key}=(\d+)', profile_content, re.MULTILINE)
        stat_baseline[key] = int(m.group(1)) if m else DEFAULT_SECONDARY_BASELINE

    return SpecConfig(
        profile_name=profile_name,
        class_keyword=class_keyword,
        character_name=character_name,
        spec=spec,
        race=race,
        primary_stat=primary_stat,
        stat_baseline=stat_baseline,
        profile_content=profile_content,
    )


def get_spec_config(profile_name: str, force_download: bool = False) -> SpecConfig:
    return parse_spec_config(profile_name, load_spec_profile(profile_name, force_download=force_download))



def _normalize_profile_content_for_signature(profile_content: str) -> str:
    # Ignore save/source directives since they are generator/emission controls, not simulation behavior.
    content = re.sub(r"^\s*save\s*=.*\n?", "", profile_content, flags=re.MULTILINE)
    content = re.sub(r"^\s*source\s*=.*\n?", "", content, flags=re.MULTILINE)
    # Normalize line endings and trim trailing whitespace for stable hashing.
    lines = [line.rstrip() for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def compute_simulation_signature(profile_content: str) -> str:
    normalized = _normalize_profile_content_for_signature(profile_content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _extract_game_data_signature(sim_json: dict[str, Any]) -> str | None:
    options = sim_json.get("sim", {}).get("options", {})
    dbc = options.get("dbc") if isinstance(options, dict) else None
    if not isinstance(dbc, dict):
        return None

    version_used = dbc.get("version_used")
    selected = dbc.get(version_used) if isinstance(version_used, str) else None
    selected_dict = selected if isinstance(selected, dict) else {}

    signature_payload = {
        "version_used": version_used,
        "selected_build_level": selected_dict.get("build_level"),
        "selected_hotfix_date": selected_dict.get("hotfix_date"),
        "selected_hotfix_build": selected_dict.get("hotfix_build"),
        "selected_hotfix_hash": selected_dict.get("hotfix_hash"),
        "live_build_level": dbc.get("Live", {}).get("build_level") if isinstance(dbc.get("Live"), dict) else None,
        "live_hotfix_hash": dbc.get("Live", {}).get("hotfix_hash") if isinstance(dbc.get("Live"), dict) else None,
        "ptr_build_level": dbc.get("PTR", {}).get("build_level") if isinstance(dbc.get("PTR"), dict) else None,
        "ptr_hotfix_hash": dbc.get("PTR", {}).get("hotfix_hash") if isinstance(dbc.get("PTR"), dict) else None,
    }
    payload_text = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def detect_game_data_signature_in_dir(results_dir: str) -> str | None:
    json_files = sorted(Path(results_dir).glob("*.json"))
    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            signature = _extract_game_data_signature(data)
            if signature:
                return signature
        except Exception:
            continue
    return None


def detect_game_data_signature_from_chunk_plans(chunk_plans: list[ChunkPlan]) -> str | None:
    for cp in chunk_plans:
        signature = detect_game_data_signature_in_dir(cp.local_chunk_dir)
        if signature:
            return signature
    return None


def _load_profile_metadata() -> dict[str, Any]:
    if not os.path.exists(PROFILE_METADATA_FILE):
        return {}
    try:
        with open(PROFILE_METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_profile_metadata(metadata: dict[str, Any]) -> None:
    with open(PROFILE_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _extract_simc_version_info(sim_json: dict[str, Any]) -> dict[str, str | None]:
    sim = sim_json.get("sim", {}) if isinstance(sim_json, dict) else {}
    return {
        "simc_version": sim_json.get("version"),
        "simc_git_revision": sim_json.get("git_revision"),
        "simc_build_date": sim_json.get("build_date"),
        "simc_report_version": sim_json.get("report_version"),
        "simc_timestamp": sim_json.get("timestamp"),
        "dbc_version_used": (
            sim.get("options", {}).get("dbc", {}).get("version_used")
            if isinstance(sim, dict)
            else None
        ),
    }


def generate_staleness_drift_plot(
    history_log_path: str = DRIFT_HISTORY_LOG_FILE,
    output_plot_path: str = os.path.join(DEFAULT_EVALUATION_PLOTS_DIR, "drift_over_time.png"),
) -> None:
    """Generate drift-over-time chart from persisted drift history JSONL."""
    if not os.path.exists(history_log_path):
        return

    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    with open(history_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = str(row.get("entry_type", "")).strip().lower()
            if entry_type == "check_summary":
                if "check_timestamp" in row and "spec_name" in row and "median_abs_drift_pct" in row:
                    summary_rows.append(row)
                continue
            if "check_timestamp" in row and "spec_name" in row and "drift_pct" in row:
                sample_rows.append(row)

    if not summary_rows and not sample_rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df["check_timestamp"] = pd.to_datetime(df["check_timestamp"], errors="coerce", utc=True)
        df["median_abs_drift_pct"] = pd.to_numeric(df["median_abs_drift_pct"], errors="coerce")
        df = df.dropna(subset=["check_timestamp", "median_abs_drift_pct", "spec_name"])
        if df.empty:
            return
        grouped = (
            df.groupby(["check_timestamp", "spec_name"], as_index=False)["median_abs_drift_pct"]
            .median()
            .rename(columns={"median_abs_drift_pct": "median_drift_pct"})
        )
    else:
        # Backward compatibility for existing logs that only have sample rows.
        df = pd.DataFrame(sample_rows)
        df["check_timestamp"] = pd.to_datetime(df["check_timestamp"], errors="coerce", utc=True)
        df["drift_pct"] = pd.to_numeric(df["drift_pct"], errors="coerce")
        df = df.dropna(subset=["check_timestamp", "drift_pct", "spec_name"])
        if df.empty:
            return
        grouped = (
            df.groupby(["check_timestamp", "spec_name"], as_index=False)["drift_pct"]
            .median()
            .rename(columns={"drift_pct": "median_drift_pct"})
        )
    latest_by_spec = grouped.sort_values("check_timestamp").groupby("spec_name").tail(1)
    top_specs = latest_by_spec.reindex(
        latest_by_spec["median_drift_pct"].abs().sort_values(ascending=False).index
    )["spec_name"].head(10).tolist()

    overall = grouped.groupby("check_timestamp", as_index=False)["median_drift_pct"].median()

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.figure(figsize=(12, 6))

    for spec_name in top_specs:
        sdf = grouped[grouped["spec_name"] == spec_name].sort_values("check_timestamp")
        plt.plot(sdf["check_timestamp"], sdf["median_drift_pct"], linewidth=1.2, alpha=0.75, label=spec_name)

    plt.plot(
        overall["check_timestamp"],
        overall["median_drift_pct"],
        color="black",
        linewidth=2.4,
        linestyle="--",
        label="overall median",
    )
    plt.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
    plt.title("Staleness Drift Over Time (Median %)")
    plt.xlabel("Check Timestamp (UTC)")
    plt.ylabel("Drift %")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=140)
    plt.close()


def detect_worker_image_signature(image_ref: str = DEFAULT_SIMC_DOCKER_IMAGE) -> str | None:
    try:
        proc = _run_command(
            ["docker", "image", "inspect", image_ref, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    image_id = (proc.stdout or "").strip()
    if not image_id:
        return None
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()


def preflight_pull_simc_image(image_ref: str = DEFAULT_SIMC_DOCKER_IMAGE) -> None:
    print(f"Preflight: pulling Docker image {image_ref}...")
    try:
        proc = _run_command(["docker", "pull", image_ref], capture_output=True, text=True)
    except Exception as exc:
        print(f"  WARNING: Docker pull failed for {image_ref}: {exc}")
        return

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        err_preview = err[-1] if err else f"exit={proc.returncode}"
        print(f"  WARNING: Docker pull failed for {image_ref}: {err_preview}")
        return

    print(f"  Docker image ready: {image_ref}")


def mark_profile_generated(
    profile_name: str,
    simulation_signature: str | None = None,
    game_data_signature: str | None = None,
    worker_image_signature: str | None = None,
) -> None:
    metadata = _load_profile_metadata()
    metadata.setdefault(profile_name, {})
    metadata[profile_name]["last_generated_timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    if simulation_signature:
        metadata[profile_name]["last_generated_simulation_signature"] = simulation_signature
    if game_data_signature:
        metadata[profile_name]["last_generated_game_data_signature"] = game_data_signature
    if worker_image_signature:
        metadata[profile_name]["last_generated_worker_image_signature"] = worker_image_signature
    metadata[profile_name]["is_stale"] = False
    # Clear old staleness drift data — it's no longer relevant after regeneration
    metadata[profile_name].pop("staleness_drift_pct", None)
    metadata[profile_name].pop("staleness_checked_at", None)
    _save_profile_metadata(metadata)


def _run_local_sample_sims(work_dir: str, timeout_seconds: int) -> subprocess.CompletedProcess:
    abs_work_dir = os.path.abspath(work_dir)
    env = dict(os.environ)
    env["MSYS_NO_PATHCONV"] = "1"
    env["MSYS2_ARG_CONV_EXCL"] = "*"
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            "-v",
            f"{abs_work_dir}:/work",
            "-w",
            "/work",
            DEFAULT_SIMC_DOCKER_IMAGE,
            "-c",
            "for f in ./*.simc; do /app/SimulationCraft/simc \"$f\" || exit $?; done",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        env=env,
    )


def _submit_staleness_job(
    batch_client,
    *,
    queue_name: str,
    job_definition_name: str,
    spec_name: str,
    manifest_key: str,
    bucket: str,
    region: str,
    worker_parallel: int,
) -> str:
    env = [
        {"name": "SIMC_S3_BUCKET", "value": bucket},
        {"name": "SIMC_MANIFEST_KEY", "value": manifest_key},
        {"name": "SIMC_AWS_REGION", "value": region},
        {"name": "SIMC_SPEC_NAME", "value": spec_name},
        {"name": "SIMC_WORKER_PARALLEL", "value": str(worker_parallel)},
    ]
    raw_job_name = f"stalecheck-{spec_name.lower().replace('_', '-')}"
    submit_kwargs = {
        "jobName": sanitize_batch_job_name(raw_job_name, fallback="stalecheck"),
        "jobQueue": queue_name,
        "jobDefinition": job_definition_name,
        "containerOverrides": {"environment": env},
    }
    resp = batch_client.submit_job(**submit_kwargs)
    return resp["jobId"]


def _run_remote_sample_sims(args: argparse.Namespace, spec_name: str, work_dir: str) -> None:
    s3_client, batch_client = build_boto_clients(args.aws_region)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = s3_key_join(args.staleness_s3_prefix, f"run-{timestamp}", spec_name)
    input_key = s3_key_join(run_prefix, "inputs", "chunk_0000.zip")
    output_key = s3_key_join(run_prefix, "outputs", "chunk_0000.zip")
    manifest_key = s3_key_join(run_prefix, "manifest.json")
    keys_to_cleanup = [manifest_key, input_key, output_key]

    input_zip_path = os.path.join(work_dir, "chunk_0000_input.zip")
    zip_simc_inputs(work_dir, input_zip_path)

    manifest = {
        "run_prefix": run_prefix,
        "spec": spec_name,
        "chunk_count": 1,
        "worker_parallel": args.worker_parallel,
        "chunks": [
            {
                "chunk_id": 0,
                "input_key": input_key,
                "output_key": output_key,
            }
        ],
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(manifest, tmp)
        manifest_path = tmp.name

    try:
        s3_client.upload_file(input_zip_path, args.s3_bucket, input_key)
        s3_client.upload_file(
            manifest_path,
            args.s3_bucket,
            manifest_key,
            ExtraArgs={"ContentType": "application/json"},
        )

        job_id = _submit_staleness_job(
            batch_client,
            queue_name=args.staleness_batch_job_queue,
            job_definition_name=args.staleness_batch_job_definition,
            spec_name=spec_name,
            manifest_key=manifest_key,
            bucket=args.s3_bucket,
            region=args.aws_region,
            worker_parallel=args.worker_parallel,
        )
        parent = wait_for_array_parent(batch_client, job_id, args.poll_seconds)
        if parent["status"] != "SUCCEEDED":
            raise RuntimeError(f"staleness Batch job failed for {spec_name} (job={job_id})")

        local_output_zip = os.path.join(work_dir, "chunk_0000_output.zip")
        if not object_exists(s3_client, args.s3_bucket, output_key):
            raise RuntimeError(f"missing staleness output object s3://{args.s3_bucket}/{output_key}")
        s3_client.download_file(args.s3_bucket, output_key, local_output_zip)
        extract_zip_to_dir(local_output_zip, work_dir)
    finally:
        try:
            os.remove(manifest_path)
        except OSError:
            pass
        delete_s3_keys(s3_client, args.s3_bucket, keys_to_cleanup)


def _evaluate_profile_sample_drift(args: argparse.Namespace, spec_name: str) -> dict[str, Any] | None:
    spec_csv = os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{spec_name}.csv")
    if not os.path.exists(spec_csv):
        return None

    df = _normalize_spec_dataframe(pd.read_csv(spec_csv, encoding="utf-8"))
    required_cols = ["primary_stat", "crit", "haste", "mastery", "versatility", "dps"]
    if any(col not in df.columns for col in required_cols):
        return None

    sample_df = df[required_cols].dropna()
    if sample_df.empty:
        return None

    sample_size = min(int(args.staleness_sample_size), len(sample_df))
    if sample_size <= 0:
        return None

    sampled = sample_df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)

    spec_config = get_spec_config(spec_name)

    timeout_seconds = max(30, 20 * sample_size)
    drifts_pct: list[float] = []
    drift_samples: list[dict[str, Any]] = []
    simc_version_info: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=f"stalecheck_{spec_name}_") as temp_work_dir:
        work_dir = os.fspath(temp_work_dir)
        if isinstance(work_dir, bytes):
            work_dir = os.fsdecode(work_dir)
        profile_names: list[str] = []
        original_dps: dict[str, float] = {}
        for idx, (_, row) in enumerate(sampled.iterrows()):
            profile_idx = 2_000_000 + idx
            stat_overrides = {
                spec_config.primary_stat: int(row["primary_stat"]),
                "crit": int(row["crit"]),
                "haste": int(row["haste"]),
                "mastery": int(row["mastery"]),
                "versatility": int(row["versatility"]),
            }
            profile_name = write_simulation_profile(
                profile_idx,
                spec_config,
                stat_overrides,
                int(args.staleness_sample_iterations),
                work_dir,
            )
            profile_names.append(profile_name)
            original_dps[profile_name] = float(row["dps"])

        if args.staleness_execution == "remote":
            _run_remote_sample_sims(args, spec_name, work_dir)
        else:
            run_result = _run_local_sample_sims(work_dir, timeout_seconds=timeout_seconds)
            if run_result.returncode != 0:
                err_preview = (run_result.stderr or run_result.stdout or "").strip().splitlines()
                err_text = err_preview[-1] if err_preview else f"exit={run_result.returncode}"
                raise RuntimeError(f"local sample sims failed for {spec_name}: {err_text}")

        for profile_name in profile_names:
            old_dps = original_dps.get(profile_name)
            if old_dps is None or old_dps <= 0:
                continue

            json_path = os.path.join(work_dir, f"{profile_name}.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if not simc_version_info:
                simc_version_info = _extract_simc_version_info(data)
            new_dps = float(data["sim"]["players"][0]["collected_data"]["dps"]["mean"])
            drift_pct = (new_dps - old_dps) / old_dps * 100.0
            drifts_pct.append(drift_pct)
            drift_samples.append({
                "profile_name": profile_name,
                "old_dps": old_dps,
                "new_dps": new_dps,
                "drift_pct": drift_pct,
            })

    if not drifts_pct:
        return None

    median_abs_drift = float(abs(np.median(drifts_pct)))
    max_abs_drift = float(np.max(np.abs(drifts_pct)))
    return {
        "median_abs_drift_pct": median_abs_drift,
        "max_abs_drift_pct": max_abs_drift,
        "compared": len(drifts_pct),
        "drift_samples": drift_samples,
        "simc_version_info": simc_version_info,
    }


def get_stale_profile_reasons_from_sample(args: argparse.Namespace) -> dict[str, list[str]]:
    stale_reasons: dict[str, list[str]] = {}
    drift_results: dict[str, float] = {}  # profile -> median_abs_drift_pct
    threshold = float(args.staleness_drift_threshold_pct)
    staleness_run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")

    existing_profiles = [
        spec for spec in get_available_spec_profiles()
        if os.path.exists(os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{spec}.csv"))
    ]

    # If --specs was given with explicit names, scope the check to those only.
    # Include requested specs that have training data even if they aren't in the
    # cached profile listing (handles names with special chars like San'layn).
    if args.specs and args.specs.strip().lower() not in ("all", "existing", ""):
        requested = {s.strip() for s in args.specs.split(",") if s.strip()}
        existing_set = set(existing_profiles)
        for name in requested:
            if name not in existing_set and os.path.exists(
                os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{name}.csv")
            ):
                existing_profiles.append(name)
        existing_profiles = [p for p in existing_profiles if p in requested]

    print(
        "Running sample-based staleness check "
        f"(profiles={len(existing_profiles)}, sample_size={args.staleness_sample_size}, "
        f"iterations={args.staleness_sample_iterations}, threshold={threshold:.2f}% median drift)…"
    )

    for idx, spec_name in enumerate(existing_profiles, 1):
        try:
            drift = _evaluate_profile_sample_drift(args, spec_name)
        except Exception as exc:
            print(f"  [{idx}/{len(existing_profiles)}] {spec_name}: sample check failed ({exc})")
            continue

        if drift is None:
            print(f"  [{idx}/{len(existing_profiles)}] {spec_name}: skipped (insufficient data)")
            continue

        median_abs_drift = float(drift["median_abs_drift_pct"])
        max_abs_drift = float(drift["max_abs_drift_pct"])
        compared = int(drift["compared"])
        drift_results[spec_name] = median_abs_drift
        print(
            f"  [{idx}/{len(existing_profiles)}] {spec_name}: "
            f"median_abs={median_abs_drift:.2f}% max_abs={max_abs_drift:.2f}% n={compared}"
        )
        check_timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        check_id = f"{staleness_run_id}:{spec_name}"
        is_stale = median_abs_drift >= threshold
        for sample_idx, sample in enumerate(drift.get("drift_samples", [])):
            payload = {
                "entry_type": "sample",
                "staleness_run_id": staleness_run_id,
                "check_id": check_id,
                "check_timestamp": check_timestamp,
                "spec_name": spec_name,
                "sample_index": sample_idx,
                "sample_profile_name": sample.get("profile_name"),
                "drift_pct": round(float(sample.get("drift_pct", 0.0)), 6),
                "old_dps": float(sample.get("old_dps", 0.0)),
                "new_dps": float(sample.get("new_dps", 0.0)),
                "median_abs_drift_pct": round(median_abs_drift, 6),
                "max_abs_drift_pct": round(max_abs_drift, 6),
                "threshold_pct": round(threshold, 6),
                "is_stale": is_stale,
                "sample_size": int(args.staleness_sample_size),
                "sample_iterations": int(args.staleness_sample_iterations),
                "execution_mode": str(args.staleness_execution),
            }
            payload.update(drift.get("simc_version_info", {}))
            _append_jsonl(DRIFT_HISTORY_LOG_FILE, payload)

        summary_payload = {
            "entry_type": "check_summary",
            "staleness_run_id": staleness_run_id,
            "check_id": check_id,
            "check_timestamp": check_timestamp,
            "spec_name": spec_name,
            "median_abs_drift_pct": round(median_abs_drift, 6),
            "max_abs_drift_pct": round(max_abs_drift, 6),
            "threshold_pct": round(threshold, 6),
            "is_stale": is_stale,
            "sample_size": int(args.staleness_sample_size),
            "sample_iterations": int(args.staleness_sample_iterations),
            "execution_mode": str(args.staleness_execution),
            "compared": compared,
        }
        summary_payload.update(drift.get("simc_version_info", {}))
        _append_jsonl(DRIFT_HISTORY_LOG_FILE, summary_payload)

        if median_abs_drift >= threshold:
            stale_reasons[spec_name] = [
                "sample-drift-detected",
                f"median_abs_drift_pct={median_abs_drift:.2f}",
            ]

    # Update profile metadata with drift results so the dashboard can display them
    check_timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    metadata = _load_profile_metadata()
    for profile_name in existing_profiles:
        metadata.setdefault(profile_name, {})
        metadata[profile_name]["is_stale"] = profile_name in stale_reasons
        metadata[profile_name]["staleness_checked_at"] = check_timestamp
        if profile_name in drift_results:
            metadata[profile_name]["staleness_drift_pct"] = round(drift_results[profile_name], 2)
    _save_profile_metadata(metadata)
    generate_staleness_drift_plot()

    return stale_reasons


def generate_stat_overrides(rng: np.random.Generator, spec_config: SpecConfig) -> dict[str, int]:
    baseline = spec_config.stat_baseline
    primary_stat = spec_config.primary_stat

    primary_baseline = baseline[primary_stat]
    primary_low = max(1, int(primary_baseline * (1 - PRIMARY_SWING_DOWN_PCT)))
    primary_high = max(primary_low + 1, int(primary_baseline * (1 + PRIMARY_SWING_UP_PCT)))
    primary_val = int(rng.integers(primary_low, primary_high + 1))

    secondary_total = sum(baseline.get(s, 0) for s in SECONDARY_STATS)
    secondary_total = max(secondary_total, SECONDARY_BUDGET_MINIMUM)
    budget_low = max(100, int(secondary_total * (1 - SECONDARY_SWING_DOWN_PCT)))
    budget_high = max(budget_low + 1, int(secondary_total * (1 + SECONDARY_SWING_UP_PCT)))
    target_budget = int(rng.integers(budget_low, budget_high + 1))

    weights = rng.random(len(SECONDARY_STATS))
    if float(np.sum(weights)) == 0.0:
        weights = np.ones(len(SECONDARY_STATS), dtype=float)
    weights = weights / np.sum(weights)

    raw_alloc = weights * target_budget
    secondary_values = np.floor(raw_alloc).astype(int)
    remainder = target_budget - int(np.sum(secondary_values))
    if remainder > 0:
        fractional = raw_alloc - secondary_values
        order = np.argsort(fractional)[::-1]
        secondary_values[order[:remainder]] += 1

    stats = {primary_stat: primary_val}
    for idx, stat_name in enumerate(SECONDARY_STATS):
        stats[stat_name] = int(secondary_values[idx])
    return stats


def write_simulation_profile(index: int, spec_config: SpecConfig, stat_overrides: dict[str, int], iterations: int, output_dir: str) -> str:
    profile_name = f"sim_{index}"
    json_output_path = f"{profile_name}.json"
    unique_char_name = f"{spec_config.character_name}_{index}"

    # Strip generator directives that are not valid/needed for worker simulation runs.
    base_profile_content = re.sub(r'^\s*source\s*=.*\n?', '', spec_config.profile_content, flags=re.MULTILINE)
    modified_content = re.sub(
        rf'(^\s*(?!#){re.escape(spec_config.class_keyword)}\s*=\s*"){re.escape(spec_config.character_name)}(")',
        rf'\g<1>{unique_char_name}\g<2>',
        base_profile_content,
        flags=re.MULTILINE,
    )
    if modified_content == base_profile_content:
        modified_content = re.sub(
            rf'(^\s*(?!#){re.escape(spec_config.class_keyword)}\s*=\s*){re.escape(spec_config.character_name)}(\s*$)',
            rf'\g<1>{unique_char_name}\g<2>',
            base_profile_content,
            flags=re.MULTILINE,
        )
    # Generator profiles include save=... directives used for profile emission;
    # keep them out of worker run inputs so SimC executes simulations and writes json2 outputs.
    modified_content = re.sub(r'^\s*save\s*=.*\n?', '', modified_content, flags=re.MULTILINE)

    primary_stat = spec_config.primary_stat
    override_block = (
        f"\n# --- stat overrides for sim {index} ---\n"
        f"gear_{primary_stat}={stat_overrides[primary_stat]}\n"
        f"gear_crit_rating={stat_overrides['crit']}\n"
        f"gear_haste_rating={stat_overrides['haste']}\n"
        f"gear_mastery_rating={stat_overrides['mastery']}\n"
        f"gear_versatility_rating={stat_overrides['versatility']}\n"
        f"iterations={iterations}\n"
        "target_error=0.1\n"
        f"json2={json_output_path}\n"
    )

    simc_content = modified_content.rstrip() + "\n" + override_block
    simc_file = os.path.join(output_dir, f"{profile_name}.simc")
    with open(simc_file, "w", encoding="utf-8") as f:
        f.write(simc_content)
    return profile_name

def _stat_key(overrides: dict[str, int]) -> tuple:
    """Return a hashable key for a stat combination (for dedup).

    The primary stat name varies by spec (intellect/strength/agility) but only
    one will be present in a given overrides dict, so we just sum all three
    possible keys (two of which will be 0).
    """
    primary = (overrides.get("intellect", 0)
               + overrides.get("strength", 0)
               + overrides.get("agility", 0))
    return (primary, overrides["crit"], overrides["haste"],
            overrides["mastery"], overrides["versatility"])


def load_existing_stat_keys(spec_name: str) -> set[tuple]:
    """Load stat combos already saved for a spec, for pre-generation dedup."""
    spec_csv = os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{spec_name}.csv")
    if not os.path.exists(spec_csv):
        return set()
    try:
        df = pd.read_csv(spec_csv, encoding="utf-8")
        keys: set[tuple] = set()
        for _, row in df.iterrows():
            keys.add((
                int(row.get("primary_stat", 0)),
                int(row.get("crit", 0)),
                int(row.get("haste", 0)),
                int(row.get("mastery", 0)),
                int(row.get("versatility", 0)),
            ))
        return keys
    except Exception:
        return set()


def generate_profiles_for_range(start_index: int, sample_count: int, seed: int, iterations: int, output_dir: str, spec_config: SpecConfig, existing_keys: set[tuple] | None = None) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)

    # When continuing from existing data, offset the seed so the RNG produces
    # a fresh sequence instead of replaying indices that already exist.
    effective_seed = seed + start_index
    stat_rng = np.random.default_rng(effective_seed)

    seen: set[tuple] = set(existing_keys) if existing_keys else set()
    profile_names: list[str] = []
    sim_index = start_index
    max_attempts = sample_count * 10
    attempts = 0
    while len(profile_names) < sample_count and attempts < max_attempts:
        stat_overrides = generate_stat_overrides(stat_rng, spec_config)
        attempts += 1
        key = _stat_key(stat_overrides)
        if key in seen:
            sim_index += 1
            continue
        seen.add(key)
        profile_name = write_simulation_profile(sim_index, spec_config, stat_overrides, iterations, output_dir)
        profile_names.append(profile_name)
        sim_index += 1

    if len(profile_names) < sample_count:
        print(f"  WARNING: Could only generate {len(profile_names)}/{sample_count} unique profiles "
              f"after {attempts} attempts (stat space may be saturated)")
    return profile_names


def collect_results(spec_config: SpecConfig, target_profiles: set[str] | list[str] | None = None,
                    csv_path: str | None = None, results_dir: str = ".",
                    mark_generated_flag: bool = False) -> pd.DataFrame:
    primary_stat_name = spec_config.primary_stat
    spec_name = spec_config.profile_name

    def _extract_override_stats(simc_path: str) -> dict[str, int | None]:
        with open(simc_path, encoding="utf-8") as f:
            text = f.read()
        override_section = text
        marker = "# --- stat overrides for sim "
        marker_idx = text.rfind(marker)
        if marker_idx != -1:
            override_section = text[marker_idx:]

        def _last_active_value(section: str, key: str) -> int | None:
            matches = re.findall(rf"^\s*{re.escape(key)}=(\d+)\s*$", section, re.MULTILINE)
            if matches:
                return int(matches[-1])
            matches = re.findall(rf"^\s*{re.escape(key)}=(\d+)\s*$", text, re.MULTILINE)
            return int(matches[-1]) if matches else None

        pval = _last_active_value(override_section, f"gear_{primary_stat_name}")
        crit = _last_active_value(override_section, "gear_crit_rating")
        haste = _last_active_value(override_section, "gear_haste_rating")
        mastery = _last_active_value(override_section, "gear_mastery_rating")
        vers = _last_active_value(override_section, "gear_versatility_rating")
        return {
            "primary_stat": pval,
            "crit": crit,
            "haste": haste,
            "mastery": mastery,
            "versatility": vers,
        }

    profile_filter = set(target_profiles) if target_profiles else None
    collected_names: set[str] = set()
    dataset: list[dict[str, Any]] = []

    for file in sorted(os.listdir(results_dir)):
        if not file.endswith(".json"):
            continue
        profile_name = file[:-5]
        if profile_filter is not None and profile_name not in profile_filter:
            continue
        collected_names.add(profile_name)
        json_path = os.path.join(results_dir, file)
        simc_path = os.path.join(results_dir, file.replace(".json", ".simc"))
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        player = data["sim"]["players"][0]
        dps_stats = player["collected_data"]["dps"]
        dps = dps_stats["mean"]
        dps_mean_std_dev = dps_stats.get("mean_std_dev")
        dps_std_dev = dps_stats.get("std_dev")
        stats = _extract_override_stats(simc_path)

        dataset.append({
            "spec": spec_name,
            "primary_stat": stats.get("primary_stat"),
            "crit": stats.get("crit"),
            "haste": stats.get("haste"),
            "mastery": stats.get("mastery"),
            "versatility": stats.get("versatility"),
            "dps": dps,
            "dps_mean_std_dev": dps_mean_std_dev,
            "dps_std_dev": dps_std_dev,
            "dps_rel_mean_std_dev_pct": (float(dps_mean_std_dev) / float(dps) * 100.0)
            if dps_mean_std_dev is not None and dps else None,
        })

    if profile_filter is not None:
        missing = sorted(profile_filter - collected_names)
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(f"Missing {len(missing)} expected JSON output(s), first few: {preview}")

    df = pd.DataFrame(dataset)
    if df.empty:
        raise RuntimeError("No JSON results found. Check worker execution and downloaded shard paths.")

    if csv_path:
        csv_parent = os.path.dirname(csv_path)
        if csv_parent:
            os.makedirs(csv_parent, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    if mark_generated_flag:
        worker_image_signature = detect_worker_image_signature()
        mark_profile_generated(
            spec_config.profile_name,
            simulation_signature=compute_simulation_signature(spec_config.profile_content),
            game_data_signature=detect_game_data_signature_in_dir(results_dir),
            worker_image_signature=worker_image_signature,
        )
    return df


def _normalize_spec_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Migrate legacy format: primary_stat held stat name and one of
    # intellect/strength/agility held the numeric value.
    if "primary_stat" in df.columns and is_string_dtype(df["primary_stat"]):
        lowered = df["primary_stat"].astype(str).str.lower()
        name_candidates = {"intellect", "strength", "agility"}
        has_source_columns = any(c in df.columns for c in ("intellect", "strength", "agility"))
        if lowered.isin(name_candidates).any() and has_source_columns:
            primary_values = pd.Series([None] * len(df), index=df.index, dtype="float64")
            for stat_name in ("intellect", "strength", "agility"):
                if stat_name in df.columns:
                    mask = lowered == stat_name
                    primary_values.loc[mask] = pd.to_numeric(df.loc[mask, stat_name], errors="coerce")
            df = df.copy()
            df["primary_stat"] = primary_values

    if "primary_stat" in df.columns:
        df = df.copy()
        try:
            df["primary_stat"] = pd.to_numeric(df["primary_stat"], errors="raise")
        except (ValueError, TypeError):
            pass

    return df.drop(columns=[c for c in ("intellect", "strength", "agility") if c in df.columns], errors="ignore")


def get_saved_sim_count(spec_name: str) -> int:
    spec_csv = os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{spec_name}.csv")
    if not os.path.exists(spec_csv):
        return 0
    try:
        return len(_normalize_spec_dataframe(pd.read_csv(spec_csv, encoding="utf-8")))
    except Exception as exc:
        print(f"  WARNING: Could not read saved CSV for {spec_name}: {exc}")
        return 0


def compute_per_spec_sample_targets(
    metadata_path: str,
    baseline: int,
    extra_budget: int,
) -> dict[str, int]:
    """Compute per-spec sample targets based on model per-spec MAE.

    Specs at or below the median MAE keep the baseline target.
    Specs above the median get extra samples proportional to how far
    above they are.  The ``extra_budget`` (e.g. 200) is the maximum
    number of *new* samples the single worst spec receives; better-but-
    still-above-median specs receive a proportionally smaller share.
    There is no upper cap on accumulated samples.
    """
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Find per_spec_mae — check top-level first (local training), then
    # fall back to ensemble_candidates lookup (SageMaker training).
    per_spec_mae: dict[str, float] | None = metadata.get("per_spec_mae")
    if not per_spec_mae:
        production_trial = metadata.get("production_trial_number")
        for candidate in metadata.get("ensemble_candidates", []):
            if candidate.get("trial_number") == production_trial:
                per_spec_mae = candidate.get("per_spec_mae")
                break

    if not per_spec_mae:
        print("WARNING: Could not find production trial per_spec_mae in metadata; using flat baseline")
        return {}

    mae_values = sorted(per_spec_mae.values())
    median_mae = float(np.median(mae_values))
    max_mae = max(mae_values)

    targets: dict[str, int] = {}
    boosted_count = 0
    for spec_name, mae in per_spec_mae.items():
        if mae <= median_mae or max_mae <= median_mae:
            targets[spec_name] = baseline
        else:
            ratio = (mae - median_mae) / (max_mae - median_mae)
            extra = int(extra_budget * ratio)
            targets[spec_name] = baseline + extra
            if extra > 0:
                boosted_count += 1

    print(f"\nPer-spec sample targets (from {metadata_path}):")
    print(f"  Median MAE: {median_mae:.1f}, Max MAE: {max_mae:.1f}")
    print(f"  Baseline: {baseline}, Extra budget: {extra_budget}")
    print(f"  Specs boosted above baseline: {boosted_count}/{len(targets)}")
    for spec_name in sorted(targets, key=lambda s: targets[s], reverse=True):
        mae = per_spec_mae[spec_name]
        target = targets[spec_name]
        if target > baseline:
            print(f"    {spec_name:50s} MAE={mae:8.1f}  target={target}")

    return targets


def rebuild_combined_training_csv() -> str:
    """Concatenate all per-spec training CSVs into all_specs_training_data.csv."""
    combined_path = os.path.join(DEFAULT_TRAINING_DATA_DIR, "..", "all_specs_training_data.csv")
    combined_path = os.path.normpath(combined_path)

    spec_csvs = sorted(glob.glob(os.path.join(DEFAULT_TRAINING_DATA_DIR, "*.csv")))
    if not spec_csvs:
        raise FileNotFoundError(f"No per-spec CSVs found in {DEFAULT_TRAINING_DATA_DIR}")

    frames = []
    for csv_path in spec_csvs:
        df = pd.read_csv(csv_path, encoding="utf-8")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(combined_path, index=False, encoding="utf-8")

    print(f"\nRebuilt combined training CSV: {combined_path}")
    print(f"  {len(spec_csvs)} spec files, {len(combined)} total rows")
    return combined_path


def clear_saved_spec_outputs(spec_name: str, shard_dir: str) -> None:
    spec_csv = os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{spec_name}.csv")
    shard_spec_dir = os.path.join(shard_dir, spec_name)
    if os.path.exists(spec_csv):
        os.remove(spec_csv)
    if os.path.isdir(shard_spec_dir):
        shutil.rmtree(shard_spec_dir)


def wipe_all_data(shard_dir: str) -> None:
    """Delete ALL training data, shards, profile metadata, model artifacts, and cached profiles."""
    removed = []

    # Training data CSVs
    if os.path.isdir(DEFAULT_TRAINING_DATA_DIR):
        for f in os.listdir(DEFAULT_TRAINING_DATA_DIR):
            fp = os.path.join(DEFAULT_TRAINING_DATA_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
                removed.append(f"training_data/{f}")
        print(f"  Cleared training_data/ ({len(removed)} files)")

    # Distributed shards
    shard_count = 0
    if os.path.isdir(shard_dir):
        for entry in os.listdir(shard_dir):
            entry_path = os.path.join(shard_dir, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
                shard_count += 1
        print(f"  Cleared shard directory ({shard_count} spec dirs)")

    # Combined CSV
    combined_csv = os.path.join(os.path.dirname(DEFAULT_TRAINING_DATA_DIR), "all_specs_training_data.csv")
    if os.path.exists(combined_csv):
        os.remove(combined_csv)
        print("  Removed all_specs_training_data.csv")

    # Profile metadata (staleness/drift tracking)
    if os.path.exists(PROFILE_METADATA_FILE):
        os.remove(PROFILE_METADATA_FILE)
        print("  Removed profile_metadata.json")

    # Model artifacts (deep_nn/)
    model_dir = os.path.join("local", "nn_website_model", "deep_nn")
    model_count = 0
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            fp = os.path.join(model_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
                model_count += 1
        print(f"  Cleared model artifacts ({model_count} files)")

    # Evaluation plots
    plots_dir = os.path.join("local", "nn_website_model", "evaluation_plots")
    plot_count = 0
    if os.path.isdir(plots_dir):
        for f in os.listdir(plots_dir):
            fp = os.path.join(plots_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
                plot_count += 1
        print(f"  Cleared evaluation plots ({plot_count} files)")

    # Cached spec profiles
    cache_count = 0
    if os.path.isdir(SPEC_PROFILES_DIR):
        for f in os.listdir(SPEC_PROFILES_DIR):
            fp = os.path.join(SPEC_PROFILES_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
                cache_count += 1
        print(f"  Cleared spec_profiles/ cache ({cache_count} files)")

    print("  Wipe complete. All simulation data has been removed.")


def zip_simc_inputs(chunk_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for simc_path in sorted(Path(chunk_dir).glob("*.simc")):
            zf.write(simc_path, arcname=simc_path.name)


def extract_zip_to_dir(zip_path: str, target_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


# ── AWS BATCH + S3 HELPERS ───────────────────────────────────────────────────

def build_boto_clients(region: str):
    session = boto3.Session(region_name=region)
    s3 = session.client(
        "s3",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
        ),
    )
    batch = session.client("batch")
    return s3, batch


def s3_key_join(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


def sanitize_batch_job_name(raw_name: str, fallback: str = "simc-job") -> str:
    # AWS Batch job name allows only letters, numbers, hyphens, and underscores.
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_name)
    sanitized = re.sub(r"[-_]{2,}", "-", sanitized).strip("-_")
    if not sanitized:
        sanitized = fallback
    if len(sanitized) > 128:
        sanitized = sanitized[:128].rstrip("-_")
        if not sanitized:
            sanitized = fallback
    return sanitized


def wait_for_array_parent(batch_client, parent_job_id: str, poll_seconds: int) -> dict[str, Any]:
    while True:
        resp = batch_client.describe_jobs(jobs=[parent_job_id])
        if not resp["jobs"]:
            raise RuntimeError(f"Batch job {parent_job_id} not found")
        job = resp["jobs"][0]
        status = job["status"]
        print(f"    Batch parent {parent_job_id}: {status}")
        if status in {"SUCCEEDED", "FAILED"}:
            return job
        if CANCEL_REQUESTED.is_set():
            raise RuntimeError("Cancelled while waiting for Batch array job")
        time.sleep(poll_seconds)


def list_failed_children(batch_client, parent_job_id: str) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    next_token = None
    while True:
        kwargs = {"arrayJobId": parent_job_id, "jobStatus": "FAILED"}
        if next_token:
            kwargs["nextToken"] = next_token
        resp = batch_client.list_jobs(**kwargs)
        failed.extend(resp.get("jobSummaryList", []))
        next_token = resp.get("nextToken")
        if not next_token:
            break
    return failed


def delete_s3_keys(s3_client, bucket: str, keys: list[str]) -> None:
    keys = [k for k in keys if k]
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )


def object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def prepare_spec_chunks(args, s3_client, run_prefix: str, spec_config: SpecConfig, start_index: int, sims_to_run: int) -> tuple[list[ChunkPlan], str, str]:
    spec_state_dir = os.path.join(args.shard_dir, spec_config.profile_name)
    os.makedirs(spec_state_dir, exist_ok=True)

    # Load existing stat combos so we don't regenerate duplicates
    existing_keys = load_existing_stat_keys(spec_config.profile_name)
    seen_keys: set[tuple] = set(existing_keys)

    chunk_plans: list[ChunkPlan] = []
    next_start = start_index
    remaining = sims_to_run
    chunk_id = 0

    while remaining > 0:
        csize = min(args.chunk_size, remaining)
        chunk_name = f"chunk_{chunk_id:04d}"
        local_chunk_dir = os.path.join(spec_state_dir, chunk_name)
        if os.path.isdir(local_chunk_dir):
            shutil.rmtree(local_chunk_dir)
        os.makedirs(local_chunk_dir, exist_ok=True)

        profile_names = generate_profiles_for_range(
            next_start,
            csize,
            args.seed,
            args.iterations,
            local_chunk_dir,
            spec_config,
            existing_keys=seen_keys,
        )
        input_zip_path = os.path.join(local_chunk_dir, f"{chunk_name}_input.zip")
        zip_simc_inputs(local_chunk_dir, input_zip_path)

        input_key = s3_key_join(run_prefix, spec_config.profile_name, "inputs", f"{chunk_name}.zip")
        output_key = s3_key_join(run_prefix, spec_config.profile_name, "outputs", f"{chunk_name}.zip")
        shard_csv_path = os.path.join(spec_state_dir, f"{chunk_name}.csv")
        s3_client.upload_file(input_zip_path, args.s3_bucket, input_key)

        chunk_plans.append(
            ChunkPlan(
                chunk_id=chunk_id,
                start_index=next_start,
                sample_count=len(profile_names),
                profile_names=profile_names,
                local_chunk_dir=local_chunk_dir,
                input_zip_path=input_zip_path,
                input_key=input_key,
                output_key=output_key,
                shard_csv_path=shard_csv_path,
            )
        )

        next_start += len(profile_names)
        remaining -= len(profile_names)
        chunk_id += 1

    manifest_key = s3_key_join(run_prefix, spec_config.profile_name, "manifest.json")
    manifest = {
        "run_prefix": run_prefix,
        "spec": spec_config.profile_name,
        "chunk_count": len(chunk_plans),
        "worker_parallel": args.worker_parallel,
        "chunks": [
            {
                "chunk_id": cp.chunk_id,
                "input_key": cp.input_key,
                "output_key": cp.output_key,
            }
            for cp in chunk_plans
        ],
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(manifest, tmp)
        tmp_path = tmp.name
    try:
        s3_client.upload_file(tmp_path, args.s3_bucket, manifest_key, ExtraArgs={"ContentType": "application/json"})
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    
    return chunk_plans, manifest_key, args.s3_bucket


def submit_array_job_for_spec(args, batch_client, spec_config: SpecConfig, manifest_key: str, chunk_count: int) -> str:
    env = [
        {"name": "SIMC_S3_BUCKET", "value": args.s3_bucket},
        {"name": "SIMC_MANIFEST_KEY", "value": manifest_key},
        {"name": "SIMC_AWS_REGION", "value": args.aws_region},
        {"name": "SIMC_SPEC_NAME", "value": spec_config.profile_name},
        {"name": "SIMC_WORKER_PARALLEL", "value": str(args.worker_parallel)},
    ]
    raw_job_name = f"{args.job_name_prefix}-{spec_config.profile_name.lower().replace('_', '-')}"
    sanitized_job_name = sanitize_batch_job_name(raw_job_name)
    if chunk_count < 1:
        raise ValueError("chunk_count must be >= 1")

    submit_kwargs = {
        "jobName": sanitized_job_name,
        "jobQueue": args.batch_job_queue,
        "jobDefinition": args.batch_job_definition,
        "containerOverrides": {"environment": env},
    }
    # Some Batch environments reject array size 1. Use a regular job for single-chunk runs.
    if chunk_count >= 2:
        submit_kwargs["arrayProperties"] = {"size": chunk_count}

    resp = batch_client.submit_job(**submit_kwargs)
    return resp["jobId"]


def download_and_merge_spec_outputs(args, s3_client, spec_config: SpecConfig, chunk_plans: list[ChunkPlan]) -> pd.DataFrame:
    shard_frames: list[pd.DataFrame] = []
    for cp in chunk_plans:
        local_output_zip = os.path.join(cp.local_chunk_dir, f"chunk_{cp.chunk_id:04d}_output.zip")
        if not object_exists(s3_client, args.s3_bucket, cp.output_key):
            raise RuntimeError(f"Expected output object missing from S3: s3://{args.s3_bucket}/{cp.output_key}")
        s3_client.download_file(args.s3_bucket, cp.output_key, local_output_zip)
        extract_zip_to_dir(local_output_zip, cp.local_chunk_dir)
        df = collect_results(
            spec_config=spec_config,
            target_profiles=set(cp.profile_names),
            csv_path=cp.shard_csv_path,
            results_dir=cp.local_chunk_dir,
        )
        shard_frames.append(df)

    if not shard_frames:
        raise RuntimeError("No shard outputs produced by AWS Batch worker jobs")

    merged = _normalize_spec_dataframe(pd.concat(shard_frames, ignore_index=True))
    spec_csv = os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{spec_config.profile_name}.csv")
    os.makedirs(DEFAULT_TRAINING_DATA_DIR, exist_ok=True)
    if os.path.exists(spec_csv):
        existing_spec_df = _normalize_spec_dataframe(pd.read_csv(spec_csv, encoding="utf-8"))
        merged = pd.concat([existing_spec_df, merged], ignore_index=True)
    _dedup_cols = ["primary_stat", "crit", "haste", "mastery", "versatility"]
    before = len(merged)
    merged = merged.drop_duplicates(subset=_dedup_cols, keep="last").reset_index(drop=True)
    if len(merged) < before:
        print(f"  Deduped {before} -> {len(merged)} rows (dropped {before - len(merged)} duplicates)")
    merged.to_csv(spec_csv, index=False, encoding="utf-8")
    print(f"  Merged {len(merged)} rows -> {spec_csv}")
    worker_image_signature = detect_worker_image_signature()
    mark_profile_generated(
        spec_config.profile_name,
        simulation_signature=compute_simulation_signature(spec_config.profile_content),
        game_data_signature=detect_game_data_signature_from_chunk_plans(chunk_plans),
        worker_image_signature=worker_image_signature,
    )
    return merged


def download_and_merge_available_spec_outputs(args, s3_client, spec_config: SpecConfig, chunk_plans: list[ChunkPlan]) -> pd.DataFrame | None:
    shard_frames: list[pd.DataFrame] = []
    missing_chunks: list[int] = []
    failed_chunks: list[int] = []

    for cp in chunk_plans:
        local_output_zip = os.path.join(cp.local_chunk_dir, f"chunk_{cp.chunk_id:04d}_output.zip")
        if not object_exists(s3_client, args.s3_bucket, cp.output_key):
            missing_chunks.append(cp.chunk_id)
            continue

        try:
            s3_client.download_file(args.s3_bucket, cp.output_key, local_output_zip)
            extract_zip_to_dir(local_output_zip, cp.local_chunk_dir)
            df = collect_results(
                spec_config=spec_config,
                target_profiles=set(cp.profile_names),
                csv_path=cp.shard_csv_path,
                results_dir=cp.local_chunk_dir,
            )
            shard_frames.append(df)
        except Exception as exc:
            failed_chunks.append(cp.chunk_id)
            print(f"  WARNING: Failed processing chunk {cp.chunk_id} for {spec_config.profile_name}: {exc}")

    if not shard_frames:
        print(
            f"  WARNING: No usable shard outputs for {spec_config.profile_name}. "
            f"Missing chunks: {len(missing_chunks)}, failed chunks: {len(failed_chunks)}"
        )
        return None

    merged = _normalize_spec_dataframe(pd.concat(shard_frames, ignore_index=True))
    spec_csv = os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{spec_config.profile_name}.csv")
    os.makedirs(DEFAULT_TRAINING_DATA_DIR, exist_ok=True)
    if os.path.exists(spec_csv):
        existing_spec_df = _normalize_spec_dataframe(pd.read_csv(spec_csv, encoding="utf-8"))
        merged = pd.concat([existing_spec_df, merged], ignore_index=True)
    _dedup_cols = ["primary_stat", "crit", "haste", "mastery", "versatility"]
    before = len(merged)
    merged = merged.drop_duplicates(subset=_dedup_cols, keep="last").reset_index(drop=True)
    if len(merged) < before:
        print(f"  Deduped {before} -> {len(merged)} rows (dropped {before - len(merged)} duplicates)")
    merged.to_csv(spec_csv, index=False, encoding="utf-8")

    print(
        f"  Partial merge complete for {spec_config.profile_name}: {len(merged)} total rows. "
        f"Missing chunks: {len(missing_chunks)}, failed chunks: {len(failed_chunks)}"
    )
    worker_image_signature = detect_worker_image_signature()
    mark_profile_generated(
        spec_config.profile_name,
        simulation_signature=compute_simulation_signature(spec_config.profile_content),
        game_data_signature=detect_game_data_signature_from_chunk_plans(chunk_plans),
        worker_image_signature=worker_image_signature,
    )
    return merged


def run_batch_array_scheduler(args, s3_client, batch_client, spec_config: SpecConfig, start_index: int, sample_count: int) -> pd.DataFrame | None:
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = s3_key_join(args.s3_prefix, f"run-{timestamp}")
    print(f"  Preparing {sample_count} sims as Batch array chunks (chunk_size={args.chunk_size})")

    chunk_plans, manifest_key, _ = prepare_spec_chunks(args, s3_client, run_prefix, spec_config, start_index, sample_count)
    print(f"  Uploaded {len(chunk_plans)} input chunk zip(s) to s3://{args.s3_bucket}/{run_prefix}/")

    parent_job_id = submit_array_job_for_spec(args, batch_client, spec_config, manifest_key, len(chunk_plans))
    print(f"  Submitted Batch array job parent: {parent_job_id}")

    parent = wait_for_array_parent(batch_client, parent_job_id, args.poll_seconds)
    if parent["status"] != "SUCCEEDED":
        failed_children = list_failed_children(batch_client, parent_job_id)
        fail_preview = ", ".join(f"{x['jobId']}:{x['status']}" for x in failed_children[:10])
        raise RuntimeError(
            f"AWS Batch array job failed for {spec_config.profile_name}. "
            f"Parent={parent_job_id}, failed children preview: {fail_preview or 'none reported'}"
        )

    merged = download_and_merge_spec_outputs(args, s3_client, spec_config, chunk_plans)

    temp_keys = [manifest_key] + [cp.input_key for cp in chunk_plans] + [cp.output_key for cp in chunk_plans]
    delete_s3_keys(s3_client, args.s3_bucket, temp_keys)
    print(f"  Deleted temporary S3 objects under s3://{args.s3_bucket}/{run_prefix}/")
    return merged


def submit_spec_batch_run(args, s3_client, batch_client, spec_config: SpecConfig, start_index: int, sample_count: int) -> ActiveSpecRun:
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = s3_key_join(args.s3_prefix, f"run-{timestamp}")
    print(f"  Preparing {sample_count} sims as Batch array chunks (chunk_size={args.chunk_size})")

    chunk_plans, manifest_key, _ = prepare_spec_chunks(
        args, s3_client, run_prefix, spec_config, start_index, sample_count
    )
    print(f"  Uploaded {len(chunk_plans)} input chunk zip(s) to s3://{args.s3_bucket}/{run_prefix}/")

    parent_job_id = submit_array_job_for_spec(args, batch_client, spec_config, manifest_key, len(chunk_plans))
    print(f"  Submitted Batch array job parent: {parent_job_id}")

    return ActiveSpecRun(
        spec_name=spec_config.profile_name,
        spec_config=spec_config,
        start_index=start_index,
        sample_count=sample_count,
        run_prefix=run_prefix,
        chunk_plans=chunk_plans,
        manifest_key=manifest_key,
        parent_job_id=parent_job_id,
    )


def finalize_spec_batch_run(args, s3_client, batch_client, run: ActiveSpecRun, parent_status: str) -> FinalizeOutcome:
    temp_keys = [run.manifest_key] + [cp.input_key for cp in run.chunk_plans] + [cp.output_key for cp in run.chunk_plans]

    if parent_status != "SUCCEEDED":
        failed_children = list_failed_children(batch_client, run.parent_job_id)
        fail_preview = ", ".join(f"{x['jobId']}:{x['status']}" for x in failed_children[:10])
        print(
            f"  WARNING: AWS Batch array job failed for {run.spec_name}. "
            f"Parent={run.parent_job_id}, failed children preview: {fail_preview or 'none reported'}"
        )
        merged = download_and_merge_available_spec_outputs(args, s3_client, run.spec_config, run.chunk_plans)
        delete_s3_keys(s3_client, args.s3_bucket, temp_keys)
        print(f"  Deleted temporary S3 objects under s3://{args.s3_bucket}/{run.run_prefix}/")
        remaining = max(0, args.samples - get_saved_sim_count(run.spec_name))
        return FinalizeOutcome(
            merged=merged,
            should_retry=remaining > 0,
            reason=f"parent-failed; remaining={remaining}",
        )

    try:
        merged = download_and_merge_spec_outputs(args, s3_client, run.spec_config, run.chunk_plans)
    except Exception as exc:
        print(f"  WARNING: Full merge failed for {run.spec_name}: {exc}")
        merged = download_and_merge_available_spec_outputs(args, s3_client, run.spec_config, run.chunk_plans)
        delete_s3_keys(s3_client, args.s3_bucket, temp_keys)
        print(f"  Deleted temporary S3 objects under s3://{args.s3_bucket}/{run.run_prefix}/")
        remaining = max(0, args.samples - get_saved_sim_count(run.spec_name))
        return FinalizeOutcome(
            merged=merged,
            should_retry=remaining > 0,
            reason=f"merge-incomplete; remaining={remaining}",
        )

    delete_s3_keys(s3_client, args.s3_bucket, temp_keys)
    print(f"  Deleted temporary S3 objects under s3://{args.s3_bucket}/{run.run_prefix}/")
    return FinalizeOutcome(merged=merged, should_retry=False)


# ── CLI / MAIN ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local SimulationCraft orchestrator that uses AWS Batch array jobs as remote workers.")
    parser.add_argument("--specs", type=str, required=False, default=None,
                        help="Optional override for profile selection (all, existing, or comma-separated names)")
    parser.add_argument("--profile-limit", type=int, default=None,
                        help="Optional cap for how many profiles to queue this run")
    parser.add_argument("--exclude-specs", type=str, default=None,
                        help="Comma-separated spec names to exclude (e.g. Druid_Balance,Druid_Balance_Elune)")
    parser.add_argument("--download-profiles", action="store_true")
    parser.add_argument("--skip-profile-refresh", action="store_true",
                        help="Skip GitHub profile refresh; use local cache only. "
                             "Useful when a prior pipeline step already refreshed.")
    parser.add_argument("--check-staleness", action="store_true")
    parser.add_argument("--regenerate-stale", action="store_true")
    parser.add_argument("--skip-staleness-check", action="store_true",
                        help="With --regenerate-stale: skip the drift check and regenerate specs "
                             "already marked stale in profile_metadata.json")
    parser.add_argument("--staleness-execution", type=str, choices=["remote", "local"], default="remote",
                        help="Where staleness checks run: remote Batch worker (recommended) or local Docker")
    parser.add_argument("--staleness-sample-size", type=int, default=DEFAULT_STALENESS_SAMPLE_SIZE,
                        help="How many existing points to re-sim per profile for drift detection")
    parser.add_argument("--staleness-sample-iterations", type=int, default=DEFAULT_STALENESS_SAMPLE_ITERATIONS,
                        help="Iterations per sample re-sim for drift detection")
    parser.add_argument("--staleness-drift-threshold-pct", type=float, default=DEFAULT_STALENESS_DRIFT_THRESHOLD_PCT,
                        help="Mark profile stale when median absolute sample drift percent exceeds this threshold")
    parser.add_argument("--staleness-batch-job-queue", type=str, default="simc-batch-array-staleness-queue",
                        help="Batch queue for remote staleness checks")
    parser.add_argument("--staleness-batch-job-definition", type=str, default="simc-batch-array-staleness-worker",
                        help="Batch job definition for remote sample staleness checks")
    parser.add_argument("--staleness-s3-prefix", type=str, default="simc-batch-staleness-temp",
                        help="S3 prefix used by remote sample staleness checks")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_COUNT,
                        help="Target saved samples per spec in training_data/<spec>.csv")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--clean-spec-output", action="store_true")
    parser.add_argument("--wipe-all-data", action="store_true",
                        help="Delete ALL training data, shards, profile metadata, and cached profiles. Full reset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_GEAR_RANDOM_SEED)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--shard-dir", type=str, default=DEFAULT_CHUNK_STATE_DIR)
    parser.add_argument("--s3-bucket", type=str, default="simc-batch-bucket")
    parser.add_argument("--s3-prefix", type=str, default="simc-batch-temp")
    parser.add_argument("--presign-seconds", type=int, default=DEFAULT_PRESIGN_SECONDS,
                        help="Deprecated no-op kept for CLI compatibility")
    parser.add_argument("--batch-job-queue", type=str, default="simc-batch-array-queue")
    parser.add_argument("--batch-job-definition", type=str, default="simc-batch-array-worker")
    parser.add_argument("--job-name-prefix", type=str, default="simc")
    parser.add_argument("--worker-parallel", type=int, default=DEFAULT_WORKER_PARALLEL)
    parser.add_argument("--max-active-profiles", type=int, default=DEFAULT_MAX_ACTIVE_PROFILES,
                        help="How many profile array jobs to keep in-flight concurrently")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-spec-retries", type=int, default=2,
                        help="Max automatic same-run retries per spec when child jobs fail or outputs are incomplete")
    parser.add_argument("--aws-region", type=str, default="eu-north-1")
    parser.add_argument("--boost-from-model", type=str, default=None,
                        help="Path to nn_metadata.json; enables per-spec adaptive sample targets based on model MAE")
    parser.add_argument("--boost-extra-budget", type=int, default=200,
                        help="Max extra samples the worst spec receives per boost pass (default: 200)")
    parser.add_argument("--trigger-training", action="store_true",
                        help="After simulation, rebuild combined CSV and launch SageMaker training")
    parser.add_argument("--training-s3-bucket", type=str, default=None,
                        help="S3 bucket for SageMaker training (required with --trigger-training)")
    parser.add_argument("--training-role-arn", type=str, default=None,
                        help="SageMaker execution role ARN (required with --trigger-training)")
    return parser.parse_args()


def resolve_specs(args: argparse.Namespace, per_spec_targets: dict[str, int] | None = None) -> list[str]:
    all_profiles = get_available_spec_profiles()

    if args.check_staleness or args.regenerate_stale:
        if args.regenerate_stale and args.skip_staleness_check:
            # Read already-stale specs from profile_metadata.json
            metadata = _load_profile_metadata()
            stale_profiles = sorted(
                name for name, meta in metadata.items()
                if meta.get("is_stale", False)
            )
            # If --specs was also given, filter to only those specs
            if args.specs and args.specs.strip().lower() not in ("all", "existing", ""):
                requested = {s.strip() for s in args.specs.split(",") if s.strip()}
                stale_profiles = [s for s in stale_profiles if s in requested]
            if stale_profiles:
                print(f"\nFound {len(stale_profiles)} previously-stale profile(s) in metadata (skipping drift check):")
                for sp in stale_profiles:
                    drift = metadata[sp].get("staleness_drift_pct")
                    drift_str = f"{drift:.2f}% drift" if drift is not None else "no drift data"
                    print(f"  - {sp} [{drift_str}]")
                args.specs = ",".join(stale_profiles)
            else:
                print("\nNo stale profiles found in metadata. Nothing to regenerate.")
                sys.exit(0)
        else:
            stale_reasons = get_stale_profile_reasons_from_sample(args)
            stale_profiles = sorted(stale_reasons.keys())

            if stale_profiles:
                print(f"\nDetected {len(stale_profiles)} stale profile(s):")
                for sp in stale_profiles:
                    reasons = ", ".join(stale_reasons.get(sp, [])) or "unknown"
                    print(f"  - {sp} [{reasons}]")
                if args.regenerate_stale:
                    args.specs = ",".join(stale_profiles)
                else:
                    print("\nUse --regenerate-stale to regenerate these profiles.")
                    sys.exit(0)
            else:
                print("\nNo stale profiles detected. All profiles are up-to-date.")
                sys.exit(0)

    selected: list[str]
    if args.specs:
        raw = args.specs.strip().lower()
        if raw == "all":
            selected = list(all_profiles)
        elif raw == "existing":
            selected = [
                spec for spec in all_profiles
                if os.path.exists(os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{spec}.csv"))
            ]
            if not selected:
                print("ERROR: No existing spec profiles found in ./training_data/")
                sys.exit(1)
            print(f"Auto-detected {len(selected)} existing spec profile(s): {', '.join(selected)}")
        else:
            selected = [s.strip() for s in args.specs.split(",") if s.strip()]
            # Validate against known profiles; also accept specs that have
            # training data (handles names with special chars like San'layn
            # that may not appear in the cached profile listing).
            known = set(all_profiles)
            invalid = [
                s for s in selected
                if s not in known
                and not os.path.exists(os.path.join(DEFAULT_TRAINING_DATA_DIR, f"{s}.csv"))
            ]
            if invalid:
                print(f"ERROR: Unknown spec profile(s): {', '.join(invalid)}")
                sys.exit(1)
    else:
        if per_spec_targets:
            selected = [
                spec for spec in all_profiles
                if get_saved_sim_count(spec) < (
                    per_spec_targets[spec]
                    if spec in per_spec_targets and per_spec_targets[spec] is not None
                    else args.samples
                )
            ]
        else:
            selected = [spec for spec in all_profiles if get_saved_sim_count(spec) < args.samples]
        if selected:
            print(
                f"Auto-selected {len(selected)} incomplete profile(s) "
                f"with saved rows below their target."
            )

    if args.exclude_specs:
        excluded = {s.strip() for s in args.exclude_specs.split(",") if s.strip()}
        before_count = len(selected)
        selected = [s for s in selected if s not in excluded]
        actually_excluded = before_count - len(selected)
        if actually_excluded:
            print(f"Excluded {actually_excluded} spec(s): {', '.join(sorted(excluded & set(all_profiles)))}")

    if args.profile_limit is not None:
        if args.profile_limit < 1:
            raise ValueError("--profile-limit must be >= 1")
        if len(selected) > args.profile_limit:
            selected = selected[:args.profile_limit]
            print(f"Applying --profile-limit={args.profile_limit}; queueing first {len(selected)} profiles.")

    return selected


def main() -> None:
    args = parse_args()

    # Full data wipe — exits after completion
    if args.wipe_all_data:
        print("\n=== WIPING ALL SIMULATION DATA ===")
        wipe_all_data(args.shard_dir)
        sys.exit(0)

    # Download-only mode: refresh all profiles and exit
    if args.download_profiles:
        print("Refreshing all profiles from GitHub (profiles/MID1)...")
        download_all_profiles(force=True)
        sys.exit(0)

    # Pull local Docker image only when running local staleness checks
    if (args.check_staleness or args.regenerate_stale) and args.staleness_execution == "local":
        preflight_pull_simc_image()
    if args.max_active_profiles < 1:
        raise ValueError("--max-active-profiles must be >= 1")
    if args.max_spec_retries < 0:
        raise ValueError("--max-spec-retries must be >= 0")
    if args.trigger_training:
        if not args.training_s3_bucket or not args.training_role_arn:
            print("ERROR: --trigger-training requires --training-s3-bucket and --training-role-arn")
            sys.exit(1)

    # Compute per-spec sample targets when --boost-from-model is used
    per_spec_targets: dict[str, int] | None = None
    if args.boost_from_model:
        if not os.path.exists(args.boost_from_model):
            print(f"ERROR: Model metadata not found at {args.boost_from_model}")
            sys.exit(1)
        per_spec_targets = compute_per_spec_sample_targets(
            metadata_path=args.boost_from_model,
            baseline=args.samples,
            extra_budget=args.boost_extra_budget,
        )

    selected_specs = resolve_specs(args, per_spec_targets=per_spec_targets)
    if not selected_specs:
        print("No profiles currently missing samples; nothing to run.")
        return

    # Refresh only the profiles we actually need (skip if a prior pipeline step
    # already refreshed everything via --download-profiles).
    if not args.skip_profile_refresh:
        try:
            download_profiles_for_specs(selected_specs, force=True)
        except Exception as exc:
            print(f"  WARNING: Profile refresh failed, using local cache: {exc}")

    # When regenerating stale specs, delete their existing data so they
    # don't get skipped for already being at target.
    if args.regenerate_stale:
        print(f"\nCleaning training data for {len(selected_specs)} stale spec(s) before regeneration...")
        for spec_name in selected_specs:
            clear_saved_spec_outputs(spec_name, args.shard_dir)
            print(f"  Deleted: {spec_name}")

    s3_client, batch_client = build_boto_clients(args.aws_region)

    print("=" * 78)
    print("SIMULATIONCRAFT ORCHESTRATOR — LOCAL ORCHESTRATOR + AWS BATCH ARRAY WORKERS")
    print("=" * 78)
    print(f"  Specs to simulate   : {len(selected_specs)}")
    print(f"  Iterations/sim      : {args.iterations}")
    print(f"  Target sims/spec    : {args.samples} (baseline)")
    if per_spec_targets:
        boosted = {s: t for s, t in per_spec_targets.items() if t > args.samples}
        print(f"  Boost mode          : ON ({len(boosted)} specs boosted, +{args.boost_extra_budget} max extra)")
    else:
        print(f"  Boost mode          : OFF")
    print(f"  Chunk size          : {args.chunk_size}")
    print(f"  Worker parallelism  : {args.worker_parallel}")
    print(f"  Max active profiles : {args.max_active_profiles}")
    print(f"  Profile limit       : {args.profile_limit if args.profile_limit is not None else 'none'}")
    print(f"  S3 bucket           : {args.s3_bucket}")
    print(f"  S3 prefix           : {args.s3_prefix}")
    print(f"  Batch queue         : {args.batch_job_queue}")
    print(f"  Batch job def       : {args.batch_job_definition}")
    print(f"  Region              : {args.aws_region}")
    print("=" * 78)

    pending_specs = collections.deque(selected_specs)
    active_runs: dict[str, ActiveSpecRun] = {}
    retry_counts: dict[str, int] = {}
    submitted_specs = 0

    while pending_specs or active_runs:
        s3_client, batch_client = build_boto_clients(args.aws_region)

        while pending_specs and len(active_runs) < args.max_active_profiles:
            spec_name = pending_specs.popleft()
            submitted_specs += 1
            print(f"\n[{submitted_specs}/{len(selected_specs)}] Queueing spec: {spec_name}")

            try:
                spec_config = get_spec_config(spec_name)
            except RuntimeError as exc:
                print(f"  SKIP – could not load spec config: {exc}")
                continue

            print(
                f"  Class: {spec_config.class_keyword}, "
                f"Spec: {spec_config.spec}, "
                f"Primary stat: {spec_config.primary_stat}, "
                f"Race: {spec_config.race}"
            )

            if args.clean_spec_output:
                print(f"  Cleaning existing saved outputs for: {spec_name}")
                clear_saved_spec_outputs(spec_name, args.shard_dir)

            existing_completed = get_saved_sim_count(spec_name)
            target_total = args.samples
            if per_spec_targets is not None:
                maybe_target = per_spec_targets.get(spec_name)
                if maybe_target is not None:
                    target_total = maybe_target
            sims_to_run = max(0, target_total - existing_completed)
            print(f"  Saved sims: {existing_completed} / {target_total}")
            print(f"  Sims to run this pass: {sims_to_run}")

            if args.start_index is not None:
                if args.start_index < 0:
                    raise ValueError("--start-index must be >= 0")
                start_index = args.start_index
                print(f"  Using explicit start index: {start_index}")
            else:
                start_index = existing_completed
                print(f"  Auto-detected start index from saved rows: {start_index}")

            if sims_to_run == 0:
                print("  Target already reached; no new sims needed.")
                continue

            run = submit_spec_batch_run(args, s3_client, batch_client, spec_config, start_index, sims_to_run)
            active_runs[run.parent_job_id] = run

        if not active_runs:
            continue

        if CANCEL_REQUESTED.is_set():
            raise RuntimeError("Cancelled while polling active Batch array jobs")

        describe = batch_client.describe_jobs(jobs=list(active_runs.keys()))
        jobs = {job["jobId"]: job for job in describe.get("jobs", [])}
        completed: list[tuple[ActiveSpecRun, str]] = []

        for parent_job_id, run in active_runs.items():
            job = jobs.get(parent_job_id)
            if not job:
                raise RuntimeError(f"Batch job {parent_job_id} not found while polling active runs")
            status = job["status"]
            if run.last_status != status:
                print(f"    Batch parent {parent_job_id} ({run.spec_name}): {status}")
                run.last_status = status
            if status in {"SUCCEEDED", "FAILED"}:
                completed.append((run, status))

        if not completed:
            time.sleep(args.poll_seconds)
            continue

        for run, status in completed:
            active_runs.pop(run.parent_job_id, None)
            print(f"  Finalizing {run.spec_name} (parent {run.parent_job_id}) with status {status}")
            try:
                outcome = finalize_spec_batch_run(args, s3_client, batch_client, run, status)
            except Exception as exc:
                print(f"  ERROR: Finalize failed for {run.spec_name}, continuing: {exc}")
                saved = get_saved_sim_count(run.spec_name)
                remaining = max(0, args.samples - saved)
                outcome = FinalizeOutcome(merged=None, should_retry=remaining > 0, reason=f"finalize-exception; remaining={remaining}")

            if outcome.should_retry and not CANCEL_REQUESTED.is_set():
                retries_used = retry_counts.get(run.spec_name, 0)
                if retries_used >= args.max_spec_retries:
                    print(
                        f"  Retry limit reached for {run.spec_name} "
                        f"({retries_used}/{args.max_spec_retries}); saving completed work and moving on"
                    )
                    continue

                saved = get_saved_sim_count(run.spec_name)
                spec_target = args.samples
                if per_spec_targets is not None:
                    maybe_target = per_spec_targets.get(run.spec_name)
                    if maybe_target is not None:
                        spec_target = maybe_target
                sims_to_run = max(0, spec_target - saved)
                if sims_to_run == 0:
                    continue

                print(
                    f"  Retrying {run.spec_name}: {sims_to_run} sim(s) remaining "
                    f"from start index {saved} (attempt {retries_used + 1}/{args.max_spec_retries})"
                )
                retry_run = submit_spec_batch_run(
                    args,
                    s3_client,
                    batch_client,
                    run.spec_config,
                    saved,
                    sims_to_run,
                )
                active_runs[retry_run.parent_job_id] = retry_run
                retry_counts[run.spec_name] = retries_used + 1

    # ── Post-run: trigger SageMaker training ─────────────────────────────────
    if args.trigger_training:
        print("\n" + "=" * 78)
        print("POST-RUN: TRIGGERING SAGEMAKER TRAINING")
        print("=" * 78)

        # Step 1: Rebuild combined training CSV
        combined_csv = rebuild_combined_training_csv()

        # Step 2: Launch full training on SageMaker
        sagemaker_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sagemaker")
        launch_script = os.path.join(sagemaker_dir, "launch_training.py")
        download_script = os.path.join(sagemaker_dir, "download_model.py")

        launch_cmd = [
            sys.executable, launch_script,
            "--s3-bucket", args.training_s3_bucket,
            "--role-arn", args.training_role_arn,
            "--data-file", combined_csv,
        ]

        print(f"\nRunning: {' '.join(launch_cmd)}")
        result = subprocess.run(launch_cmd, check=False, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print(f"ERROR: SageMaker training launch failed (exit code {result.returncode})")
            return

        # Step 3: Download the new model
        # Parse the model artifact S3 URI from launch_training output
        model_s3_uri = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Model artifact:"):
                model_s3_uri = line.split(":", 1)[1].strip()
                break

        if model_s3_uri:
            download_cmd = [
                sys.executable, download_script,
                "--s3-uri", model_s3_uri,
                "--region", args.aws_region,
            ]
        else:
            print("WARNING: Could not parse model S3 URI from training output.")
            print("  You can download manually with: python sagemaker/download_model.py --job-name <job-name>")
            return

        print(f"\nRunning: {' '.join(download_cmd)}")
        result = subprocess.run(download_cmd, check=False)
        if result.returncode != 0:
            print(f"WARNING: Model download failed (exit code {result.returncode})")
            print("  You can download manually with: python sagemaker/download_model.py --job-name <job-name>")

        # Step 4: Print improvement report from new metadata
        new_metadata_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "nn_website_model", "deep_nn", "nn_metadata.json",
        )
        if os.path.exists(new_metadata_path):
            with open(new_metadata_path, encoding="utf-8") as f:
                new_meta = json.load(f)
            improvement = new_meta.get("improvement")
            if improvement:
                print("\n" + "=" * 78)
                print("BOOST ITERATION IMPROVEMENT SUMMARY")
                print("=" * 78)
                improved = [s for s, v in improvement.items() if v["delta"] < 0]
                regressed = [s for s, v in improvement.items() if v["delta"] > 0]
                avg_pct = sum(v["pct"] for v in improvement.values()) / len(improvement)
                print(f"  {len(improved)} specs improved, {len(regressed)} regressed")
                print(f"  Average improvement: {avg_pct:+.1f}%")
                print(f"  New test MAE: {new_meta.get('test_mae', 'N/A')}")
                print("=" * 78)
            else:
                print(f"\nNew model trained (test MAE: {new_meta.get('test_mae', 'N/A')})")
                print("  No previous model comparison available.")


if __name__ == "__main__":
    main()
