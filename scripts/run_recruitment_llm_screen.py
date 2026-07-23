#!/usr/bin/env python3
"""Dry-run or explicitly execute the frozen, resumable E2b Luna screen."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.llm.eval_utils.client import create_llm_client  # noqa: E402
from baselines.llm.eval_utils.recruitment_selection import (  # noqa: E402
    true_information_oracle,
)
from baselines.llm.eval_utils.team_formation import (  # noqa: E402
    MAX_CONTROL_BYTES,
    RecordKind,
    RecruitmentMethod,
    RecruitmentRecord,
    TaskPhase,
    TeamDirectory,
    parse_tfp1,
)
from baselines.llm.recruitment_arena import (  # noqa: E402
    AGENT_IDS,
    DEFAULT_ROUNDS,
    ScenarioFamily,
    generate_scenario,
)
from baselines.llm.recruitment_llm_screen import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_PROMPT_BYTES,
    DEFAULT_MODEL,
    DEFAULT_PROMPT_FRAMING_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SEMANTIC_REPAIRS,
    DEFAULT_STALL_ROUNDS,
    DEFAULT_TRANSPORT_RETRIES,
    SCHEMA_VERSION,
    SUPPORTED_METHODS,
    AgentPromptView,
    CampaignBudget,
    ScreenConfig,
    build_messages,
    canonical_json,
    estimate_campaign,
    redact_failure_excerpt,
    resolved_model_is_accepted,
    run_llm_recruitment_episode,
    sha256_text,
)

FROZEN_SEEDS = (22000, 22001, 22002)
FROZEN_FAMILIES = tuple(ScenarioFamily)
FROZEN_METHODS = SUPPORTED_METHODS
CANARY_FAMILIES = (
    ScenarioFamily.SINGLE_COMPLEMENTARY,
    ScenarioFamily.TWO_DISJOINT,
)
CANARY_CELLS = tuple(
    (22000, family, method)
    for family in CANARY_FAMILIES
    for method in FROZEN_METHODS
)
DEFAULT_LOGICAL_CALL_CAP = 2_160
DEFAULT_PROVIDER_ATTEMPT_CAP = 4_320
DEFAULT_TOKEN_EXPOSURE_CAP = 91_238_400
DEFAULT_MAX_RETRY_AFTER_SECONDS = 30.0
DEFAULT_OUTPUT = Path("outputs/recruitment_llm/e2b_luna_screen_v3")
PROTOCOL_PATH = Path("reports/agent_scaling_recruitment/e2b_protocol.md")
UV_LOCK_PATH = Path("uv.lock")
KNOWN_DIRTY_ALLOWLIST = (Path("Results/replays/nano_high_source_full_world.mp4"),)
SOURCE_PATHS = (
    Path("baselines/llm/eval_utils/client.py"),
    Path("baselines/llm/eval_utils/openai_responses.py"),
    Path("baselines/llm/eval_utils/prompt_builder.py"),
    Path("baselines/llm/eval_utils/recruitment_selection.py"),
    Path("baselines/llm/eval_utils/team_formation.py"),
    Path("baselines/llm/recruitment_arena.py"),
    Path("baselines/llm/recruitment_llm_screen.py"),
    Path("scripts/run_recruitment_llm_screen.py"),
    Path("scripts/summarize_recruitment_llm_failure.py"),
    PROTOCOL_PATH,
)


def _git_required(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"required Git query failed: {' '.join(args)}") from exc
    return result.stdout.rstrip("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    missing = [str(path) for path in SOURCE_PATHS if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required E2b source files are missing: {missing}")
    return {str(path): _sha256(PROJECT_ROOT / path) for path in SOURCE_PATHS}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(path)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path) -> None:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"managed path contains a symlink component: {current}")


def _ensure_directory(path: Path) -> None:
    """Create each missing ancestor durably and reject link-based aliases."""

    absolute = _absolute_path(path)
    _assert_no_symlink_components(absolute)
    missing = []
    current = absolute
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise RuntimeError(f"cannot find an existing ancestor for {absolute}")
            current = parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(current)
        break
    for directory in reversed(missing):
        parent = directory.parent
        try:
            os.mkdir(directory, mode=0o700)
        except FileExistsError:
            metadata = directory.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise NotADirectoryError(directory)
        _fsync_directory(directory)
        _fsync_directory(parent)
    _assert_no_symlink_components(absolute)


def _assert_managed_path(
    root: Path,
    path: Path,
    *,
    require_file: bool = False,
) -> tuple[Path, Path]:
    root_absolute = _absolute_path(root)
    path_absolute = _absolute_path(path)
    if Path(os.path.realpath(root_absolute)) != root_absolute:
        raise RuntimeError(f"managed output root aliases another path: {root_absolute}")
    try:
        if os.path.commonpath((root_absolute, path_absolute)) != str(root_absolute):
            raise RuntimeError(f"managed path escapes output root: {path_absolute}")
    except ValueError as exc:
        raise RuntimeError("managed path and output root are on different roots") from exc
    _assert_no_symlink_components(root_absolute)
    _assert_no_symlink_components(path_absolute.parent)
    try:
        metadata = path_absolute.lstat()
    except FileNotFoundError:
        if require_file:
            raise
    else:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"managed file is not a regular no-follow file: {path_absolute}")
        if metadata.st_nlink != 1:
            raise RuntimeError(f"managed file has multiple hard links: {path_absolute}")
        if Path(os.path.realpath(path_absolute)) != path_absolute:
            raise RuntimeError(f"managed file aliases another path: {path_absolute}")
    return root_absolute, path_absolute


def _open_managed_fd(
    root: Path,
    path: Path,
    flags: int,
    *,
    mode: int = 0o600,
) -> int:
    _, absolute = _assert_managed_path(
        root,
        path,
        require_file=not bool(flags & os.O_CREAT),
    )
    descriptor = os.open(
        absolute,
        flags | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"managed descriptor is linked or non-regular: {absolute}")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _managed_sha256(root: Path, path: Path) -> str:
    descriptor = _open_managed_fd(root, path, os.O_RDONLY)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_managed_bytes(root: Path, path: Path) -> bytes:
    descriptor = _open_managed_fd(root, path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()


def _read_managed_json(root: Path, path: Path) -> Any:
    return json.loads(_read_managed_bytes(root, path).decode("utf-8"))


def _normalize_output_root(path: Path) -> Path:
    absolute = _absolute_path(path)
    _assert_no_symlink_components(absolute)
    if Path(os.path.realpath(absolute)) != absolute:
        raise RuntimeError(f"output root aliases another path: {absolute}")
    if absolute.exists():
        metadata = absolute.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(absolute)
    return absolute


def _working_tree_binding() -> dict[str, Any]:
    """Fail closed except for the one known user replay artifact."""

    top_level = Path(_git_required("rev-parse", "--show-toplevel")).resolve()
    if top_level != PROJECT_ROOT.resolve():
        raise RuntimeError("Git top level does not match the E2b project root")
    raw_status = _git_required(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
    )
    allowed = {str(path): path for path in KNOWN_DIRTY_ALLOWLIST}
    observed: dict[str, str] = {}
    unexpected = []
    for line in raw_status.splitlines():
        if not line:
            continue
        if len(line) < 4:
            unexpected.append(line)
            continue
        code = line[:2]
        raw_path = line[3:]
        allowed_path = allowed.get(raw_path)
        if code != "??" or allowed_path is None:
            unexpected.append(line)
            continue
        absolute = PROJECT_ROOT / allowed_path
        if not absolute.is_file() or absolute.is_symlink():
            unexpected.append(line)
            continue
        observed[raw_path] = _sha256(absolute)
    if unexpected:
        raise RuntimeError(
            "hosted E2b requires a clean tree except the hashed replay allowlist: "
            + "; ".join(unexpected)
        )
    return {
        "porcelain_v1": raw_status,
        "allowed_untracked_sha256": dict(sorted(observed.items())),
    }


def _launch_binding(protocol: dict[str, Any]) -> dict[str, Any]:
    head = _git_required("rev-parse", "--verify", "HEAD")
    if re.fullmatch(r"[a-f0-9]{40,64}", head) is None:
        raise RuntimeError("Git HEAD is not a canonical commit hash")
    uv_lock = PROJECT_ROOT / UV_LOCK_PATH
    if not uv_lock.is_file():
        raise FileNotFoundError("uv.lock is required for hosted E2b")
    return {
        "git_head": head,
        "working_tree": _working_tree_binding(),
        "uv_lock_path": str(UV_LOCK_PATH),
        "uv_lock_sha256": _sha256(uv_lock),
        "source_hashes": _source_hashes(),
        "config_sha256": sha256_text(canonical_json(protocol)),
    }


def _require_hosted_credentials() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError("OPENAI_API_KEY must be present before hosted E2b execution")


@contextmanager
def _output_root_lock(output: Path):
    """Hold a nonblocking advisory lock for the entire hosted invocation."""

    output = _normalize_output_root(output)
    _ensure_directory(output)
    lock_path = output / ".e2b-hosted.lock"
    existed = lock_path.exists()
    descriptor = _open_managed_fd(
        output,
        lock_path,
        os.O_RDWR | os.O_APPEND | os.O_CREAT,
    )
    handle = os.fdopen(descriptor, "a+b")
    if not existed:
        os.fsync(handle.fileno())
        _fsync_directory(output)
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another hosted E2b invocation holds {lock_path}") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _atomic_write_bytes(root: Path, path: Path, payload: bytes) -> None:
    root = _normalize_output_root(root)
    _, path = _assert_managed_path(root, path)
    _ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    descriptor = _open_managed_fd(
        root,
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_managed_path(root, path)
        os.replace(temporary, path)
        _assert_managed_path(root, path, require_file=True)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(root: Path, path: Path, payload: Any) -> None:
    """Write JSON and atomically replace the destination in one directory."""

    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    ).encode("ascii") + b"\n"
    _atomic_write_bytes(root, path, serialized)


class DurableReservationLedger:
    """Append-only dispatch reservations with hash-chain crash reconciliation."""

    SCHEMA_VERSION = "alem-dice-e2b-reservation-ledger-v2"
    ANCHOR_SCHEMA_VERSION = "alem-dice-e2b-reservation-anchor-v1"

    def __init__(self, output: Path, *, launch_binding: dict[str, Any]):
        self.output = _normalize_output_root(output)
        self.path = self.output / "reservation_ledger.jsonl"
        self.anchor_dir = self.output / "reservation_anchors"
        self.binding_sha256 = sha256_text(canonical_json(launch_binding))
        self._lock = threading.Lock()
        self._records = self._read_records()
        self._anchors = self._read_anchors(self._records)
        self._last_hash = "0" * 64 if not self._records else str(self._records[-1]["record_sha256"])
        self._last_anchor_hash = (
            "0" * 64 if not self._anchors else str(self._anchors[-1]["anchor_sha256"])
        )

    @staticmethod
    def _key(payload: Mapping[str, Any]) -> str:
        key = payload.get("reservation_key")
        if not isinstance(key, Mapping) or set(key) != {
            "seed",
            "family",
            "method",
            "round_index",
            "agent_id",
            "semantic_attempt",
        }:
            raise ValueError("reservation key has the wrong fields")
        if (
            not isinstance(key["seed"], int)
            or key["seed"] not in FROZEN_SEEDS
            or ScenarioFamily(key["family"]) not in FROZEN_FAMILIES
            or RecruitmentMethod(key["method"]) not in FROZEN_METHODS
            or any(
                isinstance(key[name], bool) or not isinstance(key[name], int) or key[name] < 0
                for name in ("round_index", "agent_id", "semantic_attempt")
            )
            or key["round_index"] >= DEFAULT_ROUNDS
            or key["agent_id"] not in AGENT_IDS
            or key["semantic_attempt"] > DEFAULT_SEMANTIC_REPAIRS
        ):
            raise ValueError("reservation key has invalid values")
        return canonical_json(key)

    @classmethod
    def _validate_payload(cls, event: str, payload: Mapping[str, Any]) -> None:
        cls._key(payload)
        if event == "reserve":
            expected = {
                "reservation_key",
                "logical_calls",
                "provider_attempts",
                "tokens",
                "prompt_bytes",
                "prompt_framing_tokens",
                "max_output_tokens",
            }
            if set(payload) != expected:
                raise ValueError("reservation payload has unexpected fields")
            integer_fields = (
                "logical_calls",
                "provider_attempts",
                "tokens",
                "prompt_bytes",
                "prompt_framing_tokens",
                "max_output_tokens",
            )
            if any(
                isinstance(payload[name], bool)
                or not isinstance(payload[name], int)
                or payload[name] < 0
                for name in integer_fields
            ):
                raise ValueError("reservation payload has invalid numeric fields")
            if (
                payload["logical_calls"] != 1
                or payload["provider_attempts"] < 1
                or payload["tokens"] < 1
                or payload["max_output_tokens"] < 1
                or payload["tokens"]
                != (
                    payload["prompt_bytes"]
                    + payload["prompt_framing_tokens"]
                    + payload["max_output_tokens"]
                )
                * payload["provider_attempts"]
            ):
                raise ValueError("reservation payload does not match its exposure formula")
            return
        if event != "resolve":
            raise ValueError("unknown reservation ledger event")
        expected = {
            "reservation_key",
            "provider_attempts_actual",
            "input_tokens",
            "output_tokens",
            "outcome",
        }
        if set(payload) != expected:
            raise ValueError("resolution payload has unexpected fields")
        for name in ("provider_attempts_actual", "input_tokens", "output_tokens"):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("resolution payload has invalid usage")
        if payload["provider_attempts_actual"] < 1:
            raise ValueError("resolution provider attempts must be positive")
        if not isinstance(payload["outcome"], str) or not payload["outcome"]:
            raise ValueError("resolution outcome must be nonempty")

    def _read_records(self) -> list[dict[str, Any]]:
        try:
            self.path.lstat()
        except FileNotFoundError:
            return []
        records = []
        previous_hash = "0" * 64
        try:
            descriptor = _open_managed_fd(self.output, self.path, os.O_RDONLY)
            with os.fdopen(descriptor, "r", encoding="utf-8", newline="\n") as handle:
                for sequence, line in enumerate(handle):
                    if not line.endswith("\n"):
                        raise ValueError("reservation ledger has a partial final record")
                    record = json.loads(line)
                    if set(record) != {
                        "schema_version",
                        "sequence",
                        "event",
                        "binding_sha256",
                        "previous_sha256",
                        "payload",
                        "record_sha256",
                    }:
                        raise ValueError("reservation ledger record has unexpected fields")
                    if (
                        record["schema_version"] != self.SCHEMA_VERSION
                        or record["sequence"] != sequence
                        or record["binding_sha256"] != self.binding_sha256
                        or record["previous_sha256"] != previous_hash
                        or record["event"] not in {"reserve", "resolve"}
                    ):
                        raise ValueError("reservation ledger binding or sequence mismatch")
                    base = {key: value for key, value in record.items() if key != "record_sha256"}
                    if record["record_sha256"] != sha256_text(canonical_json(base)):
                        raise ValueError("reservation ledger hash mismatch")
                    self._validate_payload(record["event"], record["payload"])
                    previous_hash = record["record_sha256"]
                    records.append(record)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("reservation ledger is unreadable or corrupt") from exc
        return records

    def _anchor_path(self, sequence: int) -> Path:
        return self.anchor_dir / f"{sequence:020d}.json"

    def _read_anchors(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            metadata = self.anchor_dir.lstat()
        except FileNotFoundError:
            if records:
                raise RuntimeError("reservation ledger exists without durable anchors")
            return []
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("reservation anchor path is linked or non-directory")
        _assert_managed_path(self.output, self.anchor_dir / ".containment-check")
        entries = sorted(self.anchor_dir.iterdir(), key=lambda value: value.name)
        expected_names = [self._anchor_path(index).name for index in range(len(records))]
        if [entry.name for entry in entries] != expected_names:
            raise RuntimeError("reservation anchor high-water does not match ledger length")
        anchors = []
        previous_anchor = "0" * 64
        for sequence, (entry, record) in enumerate(zip(entries, records, strict=True)):
            anchor = _read_managed_json(self.output, entry)
            if set(anchor) != {
                "schema_version",
                "sequence",
                "binding_sha256",
                "record_sha256",
                "previous_anchor_sha256",
                "anchor_sha256",
            }:
                raise ValueError("reservation anchor has unexpected fields")
            base = {key: value for key, value in anchor.items() if key != "anchor_sha256"}
            if (
                anchor["schema_version"] != self.ANCHOR_SCHEMA_VERSION
                or anchor["sequence"] != sequence
                or anchor["binding_sha256"] != self.binding_sha256
                or anchor["record_sha256"] != record["record_sha256"]
                or anchor["previous_anchor_sha256"] != previous_anchor
                or anchor["anchor_sha256"] != sha256_text(canonical_json(base))
            ):
                raise ValueError("reservation anchor binding or hash mismatch")
            previous_anchor = anchor["anchor_sha256"]
            anchors.append(anchor)
        return anchors

    def _checkpoint_at(self, count: int) -> dict[str, Any]:
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(
            self._records
        ):
            raise ValueError("ledger checkpoint count is outside anchored history")
        prefix = "".join(canonical_json(record) + "\n" for record in self._records[:count])
        return {
            "records": count,
            "ledger_prefix_sha256": sha256_text(prefix),
            "chain_head": (
                "0" * 64 if count == 0 else self._records[count - 1]["record_sha256"]
            ),
            "anchor_count": count,
            "anchor_head": (
                "0" * 64 if count == 0 else self._anchors[count - 1]["anchor_sha256"]
            ),
        }

    def verify_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if set(checkpoint) != {
            "records",
            "ledger_prefix_sha256",
            "chain_head",
            "anchor_count",
            "anchor_head",
        }:
            raise RuntimeError("manifest ledger checkpoint schema is invalid")
        count = checkpoint["records"]
        expected = self._checkpoint_at(count)
        if checkpoint != expected:
            raise RuntimeError("manifest ledger checkpoint is not an anchored ledger prefix")

    def _append(self, event: str, payload: dict[str, Any]) -> None:
        self._validate_payload(event, payload)
        with self._lock:
            _ensure_directory(self.output)
            _ensure_directory(self.anchor_dir)
            base = {
                "schema_version": self.SCHEMA_VERSION,
                "sequence": len(self._records),
                "event": event,
                "binding_sha256": self.binding_sha256,
                "previous_sha256": self._last_hash,
                "payload": payload,
            }
            record = {
                **base,
                "record_sha256": sha256_text(canonical_json(base)),
            }
            existed = self.path.exists()
            descriptor = _open_managed_fd(
                self.output,
                self.path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            )
            with os.fdopen(descriptor, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.path.parent)
            if not existed:
                _fsync_directory(self.output)
            anchor_base = {
                "schema_version": self.ANCHOR_SCHEMA_VERSION,
                "sequence": len(self._records),
                "binding_sha256": self.binding_sha256,
                "record_sha256": record["record_sha256"],
                "previous_anchor_sha256": self._last_anchor_hash,
            }
            anchor = {
                **anchor_base,
                "anchor_sha256": sha256_text(canonical_json(anchor_base)),
            }
            anchor_path = self._anchor_path(len(self._records))
            descriptor = _open_managed_fd(
                self.output,
                anchor_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(anchor) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.anchor_dir)
            self._records.append(record)
            self._anchors.append(anchor)
            self._last_hash = record["record_sha256"]
            self._last_anchor_hash = anchor["anchor_sha256"]

    def reserve(self, payload: dict[str, Any]) -> None:
        self._append("reserve", payload)

    def resolve(self, payload: dict[str, Any]) -> None:
        self._append("resolve", payload)

    def reconcile(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
            ledger_sha256 = (
                None
                if not self.path.exists()
                else _managed_sha256(self.output, self.path)
            )
            chain_head = self._last_hash
            checkpoint = self._checkpoint_at(len(records))
        reservations: dict[str, dict[str, Any]] = {}
        resolutions: dict[str, dict[str, Any]] = {}
        for record in records:
            key = self._key(record["payload"])
            if record["event"] == "reserve":
                if key in reservations:
                    raise RuntimeError("duplicate durable reservation key")
                reservations[key] = record["payload"]
            else:
                if key not in reservations or key in resolutions:
                    raise RuntimeError("orphan or duplicate reservation resolution")
                resolutions[key] = record["payload"]
        unresolved = sorted(set(reservations).difference(resolutions))
        overages = []
        for key, resolution in resolutions.items():
            reservation = reservations[key]
            if (
                int(resolution["provider_attempts_actual"]) > int(reservation["provider_attempts"])
                or int(resolution["input_tokens"])
                > int(reservation["prompt_bytes"]) + int(reservation["prompt_framing_tokens"])
                or int(resolution["output_tokens"]) > int(reservation["max_output_tokens"])
                or resolution["outcome"] == "usage_exceeded"
            ):
                overages.append(key)
        return {
            "records": len(records),
            "reservations": reservations,
            "resolutions": resolutions,
            "unresolved": unresolved,
            "overages": sorted(overages),
            "logical_used": sum(int(value["logical_calls"]) for value in reservations.values()),
            "provider_attempts_reserved": sum(
                int(value["provider_attempts"]) for value in reservations.values()
            ),
            "tokens_reserved": sum(int(value["tokens"]) for value in reservations.values()),
            "ledger_sha256": ledger_sha256,
            "chain_head": chain_head,
            "anchor_count": checkpoint["anchor_count"],
            "anchor_head": checkpoint["anchor_head"],
            "checkpoint": checkpoint,
        }


def _artifact_paths(
    output: Path,
    *,
    seed: int,
    family: ScenarioFamily,
    method: RecruitmentMethod,
) -> tuple[Path, Path, Path]:
    stem = f"{method.value}__{family.value}__seed_{seed}"
    return (
        output / "episodes" / f"{stem}.json",
        output / "debug" / f"{stem}.calls.jsonl.gz",
        output / "markers" / f"{stem}.complete.json",
    )


def _path_exists_nofollow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _validate_managed_cell_inventory(
    output: Path,
    jobs: tuple[tuple[int, ScenarioFamily, RecruitmentMethod], ...],
) -> int:
    allowed_root_files = {
        ".e2b-hosted.lock",
        "canary_gate.json",
        "reservation_ledger.jsonl",
        "run_manifest_canary.json",
        "run_manifest_full.json",
    }
    allowed_root_directories = {
        "debug",
        "episodes",
        "markers",
        "reservation_anchors",
    }
    auxiliary_artifacts = 0
    for entry in output.iterdir():
        if entry.name in allowed_root_files:
            _assert_managed_path(output, entry, require_file=True)
            auxiliary_artifacts += int(entry.name == "canary_gate.json")
            continue
        if entry.name in allowed_root_directories:
            metadata = entry.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    f"managed output directory is linked or invalid: {entry}"
                )
            _assert_managed_path(output, entry / ".containment-check")
            continue
        raise RuntimeError(f"unexpected managed output-root entry: {entry}")

    expected = {
        path
        for seed, family, method in jobs
        for path in _artifact_paths(
            output,
            seed=seed,
            family=family,
            method=method,
        )
    }
    present = 0
    for directory_name in ("episodes", "debug", "markers"):
        directory = output / directory_name
        try:
            metadata = directory.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"managed artifact directory is linked or invalid: {directory}")
        _assert_managed_path(output, directory / ".containment-check")
        for path in directory.iterdir():
            if path not in expected:
                raise RuntimeError(f"unexpected managed cell artifact: {path}")
            _assert_managed_path(output, path, require_file=True)
            present += 1
    return present + auxiliary_artifacts


def _assert_artifacts_bound_to_ledger(
    reconciliation: Mapping[str, Any],
    managed_artifact_count: int,
) -> None:
    if int(reconciliation["records"]) == 0 and managed_artifact_count:
        raise RuntimeError(
            "managed cell artifacts exist with an empty anchored ledger; "
            "refusing to repeat provider calls"
        )


def _write_debug_shard(
    root: Path,
    path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically write deterministic gzip JSONL and return both content hashes."""

    content = "".join(canonical_json(record) + "\n" for record in records).encode(
        "ascii"
    )
    compressed = gzip.compress(content, mtime=0)
    _atomic_write_bytes(root, path, compressed)
    return {
        "debug_record_count": len(records),
        "debug_content_sha256": hashlib.sha256(content).hexdigest(),
        "debug_gzip_sha256": _managed_sha256(root, path),
    }


def _load_valid_debug_shard(
    path: Path,
    *,
    expected_count: int,
    expected_content_sha256: str,
    expected_gzip_sha256: str,
    root: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Replay embedded hashes without contacting a provider."""

    root = _normalize_output_root(root or path.parents[1])
    try:
        payload = _read_managed_bytes(root, path)
    except (OSError, RuntimeError):
        return None
    if hashlib.sha256(payload).hexdigest() != expected_gzip_sha256:
        return None
    content_digest = hashlib.sha256()
    count = 0
    records: list[dict[str, Any]] = []
    try:
        with gzip.open(io.BytesIO(payload), "rt", encoding="utf-8", newline="\n") as handle:
            for line in handle:
                content_digest.update(line.encode("ascii"))
                record = json.loads(line)
                if set(record) != {
                    "schema_version",
                    "scenario_id",
                    "family",
                    "seed",
                    "method",
                    "selector",
                    "round_index",
                    "agent_id",
                    "attempt",
                    "prompt_projection",
                    "messages",
                    "raw_completion",
                    "normalized_parse",
                    "response",
                    "hashes",
                }:
                    return None
                hashes = record["hashes"]
                projection = dict(record["prompt_projection"])
                own_private = projection.pop("own_private")
                scenario = generate_scenario(
                    ScenarioFamily(record["family"]),
                    int(record["seed"]),
                )
                profile = scenario.agents[int(record["agent_id"])]
                if own_private != {
                    "true_capabilities": list(profile.true_capabilities),
                    "task_costs": dict(sorted(profile.task_costs.items())),
                }:
                    return None
                public_projection = canonical_json(projection)
                if any(
                    forbidden in public_projection
                    for forbidden in (
                        '"true_capabilities"',
                        '"task_costs"',
                        '"pending_control"',
                        '"oracle',
                    )
                ):
                    return None
                if record["messages"][1]["content"] != (
                    f"AGENT_VIEW_JSON={canonical_json(record['prompt_projection'])}"
                ):
                    return None
                if int(record["attempt"]) > 0 and not any(
                    "Repair it once" in message["content"] for message in record["messages"]
                ):
                    return None
                if set(record["response"]) != {
                    "id",
                    "model_id",
                    "status",
                    "stop_reason",
                    "incomplete_reason",
                    "usage",
                    "transport_attempt_count",
                    "transport_error_count",
                    "transport_error_types",
                }:
                    return None
                if hashes["prompt_projection_sha256"] != sha256_text(
                    canonical_json(record["prompt_projection"])
                ):
                    return None
                if hashes["messages_sha256"] != sha256_text(canonical_json(record["messages"])):
                    return None
                completion = record["raw_completion"]
                expected_completion = None if completion is None else sha256_text(str(completion))
                if hashes["raw_completion_sha256"] != expected_completion:
                    return None
                records.append(record)
                count += 1
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None
    if count != expected_count or content_digest.hexdigest() != expected_content_sha256:
        return None
    return records


def _validate_debug_shard(
    path: Path,
    *,
    expected_count: int,
    expected_content_sha256: str,
    expected_gzip_sha256: str,
    root: Path | None = None,
) -> bool:
    return (
        _load_valid_debug_shard(
            path,
            expected_count=expected_count,
            expected_content_sha256=expected_content_sha256,
            expected_gzip_sha256=expected_gzip_sha256,
            root=root,
        )
        is not None
    )


def persist_completed_episode(
    output: Path,
    episode: dict[str, Any],
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
    launch_binding: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    reservation_reconciliation: dict[str, Any] | None = None,
    require_reservations: bool = False,
) -> dict[str, Any]:
    """Persist the artifact first and its atomic completion marker second."""

    scenario = episode["scenario"]
    method = RecruitmentMethod(episode["config"]["method"])
    family = ScenarioFamily(scenario["family"])
    artifact, debug_path, marker = _artifact_paths(
        output,
        seed=int(scenario["seed"]),
        family=family,
        method=method,
    )
    debug_records = list(episode.get("_debug_call_records", ()))
    public_episode = {key: value for key, value in episode.items() if key != "_debug_call_records"}
    atomic_write_json(output, artifact, public_episode)
    debug_hashes = _write_debug_shard(output, debug_path, debug_records)
    artifact_hash = _managed_sha256(output, artifact)
    debug_valid = _validate_debug_shard(
        debug_path,
        expected_count=debug_hashes["debug_record_count"],
        expected_content_sha256=debug_hashes["debug_content_sha256"],
        expected_gzip_sha256=debug_hashes["debug_gzip_sha256"],
        root=output,
    )
    if not debug_valid:
        raise RuntimeError("debug shard failed immediate replay/hash validation")
    calls = episode["call_ledger"]
    budget_exhausted_decisions = sum(
        str(decision.get("abstain_code", "")).startswith("budget.")
        for round_record in episode["rounds"]
        for decision in round_record["decisions"]
    )
    requested_model = str(episode["provider_model"]["requested"])
    resolved_model = episode["provider_model"]["resolved"]
    reservation_keys = [
        call["reservation_key"] for call in calls if call.get("reservation_key") is not None
    ]
    marker_payload = {
        "schema_version": "alem-dice-e2b-complete-marker-v2",
        "status": "complete",
        "seed": int(scenario["seed"]),
        "family": family.value,
        "method": method.value,
        "artifact": str(artifact.relative_to(output)),
        "artifact_sha256": artifact_hash,
        "debug_artifact": str(debug_path.relative_to(output)),
        **debug_hashes,
        "debug_replay_valid": True,
        "deterministic_episode_hash": episode["deterministic_episode_hash"],
        "terminal_state_hash": episode["terminal_state_hash"],
        "terminal_audit_chain_hash": episode["terminal_audit_chain_hash"],
        "episode_schema_version": episode["schema_version"],
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "logical_calls": len(calls),
        "provider_attempts_actual": sum(int(call["transport_attempt_count"]) for call in calls),
        "provider_attempts_reserved": sum(
            int(call["reserved_provider_attempts"]) for call in calls
        ),
        "tokens_actual": sum(
            int(call["input_tokens"]) + int(call["output_tokens"]) for call in calls
        ),
        "tokens_reserved": sum(int(call["reserved_tokens"]) for call in calls),
        "reservation_keys": reservation_keys,
        "budget_exhausted_decisions": budget_exhausted_decisions,
        "config_sha256": config_sha256,
        "source_hashes": source_hashes,
        "launch_binding": launch_binding,
    }
    atomic_write_json(output, marker, marker_payload)
    validated = load_completed_marker(
        output,
        seed=int(scenario["seed"]),
        family=family,
        method=method,
        config_sha256=config_sha256,
        source_hashes=source_hashes,
        launch_binding=launch_binding,
        protocol=protocol,
        expected_resolved_model=(None if resolved_model is None else str(resolved_model)),
        reservation_reconciliation=reservation_reconciliation,
        require_reservations=require_reservations,
    )
    if validated is None or canonical_json(validated) != canonical_json(marker_payload):
        raise RuntimeError("completed cell failed immediate comprehensive validation")
    return marker_payload


def _flatten_episode_calls(episode: dict[str, Any]) -> list[dict[str, Any]]:
    flattened = []
    for round_record in episode["rounds"]:
        round_index = int(round_record["round_index"])
        decisions = round_record["decisions"]
        if [int(value["agent_id"]) for value in decisions] != sorted(
            int(value["agent_id"]) for value in decisions
        ):
            raise ValueError("episode decisions are not in canonical agent order")
        for decision in decisions:
            if int(decision["round_index"]) != round_index:
                raise ValueError("decision round does not match its round record")
            calls = decision["calls"]
            if [int(call["attempt"]) for call in calls] != list(range(len(calls))):
                raise ValueError("semantic attempts are not contiguous from zero")
            flattened.extend(calls)
        if int(round_record["logical_model_calls"]) != sum(
            len(decision["calls"]) for decision in decisions
        ):
            raise ValueError("round logical call count disagrees with decisions")
    return flattened


def _validate_debug_calls(
    *,
    debug_records: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    seed: int,
    family: ScenarioFamily,
    method: RecruitmentMethod,
    selector: str,
    requested_model: str,
    resolved_model: str,
    scenario_id: str,
    max_output_tokens: int,
    max_prompt_bytes: int,
    max_transport_retries: int,
    expected_prompt_framing_tokens: int,
    require_reservations: bool,
) -> None:
    if len(debug_records) != len(calls):
        raise ValueError("debug and call ledgers have different lengths")
    previous_codes: dict[tuple[int, int], list[str]] = {}
    expected_call_fields = {
        "attempt",
        "semantic_repair",
        "model_id",
        "response_id_hash",
        "raw_completion_sha256",
        "completion_bytes",
        "failure_excerpt",
        "validation_code",
        "valid",
        "parse",
        "prompt_bytes",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "stop_reason",
        "status",
        "incomplete_reason",
        "latency_seconds",
        "transport_attempt_count",
        "transport_error_count",
        "transport_error_types",
        "started_monotonic",
        "ended_monotonic",
        "reservation_key",
        "reserved_provider_attempts",
        "reserved_tokens",
        "prompt_framing_tokens",
    }
    for debug, call in zip(debug_records, calls, strict=True):
        if set(call) != expected_call_fields:
            raise ValueError("call ledger record has unexpected fields")
        if (
            not isinstance(call["valid"], bool)
            or isinstance(call["completion_bytes"], bool)
            or not isinstance(call["completion_bytes"], int)
            or call["completion_bytes"] < 0
            or not isinstance(call["transport_error_types"], list)
            or not all(isinstance(value, str) for value in call["transport_error_types"])
        ):
            raise ValueError("call ledger has invalid scalar or transport types")
        prompt_projection = debug["prompt_projection"]
        round_index = int(debug["round_index"])
        agent_id = int(debug["agent_id"])
        attempt = int(debug["attempt"])
        if (
            debug["schema_version"] != "alem-dice-e2b-call-debug-v1"
            or debug["scenario_id"] != scenario_id
            or int(debug["seed"]) != seed
            or debug["family"] != family.value
            or debug["method"] != method.value
            or debug["selector"] != selector
            or round_index != int(prompt_projection["round_index"])
            or agent_id != int(prompt_projection["agent_id"])
            or attempt != int(call["attempt"])
            or bool(call["semantic_repair"]) != (attempt > 0)
        ):
            raise ValueError("debug call identity mismatch")

        view = AgentPromptView(
            protocol=prompt_projection["protocol"],
            method=prompt_projection["method"],
            policy=prompt_projection["policy"],
            round_index=round_index,
            agent_id=agent_id,
            public_roles=dict(prompt_projection["public_roles"]),
            task_cards=tuple(prompt_projection["task_cards"]),
            public_ledger=dict(prompt_projection["public_ledger"]),
            public_selector=dict(prompt_projection["public_selector"]),
            own_private=dict(prompt_projection["own_private"]),
        )
        if canonical_json(view.as_dict()) != canonical_json(prompt_projection):
            raise ValueError("prompt projection does not match the typed agent view")
        prior = previous_codes.setdefault((round_index, agent_id), [])
        if attempt != len(prior):
            raise ValueError("debug semantic attempts are not contiguous")
        repair_code = None if attempt == 0 else prior[attempt - 1]
        expected_messages = [
            {"role": message.role, "content": message.content}
            for message in build_messages(view, repair_code=repair_code)
        ]
        if canonical_json(debug["messages"]) != canonical_json(expected_messages):
            raise ValueError("archived messages do not reproduce from the prompt projection")
        prompt_bytes = sum(len(message["content"].encode("utf-8")) for message in expected_messages)
        if int(call["prompt_bytes"]) != prompt_bytes or prompt_bytes > max_prompt_bytes:
            raise ValueError("call prompt byte count does not match archived messages")

        raw = debug["raw_completion"]
        if not isinstance(raw, str):
            raise ValueError("completed provider response must archive exact text")
        expected_raw_hash = None if raw is None else sha256_text(raw)
        if (
            call["raw_completion_sha256"] != expected_raw_hash
            or debug["hashes"]["raw_completion_sha256"] != expected_raw_hash
            or call["completion_bytes"]
            != (0 if raw is None else len(raw.encode("utf-8", errors="replace")))
        ):
            raise ValueError("exact raw completion hash or byte count mismatch")
        response = debug["response"]
        response_id = response["id"]
        if set(debug["hashes"]) != {
            "prompt_projection_sha256",
            "messages_sha256",
            "raw_completion_sha256",
        } or set(debug["normalized_parse"]) != {
            "validation_code",
            "protocol_parse",
            "record",
        }:
            raise ValueError("debug hash or normalized-parse schema is not exact")
        if (
            response["model_id"] != call["model_id"]
            or response["model_id"] != resolved_model
            or not resolved_model_is_accepted(requested_model, response["model_id"])
            or response["status"] != "completed"
            or call["status"] != response["status"]
            or call["stop_reason"] != response["stop_reason"]
            or response["incomplete_reason"] is not None
            or call["incomplete_reason"] is not None
            or not isinstance(response_id, str)
            or not response_id.strip()
            or call["response_id_hash"] != sha256_text(response_id)
        ):
            raise ValueError("provider response binding is invalid")
        usage = response["usage"]
        if set(usage) != {
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "cache_write_tokens",
        }:
            raise ValueError("provider usage schema is not exact")
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "cache_write_tokens",
        ):
            if (
                isinstance(call[name], bool)
                or not isinstance(call[name], int)
                or isinstance(usage[name], bool)
                or not isinstance(usage[name], int)
                or call[name] != usage[name]
                or call[name] < 0
            ):
                raise ValueError("call and provider usage ledgers disagree")
        for name in ("transport_attempt_count", "transport_error_count"):
            if (
                isinstance(response[name], bool)
                or not isinstance(response[name], int)
                or response[name] < 0
            ):
                raise ValueError("provider transport accounting has invalid types")
        if (
            isinstance(call["transport_attempt_count"], bool)
            or not isinstance(call["transport_attempt_count"], int)
            or isinstance(call["transport_error_count"], bool)
            or not isinstance(call["transport_error_count"], int)
            or int(call["transport_attempt_count"]) < 1
            or int(call["transport_error_count"]) < 0
            or int(call["transport_error_count"]) != len(call["transport_error_types"])
            or int(call["transport_error_count"]) > int(call["transport_attempt_count"]) - 1
            or int(call["transport_attempt_count"]) != int(response["transport_attempt_count"])
            or int(call["transport_error_count"]) != int(response["transport_error_count"])
            or call["transport_error_types"] != response["transport_error_types"]
            or int(call["transport_attempt_count"]) > int(call["reserved_provider_attempts"])
            or int(call["reserved_provider_attempts"]) != 1 + max_transport_retries
            or int(call["prompt_framing_tokens"]) != expected_prompt_framing_tokens
            or int(call["reserved_tokens"])
            != (prompt_bytes + expected_prompt_framing_tokens + max_output_tokens)
            * (1 + max_transport_retries)
            or int(call["input_tokens"]) > prompt_bytes + expected_prompt_framing_tokens
            or int(call["output_tokens"]) > max_output_tokens
        ):
            raise ValueError("actual provider usage exceeds its reservation")
        expected_key = {
            "seed": seed,
            "family": family.value,
            "method": method.value,
            "round_index": round_index,
            "agent_id": agent_id,
            "semantic_attempt": attempt,
        }
        reservation_key = call["reservation_key"]
        if require_reservations and reservation_key != expected_key:
            raise ValueError("hosted call does not have its exact cell reservation key")
        if reservation_key is not None and reservation_key != expected_key:
            raise ValueError("call reservation key does not match its identity")

        normalized = debug["normalized_parse"]
        if normalized["validation_code"] != call["validation_code"]:
            raise ValueError("normalized validation codes disagree")
        if bool(call["valid"]) != (call["validation_code"] in {"valid", "abstain"}):
            raise ValueError("call valid flag disagrees with its validation code")
        expected_excerpt = None if call["valid"] else redact_failure_excerpt(raw)
        if call["failure_excerpt"] != expected_excerpt:
            raise ValueError("failure excerpt does not reproduce from exact raw text")
        prior.append(str(call["validation_code"]))
        if raw == "ABSTAIN":
            if call["validation_code"] != "abstain" or normalized["protocol_parse"] is not None:
                raise ValueError("ABSTAIN was not archived canonically")
            continue
        if raw is None:
            raise ValueError("completed cells may not contain transport-exception calls")
        reparsed = parse_tfp1(raw, max_bytes=MAX_CONTROL_BYTES)
        reparse_projection = {
            "valid": reparsed.valid,
            "code": reparsed.code,
            "payload_bytes": reparsed.payload_bytes,
        }
        if (
            normalized["protocol_parse"] != reparse_projection
            or call["parse"] != reparse_projection
        ):
            raise ValueError("archived completion does not reproduce its TFP1 parse")
        normalized_record = normalized["record"]
        if reparsed.valid:
            if (
                reparsed.record is None
                or normalized_record is None
                or normalized_record["canonical"] != reparsed.record.render()
                or normalized_record["typed"] != reparsed.record.as_dict()
            ):
                raise ValueError("archived typed record does not match raw completion")
        elif normalized_record is not None:
            raise ValueError("invalid raw completion has a normalized record")


def _record_from_projection(value: dict[str, Any]) -> RecruitmentRecord:
    fields = dict(value)
    kind = RecordKind(fields.pop("kind"))
    task_id = fields.pop("task_id")
    for name in ("members", "capabilities", "demand"):
        if name in fields:
            fields[name] = tuple(fields[name])
    record = RecruitmentRecord(kind=kind, task_id=task_id, **fields)
    if record.as_dict() != value:
        raise ValueError("decision record is not a canonical typed TFP1 projection")
    return record


def _peak_call_concurrency(calls: list[dict[str, Any]]) -> int:
    boundaries = []
    for call in calls:
        started = float(call["started_monotonic"])
        ended = float(call["ended_monotonic"])
        if not math.isfinite(started) or not math.isfinite(ended) or ended < started:
            raise ValueError("call timing is non-finite or reversed")
        boundaries.extend(((started, 1), (ended, -1)))
    boundaries.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    peak = 0
    for _, delta in boundaries:
        active += delta
        peak = max(peak, active)
    return peak


def _validate_episode_counts(episode: dict[str, Any], config: ScreenConfig) -> None:
    rounds = episode["rounds"]
    if [int(value["round_index"]) for value in rounds] != list(range(len(rounds))):
        raise ValueError("episode round indices are not contiguous from zero")
    all_calls = []
    for round_record in rounds:
        round_index = int(round_record["round_index"])
        eligible = [int(value) for value in round_record["eligible_agents"]]
        decision_ids = [int(value["agent_id"]) for value in round_record["decisions"]]
        if (
            eligible != sorted(set(eligible))
            or not set(eligible).issubset(AGENT_IDS)
            or decision_ids != eligible
        ):
            raise ValueError("eligible-agent and decision identities disagree")
        round_calls = []
        for decision in round_record["decisions"]:
            calls = decision["calls"]
            record_projection = decision["record"]
            record = (
                None if record_projection is None else _record_from_projection(record_projection)
            )
            valid_records = [call for call in calls if call["validation_code"] == "valid"]
            if (
                (record is None and valid_records)
                or (record is not None and (not calls or calls[-1]["validation_code"] != "valid"))
                or (record is not None and decision["abstain_code"] is not None)
                or (record is None and not isinstance(decision["abstain_code"], str))
            ):
                raise ValueError("decision outcome disagrees with its call sequence")
            semantic_repairs = sum(int(call["attempt"]) > 0 for call in calls)
            if int(decision["semantic_repairs"]) != semantic_repairs:
                raise ValueError("decision semantic-repair count disagrees")
            semantic_payload = {
                "agent_id": int(decision["agent_id"]),
                "round_index": round_index,
                "record": None if record is None else record.render(),
                "abstain_code": decision["abstain_code"],
                "completion_hashes": [call["raw_completion_sha256"] for call in calls],
                "validation_codes": [call["validation_code"] for call in calls],
            }
            if decision["deterministic_hash"] != sha256_text(canonical_json(semantic_payload)):
                raise ValueError("decision deterministic hash does not reproduce")
            round_calls.extend(calls)
        if (
            int(round_record["logical_model_calls"]) != len(round_calls)
            or int(round_record["provider_requests"])
            != sum(int(call["transport_attempt_count"]) for call in round_calls)
            or int(round_record["peak_concurrent_calls"]) != _peak_call_concurrency(round_calls)
        ):
            raise ValueError("round call counts or concurrency disagree")
        all_calls.extend(round_calls)
    early = episode["early_stop"]
    drain = episode["drain"]
    if (
        int(early["requested_acting_rounds"]) != config.rounds
        or int(early["executed_acting_rounds"]) != len(rounds)
        or int(drain["round_index"]) != len(rounds)
        or len(rounds) > config.rounds
    ):
        raise ValueError("acting horizon, executed rounds, or drain round disagree")
    retry = episode["retry_ledger"]
    if retry != {
        "semantic_repairs": sum(int(call["attempt"]) > 0 for call in all_calls),
        "transport_attempts": sum(int(call["transport_attempt_count"]) for call in all_calls),
        "transport_errors": sum(int(call["transport_error_count"]) for call in all_calls),
    }:
        raise ValueError("episode retry ledger disagrees with calls")
    expected_tokens = {
        key: sum(int(call[key]) for call in all_calls)
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "cache_write_tokens",
        )
    }
    if episode["token_ledger"] != expected_tokens:
        raise ValueError("episode token ledger disagrees with calls")
    if (
        episode["call_caps"]["episode_limit"] != config.max_calls_per_episode
        or episode["call_caps"]["episode_used"] != len(all_calls)
        or len(all_calls) > int(config.max_calls_per_episode)
    ):
        raise ValueError("episode call budget ledger disagrees with calls")
    expected_latency_sum = sum(float(call["latency_seconds"]) for call in all_calls)
    if not math.isclose(
        float(episode["latency_ledger"]["sum_call_latency_seconds"]),
        expected_latency_sum,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("episode latency ledger disagrees with calls")


def _expected_analysis(scenario: Any, directory: TeamDirectory) -> dict[str, Any]:
    locked = {
        task_id: list(state.lease.members)
        for task_id, state in sorted(directory.tasks.items())
        if state.phase is TaskPhase.LOCKED and state.lease is not None
    }
    lock_rounds = {
        str(record["payload"]["task_id"]): int(record["payload"]["round_index"])
        for record in directory.audit_records()
        if record["category"] == "transition"
        and record["payload"]["accepted"]
        and record["payload"]["code"] == "team.locked"
    }
    profiles = {profile.agent_id: profile for profile in scenario.agents}
    tasks = {task.task_id: task for task in scenario.tasks}
    true_feasible = {}
    for task_id, members in locked.items():
        task = tasks[task_id]
        true_feasible[task_id] = (
            len(members) == task.required_size
            and len(set(members)) == len(members)
            and all(
                sum(profiles[member].true_capabilities[dimension] for member in members) >= demand
                for dimension, demand in enumerate(task.demand)
                if demand > 0
            )
        )
    oracle = true_information_oracle(scenario.tasks, scenario.agents)
    oracle_reward = int(oracle.total_completed_reward)
    oracle_completed_tasks = len(oracle.completed_tasks)
    true_feasible_locked_tasks = sum(true_feasible.values())
    achieved_reward = sum(
        task.reward for task in scenario.tasks if true_feasible.get(task.task_id, False)
    )
    return {
        "locked_rosters": locked,
        "lock_rounds": lock_rounds,
        "true_feasible_locks": true_feasible,
        "true_feasible_locked_tasks": true_feasible_locked_tasks,
        "oracle_completed_tasks": oracle_completed_tasks,
        "oracle_allocation_coverage": (
            true_feasible_locked_tasks / oracle_completed_tasks if oracle_completed_tasks else 1.0
        ),
        "oracle_reward": oracle_reward,
        "achieved_reward": achieved_reward,
        "normalized_reward": (
            float(Fraction(achieved_reward, oracle_reward)) if oracle_reward else 1.0
        ),
    }


def _strict_completed_cell(
    output: Path,
    *,
    seed: int,
    family: ScenarioFamily,
    method: RecruitmentMethod,
    config_sha256: str,
    source_hashes: dict[str, str],
    launch_binding: dict[str, Any] | None,
    protocol: dict[str, Any] | None,
    expected_resolved_model: str | None,
    reservation_reconciliation: dict[str, Any] | None,
    require_reservations: bool,
) -> dict[str, Any]:
    artifact, debug_path, marker_path = _artifact_paths(
        output,
        seed=seed,
        family=family,
        method=method,
    )
    for path in (marker_path, artifact, debug_path):
        _assert_managed_path(output, path, require_file=True)
    marker = _read_managed_json(output, marker_path)
    expected_marker_fields = {
        "schema_version",
        "status",
        "seed",
        "family",
        "method",
        "artifact",
        "artifact_sha256",
        "debug_artifact",
        "debug_record_count",
        "debug_content_sha256",
        "debug_gzip_sha256",
        "debug_replay_valid",
        "deterministic_episode_hash",
        "terminal_state_hash",
        "terminal_audit_chain_hash",
        "episode_schema_version",
        "requested_model",
        "resolved_model",
        "logical_calls",
        "provider_attempts_actual",
        "provider_attempts_reserved",
        "tokens_actual",
        "tokens_reserved",
        "reservation_keys",
        "budget_exhausted_decisions",
        "config_sha256",
        "source_hashes",
        "launch_binding",
    }
    expected_artifact = str(artifact.relative_to(output))
    expected_debug = str(debug_path.relative_to(output))
    if (
        not isinstance(marker, dict)
        or set(marker) != expected_marker_fields
        or marker.get("schema_version") != "alem-dice-e2b-complete-marker-v2"
        or marker.get("status") != "complete"
        or marker.get("seed") != seed
        or marker.get("family") != family.value
        or marker.get("method") != method.value
        or marker.get("artifact") != expected_artifact
        or marker.get("debug_artifact") != expected_debug
        or marker.get("config_sha256") != config_sha256
        or marker.get("source_hashes") != source_hashes
        or marker.get("launch_binding") != launch_binding
        or marker.get("artifact_sha256") != _managed_sha256(output, artifact)
        or marker.get("debug_replay_valid") is not True
    ):
        raise ValueError("completed marker identity or binding mismatch")
    episode = _read_managed_json(output, artifact)
    expected_episode_fields = {
        "schema_version",
        "scenario",
        "config",
        "provider_model",
        "selector",
        "rounds",
        "drain",
        "early_stop",
        "transition_ledger",
        "call_ledger",
        "retry_ledger",
        "token_ledger",
        "latency_ledger",
        "call_caps",
        "analysis_only",
        "directory_replay",
        "replay_hash_match",
        "terminal_state_hash",
        "terminal_audit_chain_hash",
        "deterministic_episode_hash",
    }
    scenario = generate_scenario(family, seed)
    expected_scenario = {
        "scenario_id": scenario.scenario_id,
        "family": family.value,
        "seed": seed,
    }
    if (
        not isinstance(episode, dict)
        or set(episode) != expected_episode_fields
        or episode.get("schema_version") != SCHEMA_VERSION
        or marker.get("episode_schema_version") != SCHEMA_VERSION
        or episode.get("scenario") != expected_scenario
        or episode.get("config", {}).get("method") != method.value
        or episode.get("selector")
        != (
            "joint_exact_allocation"
            if method is RecruitmentMethod.OPEN_VOLUNTEER
            else "native_mutual_reciprocal"
        )
    ):
        raise ValueError("episode schema, scenario, method, or selector mismatch")
    config = ScreenConfig(**episode["config"])
    if config.as_dict() != episode["config"]:
        raise ValueError("episode config is not a canonical ScreenConfig")
    if protocol is not None:
        expected_config = ScreenConfig(method=method).as_dict()
        protocol_projection = {
            "rounds": protocol["rounds"],
            "model_id": protocol["model_id"],
            "reasoning_effort": protocol["reasoning_effort"],
            "max_output_tokens": protocol["max_output_tokens"],
            "max_prompt_bytes": protocol["max_prompt_bytes"],
            "max_semantic_repairs": protocol["max_semantic_repairs"],
            "max_transport_retries": protocol["max_transport_retries"],
        }
        if episode["config"] != expected_config or any(
            expected_config[key] != value for key, value in protocol_projection.items()
        ):
            raise ValueError("episode generation config differs from protocol")
    requested_model = str(episode["provider_model"]["requested"])
    resolved_model = episode["provider_model"]["resolved"]
    if (
        set(episode["provider_model"]) != {"requested", "resolved"}
        or requested_model != episode["config"]["model_id"]
        or marker.get("requested_model") != requested_model
        or not isinstance(resolved_model, str)
        or marker.get("resolved_model") != resolved_model
        or not resolved_model_is_accepted(requested_model, resolved_model)
        or (expected_resolved_model is not None and resolved_model != expected_resolved_model)
    ):
        raise ValueError("requested/resolved provider model binding mismatch")

    calls = episode["call_ledger"]
    _validate_episode_counts(episode, config)
    flattened = _flatten_episode_calls(episode)
    if canonical_json(flattened) != canonical_json(calls):
        raise ValueError("episode call ledger differs from its round decisions")
    debug_count = int(marker.get("debug_record_count", -1))
    debug_records = _load_valid_debug_shard(
        debug_path,
        expected_count=debug_count,
        expected_content_sha256=str(marker.get("debug_content_sha256", "")),
        expected_gzip_sha256=str(marker.get("debug_gzip_sha256", "")),
        root=output,
    )
    if debug_records is None:
        raise ValueError("debug shard failed hash/privacy validation")
    _validate_debug_calls(
        debug_records=debug_records,
        calls=calls,
        seed=seed,
        family=family,
        method=method,
        selector=episode["selector"],
        requested_model=requested_model,
        resolved_model=resolved_model,
        scenario_id=scenario.scenario_id,
        max_output_tokens=config.max_output_tokens,
        max_prompt_bytes=config.max_prompt_bytes,
        max_transport_retries=config.max_transport_retries,
        expected_prompt_framing_tokens=(
            DEFAULT_PROMPT_FRAMING_TOKENS
            if protocol is None
            else int(protocol["prompt_framing_tokens"])
        ),
        require_reservations=require_reservations,
    )
    replay_payload = episode["directory_replay"]
    replayed = TeamDirectory.replay(replay_payload)
    reported_transitions = [
        transition
        for round_record in episode["rounds"]
        for transition in round_record["delivery_transitions"]
    ] + list(episode["drain"]["transitions"])
    replay_transitions = [
        record["payload"]
        for record in replayed.audit_records()
        if record["category"] == "transition" and record["payload"]["code"] != "task.registered"
    ]
    reported_ordinary = [
        delivery
        for round_record in episode["rounds"]
        for delivery in round_record["ordinary_deliveries"]
    ] + list(episode["drain"]["ordinary_deliveries"])
    replay_ordinary = [
        record["payload"]
        for record in replayed.audit_records()
        if record["category"] == "ordinary_delivery"
    ]
    expected_task_cards = sorted(
        (task.as_dict() for task in scenario.tasks),
        key=lambda value: value["task_id"],
    )
    replay_task_cards = sorted(
        (state.card.as_dict() for state in replayed.tasks.values()),
        key=lambda value: value["task_id"],
    )
    if (
        replay_payload["agent_ids"] != list(AGENT_IDS)
        or replay_payload["method"] != method.value
        or int(replay_payload["seed"]) != seed
        or int(replay_payload["max_control_bytes"]) != MAX_CONTROL_BYTES
        or replay_task_cards != expected_task_cards
        or episode["transition_ledger"] != reported_transitions
        or reported_transitions != replay_transitions
        or reported_ordinary != replay_ordinary
        or episode["analysis_only"] != _expected_analysis(scenario, replayed)
    ):
        raise ValueError("directory replay identity, tasks, or derived analysis disagree")
    deterministic_projection = {
        "scenario_id": scenario.scenario_id,
        "method": method.value,
        "selector": episode["selector"],
        "decisions": [
            decision["deterministic_hash"]
            for round_record in episode["rounds"]
            for decision in round_record["decisions"]
        ],
        "early_stop": {
            "executed_acting_rounds": episode["early_stop"]["executed_acting_rounds"],
            "stall_rounds": episode["early_stop"]["stall_rounds"],
            "reason": episode["early_stop"]["reason"],
            "drain_round": episode["drain"]["round_index"],
        },
        "terminal_state_hash": replayed.state_hash(),
        "audit_chain_hash": replayed.audit_chain_hash,
        "resolved_model": resolved_model,
    }
    reproduced_episode_hash = sha256_text(canonical_json(deterministic_projection))
    if (
        episode.get("replay_hash_match") is not True
        or replayed.state_hash() != episode["terminal_state_hash"]
        or replayed.audit_chain_hash != episode["terminal_audit_chain_hash"]
        or marker.get("terminal_state_hash") != episode["terminal_state_hash"]
        or marker.get("terminal_audit_chain_hash") != episode["terminal_audit_chain_hash"]
        or episode["deterministic_episode_hash"] != reproduced_episode_hash
        or marker.get("deterministic_episode_hash") != episode["deterministic_episode_hash"]
    ):
        raise ValueError("directory replay or terminal hashes disagree")
    budget_exhausted = sum(
        str(decision.get("abstain_code", "")).startswith("budget.")
        for round_record in episode["rounds"]
        for decision in round_record["decisions"]
    )
    expected_metrics = {
        "logical_calls": len(calls),
        "provider_attempts_actual": sum(int(call["transport_attempt_count"]) for call in calls),
        "provider_attempts_reserved": sum(
            int(call["reserved_provider_attempts"]) for call in calls
        ),
        "tokens_actual": sum(
            int(call["input_tokens"]) + int(call["output_tokens"]) for call in calls
        ),
        "tokens_reserved": sum(int(call["reserved_tokens"]) for call in calls),
        "budget_exhausted_decisions": budget_exhausted,
    }
    if any(marker.get(key) != value for key, value in expected_metrics.items()):
        raise ValueError("marker call or budget counts disagree with episode")
    reservation_keys = [
        call["reservation_key"] for call in calls if call.get("reservation_key") is not None
    ]
    if marker.get("reservation_keys") != reservation_keys or len(
        {canonical_json(key) for key in reservation_keys}
    ) != len(reservation_keys):
        raise ValueError("marker reservation keys are missing, duplicate, or reordered")
    if require_reservations and len(reservation_keys) != len(calls):
        raise ValueError("hosted completed cell has a call without a durable reservation")
    if reservation_reconciliation is not None:
        reservations = reservation_reconciliation["reservations"]
        resolutions = reservation_reconciliation["resolutions"]
        cell_keys = {
            key
            for key, value in reservations.items()
            if value["reservation_key"]["seed"] == seed
            and value["reservation_key"]["family"] == family.value
            and value["reservation_key"]["method"] == method.value
        }
        expected_keys = {canonical_json(key) for key in reservation_keys}
        if (
            cell_keys != expected_keys
            or not expected_keys.issubset(resolutions)
            or expected_keys.intersection(reservation_reconciliation.get("unresolved", ()))
            or expected_keys.intersection(reservation_reconciliation.get("overages", ()))
        ):
            raise ValueError("durable reservation ledger does not exactly cover the cell")
        for call in calls:
            key = canonical_json(call["reservation_key"])
            reservation = reservations[key]
            resolution = resolutions[key]
            if (
                int(reservation["provider_attempts"]) != int(call["reserved_provider_attempts"])
                or int(reservation["tokens"]) != int(call["reserved_tokens"])
                or int(reservation["prompt_bytes"]) != int(call["prompt_bytes"])
                or int(reservation["prompt_framing_tokens"]) != int(call["prompt_framing_tokens"])
                or int(reservation["max_output_tokens"]) != config.max_output_tokens
                or int(resolution["provider_attempts_actual"])
                != int(call["transport_attempt_count"])
                or int(resolution["input_tokens"]) != int(call["input_tokens"])
                or int(resolution["output_tokens"]) != int(call["output_tokens"])
                or resolution["outcome"] != call["validation_code"]
            ):
                raise ValueError("call reservation differs from durable ledger")
    return marker


def load_completed_marker(
    output: Path,
    *,
    seed: int,
    family: ScenarioFamily,
    method: RecruitmentMethod,
    config_sha256: str,
    source_hashes: dict[str, str],
    launch_binding: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    expected_resolved_model: str | None = None,
    reservation_reconciliation: dict[str, Any] | None = None,
    require_reservations: bool = False,
) -> dict[str, Any] | None:
    """Return a comprehensively validated completed cell, else fail closed."""

    try:
        return _strict_completed_cell(
            output,
            seed=seed,
            family=family,
            method=method,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
            launch_binding=launch_binding,
            protocol=protocol,
            expected_resolved_model=expected_resolved_model,
            reservation_reconciliation=reservation_reconciliation,
            require_reservations=require_reservations,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        return None


def _load_completed_or_pending(
    output: Path,
    *,
    seed: int,
    family: ScenarioFamily,
    method: RecruitmentMethod,
    config_sha256: str,
    source_hashes: dict[str, str],
    launch_binding: dict[str, Any],
    protocol: dict[str, Any],
    expected_resolved_model: str | None,
    reservation_reconciliation: dict[str, Any],
) -> dict[str, Any] | None:
    paths = _artifact_paths(
        output,
        seed=seed,
        family=family,
        method=method,
    )
    presence = [_path_exists_nofollow(path) for path in paths]
    if not any(presence):
        return None
    if not all(presence):
        raise RuntimeError(
            "partial managed cell artifacts exist; refusing to classify the cell as pending"
        )
    marker = load_completed_marker(
        output,
        seed=seed,
        family=family,
        method=method,
        config_sha256=config_sha256,
        source_hashes=source_hashes,
        launch_binding=launch_binding,
        protocol=protocol,
        expected_resolved_model=expected_resolved_model,
        reservation_reconciliation=reservation_reconciliation,
        require_reservations=True,
    )
    if marker is None:
        raise RuntimeError(
            "existing managed cell artifacts failed comprehensive validation; "
            "refusing to repeat provider calls"
        )
    return marker


def _compute_canary_gate(
    output: Path,
    markers: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
    launch_binding: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    reservation_reconciliation: dict[str, Any] | None = None,
    require_reservations: bool = False,
) -> dict[str, Any]:
    expected_cells = set(CANARY_CELLS)
    supplied: dict[
        tuple[int, ScenarioFamily, RecruitmentMethod],
        dict[str, Any],
    ] = {}
    for marker in markers:
        try:
            key = (
                int(marker["seed"]),
                ScenarioFamily(marker["family"]),
                RecruitmentMethod(marker["method"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("canary marker identity is invalid") from exc
        if key not in expected_cells or key in supplied:
            raise ValueError("canary marker set has an unexpected or duplicate cell")
        supplied[key] = marker
    if set(supplied) != expected_cells:
        raise ValueError("canary requires the exact frozen four-cell matrix")

    cell_payloads = []
    requested_models = set()
    resolved_models = set()
    for seed, family, method in CANARY_CELLS:
        marker = supplied[(seed, family, method)]
        validated_marker = load_completed_marker(
            output,
            seed=seed,
            family=family,
            method=method,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
            launch_binding=launch_binding,
            protocol=protocol,
            expected_resolved_model=None,
            reservation_reconciliation=reservation_reconciliation,
            require_reservations=require_reservations,
        )
        if validated_marker is None or canonical_json(marker) != canonical_json(
            validated_marker
        ):
            raise ValueError(
                "canary marker is not canonically equal to its validated cell"
            )
        artifact = output / validated_marker["artifact"]
        episode = _read_managed_json(output, artifact)
        calls = episode["call_ledger"]
        transport_attempts = sum(
            int(call["transport_attempt_count"]) for call in calls
        )
        transport_errors = sum(
            int(call["transport_error_count"]) for call in calls
        )
        invalid_calls = sum(not bool(call["valid"]) for call in calls)
        initial_calls = sum(int(call["attempt"]) == 0 for call in calls)
        repair_calls = sum(int(call["attempt"]) > 0 for call in calls)
        max_output_truncations = sum(
            call["status"] == "incomplete"
            and call["incomplete_reason"] == "max_output_tokens"
            for call in calls
        )
        gates = {
            "marker_schema_bound": (
                validated_marker["schema_version"]
                == "alem-dice-e2b-complete-marker-v2"
            ),
            "episode_schema_bound": episode["schema_version"] == SCHEMA_VERSION,
            "expected_cell_bound": (
                validated_marker["seed"] == seed
                and validated_marker["family"] == family.value
                and validated_marker["method"] == method.value
                and episode["scenario"]["seed"] == seed
                and episode["scenario"]["family"] == family.value
                and episode["config"]["method"] == method.value
            ),
            "requested_model_bound": (
                validated_marker["requested_model"] == DEFAULT_MODEL
                and episode["provider_model"]["requested"] == DEFAULT_MODEL
            ),
            "resolved_model_accepted": (
                validated_marker["resolved_model"]
                == episode["provider_model"]["resolved"]
                and resolved_model_is_accepted(
                    validated_marker["requested_model"],
                    validated_marker["resolved_model"],
                )
            ),
            "replay_hash_match": bool(episode["replay_hash_match"]),
            "debug_replay_valid": bool(validated_marker["debug_replay_valid"]),
            "formed_true_feasible_team": (
                int(episode["analysis_only"]["true_feasible_locked_tasks"]) >= 1
            ),
            "no_max_output_truncation": max_output_truncations == 0,
            "invalid_call_rate_at_most_25_percent": (
                invalid_calls / len(calls) <= 0.25 if calls else False
            ),
            "semantic_repair_rate_at_most_25_percent": (
                repair_calls / initial_calls <= 0.25 if initial_calls else False
            ),
            "transport_error_rate_at_most_5_percent": (
                transport_errors / transport_attempts <= 0.05
                if transport_attempts
                else False
            ),
            "no_budget_exhaustion": (
                int(validated_marker["budget_exhausted_decisions"]) == 0
            ),
        }
        requested_models.add(validated_marker["requested_model"])
        resolved_models.add(validated_marker["resolved_model"])
        cell_payloads.append(
            {
                "cell": {
                    "seed": seed,
                    "family": family.value,
                    "method": method.value,
                },
                "status": "pass" if all(gates.values()) else "fail",
                "gates": gates,
                "diagnostics": {
                    "logical_calls": len(calls),
                    "initial_calls": initial_calls,
                    "semantic_repair_calls": repair_calls,
                    "invalid_calls": invalid_calls,
                    "max_output_truncations": max_output_truncations,
                    "transport_attempts": transport_attempts,
                    "transport_errors": transport_errors,
                    "executed_acting_rounds": episode["early_stop"][
                        "executed_acting_rounds"
                    ],
                    "early_stop_reason": episode["early_stop"]["reason"],
                    "true_feasible_locked_tasks": episode["analysis_only"][
                        "true_feasible_locked_tasks"
                    ],
                    "oracle_allocation_coverage": episode["analysis_only"][
                        "oracle_allocation_coverage"
                    ],
                },
                "marker_sha256": sha256_text(canonical_json(validated_marker)),
                "artifact_sha256": validated_marker["artifact_sha256"],
                "debug_content_sha256": validated_marker[
                    "debug_content_sha256"
                ],
                "debug_gzip_sha256": validated_marker["debug_gzip_sha256"],
            }
        )

    global_gates = {
        "exact_four_cell_matrix": len(cell_payloads) == len(CANARY_CELLS),
        "both_methods_covered": {
            cell["cell"]["method"] for cell in cell_payloads
        }
        == {method.value for method in FROZEN_METHODS},
        "single_and_two_disjoint_covered": {
            cell["cell"]["family"] for cell in cell_payloads
        }
        == {family.value for family in CANARY_FAMILIES},
        "all_cells_pass": all(
            cell["status"] == "pass" for cell in cell_payloads
        ),
        "requested_model_consistent": requested_models == {DEFAULT_MODEL},
        "resolved_model_consistent": len(resolved_models) == 1,
    }
    resolved_model = (
        next(iter(resolved_models)) if len(resolved_models) == 1 else None
    )
    return {
        "schema_version": "alem-dice-e2b-canary-gate-v3",
        "status": "pass" if all(global_gates.values()) else "fail",
        "canary_design": {
            "seed": 22000,
            "families": [family.value for family in CANARY_FAMILIES],
            "methods": [method.value for method in FROZEN_METHODS],
            "cell_count": len(CANARY_CELLS),
            "promotion_requires_every_cell": True,
        },
        "model_binding": {
            "requested": DEFAULT_MODEL,
            "resolved": resolved_model,
        },
        "gates": global_gates,
        "cells": cell_payloads,
        "config_sha256": config_sha256,
        "source_hashes": source_hashes,
        "launch_binding": launch_binding,
    }


def _write_canary_gate(
    output: Path,
    markers: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
    launch_binding: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    reservation_reconciliation: dict[str, Any] | None = None,
    require_reservations: bool = False,
) -> dict[str, Any]:
    payload = _compute_canary_gate(
        output,
        markers,
        config_sha256=config_sha256,
        source_hashes=source_hashes,
        launch_binding=launch_binding,
        protocol=protocol,
        reservation_reconciliation=reservation_reconciliation,
        require_reservations=require_reservations,
    )
    atomic_write_json(output, output / "canary_gate.json", payload)
    return payload


def _load_passing_canary_gate(
    output: Path,
    markers: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
    launch_binding: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    reservation_reconciliation: dict[str, Any] | None = None,
    require_reservations: bool = False,
) -> dict[str, Any] | None:
    path = output / "canary_gate.json"
    if not path.exists():
        return None
    try:
        payload = _read_managed_json(output, path)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        expected = _compute_canary_gate(
            output,
            markers,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
            launch_binding=launch_binding,
            protocol=protocol,
            reservation_reconciliation=reservation_reconciliation,
            require_reservations=require_reservations,
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if (
        payload.get("status") != "pass"
        or not all(payload.get("gates", {}).values())
        or canonical_json(payload) != canonical_json(expected)
    ):
        return None
    return payload


def _protocol_config(
    *,
    logical_call_cap: int,
    provider_attempt_cap: int,
    token_exposure_cap: int,
) -> dict[str, Any]:
    estimate = estimate_campaign(
        seed_count=len(FROZEN_SEEDS),
        family_count=len(FROZEN_FAMILIES),
        method_count=len(FROZEN_METHODS),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_revision": "e2b-v3-high-reasoning-four-cell-canary",
        "seeds": list(FROZEN_SEEDS),
        "scenario_families": [family.value for family in FROZEN_FAMILIES],
        "methods": [method.value for method in FROZEN_METHODS],
        "canary_cells": [
            {
                "seed": seed,
                "family": family.value,
                "method": method.value,
            }
            for seed, family, method in CANARY_CELLS
        ],
        "promotion_requires_every_canary_cell": True,
        "policy": "public_sweep",
        "rounds": DEFAULT_ROUNDS,
        "agent_count": 6,
        "model_id": DEFAULT_MODEL,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "max_prompt_bytes": DEFAULT_MAX_PROMPT_BYTES,
        "prompt_framing_tokens": DEFAULT_PROMPT_FRAMING_TOKENS,
        "max_semantic_repairs": DEFAULT_SEMANTIC_REPAIRS,
        "max_transport_retries": DEFAULT_TRANSPORT_RETRIES,
        "max_retry_after_seconds": DEFAULT_MAX_RETRY_AFTER_SECONDS,
        "preserve_completion_whitespace": True,
        "strict_response_envelope": True,
        "resolved_model_acceptance": "exact_or_exact_plus_yyyy_mm_dd",
        "stall_rounds": DEFAULT_STALL_ROUNDS,
        "selectors": {
            RecruitmentMethod.OPEN_VOLUNTEER.value: "joint_exact_allocation",
            RecruitmentMethod.MUTUAL_NOMINATION.value: "native_mutual_reciprocal",
        },
        "hard_caps": {
            "logical_calls": logical_call_cap,
            "provider_attempts": provider_attempt_cap,
            "provider_attempt_token_exposure": token_exposure_cap,
        },
        "estimate": estimate,
    }


def _client_factory(config: ScreenConfig):
    client_config = SimpleNamespace(
        client_name="openai_responses",
        model_id=config.model_id,
        base_url=None,
        timeout=180,
        generate_kwargs={
            "max_output_tokens": config.max_output_tokens,
            "reasoning_effort": config.reasoning_effort,
            "prompt_cache_key": f"alem-e2b-v3-{config.method.value}",
            "prompt_cache_traffic_shards": 6,
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "prompt_cache_retention": "24h",
            "store": False,
            "preserve_completion_whitespace": True,
            "strict_response_envelope": True,
            "max_retry_after_seconds": DEFAULT_MAX_RETRY_AFTER_SECONDS,
        },
        max_retries=config.max_transport_retries,
        delay=1,
        alternate_roles=False,
        enable_thinking=False,
    )
    create = create_llm_client(client_config)
    return lambda agent_id: create()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-hosted",
        action="store_true",
        help="Make the preregistered hosted Responses calls; omitted means estimate-only.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--parallel-cells",
        action="store_true",
        help="Explicitly permit bounded in-process concurrency across isolated full-stage cells.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Cell workers; values above one require --parallel-cells (default: 1).",
    )
    parser.add_argument("--stage", choices=("canary", "full"), default="canary")
    parser.add_argument(
        "--max-logical-calls",
        type=int,
        default=DEFAULT_LOGICAL_CALL_CAP,
    )
    parser.add_argument(
        "--max-provider-attempts",
        type=int,
        default=DEFAULT_PROVIDER_ATTEMPT_CAP,
    )
    parser.add_argument(
        "--max-token-exposure",
        type=int,
        default=DEFAULT_TOKEN_EXPOSURE_CAP,
        help=(
            "Hard provider-attempt exposure reservation: prompt bytes plus "
            "framing and output allowances."
        ),
    )
    return parser.parse_args()


def _launch_projection(episode_count: int) -> dict[str, int]:
    """Projection with the canary's maximum promotable 25% repair rate."""

    initial_calls = episode_count * DEFAULT_ROUNDS * 6
    repair_allowance = (initial_calls + 3) // 4
    logical_calls = initial_calls + repair_allowance
    provider_attempts = logical_calls * (1 + DEFAULT_TRANSPORT_RETRIES)
    token_exposure = provider_attempts * (
        DEFAULT_MAX_PROMPT_BYTES + DEFAULT_PROMPT_FRAMING_TOKENS + DEFAULT_MAX_OUTPUT_TOKENS
    )
    return {
        "episodes": episode_count,
        "initial_logical_calls": initial_calls,
        "semantic_repair_allowance": repair_allowance,
        "logical_calls": logical_calls,
        "provider_attempts_reserved": provider_attempts,
        "provider_attempt_token_exposure": token_exposure,
    }


def _caps_conflicts(
    *,
    protocol: dict[str, Any],
    projection: dict[str, int],
    logical_used: int = 0,
    provider_attempts_reserved: int = 0,
    tokens_reserved: int = 0,
) -> list[str]:
    caps = protocol["hard_caps"]
    conflicts = []
    checks = (
        ("logical_calls", logical_used + projection["logical_calls"]),
        (
            "provider_attempts",
            provider_attempts_reserved + projection["provider_attempts_reserved"],
        ),
        (
            "provider_attempt_token_exposure",
            tokens_reserved + projection["provider_attempt_token_exposure"],
        ),
    )
    for name, projected_total in checks:
        if projected_total > int(caps[name]):
            conflicts.append(f"{name}: projected {projected_total} > hard cap {caps[name]}")
    return conflicts


def _print_estimate(
    protocol: dict[str, Any],
    *,
    stage: str,
    projection: dict[str, int],
    execute_hosted: bool,
) -> None:
    estimate = protocol["estimate"]
    print(
        "E2b PRELAUNCH ESTIMATE — hosted execution requested"
        if execute_hosted
        else "E2b DRY RUN — no provider calls"
    )
    print(
        f"{estimate['episodes']} episodes = {len(FROZEN_SEEDS)} seeds × "
        f"{len(FROZEN_FAMILIES)} families × {len(FROZEN_METHODS)} methods"
    )
    print(
        "Selectors: "
        + ", ".join(
            f"{method}={selector}" for method, selector in sorted(protocol["selectors"].items())
        )
    )
    print(
        f"Maximum logical calls: {estimate['max_logical_calls']} "
        f"({estimate['initial_logical_calls']} initial + "
        f"{estimate['semantic_repair_calls']} repairs)"
    )
    print(f"Maximum provider attempts: {estimate['max_provider_attempts']}")
    print(
        "Maximum recorded successful-call usage: "
        f"{estimate['max_recorded_input_tokens']} input + "
        f"{estimate['max_recorded_output_tokens']} output = "
        f"{estimate['max_recorded_total_tokens']} total"
    )
    print(
        "Maximum provider-attempt exposure if retries are also billable: "
        f"{estimate['max_provider_input_token_exposure']} input + "
        f"{estimate['max_provider_output_token_exposure']} output = "
        f"{estimate['max_provider_total_token_exposure']} total"
    )
    print(
        "The input ceiling charges one token per permitted prompt byte plus "
        f"{protocol['prompt_framing_tokens']} framing tokens per provider attempt."
    )
    print(
        f"{stage.capitalize()} launch projection (25% repair allowance): "
        f"{projection['episodes']} episode(s), {projection['logical_calls']} logical "
        f"({projection['initial_logical_calls']} initial + "
        f"{projection['semantic_repair_allowance']} repair), "
        f"{projection['provider_attempts_reserved']} provider reservations, "
        f"{projection['provider_attempt_token_exposure']} token exposure"
    )
    caps = protocol["hard_caps"]
    print(
        "Hard campaign caps: "
        f"{caps['logical_calls']} logical, {caps['provider_attempts']} provider, "
        f"{caps['provider_attempt_token_exposure']} token exposure"
    )


def _load_bound_manifest(
    path: Path,
    *,
    stage: str,
    protocol: dict[str, Any],
    launch_binding: dict[str, Any],
) -> dict[str, Any]:
    try:
        manifest = _read_managed_json(path.parent, path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"bound resume manifest is unreadable: {path}") from exc
    if (
        manifest.get("schema_version") != "alem-dice-e2b-campaign-v3"
        or manifest.get("stage") != stage
        or canonical_json(manifest.get("protocol")) != canonical_json(protocol)
        or canonical_json(manifest.get("launch_binding")) != canonical_json(launch_binding)
        or manifest.get("config_sha256") != launch_binding["config_sha256"]
        or manifest.get("source_commit") != launch_binding["git_head"]
        or manifest.get("uv_lock_sha256") != launch_binding["uv_lock_sha256"]
    ):
        raise RuntimeError("resume manifest does not match Git/lock/source/config binding")
    return manifest


def _assert_reservation_coverage(
    reconciliation: dict[str, Any],
    markers: list[dict[str, Any]],
) -> None:
    if reconciliation.get("overages"):
        raise RuntimeError("durable reservation ledger contains provider usage overages")
    if reconciliation["unresolved"]:
        raise RuntimeError(
            "unresolved pre-dispatch reservations are treated as spent; "
            "stop and reconcile before any further request"
        )
    covered = {
        canonical_json(key) for marker in markers for key in marker.get("reservation_keys", ())
    }
    reserved = set(reconciliation["reservations"])
    if covered != reserved:
        raise RuntimeError(
            "durable reservations are not exactly covered by validated completed cells"
        )


def _run_hosted_locked(
    *,
    args: argparse.Namespace,
    output: Path,
    jobs: tuple[tuple[int, ScenarioFamily, RecruitmentMethod], ...],
    protocol: dict[str, Any],
    launch_binding: dict[str, Any],
) -> int:
    """Execute only after the caller holds the whole-output-root lock."""

    # Recheck mutable preconditions after acquiring the lock and before any
    # campaign write or client construction.
    _require_hosted_credentials()
    rebound = _launch_binding(protocol)
    if canonical_json(rebound) != canonical_json(launch_binding):
        raise RuntimeError("launch binding changed while acquiring the output lock")
    source_hashes = launch_binding["source_hashes"]
    config_sha256 = launch_binding["config_sha256"]
    manifest_path = output / f"run_manifest_{args.stage}.json"
    stage_manifest: dict[str, Any] | None = None
    if args.resume and manifest_path.exists():
        stage_manifest = _load_bound_manifest(
            manifest_path,
            stage=args.stage,
            protocol=protocol,
            launch_binding=launch_binding,
        )

    ledger = DurableReservationLedger(output, launch_binding=launch_binding)
    reconciliation = ledger.reconcile()
    managed_artifacts = _validate_managed_cell_inventory(output, jobs)
    _assert_artifacts_bound_to_ledger(reconciliation, managed_artifacts)
    if reconciliation["unresolved"] or reconciliation["overages"]:
        _assert_reservation_coverage(reconciliation, [])
    if stage_manifest is not None:
        ledger.verify_checkpoint(
            stage_manifest["reservation_ledger_at_launch"]["checkpoint"]
        )
        if "reservation_ledger_final" in stage_manifest:
            ledger.verify_checkpoint(
                stage_manifest["reservation_ledger_final"]["checkpoint"]
            )
        prior_budget = stage_manifest.get("campaign_budget", {})
        if stage_manifest.get("status") in {"failed", "canary_failed"} or (
            isinstance(prior_budget, dict) and prior_budget.get("poisoned") is True
        ):
            raise RuntimeError(
                "bound campaign manifest records a poisoned or failed run; "
                "resume is forbidden before any provider dispatch"
            )

    expected_resolved_model: str | None = None
    canary_manifest: dict[str, Any] | None = None
    if args.stage == "full":
        canary_manifest = _load_bound_manifest(
            output / "run_manifest_canary.json",
            stage="canary",
            protocol=protocol,
            launch_binding=launch_binding,
        )
        if canary_manifest.get("status") != "complete":
            raise RuntimeError("full E2b requires a completed bound canary manifest")
        ledger.verify_checkpoint(
            canary_manifest["reservation_ledger_final"]["checkpoint"]
        )

    completed: list[dict[str, Any]] = []
    pending: list[tuple[int, ScenarioFamily, RecruitmentMethod]] = []
    for seed, family, method in jobs:
        marker = _load_completed_or_pending(
            output,
            seed=seed,
            family=family,
            method=method,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
            launch_binding=launch_binding,
            protocol=protocol,
            expected_resolved_model=expected_resolved_model,
            reservation_reconciliation=reconciliation,
        )
        if marker is None:
            pending.append((seed, family, method))
        else:
            completed.append(marker)

    canary_markers = [
        marker
        for marker in completed
        if (
            marker["seed"],
            ScenarioFamily(marker["family"]),
            RecruitmentMethod(marker["method"]),
        )
        in set(CANARY_CELLS)
    ]
    if args.stage == "full":
        if {
            (
                marker["seed"],
                ScenarioFamily(marker["family"]),
                RecruitmentMethod(marker["method"]),
            )
            for marker in canary_markers
        } != set(CANARY_CELLS):
            raise RuntimeError(
                "full E2b requires all four comprehensively validated canary cells"
            )
        canary_gate = _load_passing_canary_gate(
            output,
            canary_markers,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
            launch_binding=launch_binding,
            protocol=protocol,
            reservation_reconciliation=reconciliation,
            require_reservations=True,
        )
        if canary_gate is None:
            raise RuntimeError(
                "full E2b refused before any call: pure canary recomputation did not pass"
            )
        if (
            canary_manifest is None
            or canonical_json(canary_manifest.get("canary_gate")) != canonical_json(canary_gate)
            or canary_manifest.get("resolved_model_binding")
            != canary_gate["model_binding"]["resolved"]
        ):
            raise RuntimeError("canary manifest does not canonically bind the recomputed gate")
        expected_resolved_model = canary_gate["model_binding"]["resolved"]
        if args.resume and manifest_path.exists():
            existing_full = _load_bound_manifest(
                manifest_path,
                stage="full",
                protocol=protocol,
                launch_binding=launch_binding,
            )
            if existing_full.get("resolved_model_binding") != expected_resolved_model:
                raise RuntimeError("full resume manifest resolved-model binding changed")
        # Revalidate every already completed full cell against the resolved
        # canary snapshot, including copied markers.
        completed = []
        pending = []
        for seed, family, method in jobs:
            marker = _load_completed_or_pending(
                output,
                seed=seed,
                family=family,
                method=method,
                config_sha256=config_sha256,
                source_hashes=source_hashes,
                launch_binding=launch_binding,
                protocol=protocol,
                expected_resolved_model=expected_resolved_model,
                reservation_reconciliation=reconciliation,
            )
            if marker is None:
                pending.append((seed, family, method))
            else:
                completed.append(marker)

    _assert_reservation_coverage(reconciliation, completed)
    pending_projection = _launch_projection(len(pending))
    resume_conflicts = _caps_conflicts(
        protocol=protocol,
        projection=pending_projection,
        logical_used=int(reconciliation["logical_used"]),
        provider_attempts_reserved=int(reconciliation["provider_attempts_reserved"]),
        tokens_reserved=int(reconciliation["tokens_reserved"]),
    )
    if resume_conflicts:
        raise ValueError(
            "remaining launch projection conflicts with hard caps before any "
            "provider call: " + "; ".join(resume_conflicts)
        )
    caps = protocol["hard_caps"]
    campaign_budget = CampaignBudget(
        logical_limit=int(caps["logical_calls"]),
        provider_attempt_limit=int(caps["provider_attempts"]),
        token_limit=int(caps["provider_attempt_token_exposure"]),
        logical_used=int(reconciliation["logical_used"]),
        provider_attempts_reserved=int(reconciliation["provider_attempts_reserved"]),
        tokens_reserved=int(reconciliation["tokens_reserved"]),
        prompt_framing_tokens=int(protocol["prompt_framing_tokens"]),
        reservation_callback=ledger.reserve,
        resolution_callback=ledger.resolve,
    )
    cell_workers = min(
        args.workers if args.parallel_cells else 1,
        max(1, len(pending)),
    )
    manifest = {
        "schema_version": "alem-dice-e2b-campaign-v3",
        "stage": args.stage,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "stage_launch_projection": pending_projection,
        "config_sha256": config_sha256,
        "launch_binding": launch_binding,
        "source_commit": launch_binding["git_head"],
        "source_hashes": source_hashes,
        "uv_lock_sha256": launch_binding["uv_lock_sha256"],
        "resolved_model_binding": expected_resolved_model,
        "python": platform.python_version(),
        "parallel_cells": bool(args.parallel_cells),
        "cell_workers": cell_workers,
        "completed_at_launch": len(completed),
        "reservation_ledger_at_launch": {
            key: reconciliation[key]
            for key in (
                "records",
                "logical_used",
                "provider_attempts_reserved",
                "tokens_reserved",
                "ledger_sha256",
                "chain_head",
                "anchor_count",
                "anchor_head",
            )
        }
        | {"checkpoint": reconciliation["checkpoint"]},
    }
    atomic_write_json(output, manifest_path, manifest)

    def run_job(job: tuple[int, ScenarioFamily, RecruitmentMethod]) -> dict[str, Any]:
        seed, family, method = job
        scenario = generate_scenario(family, seed)
        config = ScreenConfig(method=method)
        episode = run_llm_recruitment_episode(
            scenario,
            config,
            client_factory=_client_factory(config),
            campaign_budget=campaign_budget,
            expected_resolved_model=expected_resolved_model,
        )
        return persist_completed_episode(
            output,
            episode,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
            launch_binding=launch_binding,
            protocol=protocol,
            reservation_reconciliation=ledger.reconcile(),
            require_reservations=True,
        )

    failures: list[dict[str, str]] = []
    cancelled_cells: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=cell_workers) as pool:
        futures = {pool.submit(run_job, job): job for job in pending}
        for future in as_completed(futures):
            seed, family, method = futures[future]
            if future.cancelled():
                cancelled_cells.append(
                    {
                        "seed": str(seed),
                        "family": family.value,
                        "method": method.value,
                    }
                )
                continue
            try:
                completed.append(future.result())
            except Exception as exc:
                campaign_budget.poison(f"cell_failure:{type(exc).__name__}")
                failures.append(
                    {
                        "seed": str(seed),
                        "family": family.value,
                        "method": method.value,
                        "error_type": type(exc).__name__,
                        "error_sha256": sha256_text(str(exc)),
                    }
                )
                for queued in futures:
                    if queued is not future:
                        queued.cancel()

    final_reconciliation = ledger.reconcile()
    if not failures:
        _assert_reservation_coverage(final_reconciliation, completed)
    manifest.update(
        {
            "status": (
                "complete"
                if not failures
                and not cancelled_cells
                and len(completed) == len(jobs)
                and not final_reconciliation["unresolved"]
                and not final_reconciliation["overages"]
                and not campaign_budget.snapshot()["poisoned"]
                and all(int(marker["budget_exhausted_decisions"]) == 0 for marker in completed)
                else "failed"
            ),
            "finished_at": datetime.now(UTC).isoformat(),
            "completed_episodes": len(completed),
            "expected_episodes": len(jobs),
            "failed_episodes": failures,
            "cancelled_cells": cancelled_cells,
            "campaign_budget": campaign_budget.snapshot(),
            "reservation_ledger_final": {
                key: final_reconciliation[key]
                for key in (
                    "records",
                    "logical_used",
                    "provider_attempts_reserved",
                    "tokens_reserved",
                    "unresolved",
                    "overages",
                    "ledger_sha256",
                    "chain_head",
                    "anchor_count",
                    "anchor_head",
                )
            }
            | {"checkpoint": final_reconciliation["checkpoint"]},
            "markers": sorted(
                completed,
                key=lambda value: (
                    value["seed"],
                    value["family"],
                    value["method"],
                ),
            ),
        }
    )
    if args.stage == "canary" and manifest["status"] == "complete":
        canary_gate = _write_canary_gate(
            output,
            completed,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
            launch_binding=launch_binding,
            protocol=protocol,
            reservation_reconciliation=final_reconciliation,
            require_reservations=True,
        )
        manifest["canary_gate"] = canary_gate
        manifest["resolved_model_binding"] = canary_gate["model_binding"]["resolved"]
        if canary_gate["status"] != "pass":
            manifest["status"] = "canary_failed"
    atomic_write_json(output, manifest_path, manifest)
    if manifest["status"] != "complete":
        raise RuntimeError(
            f"E2b {args.stage} status={manifest['status']}: "
            f"{len(completed)}/{len(jobs)} complete, {len(failures)} failed, "
            f"{len(cancelled_cells)} cancelled"
        )
    return 0


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.parallel_cells and args.workers < 2:
        raise ValueError("--parallel-cells requires --workers of at least 2")
    if not args.parallel_cells and args.workers != 1:
        raise ValueError("--workers above one requires explicit --parallel-cells")
    if (
        min(
            args.max_logical_calls,
            args.max_provider_attempts,
            args.max_token_exposure,
        )
        < 1
    ):
        raise ValueError("all hard campaign caps must be positive")
    all_jobs = tuple(
        (seed, family, method)
        for seed in FROZEN_SEEDS
        for family in FROZEN_FAMILIES
        for method in FROZEN_METHODS
    )
    jobs = CANARY_CELLS if args.stage == "canary" else all_jobs
    protocol = _protocol_config(
        logical_call_cap=args.max_logical_calls,
        provider_attempt_cap=args.max_provider_attempts,
        token_exposure_cap=args.max_token_exposure,
    )
    initial_projection = _launch_projection(len(jobs))
    _print_estimate(
        protocol,
        stage=args.stage,
        projection=initial_projection,
        execute_hosted=args.execute_hosted,
    )
    initial_conflicts = _caps_conflicts(
        protocol=protocol,
        projection=initial_projection,
    )
    if initial_conflicts:
        raise ValueError(
            "launch projection conflicts with hard caps before any provider call: "
            + "; ".join(initial_conflicts)
        )
    if not args.execute_hosted:
        return 0

    _require_hosted_credentials()
    launch_binding = _launch_binding(protocol)
    output = _normalize_output_root(args.output)
    if args.stage == "full" and (not output.exists() or not args.resume):
        raise FileNotFoundError("the full stage requires the canary output and --resume")
    if args.stage == "canary" and output.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse {output}; pass --resume to verify atomic markers")
    if args.resume:
        stage_manifest = output / f"run_manifest_{args.stage}.json"
        if stage_manifest.exists():
            _load_bound_manifest(
                stage_manifest,
                stage=args.stage,
                protocol=protocol,
                launch_binding=launch_binding,
            )
    if args.stage == "full":
        _load_bound_manifest(
            output / "run_manifest_canary.json",
            stage="canary",
            protocol=protocol,
            launch_binding=launch_binding,
        )
    with _output_root_lock(output):
        return _run_hosted_locked(
            args=args,
            output=output,
            jobs=jobs,
            protocol=protocol,
            launch_binding=launch_binding,
        )


if __name__ == "__main__":
    raise SystemExit(main())
