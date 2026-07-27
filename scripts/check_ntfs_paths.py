#!/usr/bin/env python3
"""Reject repository paths that cannot be checked out on a normal NTFS volume."""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CHARACTERS = frozenset('<>:"\\|?*')
RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


def _repository_paths() -> list[str]:
    """Return tracked and nonignored untracked paths as repository-relative names."""

    listed_command = [
        "git",
        "-C",
        str(REPO_ROOT),
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    deleted_command = ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--deleted"]
    try:
        listed = subprocess.run(listed_command, check=True, capture_output=True)
        deleted = subprocess.run(deleted_command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to enumerate repository paths") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git could not enumerate repository paths: {detail}") from exc

    deleted_paths = set(deleted.stdout.split(b"\0"))
    return [
        raw_path.decode("utf-8", errors="surrogateescape")
        for raw_path in listed.stdout.split(b"\0")
        if raw_path and raw_path not in deleted_paths
    ]


def _component_errors(component: str) -> list[str]:
    errors: list[str] = []
    forbidden = sorted({character for character in component if character in FORBIDDEN_CHARACTERS})
    if forbidden:
        errors.append(f"contains forbidden character(s) {''.join(forbidden)!r}")
    if any(ord(character) < 32 for character in component):
        errors.append("contains an ASCII control character")
    if component.endswith((" ", ".")):
        errors.append("ends with a space or period")

    # Windows reserves these device names even when an extension is present.
    stem = component.rstrip(" .").split(".", maxsplit=1)[0].upper()
    if stem in RESERVED_STEMS:
        errors.append(f"uses reserved Windows device name {stem!r}")
    return errors


def _collision_key(path: str) -> str:
    normalized = unicodedata.normalize("NFC", path)
    return normalized.replace("\\", "/").casefold()


def validate_paths(paths: list[str]) -> list[str]:
    errors: list[str] = []
    collision_groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        collision_groups[_collision_key(path)].append(path)
        for component in re.split(r"[/\\]", path):
            if not component:
                errors.append(f"{path!r}: contains an empty path component")
                continue
            for error in _component_errors(component):
                errors.append(f"{path!r}: component {component!r} {error}")

    for group in collision_groups.values():
        distinct = sorted(set(group))
        if len(distinct) > 1:
            errors.append(
                "case-insensitive NTFS collision: " + ", ".join(repr(path) for path in distinct)
            )
    return sorted(errors)


def main() -> int:
    try:
        paths = _repository_paths()
    except RuntimeError as exc:
        print(f"NTFS path check failed: {exc}", file=sys.stderr)
        return 2

    errors = validate_paths(paths)
    if errors:
        print("NTFS-incompatible repository paths found:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"NTFS path check passed ({len(paths)} repository paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
