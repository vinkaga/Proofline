#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Set the release version in every local project and regenerate its uv lock.

Usage::

    python scripts/set_versions.py 0.1.2
    python scripts/set_versions.py 0.1.2 --check
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATHS = (
    Path("."),
    Path("reference-demo"),
    Path("examples/custom-loop"),
    Path("examples/fastapi-host"),
    Path("examples/host"),
    Path("examples/langgraph"),
    Path("examples/llamaindex"),
    Path("examples/pydantic-ai"),
)
VERSION_PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"]+)("\s*)$')
RELEASE_VERSION_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?$",
    re.IGNORECASE,
)


def project_directories(root: Path) -> tuple[Path, ...]:
    """Return every project whose manifest and lock are versioned together."""

    projects = tuple(root / relative_path for relative_path in PROJECT_PATHS)
    missing = [str(project) for project in projects if not (project / "pyproject.toml").is_file()]
    if missing:
        raise ValueError("missing project manifest: " + ", ".join(missing))
    return projects


def validate_version(version: str) -> str:
    """Accept the release versions this repository publishes."""

    if not RELEASE_VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            "version must use a release form such as 0.1.2, 0.1.2rc1, or 0.1.2.post1"
        )
    return version


def manifest_version(manifest: Path) -> str:
    """Read the single project version without depending on a TOML writer."""

    matches = VERSION_PATTERN.findall(manifest.read_text())
    if len(matches) != 1:
        raise ValueError(f"expected exactly one project version in {manifest}")
    return matches[0][1]


def write_manifest_version(manifest: Path, version: str) -> None:
    """Replace the one project-version value while preserving file formatting."""

    content = manifest.read_text()
    updated, replacements = VERSION_PATTERN.subn(rf'\g<1>{version}\g<3>', content)
    if replacements != 1:
        raise ValueError(f"expected exactly one project version in {manifest}")
    manifest.write_text(updated)


def run_uv_lock(project: Path, *, check: bool) -> None:
    """Regenerate, or verify, the project lockfile through uv rather than editing it."""

    command = ["uv", "lock", "--check"] if check else ["uv", "lock"]
    subprocess.run(command, cwd=project, check=True)


def set_versions(root: Path, version: str, *, check: bool) -> None:
    """Synchronize manifests and locks, or verify that they already match."""

    version = validate_version(version)
    projects = project_directories(root)
    manifests = tuple(project / "pyproject.toml" for project in projects)
    mismatched = [manifest for manifest in manifests if manifest_version(manifest) != version]

    if check:
        if mismatched:
            paths = ", ".join(str(manifest.relative_to(root)) for manifest in mismatched)
            raise ValueError(f"project versions do not match {version}: {paths}")
        for project in projects:
            run_uv_lock(project, check=True)
        return

    for manifest in manifests:
        write_manifest_version(manifest, version)
    for project in projects:
        run_uv_lock(project, check=False)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the intentionally small release-tool interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="lock-step release version, for example 0.1.2")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify project versions and uv locks without modifying files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the release update and return a shell-friendly status code."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        set_versions(ROOT, args.version, check=args.check)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"version update failed: {error}", file=sys.stderr)
        return 1
    action = "Verified" if args.check else "Updated"
    print(f"{action} {len(PROJECT_PATHS)} project versions and uv locks at {args.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
