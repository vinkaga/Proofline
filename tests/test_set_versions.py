# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_tool():  # noqa: ANN202
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("set_versions", root / "scripts/set_versions.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_projects(root: Path, tool, version: str) -> tuple[Path, ...]:  # noqa: ANN001
    projects = tuple(root / relative_path for relative_path in tool.PROJECT_PATHS)
    for project in projects:
        project.mkdir(parents=True, exist_ok=True)
        (project / "pyproject.toml").write_text(
            "[project]\nname = \"test\"\nversion = \"" + version + "\"\n"
        )
    return projects


def test_set_versions_check_requires_matching_manifests_and_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    projects = _create_projects(tmp_path, tool, "0.1.2")
    locked: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        tool, "run_uv_lock", lambda project, *, check: locked.append((project, check))
    )

    tool.set_versions(tmp_path, "0.1.2", check=True)

    assert locked == [(project, True) for project in projects]


def test_set_versions_check_reports_manifest_drift(tmp_path: Path) -> None:
    tool = _load_tool()
    projects = _create_projects(tmp_path, tool, "0.1.2")
    (projects[-1] / "pyproject.toml").write_text(
        "[project]\nname = \"test\"\nversion = \"0.1.1\"\n"
    )

    with pytest.raises(ValueError, match="project versions do not match 0.1.2"):
        tool.set_versions(tmp_path, "0.1.2", check=True)


@pytest.mark.parametrize("version", ["0.1", "v0.1.2", "0.1.2 invalid"])
def test_set_versions_rejects_invalid_versions(version: str) -> None:
    tool = _load_tool()

    with pytest.raises(ValueError, match="version must use a release form"):
        tool.validate_version(version)
