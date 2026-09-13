# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

from pathlib import Path
from subprocess import run
from zipfile import ZipFile

import proofline


def test_package_exposes_a_version() -> None:
    assert proofline.__version__ == "0.1.0"


def test_built_wheel_includes_the_inline_typing_marker(tmp_path) -> None:
    """Downstream type checkers only see the marker when it is in the wheel."""

    root = Path(__file__).parents[1]
    completed = run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    wheel = next(tmp_path.glob("proofline-*.whl"))
    with ZipFile(wheel) as archive:
        assert "proofline/py.typed" in archive.namelist()
        archive.extractall(tmp_path / "site-packages")

    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        """
from proofline import RetrievalScope, scoped

def sync_resolver(context: str) -> RetrievalScope:
    return RetrievalScope.root(principal=context, filters={"resource_id": ["guide"]})

def sync_backend(query: str, *, filters: object, limit: int) -> list[int]:
    return [len(query)]

async def async_resolver(context: str) -> RetrievalScope:
    return RetrievalScope.root(principal=context, filters={"resource_id": ["guide"]})

async def async_backend(query: str, *, filters: object, limit: int) -> list[str]:
    return [query]

sync_retriever = scoped(sync_backend, resolve_scope=sync_resolver)
async_retriever = scoped(async_backend, resolve_scope=async_resolver)

async def check() -> None:
    sync_result = await sync_retriever.search("query", context="user:ana")
    async_result = await async_retriever.search("query", context="user:ana")
    reveal_type(sync_result.items)
    reveal_type(async_result.items)
"""
    )
    config = tmp_path / "pyrightconfig.json"
    config.write_text('{"extraPaths": ["site-packages"]}\n')
    checked = run(
        ["uv", "run", "pyright", "--project", str(config), str(consumer)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert 'Type of "sync_result.items" is "tuple[int, ...]"' in checked.stdout
    assert 'Type of "async_result.items" is "tuple[str, ...]"' in checked.stdout
