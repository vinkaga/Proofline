# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import sys
from pathlib import Path
from subprocess import run


def test_hello_scopeanchor_compares_equally_protected_paths() -> None:
    root = Path(__file__).parents[1]

    completed = run(
        [sys.executable, "examples/hello_scopeanchor.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Unsafe: proposal controls the filter" in completed.stdout
    assert "Result: Beta rollout plan" in completed.stdout
    assert "Manual ACL filtering on every hop" in completed.stdout
    assert "Result: Acme rollout plan" in completed.stdout
    assert "Scope-bearing fields: ignored; trusted code chose the filter" in completed.stdout
    assert "ScopeAnchor" in completed.stdout
    assert "rejected before retrieval (resource_id)" in completed.stdout
    assert completed.stdout.count("Data-only follow-up") == 3
    assert "Inherited scope record: yes" in completed.stdout
    assert "Verified lineage record: yes" in completed.stdout
    assert "Comparison for this one trace" in completed.stdout
    assert (
        "No unauthorized evidence exposed             no      yes         yes" in completed.stdout
    )
    assert (
        "Scope-bearing proposal rejected before search no      no          yes" in completed.stdout
    )
    assert (
        "Query-only follow-up lineage complete         no      yes         yes" in completed.stdout
    )
    assert "Fair reading" in completed.stdout
    assert "Manual ACL filtering and ScopeAnchor both prevent that leak here." in completed.stdout
    assert (
        "ScopeAnchor additionally rejects a scope-bearing proposal before search."
        in completed.stdout
    )
