# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Expose Proofline package metadata without coupling callers to implementation modules.

The package keeps its public identity here so the CLI, evaluation reports, and
future integrations can report a single versioned project name.
"""

__version__ = "0.1.0"
