# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import proofline


def test_package_exposes_a_version() -> None:
    assert proofline.__version__ == "0.1.0"
