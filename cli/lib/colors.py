# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""ANSI color helpers for terminal output."""

import sys

# Detect color support
_NO_COLOR = not (sys.stdout.isatty() and sys.platform != "win32")

RED = "\033[0;31m" if not _NO_COLOR else ""
GREEN = "\033[0;32m" if not _NO_COLOR else ""
YELLOW = "\033[1;33m" if not _NO_COLOR else ""
CYAN = "\033[0;36m" if not _NO_COLOR else ""
BOLD = "\033[1m" if not _NO_COLOR else ""
DIM = "\033[2m" if not _NO_COLOR else ""
NC = "\033[0m" if not _NO_COLOR else ""


def ok(msg: str) -> str:
    return f"  {GREEN}✓{NC} {msg}"


def fail(msg: str) -> str:
    return f"  {RED}✗{NC} {msg}"


def warn(msg: str) -> str:
    return f"  {YELLOW}!{NC} {msg}"


def info(msg: str) -> str:
    return f"  {CYAN}ℹ{NC} {msg}"
