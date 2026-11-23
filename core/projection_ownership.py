# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Carry the active outbox claim to the index mutation boundary."""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

_owner: ContextVar[Callable[[], None] | None] = ContextVar("projection_owner", default=None)


@contextmanager
def projection_owner(check):
    token = _owner.set(check)
    try:
        yield
    finally:
        _owner.reset(token)


def current_projection_owner():
    return _owner.get()


def check_projection_owner():
    check = current_projection_owner()
    if check is not None:
        check()
