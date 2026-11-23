# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Source revision checks for memory consolidation plans."""

from collections.abc import Mapping, Sequence

from core.errors import ConcurrentModificationError, NodeNotFoundError, ValidationError
from core.interfaces import ContextFS
from core.models import ContextNode, RequestContext


def collect_input_versions(nodes: Sequence[ContextNode]) -> dict[str, int]:
    """Record the persisted revisions of the nodes used to generate a plan."""
    versions: dict[str, int] = {}
    for node in nodes:
        version = node.metadata.get("version")
        if type(version) is not int or version < 1:
            raise ValidationError("version", "plan inputs require a positive persisted version")
        if node.uri in versions and versions[node.uri] != version:
            raise ConcurrentModificationError(node.uri, versions[node.uri], version)
        versions[node.uri] = version
    return dict(sorted(versions.items()))


def find_changed_inputs(
    fs: ContextFS, versions: Mapping[str, int], ctx: RequestContext
) -> dict[str, int | None]:
    """Return changed URI/current-version pairs; unavailable nodes map to None.

    This is a preflight read. Applying a plan still uses the storage layer's
    expected_version checks to serialize writes against concurrent updates.
    """
    changed: dict[str, int | None] = {}
    for uri, expected_version in sorted(versions.items()):
        try:
            node = fs.read_node(uri, ctx)
        except NodeNotFoundError:
            changed[uri] = None
            continue
        current_version = collect_input_versions([node])[node.uri]
        if current_version != expected_version:
            changed[uri] = current_version
    return changed
