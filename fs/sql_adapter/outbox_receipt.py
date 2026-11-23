# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Projection receipt expression shared by SQL acknowledgement and resource erasure."""

# JSONB text retains JSON quoting when assembling the portable receipt.
PROJECTION_RECEIPT_SQL = """CASE
    WHEN jsonb_typeof(payload->'task_owner_spaces') = 'array' THEN
        ('{"task_owner_spaces":' || (payload->'task_owner_spaces')::text || '}')::jsonb
    ELSE coalesce(
        ('{"filters":{"owner_space":' ||
         (CASE WHEN jsonb_typeof(payload->'filters') = 'object'
               THEN payload #> '{filters,owner_space}'
               ELSE payload #> '{records,0,filters,owner_space}' END)::text || '}}')::jsonb,
        '{}'::jsonb)
    END"""
