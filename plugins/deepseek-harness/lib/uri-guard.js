/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * oGMemory is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

const LOCAL_IO_TOOLS = new Set([
  'read',
  'glob',
  'grep',
  'bash',
  'edit',
  'write',
  'str_replace_editor',
])

export async function guardOgMemoryUri(exec, next) {
  if (!LOCAL_IO_TOOLS.has(exec.name)) return next()
  const uri = findOgMemoryUri(exec.arguments)
  if (!uri) return next()
  return {
    kind: 'deny',
    reason: `${uri} is an oGMemory URI, not a local path. Use ogm_read_memory for an exact URI or ogm_query_memory to search.`,
  }
}

export function findOgMemoryUri(value) {
  const stack = [value]
  while (stack.length > 0) {
    const current = stack.pop()
    if (typeof current === 'string') {
      const match = current.match(/ogmemory:\/\/[^\s"'`]+/i)
      if (match) return match[0]
    } else if (Array.isArray(current)) {
      stack.push(...current)
    } else if (current && typeof current === 'object') {
      stack.push(...Object.values(current))
    }
  }
  return ''
}
