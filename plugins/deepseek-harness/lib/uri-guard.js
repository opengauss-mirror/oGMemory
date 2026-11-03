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
