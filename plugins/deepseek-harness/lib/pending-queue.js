import { createHash, randomUUID } from 'node:crypto'
import { mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'

export function defaultPendingDir() {
  return join(homedir(), '.ogmemory', 'dsh-pending')
}

export function scopedPendingDir(directory, { baseUrl, accountId, userId, agentId }) {
  const scope = createHash('sha256')
    .update(`${baseUrl}\0${accountId}\0${userId}\0${agentId}`)
    .digest('hex')
    .slice(0, 24)
  return join(resolve(directory || defaultPendingDir()), scope)
}

export class PendingStore {
  constructor(directory = '') {
    this.directory = resolve(directory || defaultPendingDir())
    this.lastCreatedAt = 0
  }

  async list() {
    await this.ensureDirectory()
    const names = (await readdir(this.directory))
      .filter(name => name.endsWith('.json'))
      .sort()
    const entries = []
    for (const name of names) {
      const path = join(this.directory, name)
      try {
        const entry = await this.read(path)
        if (entry) entries.push(entry)
      } catch (error) {
        entries.push({ path, error })
      }
    }
    return entries
  }

  async read(path) {
    let text
    try {
      text = await readFile(path, 'utf8')
    } catch (error) {
      if (error.code === 'ENOENT') return null
      throw error
    }
    const value = JSON.parse(text)
    if (typeof value?.sessionId !== 'string' || !isRecord(value.payload)) {
      throw new TypeError('invalid pending entry')
    }
    return { ...value, path }
  }

  async put(sessionId, payload) {
    await this.ensureDirectory()
    const createdAt = Math.max(Date.now(), this.lastCreatedAt + 1)
    this.lastCreatedAt = createdAt
    const id = randomUUID()
    const queuedPayload = { ...payload, pendingId: id }
    const name = `${String(createdAt).padStart(13, '0')}-${id}.json`
    const path = join(this.directory, name)
    const temporary = `${path}.tmp`
    await writeFile(temporary, JSON.stringify({
      id,
      sessionId,
      createdAt,
      payload: queuedPayload,
    }), {
      encoding: 'utf8',
      mode: 0o600,
    })
    await rename(temporary, path)
    return { path, payload: queuedPayload }
  }

  async remove(path) {
    if (path) await rm(path, { force: true })
  }

  ensureDirectory() {
    return mkdir(this.directory, { recursive: true, mode: 0o700 })
  }
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
