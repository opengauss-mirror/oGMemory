import assert from 'node:assert/strict'
import test from 'node:test'

import { findOgMemoryUri, guardOgMemoryUri } from '../lib/uri-guard.js'

test('finds nested native memory URIs', () => {
  assert.equal(
    findOgMemoryUri({ command: ['cat', { path: 'ogmemory://memories/preference/1' }] }),
    'ogmemory://memories/preference/1',
  )
})

test('blocks local IO tools from treating native memory URIs as paths', async () => {
  let continued = false
  const decision = await guardOgMemoryUri({
    name: 'read',
    arguments: { path: 'ogmemory://memories/preference/1' },
  }, async () => { continued = true })
  assert.equal(decision.kind, 'deny')
  assert.match(decision.reason, /ogm_read_memory/)
  assert.equal(continued, false)
})

test('allows ordinary paths and native oGMemory tools', async () => {
  const next = async () => ({ kind: 'allow' })
  assert.deepEqual(await guardOgMemoryUri({
    name: 'read', arguments: { path: 'README.md' },
  }, next), { kind: 'allow' })
  assert.deepEqual(await guardOgMemoryUri({
    name: 'ogm_read_memory', arguments: { uri: 'ogmemory://memories/preference/1' },
  }, next), { kind: 'allow' })
})
