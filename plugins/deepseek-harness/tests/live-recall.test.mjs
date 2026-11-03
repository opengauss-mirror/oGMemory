import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import test from 'node:test'

const run = promisify(execFile)
const enabled = process.env.OGMEM_E2E === '1'

test('live backend stores and recalls a unique marker', {
  skip: enabled ? false : 'set OGMEM_E2E=1 to run against a real oGMemory server',
  timeout: 90_000,
}, async () => {
  const { stdout } = await run(process.execPath, ['scripts/acceptance-wsl.mjs'], {
    cwd: new URL('..', import.meta.url),
    env: process.env,
    timeout: 80_000,
  })
  const result = JSON.parse(stdout)
  assert.equal(result.ok, true)
  assert.equal(result.markerRecovered, true)
})
