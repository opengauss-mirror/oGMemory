import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdir, mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import test from 'node:test'

const root = resolve(import.meta.dirname, '..')

test('each tarball contains its selected adapter alongside the common bundle', async t => {
  const out = await mkdtemp(join(tmpdir(), 'ogmemory-pack-'))
  t.after(() => rm(out, { recursive: true, force: true }))
  execFileSync(process.execPath, [join(root, 'scripts/pack.mjs'), 'all', out])

  const source = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
  const cases = [
    [source.version, source.peerDependencies, 'lib/adapter.js'],
    ['0.2.1', {
      ...source.peerDependencies,
      '@deepseek-ai/cordis': '~4.0.5',
      '@deepseek-ai/dsh-llm': '0.2.1-rc.2',
      '@deepseek-ai/dsh-tools': '0.2.1-rc.2',
    }, 'adapters/dsh-0.2.1-rc.2.js'],
  ]
  for (const [version, peers, adapter] of cases) {
    const archive = join(out, `${source.name.slice(1).replace('/', '-')}-${version}.tgz`)
    const unpacked = join(out, version)
    await mkdir(unpacked)
    execFileSync('tar', ['-xzf', archive, '-C', unpacked])
    const pkg = join(unpacked, 'package')
    const manifest = JSON.parse(await readFile(join(pkg, 'package.json'), 'utf8'))
    assert.equal(manifest.name, source.name)
    assert.equal(manifest.version, version)
    assert.deepEqual(manifest.peerDependencies, peers)
    assert.deepEqual(await readFile(join(pkg, 'lib/adapter.js')), await readFile(join(root, adapter)))
    assert.deepEqual(await readFile(join(pkg, manifest.exports)), await readFile(join(root, 'index.js')))
    assert.deepEqual(await readFile(join(pkg, manifest.dsh.bundle.patch)), await readFile(join(root, 'cordis.patch.yml')))
    assert.deepEqual((await readdir(join(pkg, 'lib'))).sort(), (await readdir(join(root, 'lib'))).sort())
  }
})
