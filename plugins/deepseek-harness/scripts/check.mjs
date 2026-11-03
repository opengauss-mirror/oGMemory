import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const root = resolve(import.meta.dirname, '..')
const manifest = JSON.parse(await readFile(resolve(root, 'package.json'), 'utf8'))
if (manifest.dsh?.bundle?.patch !== './cordis.patch.yml') {
  throw new Error('package.json must declare dsh.bundle.patch')
}
const patch = await readFile(resolve(root, 'cordis.patch.yml'), 'utf8')
if (!patch.includes("name: '@opengauss/ogmemory-dsh-plugin'")) {
  throw new Error('cordis.patch.yml does not load the package entry point')
}
await import(pathToFileURL(resolve(root, 'index.js')).href)
console.log('ogmemory dsh plugin check passed')
