import { execFileSync } from 'node:child_process'
import { copyFile, cp, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const { devDependencies, scripts, files, ...base } = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
const newDsh = '0.2.1-rc.2'
const targets = {
  [base.peerDependencies['@deepseek-ai/dsh-llm']]: { manifest: base },
  [newDsh]: {
    adapter: `adapters/dsh-${newDsh}.js`,
    manifest: {
      ...base,
      version: '0.2.1',
      peerDependencies: {
        ...base.peerDependencies,
        '@deepseek-ai/cordis': '~4.0.5',
        '@deepseek-ai/dsh-llm': newDsh,
        '@deepseek-ai/dsh-tools': newDsh,
      },
    },
  },
}

const [selected = 'all', output = join(root, 'dist'), ...extra] = process.argv.slice(2)
if (extra.length || (selected !== 'all' && !Object.hasOwn(targets, selected))) {
  throw new Error(`Usage: node scripts/pack.mjs [all|${Object.keys(targets).join('|')}] [output-directory]`)
}

const outputDir = resolve(output)
await mkdir(outputDir, { recursive: true })
for (const dsh of selected === 'all' ? Object.keys(targets) : [selected]) {
  const { adapter, manifest } = targets[dsh]
  const staging = await mkdtemp(join(outputDir, '.pack-'))
  try {
    const pkg = join(staging, 'package')
    await mkdir(pkg)
    await cp(join(root, 'lib'), join(pkg, 'lib'), { recursive: true })
    if (adapter) await copyFile(join(root, adapter), join(pkg, 'lib/adapter.js'))
    await copyFile(join(root, 'index.js'), join(pkg, 'index.js'))
    await copyFile(join(root, 'cordis.patch.yml'), join(pkg, 'cordis.patch.yml'))
    await copyFile(resolve(root, '../../LICENSE'), join(pkg, 'LICENSE'))
    await writeFile(join(pkg, 'package.json'), JSON.stringify(manifest, null, 2) + '\n')
    const filename = `${manifest.name.slice(1).replace('/', '-')}-${manifest.version}.tgz`
    execFileSync('tar', ['-czf', filename, 'package'], { cwd: staging, stdio: 'inherit' })
    await rename(join(staging, filename), join(outputDir, filename))
    console.log(`DSH ${dsh}: ${join(outputDir, filename)}`)
  } finally {
    await rm(staging, { recursive: true, force: true })
  }
}
