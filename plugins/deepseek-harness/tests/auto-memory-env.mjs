const AUTO_MEMORY_ENV = ['OGMEM_AUTO_CAPTURE', 'OGMEM_AUTO_RECALL']

export function isolateAutoMemoryEnv(t) {
  const saved = AUTO_MEMORY_ENV.map(name => [name, process.env[name]])
  t.after(() => {
    for (const [name, value] of saved) {
      if (value === undefined) delete process.env[name]
      else process.env[name] = value
    }
  })
  for (const name of AUTO_MEMORY_ENV) delete process.env[name]
}
