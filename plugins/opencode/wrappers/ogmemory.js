async function OgMemoryPlugin(input) {
  const plugin = await import("./ogmemory/index.mjs");
  return plugin.OgMemoryPlugin(input);
}

module.exports = OgMemoryPlugin;
module.exports.OgMemoryPlugin = OgMemoryPlugin;
