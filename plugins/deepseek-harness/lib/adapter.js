export function recallSource(pluginName) {
  return { kind: 'plugin', plugin: pluginName }
}

export function readToolResult(message) {
  if (message?.source?.kind !== 'tool') return
  const blocks = message.content.filter(block => block.type === 'tool-result')
  return {
    callId: message.source.callId,
    content: blocks.flatMap(block => block.content),
    failed: blocks.some(block => block.isError === true),
  }
}
