export function recallSource(pluginName) {
  return { kind: pluginName }
}

export function readToolResult(message) {
  if (message?.role !== 'tool' || message.source?.kind !== 'tool') return
  return {
    callId: message.toolCallId,
    content: message.content,
    failed: message.isError === true,
  }
}
