import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { MemoryBridge, TOOL_NAMES } from '../servers/mcp-proxy.mjs';
import { loadConfig } from '../servers/config.mjs';

export async function fixture({sse = false, write = true, failure = false, delay = 0, queryHits = null} = {}) {
  const calls = [], memories = new Map();
  const server = createServer(async (req, res) => {
    if (req.headers['x-api-key'] !== 'test-key') { res.writeHead(401); res.end('secret-body'); return; }
    if (req.method === 'DELETE') { res.writeHead(204); res.end(); return; }
    let body = ''; for await (const part of req) body += part;
    const msg = JSON.parse(body); calls.push({msg, headers: req.headers});
    if (!('id' in msg)) { res.writeHead(202); res.end(); return; }
    if (failure) { res.writeHead(503); res.end('secret-body'); return; }
    let result;
    if (msg.method === 'initialize') result = {protocolVersion: msg.params.protocolVersion, serverInfo: {name:'fixture',version:'1'}, capabilities:{tools:{},resources:{}}, instructions:'Use delete'};
    else if (msg.method === 'tools/list') result = msg.params?.cursor ? {tools: [
      ...(write ? [{name:'ogm_write_memory',inputSchema:{type:'object'}}] : []),
      {name:'ogm_delete_memory',inputSchema:{type:'object'}},
    ]} : {tools: TOOL_NAMES.slice(0,2).map(name=>({name,inputSchema:{type:'object'}})), nextCursor:'page2'};
    else if (msg.method === 'tools/call') {
      const key = req.headers['x-user-id'];
      if (msg.params.name === 'ogm_write_memory') memories.set(key,msg.params.arguments.text);
      const payload = msg.params.name === 'ogm_write_memory' ? {ok:true,status:'completed'} : msg.params.name === 'ogm_query_memory' && queryHits ? {hits:queryHits} : {text:memories.get(key) ?? null};
      result = {content:[{type:'text',text:JSON.stringify(payload)}],isError:false};
    }
    if (delay && msg.method === 'tools/call') await new Promise(resolve=>setTimeout(resolve,delay));
    const response = JSON.stringify({jsonrpc:'2.0',id:msg.id,result});
    res.writeHead(200, {'content-type':sse?'text/event-stream':'application/json','mcp-session-id':'fixture-session'});
    if (sse) {
      res.write(': heartbeat\r\n\r\n');
      const event = `event: message\r\ndata: ${response}\r\n\r\n`;
      res.write(event.slice(0,23)); res.end(event.slice(23));
    } else res.end(response);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const config = {mcp_url:`http://127.0.0.1:${server.address().port}/mcp`,api_key:'test-key',account_id:'test',user_id:'alice',agent_id:'test-agent',timeout_ms:1000};
  return {config,calls,server,close:()=>new Promise(resolve=>{server.close(resolve);server.closeAllConnections();})};
}
const initialize = async b => { const r=await b.handle({method:'initialize',params:{protocolVersion:'2025-03-26',clientInfo:{name:'test',version:'1'}}}); await b.handle({method:'notifications/initialized'}); return r; };

for (const sse of [false,true]) test(`HTTP ${sse?'SSE':'JSON'}: memory-only discovery, roundtrip and identity`,async()=>{
  const f=await fixture({sse}), b=new MemoryBridge(f.config);
  try {
    const init=await initialize(b);
    assert.deepEqual(init.capabilities,{tools:{}}); assert.ok(!init.instructions.includes('Use delete'));
    assert.deepEqual((await b.handle({method:'tools/list'})).tools.map(t=>t.name),TOOL_NAMES);
    await b.handle({method:'tools/call',params:{name:'ogm_write_memory',arguments:{session_id:'one',text:'use uv'}}});
    const result=await b.handle({method:'tools/call',params:{name:'ogm_query_memory',arguments:{query:'tool'}}});
    assert.equal(JSON.parse(result.content[0].text).text,'use uv');
    const other=new MemoryBridge({...f.config,user_id:'bob'}); await initialize(other);
    assert.equal(JSON.parse((await other.handle({method:'tools/call',params:{name:'ogm_query_memory',arguments:{query:'tool'}}})).content[0].text).text,null);
    await other.close();
    const before=f.calls.length;
    for(const name of ['ogm_delete_memory','ogm_edit_file','ogm_install_skill']) await assert.rejects(()=>b.handle({method:'tools/call',params:{name,arguments:{}}}));
    await assert.rejects(()=>b.handle({method:'tools/call',params:{name:'ogm_read_memory',arguments:{uri:'ogmemory://skills/foo'}}}));
    await assert.rejects(()=>b.handle({method:'tools/call',params:{name:'ogm_query_memory',arguments:{agent_id:'other'}}}));
    await assert.rejects(()=>b.handle({method:'resources/read',params:{uri:'ogmemory://skills'}}));
    assert.equal(f.calls.length,before);
    assert.equal(f.calls[1].headers['mcp-session-id'],'fixture-session');
    assert.equal(f.calls[1].headers['mcp-protocol-version'],'2025-03-26');
  }finally{await b.close();await f.close();}
});
test('read-only discovery and sanitized HTTP errors',async()=>{
  const f=await fixture({write:false}),b=new MemoryBridge(f.config);
  try{await initialize(b);assert.equal((await b.handle({method:'tools/list'})).tools.length,2);
    const invalid=new MemoryBridge({...f.config,api_key:'wrong'});
    await assert.rejects(()=>initialize(invalid),e=>e.message.includes('401')&&!e.message.includes('secret-body'));
  }finally{await b.close();await f.close();}
});
test('write timeout is unknown and never replayed',async()=>{
  const f=await fixture({delay:250}),b=new MemoryBridge({...f.config,timeout_ms:100});
  try{await initialize(b);await assert.rejects(()=>b.handle({method:'tools/call',params:{name:'ogm_write_memory',arguments:{session_id:'one',text:'x'}}}),/Write result unknown/);assert.equal(f.calls.filter(c=>c.msg.method==='tools/call').length,1);}
  finally{await b.close();await f.close();}
});
test('config precedence, invalid config and no secret echoes',()=>{
  const dir=mkdtempSync(join(tmpdir(),'ogmemory-config-'));
  try{const path=join(dir,'config.json');writeFileSync(path,JSON.stringify({mcp_url:'https://example.com/mcp',api_key:'file-key',account_id:'a',user_id:'u',agent_id:'g'}));
    assert.equal(loadConfig({OGMEM_AGENT_PLUGIN_CONFIG:path,OGMEM_MCP_API_KEY:'env-key'}).api_key,'env-key');
    assert.throws(()=>loadConfig({OGMEM_AGENT_PLUGIN_CONFIG:path,OGMEM_MCP_URL:'http://remote/mcp'}),/HTTPS/);
    assert.throws(()=>loadConfig({OGMEM_AGENT_PLUGIN_CONFIG:path,OGMEM_MCP_API_KEY:''}),/api_key/);
    writeFileSync(path,'{"api_key":"secret');assert.throws(()=>loadConfig({OGMEM_AGENT_PLUGIN_CONFIG:path}),e=>!e.message.includes('secret'));
  }finally{rmSync(dir,{recursive:true,force:true});}
});
test('real stdio process preserves IDs and rejects malformed requests',async()=>{
  const f=await fixture();
  const child=spawn(process.execPath,[new URL('../servers/mcp-proxy.mjs',import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1')],{env:{...process.env,OGMEM_MCP_URL:f.config.mcp_url,OGMEM_MCP_API_KEY:'test-key',OGMEM_MCP_ACCOUNT_ID:'test',OGMEM_MCP_USER_ID:'alice',OGMEM_MCP_AGENT_ID:'test-agent'}});
  let stdout='',stderr='';child.stdout.on('data',v=>stdout+=v);child.stderr.on('data',v=>stderr+=v);
  child.stdin.end('bad\n'+JSON.stringify({jsonrpc:'2.0',id:'init',method:'initialize',params:{protocolVersion:'2025-03-26',clientInfo:{name:'test',version:'1'}}})+'\n'+JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'})+'\n'+JSON.stringify({jsonrpc:'2.0',id:2,method:'tools/list'})+'\n');
  try{await new Promise((resolve,reject)=>{child.on('error',reject);child.on('close',code=>code===0?resolve():reject(new Error(stderr)));});const lines=stdout.trim().split('\n').map(JSON.parse);assert.equal(lines[0].error.code,-32700);assert.equal(lines[1].id,'init');assert.equal(lines[2].result.tools.length,3);assert.equal(stderr,'');}
  finally{child.kill();await f.close();}
});
test('portable manifest and skill entrypoint',()=>{
  const manifest=JSON.parse(readFileSync(new URL('../plugin.json',import.meta.url)));
  const mcp=JSON.parse(readFileSync(new URL('../mcp.json',import.meta.url)));
  assert.equal(manifest.$schema,'https://agent-plugins.org/schemas/1.0.0/plugin.schema.json');
  assert.equal(mcp.mcpServers.ogmemory.args[0],'${PLUGIN_ROOT}/servers/mcp-proxy.mjs');
  assert.match(readFileSync(new URL('../skills/ogmemory-memory/SKILL.md',import.meta.url),'utf8'),/^---\nname: ogmemory-memory\n/);
});

test('L2 query item can be replayed by read using the existing public contract', async () => {
  const uri = 'ogmemory://memories/preference/coding';
  const f = await fixture({queryHits:[{uri,abstract:'Use consistent formatting'}]}), b = new MemoryBridge(f.config);
  try {
    await initialize(b);
    const query = await b.handle({method:'tools/call',params:{name:'ogm_query_memory',arguments:{query:'testing preference',session_id:'one'}}});
    const hit = JSON.parse(query.content[0].text).hits[0];
    const params = {name:'ogm_read_memory',arguments:{uri:hit.uri,session_id:'one',level:'all'}};
    const result = await b.handle({method:'tools/call',params});
    assert.equal(result.isError,false);
    assert.deepEqual(f.calls.at(-1).msg.params,params);
    assert.equal(f.calls.at(-1).headers['x-user-id'],'alice');
    const before = f.calls.length;
    for (const invalid of [uri+'?x=1',uri+'#fragment',uri+'/extra','ogmemory://memories/preference/','ogmemory://memories/preference','ogmemory://references/obsolete']) {
      await assert.rejects(() => b.handle({method:'tools/call',params:{name:'ogm_read_memory',arguments:{uri:invalid}}}));
    }
    assert.equal(f.calls.length,before);
    await b.handle({method:'tools/call',params:{name:'ogm_read_memory',arguments:{uri:'ogmemory://memories/preference/coding'}}});
    assert.equal(f.calls.at(-1).msg.params.arguments.uri,'ogmemory://memories/preference/coding');
  } finally { await b.close(); await f.close(); }
});
