const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const script = readFileSync(path.join(__dirname, '../../../server/memory_viewer_static/migration.js'), 'utf8');
class Element {
  constructor() { this.children = []; this.files = []; this.checked = true; this.attributes = {}; this.click = () => {}; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; if (key === 'value') delete this.value; }
}
async function viewer() {
  const nodes = new Map();
  const el = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  let response = {mode:'online', storage:'sql', vector:'opengauss', jobs:[]};
  let error = null, status = 200;
  const ctx = vm.createContext({document:{getElementById:el, createElement:() => new Element(), documentElement:{dataset:{}}}, localStorage:{getItem:()=>null},
    sessionStorage:{getItem:()=>null,setItem:()=>{},removeItem:()=>{}}, crypto:require("node:crypto").webcrypto, confirm:()=>true, setTimeout, fetch:async () => { if (error) throw error; return {ok:status < 400,status,json:async()=>response}; }});
  vm.runInContext(script, ctx);
  await new Promise(setImmediate);
  return {ctx, el, respond:data => {response = data; error = null;}, reject:e => {error = e;}, status:code => {status = code;}};
}
test('viewing historical jobs cannot unlock a running migration', async () => {
  const v = await viewer();
  v.respond({mode:'online', storage:'sql', vector:'opengauss', jobs:[{id:'old',kind:'export',status:'succeeded'}, {id:'running',kind:'import',status:'running'}]});
  await v.ctx.refresh({resume:false});
  v.el('history').children[0].onclick();
  assert.equal(v.el('export').disabled, true);
  assert.match(v.el('error').textContent, /当前任务/);
});
test('unknown submission outcome stays locked until server status is recovered', async () => {
  const v = await viewer();
  v.reject(new Error('connection lost'));
  await assert.rejects(v.ctx.submit('export'), /connection lost/);
  assert.equal(v.el('export').disabled, true);
  v.respond({mode:'online',storage:'sql',vector:'opengauss',jobs:[]});
  await v.ctx.refresh({resume:false});
  assert.equal(v.el('export').disabled, true);
});
test('restored preflight can be selected, but choosing a new file invalidates it', async () => {
  const v = await viewer();
  v.ctx.applyPlan({id:'plan',result:{artifact_id:'old-file',data:{nodes:{count:2}}}});
  assert.equal(v.el('import').disabled, false);
  v.el('file').files = [{name:'new.tar.gz'}];
  v.el('file').onchange();
  assert.equal(v.el('import').disabled, true);
  assert.equal(v.el('plan').textContent, '');
});
test('unknown totals are indeterminate and successful tasks complete the progress bar', async () => {
  const v = await viewer();
  v.ctx.showJob({kind:'import',phase:'reindexing',status:'running',index_records:12});
  assert.equal(v.el('progress').value, undefined);
  assert.match(v.el('progress-label').textContent, /12/);
  v.ctx.showJob({kind:'import',phase:'complete',status:'succeeded'});
  assert.equal(v.el('progress').value, 100);
});

test('shared-space selection and reason appear in preview and completed results', async () => {
  const v = await viewer();
  const result = {artifact_id:'package', shared_spaces:{spaces:2,nodes:3,action:'skipped',reason:'target_business_not_empty'}};
  v.ctx.applyPlan({id:'plan',result});
  assert.match(v.el('plan-summary').textContent, /共享空间 2 个、内容 3 条：整批跳过（目标已有业务数据或索引）/);
  v.ctx.showJob({kind:'import',status:'succeeded',phase:'complete',result});
  assert.match(v.el('result-summary').textContent, /整批跳过/);
});

test('expired authentication invalidates the import plan and download action', async () => {
  const v = await viewer();
  v.ctx.applyPlan({id:'plan',result:{artifact_id:'old-file'}});
  v.ctx.offerDownload('old-export');
  v.status(401);
  await assert.rejects(v.ctx.refresh(), /HTTP 401/);
  assert.equal(v.el('operations').hidden, true);
  assert.equal(v.el('import').disabled, true);
  assert.equal(v.el('download').hidden, true);
  assert.equal(v.el('plan').textContent, '');
  v.status(200);
  await v.ctx.refresh({resume:false});
  assert.equal(v.el('import').disabled, true);
});

const statusData = jobs => ({mode:'online',storage:'sql',vector:'opengauss',jobs});
const responseData = data => ({ok:true,status:200,json:async()=>data});
test('a delayed status snapshot cannot unlock a newly submitted task', async () => {
  const v = await viewer(); let finishStatus;
  v.ctx.fetch = async url => {
    if (url.endsWith('/status')) return new Promise(resolve => {finishStatus = resolve;});
    if (url.endsWith('/jobs')) return responseData({job_id:'new-job'});
    return new Promise(()=>{});
  };
  const stale = v.ctx.refresh();
  void v.ctx.submit('export');
  await new Promise(setImmediate);
  finishStatus(responseData(statusData([]))); await stale;
  assert.equal(v.el('export').disabled, true);
  assert.equal(v.el('file').disabled, true);
  assert.equal(vm.runInContext('activeJob', v.ctx), 'new-job');
});
test('a task discovered after completion is watched until controls unlock', async () => {
  const v = await viewer(); const requests = []; let statuses = 0;
  v.ctx.fetch = async url => {
    requests.push(url);
    if (url.endsWith('/status')) return responseData(statusData(statuses++ === 0 ? [{id:'next',kind:'export',status:'running'}] : []));
    return responseData({id:url.split('/').pop(),kind:'export',status:'succeeded'});
  };
  await v.ctx.watch('old');
  await new Promise(setImmediate);
  assert.equal(requests.some(url => url.endsWith('/jobs/next')), true);
  assert.equal(v.el('export').disabled, false);
  assert.equal(vm.runInContext('activeJob', v.ctx), null);
});

test('unknown instance request locks new files and cannot silently reuse another package', async () => {
  const v = await viewer();
  vm.runInContext("artifactId='package-a'",v.ctx);
  v.reject(new Error('connection lost'));
  await assert.rejects(v.ctx.submit('preflight'), /connection lost/);
  v.respond({mode:'online',storage:'sql',vector:'opengauss',jobs:[]});
  await v.ctx.refresh({resume:false});
  assert.equal(v.el('file').disabled,true);
  vm.runInContext("artifactId='package-b'",v.ctx);
  await assert.rejects(v.ctx.submit('preflight'), /上次提交尚未确认/);
  const sent=[];
  v.ctx.fetch=async (url,options) => {
    if(url.endsWith('/jobs')) {sent.push(JSON.parse(options.body));return responseData({job_id:'retry'});}
    if(url.endsWith('/jobs/retry')) return responseData({id:'retry',kind:'preflight',status:'failed',error:'test'});
    return responseData({mode:'online',storage:'sql',vector:'opengauss',jobs:[]});
  };
  await v.ctx.submit('preflight',{retry:true});
  assert.equal(sent[0].artifact_id,'package-a');
});
