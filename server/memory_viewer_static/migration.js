'use strict';
const el = (id) => document.getElementById(id);
let artifactId = null, preflightId = null, pollGeneration = 0, busy = false;
const phases = {queued:'等待执行', validating:'校验中', exporting:'导出中', restoring:'恢复数据中', reindexing:'重建索引中', verifying:'校验结果中', complete:'完成', interrupted:'进程中断'};
async function api(path, options = {}) {
  const response = await fetch('/api/migration/' + path, {...options, credentials:'same-origin'});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function summary(data) {
  if (!data) return '等待数据';
  const labels = {nodes:'记忆节点', relations:'关系', archives:'历史归档'};
  const parts = Object.entries(data.data || data).filter(([key]) => key in labels)
    .map(([key, value]) => `${labels[key]} ${value.count} 条`);
  if (data.source_backend && data.target_backend) parts.unshift(`${data.source_backend.toUpperCase()} → ${data.target_backend.toUpperCase()}`);
  if (data.index_records !== undefined) parts.push(`索引记录 ${data.index_records} 条`);
  if (data.archive_bytes !== undefined) parts.push(`迁移包 ${data.archive_bytes.toLocaleString()} 字节`);
  return parts.join(' · ') || '等待统计';
}
function showCompatibility(info) {
  const container = el('compatibility');
  container.replaceChildren();
  if (!info) return;
  const labels = {embedding:'Embedding', content_storage:'内容存储', vector_storage:'向量存储'};
  const fields = {provider:'提供方', model:'模型', dimension:'维度', model_version:'模型版本', backend:'后端', database_type:'数据库类型', database_version:'数据库版本'};
  for (const [section, values] of Object.entries(info.target)) {
    for (const [field, after] of Object.entries(values)) {
      const line = document.createElement('p');
      const before = info.source?.[section]?.[field];
      line.textContent = `${labels[section]} ${fields[field]}：${before ?? '未知／不适用'} → ${after ?? '未知／不适用'}`;
      container.append(line);
    }
  }
  for (const notice of info.notices) {
    const line = document.createElement('p'); line.textContent = notice; container.append(line);
  }
}
function fail(error) { el('error').textContent = error.message; }
function controls(value) {
  busy = value;
  for (const id of ['export','inspect','file']) el(id).disabled = value;
  el('import').disabled = value || !preflightId;
}
function confirmed() {
  if (!el('maintenance').checked) throw new Error('请先确认已停止业务进程并完成必要落盘。');
}
async function refresh() {
  const data = await api('status');
  el('operations').hidden = false;
  el('connection').textContent = `迁移专用模式 · 内容存储：${data.storage} · 向量存储：${data.vector}`;
  el('history').replaceChildren();
  for (const job of data.jobs) {
    const button = document.createElement('button');
    button.textContent = `${job.kind} · ${job.status} · ${job.id.slice(0,8)}`;
    button.onclick = () => watch(job.id).catch(fail);
    el('history').append(button);
  }
  const running = data.jobs.find((j) => j.status === 'running');
  if (running) watch(running.id).catch(fail);
}
async function watch(id) {
  const generation = ++pollGeneration;
  for (;;) {
    const job = await api('jobs/' + id);
    if (generation !== pollGeneration) return;
    controls(job.status === 'running');
    el('phase').textContent = `${phases[job.phase] || job.phase} · ${{running:'进行中', succeeded:'成功', failed:'失败'}[job.status] || job.status}`;
    el('result-summary').textContent = summary(job.result || job.counts || {index_records:job.index_records});
    el('result').textContent = JSON.stringify(job.result || job.counts || {index_records:job.index_records || 0}, null, 2);
    if (job.status === 'failed') { fail(new Error(job.error)); break; }
    if (job.status === 'succeeded') {
      if (job.kind === 'export') {
        el('download').href = `/api/migration/jobs/${id}/download`;
        el('download').hidden = false;
      } else if (job.kind === 'preflight') {
        artifactId = job.result.artifact_id; preflightId = id;
        el('upload').textContent = '预检通过，可以开始导入。';
        el('plan').textContent = JSON.stringify(job.result, null, 2);
        showCompatibility(job.result.compatibility);
        el('plan-summary').textContent = summary(job.result); controls(false);
      } else if (job.kind === 'import') {
        preflightId = null; controls(false);
        el('phase').textContent = '导入与校验完成。请配置目标身份和访问权限，停止迁移服务后启动业务服务。';
      }
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}
async function submit(kind) {
  confirmed(); el('error').textContent = '';
  const data = await api('jobs', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({kind, artifact_id:artifactId, preflight_id:preflightId, maintenance_confirmed:true})});
  return watch(data.job_id);
}
el('connect').onclick = async () => {
  try { await api('login', {method:'POST', headers:{'X-API-Key':el('key').value}}); el('key').value=''; await refresh(); }
  catch (error) { fail(error); }
};
el('export').onclick = () => submit('export').catch(fail);
el('import').onclick = () => submit('import').catch(fail);
el('file').onchange = () => { artifactId=null; preflightId=null; showCompatibility(null); el('plan').textContent=''; el('plan-summary').textContent=''; controls(busy); };
el('inspect').onclick = async () => {
  try {
    confirmed(); const file = el('file').files[0]; if (!file) throw new Error('请选择迁移包。');
    controls(true); preflightId=null; showCompatibility(null); el('error').textContent=''; el('upload').textContent='上传中…';
    const result = await api('uploads', {method:'POST', headers:{'Content-Type':'application/octet-stream'}, body:file});
    artifactId=result.artifact_id; el('upload').textContent=`上传完成，${result.size} 字节。正在预检…`;
    await submit('preflight');
  } catch(error) { controls(false); fail(error); }
};
refresh().catch(() => {});
