'use strict';
const el = (id) => document.getElementById(id);
const apiBase = '/api/v1/migration/';
const authHeaders = () => ({'X-API-Key':sessionStorage.getItem('ogmemory-viewer-api-key') || ''});
let maintenanceTask = null;
let pendingSubmission = JSON.parse(sessionStorage.getItem('ogmemory-migration-pending') || 'null');
const requestId = () => '10000000-1000-4000-8000-100000000000'.replace(/[018]/g, c => (Number(c) ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> Number(c) / 4).toString(16));
let artifactId = null, preflightId = null, activeJob = null, requestBusy = false;
let uncertain = false, pollGeneration = 0, statusRequest = 0;
const phases = {queued:'等待执行', validating:'预检', snapshotting:'创建快照', exporting:'打包', preparing:'准备数据', resolving:'处理记忆冲突', restoring:'写入数据', reindexing:'构建索引', verifying:'校验结果', publishing:'恢复业务', draining:'暂停目标业务', backing_up:'备份原数据', planning:'检查合并条件', merging:'合并记忆', recovering:'恢复备份', complete:'完成', interrupted:'进程中断'};
const kinds = {export:'导出', preflight:'预检', import:'导入', recover:'恢复'};
const states = {running:'进行中', succeeded:'成功', failed:'失败'};
const bytes = (value) => value >= 1048576 ? `${(value / 1048576).toFixed(1)} MiB` : `${value.toLocaleString()} 字节`;
function progress(done, total, label) {
  const bar = el('progress');
  if (Number.isFinite(total) && total > 0 && Number.isFinite(done)) bar.value = Math.min(100, done / total * 100);
  else bar.removeAttribute('value');
  el('progress-label').textContent = label;
}
function controls() {
  const locked = requestBusy || activeJob !== null || uncertain || pendingSubmission !== null;
  el('export').disabled = locked;
  el('inspect').disabled = locked || !el('file').files.length;
  el('file').disabled = locked;
  el('import').disabled = locked || !preflightId;
}
function fail(error) { el('error').textContent = error.message || String(error); }
async function api(path, options = {}) {
  const response = await fetch(apiBase + path, {...options, headers:{...authHeaders(), ...(options.headers || {})}, credentials:'same-origin'});
  const data = await response.json().catch(() => ({error:`服务返回异常响应（HTTP ${response.status}）`}));
  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      ++pollGeneration; uncertain = true; clearPlan();
      el('download').hidden = true; el('download').removeAttribute('href');
      el('operations').hidden = true; el('connection').textContent = '请使用实例管理员身份连接。';
      controls();
    }
    const error = new Error(data.error || `HTTP ${response.status}`); error.status = response.status; throw error;
  }
  return data;
}
function summary(data) {
  if (!data) return '等待统计';
  const labels = {nodes:'记忆节点', relations:'关系', archives:'历史归档'};
  const parts = Object.entries(data.data || data).filter(([key]) => key in labels)
    .map(([key, value]) => `${labels[key]} ${value.count ?? value} 条`);
  if (data.index_records !== undefined) parts.push(`索引 ${data.index_records} 条`);
  if (data.archive_bytes !== undefined) parts.push(`迁移包 ${bytes(data.archive_bytes)}`);
  const shared = data.shared_spaces;
  if (shared?.spaces) {
    const actions = {import:'整批迁入', exported:'已导出', skipped:'整批跳过'};
    const reasons = {target_shared_spaces_disabled:'目标未启用共享空间', content_backend_differs:'内容存储后端不同',
      database_type_differs:'数据库类型不同', shared_schema_differs:'共享空间结构不兼容',
      target_business_not_empty:'目标已有业务数据或索引', target_identity_missing:'目标缺少所需有效身份',
      target_source_authority_changed:'目标来源授权资格已变化'};
    parts.push(`共享空间 ${shared.spaces} 个${shared.nodes === undefined ? '' : `、内容 ${shared.nodes} 条`}：${actions[shared.action] || shared.action}${shared.reason ? `（${reasons[shared.reason] || shared.reason}）` : ''}`);
  }
  return parts.join(' · ') || '等待统计';
}
function showCompatibility(info) {
  const container = el('compatibility'); container.replaceChildren();
  if (!info) return;
  const labels = {embedding:'模型', content_storage:'内容存储', vector_storage:'向量存储'};
  const fields = {provider:'提供方', model:'名称', dimension:'维度', model_version:'版本', backend:'后端', database_type:'数据库', database_version:'版本'};
  const table = document.createElement('table'); table.className = 'migration-table';
  const head = table.createTHead().insertRow();
  for (const title of ['配置', '来源', '当前目标']) { const cell = document.createElement('th'); cell.textContent = title; head.append(cell); }
  const body = table.createTBody();
  for (const [section, values] of Object.entries(info.target || {})) {
    for (const [field, after] of Object.entries(values)) {
      const row = body.insertRow();
      for (const value of [`${labels[section] || section} · ${fields[field] || field}`, info.source?.[section]?.[field] ?? '未知／不适用', after ?? '未知／不适用']) row.insertCell().textContent = value;
    }
  }
  container.append(table);
  for (const notice of info.notices || []) { const line = document.createElement('p'); line.textContent = notice; container.append(line); }
}
function clearPlan() {
  artifactId = null; preflightId = null; showCompatibility(null);
  el('plan').textContent = ''; el('plan-summary').textContent = '';
}
function applyPlan(job) {
  artifactId = job.result.artifact_id; preflightId = job.id;
  el('upload').textContent = `预检通过（任务 ${job.id.slice(0,8)}），请核对目标配置后确认导入。`;
  el('plan').textContent = JSON.stringify(job.result, null, 2);
  showCompatibility(job.result.compatibility); el('plan-summary').textContent = summary(job.result) + (job.result.counts ? ` · 新增 ${job.result.counts.new} · 重复 ${job.result.counts.unchanged} · 冲突 ${job.result.counts.conflicts}` : '');
  selectTab('import'); controls();
}
function selectTab(name) {
  for (const key of ['export', 'import']) {
    el(`tab-${key}`).setAttribute('aria-selected', String(name === key));
    el(`${key}-panel`).hidden = name !== key;
  }
}
function showJob(job) {
  el('phase').textContent = `${kinds[job.kind] || job.kind} · ${phases[job.phase] || job.phase} · ${states[job.status] || job.status}`;
  el('result-summary').textContent = summary(job.result || job.counts || {index_records:job.index_records});
  el('result').textContent = JSON.stringify(job.result || job.counts || {index_records:job.index_records || 0}, null, 2);
  if (job.status === 'running') {
    const done = job.completed ?? job.index_records;
    progress(done, job.total, job.total ? `${done || 0} / ${job.total}` : done ? `已处理 ${done} 条，任务仍在进行` : '处理中，请稍候…');
  } else if (job.status === 'succeeded') progress(1, 1, '此任务已完成');
  else { el('progress').value = 0; el('progress-label').textContent = job.maintenance?.phase === 'restored' ? '导入失败，原数据已恢复。' : maintenanceTask ? '目标仍在维护，请查看失败原因并恢复维护任务。' : '任务失败；请查看原因后重试。'; }
  if (job.maintenance_started_at && job.status === 'running') el('progress-label').textContent += ` · 目标维护 ${Math.floor(Date.now()/1000-job.maintenance_started_at)} 秒`;
  if (job.status === 'succeeded' && job.result?.outcome === 'restored') el('phase').textContent = '原数据已恢复，业务已开放';
}
async function refresh({resume = true} = {}) {
  const generation = pollGeneration, request = ++statusRequest;
  const data = await api('status');
  if (generation !== pollGeneration || request !== statusRequest) return;
  uncertain = false;
  el('operations').hidden = false;
  el('connection').textContent = `已连接 · ${data.storage.toUpperCase()} 内容存储 · ${data.vector === 'none' ? '无 Embedding 检索' : data.vector + ' 向量存储'}`;
  maintenanceTask = data.maintenance?.id || null;
  el('recover').hidden = !maintenanceTask || data.maintenance?.mode !== 'import';
  const running = data.jobs.find((j) => j.status === 'running');
  activeJob = running?.id || null;
  if (pendingSubmission && data.jobs.some(job => job.id === pendingSubmission.request_id)) { pendingSubmission=null; sessionStorage.removeItem('ogmemory-migration-pending'); }
  if (!running && resume && data.jobs.length) showJob(data.jobs[0]);
  el('history').replaceChildren();
  if (!data.jobs.length) el('history').textContent = '还没有迁移任务';
  for (let job of data.jobs) {
    const button = document.createElement('button'); button.type = 'button';
    button.textContent = `${kinds[job.kind] || job.kind} · ${states[job.status] || job.status} · ${job.id.slice(0,8)}`;
    button.onclick = async () => {
      if (activeJob || requestBusy || uncertain) { fail(new Error('请先等待当前任务结束，或刷新任务状态。')); return; }
      try { job = await api('jobs/' + job.id); } catch (error) { fail(error); return; }
      showJob(job);
      if (job.status === 'failed') fail(new Error(job.error));
      if (job.kind === 'export' && job.status === 'succeeded') offerDownload(job.id);
      if (job.kind === 'preflight' && job.status === 'succeeded') applyPlan(job);
    };
    el('history').append(button);
  }
  controls();
  if (running && resume) watch(running.id).catch(handlePollError);
}
function offerDownload(id) {
  el('download').href = `${apiBase}jobs/${id}/download`; el('download').hidden = false;
  el('download').onclick = async event => {
    event.preventDefault();
    try {
      const response = await fetch(el('download').href, {headers:authHeaders()});
      if (!response.ok) throw new Error('下载失败，请检查管理员连接。');
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a'); link.href=url; link.download='ogmemory-snapshot.tar.gz'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    } catch (error) { fail(error); }
  };
}
function handlePollError(error) { uncertain = true; controls(); fail(new Error(`${error.message}。任务状态尚未确认，请刷新任务状态，勿重复提交。`)); }
async function watch(id) {
  const generation = ++pollGeneration;
  activeJob = id; controls();
  for (;;) {
    const job = await api('jobs/' + id);
    if (generation !== pollGeneration) return;
    showJob(job);
    if (job.status !== 'running') {
      activeJob = null;
      if (job.status === 'failed') { preflightId = null; fail(new Error(job.error)); }
      else if (job.kind === 'export') offerDownload(id);
      else if (job.kind === 'preflight') {
        applyPlan(job);
      } else if (job.kind === 'import') {
        preflightId = null;
        el('upload').textContent = '本次导入已完成。';
        el('phase').textContent = '导入与校验完成';
      }
      await refresh(); controls(); return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}
async function submit(kind, {retry = false} = {}) {
  if (pendingSubmission && !retry) throw new Error('上次提交尚未确认，请先刷新任务状态。');
  if (kind === 'import' && !confirm('导入期间目标全部业务读写将暂停。系统会先备份原数据，失败时恢复。确认导入？')) return;
  el('error').textContent = ''; ++pollGeneration; requestBusy = true; controls();
  if (kind === 'export') el('download').hidden = true;
  try {
    const payload = {kind, request_id:requestId(), artifact_id:artifactId, preflight_id:preflightId, recover_task:maintenanceTask, maintenance_confirmed:true};
    if (pendingSubmission) {
      if (kind !== pendingSubmission.kind) throw new Error('上次提交尚未确认，请先刷新任务状态。');
      Object.assign(payload, pendingSubmission);
    }
    else { pendingSubmission=payload; sessionStorage.setItem('ogmemory-migration-pending',JSON.stringify(payload)); }
    const result = await api('jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    pendingSubmission=null; sessionStorage.removeItem('ogmemory-migration-pending');
    activeJob = result.job_id;
    if (kind === 'import') preflightId = null;
  } catch (error) {
    if ([400,401,403,404,409,413,422].includes(error.status)) { pendingSubmission=null; sessionStorage.removeItem('ogmemory-migration-pending'); }
    uncertain = true; throw error;
  }
  finally { requestBusy = false; controls(); }
  return watch(activeJob);
}
function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open('POST', apiBase + 'uploads'); for (const [key,value] of Object.entries(authHeaders())) xhr.setRequestHeader(key,value); xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (event) => progress(event.loaded, event.lengthComputable ? event.total : null, `上传 ${bytes(event.loaded)} / ${bytes(file.size)}`);
    xhr.onerror = () => reject(new Error('上传连接中断，请重新上传。'));
    xhr.onload = () => {
      let data; try { data = JSON.parse(xhr.responseText); } catch { reject(new Error('上传返回异常响应。')); return; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data); else reject(new Error(data.error || `上传失败（HTTP ${xhr.status}）`));
    };
    xhr.send(file);
  });
}
el('export').onclick = () => submit('export').catch(handlePollError);
el('import').onclick = () => submit('import').catch(handlePollError);
el('file').onchange = () => { clearPlan(); el('upload').textContent = el('file').files[0]?.name || '尚未选择文件'; controls(); if (el('file').files.length) el('inspect').click(); };
el('inspect').onclick = async () => {
  const file = el('file').files[0]; if (!file) return;
  clearPlan(); requestBusy = true; controls(); el('error').textContent = ''; el('phase').textContent = '上传迁移包';
  try { const result = await upload(file); artifactId = result.artifact_id; el('upload').textContent = `已上传 ${bytes(result.size)}，正在预检…`; await submit('preflight'); }
  catch (error) { fail(error); }
  finally { requestBusy = false; controls(); }
};
el('retry-status').onclick = async () => {
  el('error').textContent = '';
  try { await refresh(); if (pendingSubmission && !activeJob) await submit(pendingSubmission.kind, {retry:true}); }
  catch (error) { handlePollError(error); }
};
for (const name of ['export', 'import']) el(`tab-${name}`).onclick = () => selectTab(name);
try { document.documentElement.dataset.theme = localStorage.getItem('ogmemory-viewer-theme') || 'light'; } catch {}
el('theme-button').onclick = () => { const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = next; try { localStorage.setItem('ogmemory-viewer-theme', next); } catch {} };
el('recover').onclick = () => submit('recover').catch(handlePollError);
controls(); refresh().catch(error => { fail(new Error(error.message + ' 请在记忆查看器的连接设置中使用 ROOT 管理员身份。')); });
