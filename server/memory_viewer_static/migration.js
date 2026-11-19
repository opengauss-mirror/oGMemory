'use strict';

const el = (id) => document.getElementById(id);
const jobKinds = { export: '导出', preflight: '预检', import: '导入' };
const jobStates = { running: '进行中', succeeded: '已完成', failed: '失败' };
const phases = {
  queued: '等待执行', validating: '校验中', exporting: '导出中',
  restoring: '恢复数据中', reindexing: '重建索引中', verifying: '校验结果中',
  complete: '完成', interrupted: '进程中断',
};
const number = new Intl.NumberFormat('zh-CN');
const jobs = new Map();
let artifactId = null;
let preflightId = null;
let selectedJobId = null;
let operation = '';

async function api(path, options = {}) {
  const response = await fetch('/api/migration/' + path, { ...options, credentials: 'same-origin' });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || '请求失败（HTTP ' + response.status + '）。');
    error.status = response.status;
    throw error;
  }
  return data;
}

function showError(text) {
  const target = el('error');
  target.textContent = text;
  target.hidden = !text;
}

function fail(error) {
  if (error.status === 401) {
    el('operations').hidden = true;
    el('migration-target').hidden = true;
    el('connect-form').hidden = false;
    setStatus('connection', '连接已失效', 'error');
    el('key').focus();
    showError('请重新连接迁移服务。');
  } else {
    showError(error instanceof TypeError ? '无法连接迁移服务，请检查连接后刷新任务状态。' : error.message);
  }
}

function setStatus(id, text, tone) {
  const target = el(id);
  if (target.textContent !== text) target.textContent = text;
  target.dataset.tone = tone;
}

function setKeyError(text) {
  const key = el('key');
  const error = el('key-error');
  error.textContent = text;
  error.hidden = !text;
  if (text) key.setAttribute('aria-invalid', 'true');
  else key.removeAttribute('aria-invalid');
}

function revealKey(visible) {
  el('key').type = visible ? 'text' : 'password';
  const toggle = el('toggle-key');
  toggle.textContent = visible ? '隐藏' : '显示';
  toggle.setAttribute('aria-label', visible ? '隐藏管理员密钥' : '显示管理员密钥');
}

function setOperation(kind) {
  operation = kind;
  const busy = Boolean(kind);
  for (const id of ['export', 'inspect', 'file', 'maintenance', 'refresh-jobs']) el(id).disabled = busy;
  el('import').disabled = busy || !preflightId;
  for (const [id, expected] of [['export', 'export'], ['inspect', 'preflight'], ['import', 'import'], ['refresh-jobs', 'refresh']]) {
    el(id).setAttribute('aria-busy', String(kind === expected));
  }
}

function maintenanceConfirmed() {
  if (el('maintenance').checked) return true;
  showError('请确认已停止业务进程并完成落盘。');
  el('maintenance').focus();
  return false;
}

function summary(data) {
  const labels = { nodes: '记忆节点', relations: '关系', archives: '历史归档' };
  const parts = Object.entries(data.data || data).filter(([key]) => Object.hasOwn(labels, key))
    .map(([key, value]) => labels[key] + ' ' + number.format(value.count) + ' 条');
  if (data.source_backend && data.target_backend) {
    parts.unshift(data.source_backend.toUpperCase() + ' → ' + data.target_backend.toUpperCase());
  }
  if (data.index_records != null) parts.push('索引记录 ' + number.format(data.index_records) + ' 条');
  if (data.archive_bytes != null) parts.push('迁移包 ' + number.format(data.archive_bytes) + ' 字节');
  return parts.join(' · ');
}

function showCompatibility(info) {
  const container = el('compatibility');
  container.replaceChildren();
  if (!info) return;

  const labels = { embedding: '向量模型', content_storage: '内容存储', vector_storage: '向量存储' };
  const fields = {
    provider: '提供方', model: '模型', dimension: '维度', model_version: '模型版本',
    backend: '后端', database_type: '数据库类型', database_version: '数据库版本',
  };
  const comparison = document.createElement('div');
  comparison.className = 'migration-compare';
  for (const [section, values] of Object.entries(info.target)) {
    const group = document.createElement('section');
    group.className = 'migration-compare-group';
    const heading = document.createElement('h3');
    heading.textContent = labels[section] || section;
    const list = document.createElement('dl');
    for (const [field, after] of Object.entries(values)) {
      const title = document.createElement('dt');
      title.textContent = fields[field] || field;
      const value = document.createElement('dd');
      const before = info.source?.[section]?.[field];
      value.textContent = (before ?? '未知／不适用') + ' → ' + (after ?? '未知／不适用');
      list.append(title, value);
    }
    group.append(heading, list);
    comparison.append(group);
  }
  container.append(comparison);
  if (info.notices?.length) {
    const notices = document.createElement('ul');
    notices.className = 'migration-compare-notes';
    for (const notice of info.notices) {
      const item = document.createElement('li');
      item.textContent = notice;
      notices.append(item);
    }
    container.append(notices);
  }
}

function clearPreflight() {
  artifactId = null;
  preflightId = null;
  el('upload').textContent = '';
  el('preflight-report').hidden = true;
}

function usePreflight(job) {
  artifactId = job.result.artifact_id;
  preflightId = job.id;
  el('upload').textContent = '预检通过，可以开始导入。';
  el('plan').textContent = JSON.stringify(job.result, null, 2);
  el('plan-summary').textContent = summary(job.result);
  showCompatibility(job.result.compatibility);
  el('preflight-report').hidden = false;
}

function showJob(job) {
  const { phase, status, kind } = job;
  const result = job.result || job.counts || { index_records: job.index_records };
  const text = status === 'succeeded' && kind === 'import'
    ? '导入与校验完成。请配置目标身份和访问权限，停止迁移服务后启动业务服务。'
    : [jobKinds[kind], status === 'running' || (status === 'failed' && phase === 'interrupted') ? phases[phase] : null, jobStates[status]].filter(Boolean).join(' · ');
  const tone = { running: 'active', succeeded: 'success', failed: 'error' }[status] || 'neutral';
  setStatus('phase', text, tone);
  el('task-reference').hidden = false;
  el('task-id').textContent = job.id;
  el('task-progress').hidden = status !== 'running';
  el('result-summary').textContent = status === 'failed' ? job.error || '请查看任务日志。' : summary(result);
  el('result').textContent = JSON.stringify(result, null, 2);
  el('result-details').hidden = Object.values(result).every((value) => value === undefined);
  if (status === 'succeeded' && kind === 'export') {
    el('download').href = '/api/migration/jobs/' + encodeURIComponent(job.id) + '/download';
    el('download').hidden = false;
  }
}

function selectJob(job) {
  selectedJobId = job.id;
  for (const button of el('history').querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(button.dataset.jobId === job.id));
  }
  showJob(job);
}

function updateHistory(job) {
  jobs.set(job.id, job);
  const list = el('history');
  let button = [...list.querySelectorAll('button')].find((item) => item.dataset.jobId === job.id);
  if (!button) {
    const row = document.createElement('li');
    button = document.createElement('button');
    button.type = 'button';
    button.dataset.jobId = job.id;
    const name = document.createElement('strong');
    name.textContent = jobKinds[job.kind] || job.kind;
    const status = document.createElement('span');
    status.className = 'migration-job-status';
    const id = document.createElement('code');
    id.textContent = job.id.slice(0, 8);
    button.append(name, status, id);
    button.setAttribute('aria-pressed', String(selectedJobId === job.id));
    button.onclick = () => {
      const chosen = jobs.get(job.id);
      selectJob(chosen);
      if (!operation && chosen.kind === 'preflight' && chosen.status === 'succeeded') {
        el('file').value = '';
        usePreflight(chosen);
        setOperation('');
      }
    };
    row.append(button);
    list.append(row);
  }
  const state = button.querySelector('.migration-job-status');
  state.textContent = jobStates[job.status] || job.status;
  el('history-empty').hidden = true;
  return button.closest('li');
}

async function watch(id) {
  for (;;) {
    const job = await api('jobs/' + encodeURIComponent(id));
    updateHistory(job);
    if (selectedJobId === id) showJob(job);
    if (job.status !== 'running') {
      if (job.kind === 'preflight') {
        if (job.status === 'succeeded') usePreflight(job);
        else el('upload').textContent = '预检失败。请查看任务进度。';
      }
      if (job.kind === 'import' && job.status === 'succeeded') {
        el('file').value = '';
        clearPreflight();
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

async function refresh({ focusAfterConnect = false, allowUnauthenticated = false } = {}) {
  const data = await api('status').catch((error) => {
    if (!allowUnauthenticated || error.status !== 401) throw error;
    return null;
  });
  if (data === null) return;
  el('operations').hidden = false;
  el('connect-form').hidden = true;
  el('migration-target').hidden = false;
  el('migration-storage').textContent = data.storage;
  el('migration-vector').textContent = data.vector;
  setStatus('connection', '已连接', 'success');
  const list = el('history');
  const previous = jobs.size;
  jobs.clear();
  for (const job of data.jobs) updateHistory(job);
  for (const button of list.querySelectorAll('button')) {
    if (!jobs.has(button.dataset.jobId)) button.closest('li').remove();
  }
  el('history-empty').hidden = Boolean(jobs.size);
  const running = data.jobs.find((job) => job.status === 'running');
  const selected = running || jobs.get(selectedJobId) || (!previous ? data.jobs[0] : null);
  if (selected) {
    selectJob(selected);
    if (!previous && selected.kind === 'preflight' && selected.status === 'succeeded') usePreflight(selected);
  }
  if (focusAfterConnect && running) el('phase').focus();
  if (running) {
    setOperation(running.kind);
    await watch(running.id);
  }
}

async function submit(kind) {
  const { job_id } = await api('jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, artifact_id: artifactId, preflight_id: preflightId, maintenance_confirmed: true }),
  });
  const job = { id: job_id, kind, status: 'running', phase: 'queued' };
  el('history').prepend(updateHistory(job));
  selectJob(job);
  await watch(job_id);
}

async function run(kind, action) {
  if (operation) return;
  showError('');
  setOperation(kind);
  try {
    await action();
  } catch (error) {
    if (kind === 'preflight' && !preflightId) el('upload').textContent = '未完成预检。请查看提示。';
    fail(error);
  } finally {
    setOperation('');
  }
}

el('connect-form').onsubmit = async (event) => {
  event.preventDefault();
  const connect = el('connect');
  if (connect.disabled) return;
  const key = el('key');
  if (!key.value) {
    setKeyError('请输入管理员密钥。');
    key.focus();
    return;
  }
  connect.disabled = true;
  connect.setAttribute('aria-busy', 'true');
  setKeyError('');
  showError('');
  try {
    await api('login', { method: 'POST', headers: { 'X-API-Key': key.value } });
    key.value = '';
    revealKey(false);
    await run('refresh', () => refresh({ focusAfterConnect: true }));
    if (!el('operations').hidden && document.activeElement === document.body) el('maintenance').focus();
  } catch (error) {
    if (error.status === 401) {
      setKeyError('密钥未通过验证，请检查后重试。');
      key.focus();
    } else fail(error);
  } finally {
    connect.disabled = false;
    connect.setAttribute('aria-busy', 'false');
  }
};

el('key').oninput = () => setKeyError('');
el('toggle-key').onclick = () => revealKey(el('key').type === 'password');
el('maintenance').onchange = () => showError('');
el('refresh-jobs').onclick = () => run('refresh', refresh);
el('export').onclick = () => {
  if (maintenanceConfirmed()) run('export', () => submit('export'));
};
el('file').onchange = () => {
  clearPreflight();
  setOperation('');
};
el('inspect').onclick = () => {
  if (!maintenanceConfirmed()) return;
  const file = el('file').files[0];
  if (!file) {
    el('upload').textContent = '请先选择迁移包。';
    el('file').focus();
    return;
  }
  run('preflight', async () => {
    clearPreflight();
    el('upload').textContent = '上传中…';
    const upload = await api('uploads', {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file,
    });
    artifactId = upload.artifact_id;
    el('upload').textContent = '已上传 ' + number.format(upload.size) + ' 字节，等待预检…';
    await submit('preflight');
  });
};

const dialog = el('import-dialog');
el('import').onclick = () => {
  if (maintenanceConfirmed()) {
    dialog.returnValue = '';
    dialog.showModal();
  }
};
dialog.onclose = () => {
  if (dialog.returnValue !== 'import') return;
  el('phase').focus();
  run('import', () => submit('import'));
};

setOperation('refresh');
el('connect').disabled = true;
refresh({ allowUnauthenticated: true }).catch(fail).finally(() => {
  el('connect').disabled = false;
  setOperation('');
});
