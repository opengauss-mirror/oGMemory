const USER_MEMORY_DIRECTORIES = [
  { directory: 'profile', type: 'profile', singleton: true },
  { directory: 'preferences', type: 'preference' },
  { directory: 'entities', type: 'entity' },
  { directory: 'events', type: 'event' },
  { directory: 'cases', type: 'case' },
  { directory: 'patterns', type: 'pattern' },
  { directory: 'session_summaries', type: 'session_summary' },
  { directory: 'session_archives', type: 'session_archive' },
];
const AGENT_MEMORY_DIRECTORIES = [
  { directory: 'patterns', type: 'pattern' },
  { directory: 'tools', type: 'tool_memory' },
  { directory: 'tool_units', type: 'tool_evidence' },
  { directory: 'experiences', type: 'experience' },
];
const TYPE_LABELS = {
  profile: '用户画像', preference: '偏好', entity: '实体', event: '事件', case: '案例',
  pattern: '模式', tool_memory: '工具记忆', tool_evidence: 'Tools', experience: 'Experiences', session_summary: '会话摘要',
  session_archive: '会话归档', skill: 'Skill',
};
const MONITOR_ICONS = {
  overview: '▦', service: '◇', storage: '▤', models: '◉', index: '▥', queue: '≡', retrieval: '⌕',
};
const MONITOR_STATUS = {
  ok: { label: '正常', summary: '所有组件运行正常' },
  degraded: { label: '需关注', summary: '部分组件需要关注' },
  error: { label: '异常', summary: '部分组件运行异常' },
};
const TASK_TYPE_LABELS = {
  memory_index: '记忆索引', reindex: '目录 Reindex', skill_processing: 'Skill 处理', session_commit: '会话提交',
};
const TASK_STATUS_LABELS = { pending: '等待中', running: '执行中', failed: '失败', unknown: '未知' };
const TASK_ERROR_MESSAGES = {
  TASK_RESOURCE_NOT_FOUND: {
    reason: '“{failed_step}”步骤未完成，因此没有生成资源 {resource_id}。',
    action: '确认资源已写入 ContextFS 后重新触发索引任务。',
  },
  TASK_VECTOR_DB_UNAVAILABLE: {
    reason: '向量数据库 {dependency} 无法处理资源 {resource_id}。',
    action: '检查 openGauss 服务和连接配置，恢复后重新触发索引任务。',
  },
  TASK_COMPONENT_ERROR: {
    reason: '{component}无法处理资源 {resource_id}。',
    action: '检查服务版本和任务处理日志，修复组件后重新触发任务。',
  },
  TASK_MAX_RETRIES: {
    reason: '任务重试 {retry_count} 次后仍未完成。',
    action: '查看任务依赖状态，修复根因后重新触发任务。',
  },
  TASK_DEPENDENCY_TIMEOUT: {
    reason: '{dependency}处理资源 {resource_id} 时响应超时。',
    action: '检查依赖服务连通性和负载，恢复后重新触发任务。',
  },
  TASK_DEPENDENCY_UNAVAILABLE: {
    reason: '{dependency}不可用，资源 {resource_id} 暂未索引。',
    action: '恢复依赖服务后重新触发任务。',
  },
  TASK_AUTH_FAILED: {
    reason: '{dependency}认证失败，资源 {resource_id} 暂未索引。',
    action: '更新服务凭据并验证连接后重新触发任务。',
  },
  TASK_RATE_LIMITED: {
    reason: '{dependency}请求频率受限，资源 {resource_id} 暂未索引。',
    action: '降低并发或等待限流窗口结束后重新触发任务。',
  },
  TASK_UNKNOWN: {
    reason: '{detail}',
    action: '根据失败详情修复根因后重新触发任务。',
  },
};
const API_KEY_STORAGE_KEY = 'ogmemory-viewer-api-key';
const API_KEY_CHANNEL_NAME = 'ogmemory-viewer-api-key-sync';
const SCENE_PAGE_SIZE = 6;
const storedApiKey = sessionStorage.getItem(API_KEY_STORAGE_KEY) || '';
localStorage.removeItem(API_KEY_STORAGE_KEY);

const state = {
  apiKey: storedApiKey,
  accountId: localStorage.getItem('ogmemory-viewer-account-id') || '',
  tree: null, accountTrees: new Map(), inventories: new Map(), expanded: new Set(), directories: new Map(), directorySpecs: new Map(),
  singletonSpecs: new Map(), selectedUri: '', selectedDirectoryKey: '', detailRequestId: 0, searchQuery: '', view: 'memory',
  adminMode: true, retrievalRequestId: 0, searchTargets: new Map(), lastSearchTarget: null,
  retrievalHits: [], metricsRequestId: 0, monitoringRequestId: 0, treeRequestId: 0, accountRequestId: 0,
  monitoringData: null, activeMonitor: 'overview',
  tasks: [], tasksPage: 1, tasksRequestId: 0, tasksSnapshot: '',
  taskRetryEnabled: false, selectedTaskId: '',
  sceneScopes: new Map(), scenes: [], sceneRequestId: 0, sceneOffset: 0, sceneNextOffset: null,
  selectedSceneId: '',
};
const apiKeyChannel = typeof BroadcastChannel === 'function'
  ? new BroadcastChannel(API_KEY_CHANNEL_NAME)
  : null;
let initialTreeLoadPending = Boolean(apiKeyChannel && !state.apiKey);

const $ = (selector) => document.querySelector(selector);
const elements = {
  account: $('#account-select'), connection: $('#connection'), tree: $('#hierarchy-tree'),
  treeCount: $('#tree-count'), treeRoot: $('#tree-root'), search: $('#search-input'),
  refresh: $('#refresh-button'), settings: $('#settings-dialog'), detailView: $('#detail-view'),
  detailPlaceholder: $('#detail-placeholder'), shell: $('.app-shell'), pageTitle: $('#page-title'),
  memoryView: $('#memory-view'), dashboardView: $('#dashboard-view'),
  retrievalView: $('#retrieval-view'), retrievalScope: $('#retrieval-scope'),
  monitoringView: $('#monitoring-view'), monitoringContent: $('#monitoring-content'),
  retrievalResults: $('#retrieval-results'), retrievalDetail: $('#retrieval-detail'),
  revealCurrent: $('#reveal-current-button'),
  tasksView: $('#tasks-view'), tasksPanel: $('#tasks-panel'),
  scenesView: $('#scenes-view'), scenesScope: $('#scenes-scope'), scenesList: $('#scenes-list'),
};

function requestHeaders() { return state.apiKey ? { 'X-API-Key': state.apiKey } : {}; }
function setApiKey(value) {
  state.apiKey = value.trim();
  if (state.apiKey) sessionStorage.setItem(API_KEY_STORAGE_KEY, state.apiKey);
  else sessionStorage.removeItem(API_KEY_STORAGE_KEY);
}
function loadTreeAfterApiKeySync() {
  if (!initialTreeLoadPending) return;
  initialTreeLoadPending = false;
  loadTree();
}
function apiUrl(path, values = {}) {
  const url = new URL(path, location.origin);
  Object.entries(values).forEach(([key, value]) => { if (value) url.searchParams.set(key, value); });
  return url;
}
async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { ...requestHeaders(), ...(options.headers || {}) },
  });
  const isJson = (response.headers.get('content-type') || '').includes('application/json');
  const body = isJson ? await response.json() : null;
  if (!response.ok) throw new Error(body?.error || `HTTP ${response.status}`);
  if (!body) throw new Error('服务返回了无法识别的响应');
  return body;
}

async function loadTree() {
  const requestId = ++state.treeRequestId;
  const accountRequestId = ++state.accountRequestId;
  const requestOptions = { headers: requestHeaders() };
  resetRetrieval();
  resetScenes();
  setLoading(true);
  try {
    const scope = await fetchJson(apiUrl('/api/v1/memory-viewer/scope'), requestOptions);
    const adminMode = Boolean(scope.admin_mode);
    const body = adminMode
      ? await fetchJson(apiUrl('/api/v1/memory-viewer/admin/tree', { accountId: state.accountId }), requestOptions)
      : publicTree(scope);
    if (requestId !== state.treeRequestId || accountRequestId !== state.accountRequestId) return;
    state.adminMode = adminMode;
    state.tree = body;
    state.accountId = body.account_id || '';
    state.accountTrees.clear();
    state.inventories.clear();
    if (state.accountId) state.accountTrees.set(state.accountId, body);
    state.directories.clear();
    clearDetail();
    localStorage.setItem('ogmemory-viewer-account-id', state.accountId);
    renderAccounts(body.accounts || []);
    renderSearchScopes();
    openRootDirectories();
    renderContextTree();
    renderDashboard();
    loadInventories(body.accounts || []);
    setConnection(true);
  } catch (error) {
    if (requestId !== state.treeRequestId || accountRequestId !== state.accountRequestId) return;
    showTreeError(error.message);
    setConnection(false);
    if (!state.apiKey || error.message === 'Invalid API Key') openSettings();
  } finally {
    if (requestId === state.treeRequestId) setLoading(false);
  }
}

async function loadInventories(accounts) {
  const requestId = state.treeRequestId;
  const pending = accounts.map((account) => String(account.account_id || '')).filter(Boolean);
  await Promise.all(Array.from({ length: Math.min(3, pending.length) }, async () => {
    while (pending.length && requestId === state.treeRequestId) {
      const accountId = pending.shift();
      try {
        const data = await fetchJson(apiUrl('/api/v1/memory-viewer/inventory', {
          accountId: state.adminMode ? accountId : '',
        }));
        if (requestId !== state.treeRequestId) return;
        data.directories = Object.fromEntries(Object.entries(data.directories || {}).map(([key, count]) => [
          key.replace(/^(ctx:\/\/[^/]+\/(?:users|agents)\/)([^/]+)(\/(?:memories|skills)\/)/,
            (_, prefix, owner, suffix) => `${prefix}${encodeURIComponent(owner)}${suffix}`), count,
        ]));
        data.memoryTotals = {};
        for (const [key, count] of Object.entries(data.directories)) {
          const match = key.match(/^(ctx:\/\/[^/]+\/(?:users|agents))\/[^/]+\/memories\/[^/]+$/);
          if (!match) continue;
          const ownerRoot = key.slice(0, key.lastIndexOf('/'));
          data.memoryTotals[match[1]] = (data.memoryTotals[match[1]] || 0) + Number(count);
          data.memoryTotals[ownerRoot] = (data.memoryTotals[ownerRoot] || 0) + Number(count);
        }
        state.inventories.set(accountId, data);
        renderContextTree();
        if (accountId === state.accountId) renderDashboard();
      } catch (error) {
        if (requestId === state.treeRequestId && accountId === state.accountId) {
          $('#dashboard-memories').textContent = '—';
          $('#dashboard-skills').textContent = '—';
        }
      }
    }
  }));
}

async function selectAccount(accountId, toggle = false) {
  const root = `ctx://${accountId}`;
  if (toggle && state.expanded.has(root)) {
    state.expanded.delete(root);
    renderContextTree();
    return;
  }
  const requestId = ++state.accountRequestId;
  const treeRequestId = state.treeRequestId;
  resetScenes();
  let tree = state.accountTrees.get(accountId);
  if (!tree && state.adminMode) {
    try {
      tree = await fetchJson(apiUrl('/api/v1/memory-viewer/admin/tree', { accountId }));
    } catch (error) {
      if (requestId !== state.accountRequestId || treeRequestId !== state.treeRequestId) return;
      throw error;
    }
  }
  if (!tree || requestId !== state.accountRequestId || treeRequestId !== state.treeRequestId) return;
  const resolvedAccountId = String(tree.account_id || '');
  if (!resolvedAccountId) throw new Error('无可用账户');
  state.accountTrees.set(resolvedAccountId, tree);
  state.expanded.add('ctx://');
  state.expanded.add(contextRoot(resolvedAccountId));
  if (resolvedAccountId !== state.accountId) resetRetrieval();
  state.accountId = resolvedAccountId;
  state.tree = tree;
  localStorage.setItem('ogmemory-viewer-account-id', resolvedAccountId);
  renderAccounts(tree.accounts || []);
  renderSearchScopes();
  clearDetail();
  renderContextTree();
  renderDashboard();
  if (state.view === 'dashboard') loadDashboardMetrics();
  if (state.view === 'monitoring') loadMonitoring();
  if (state.view === 'tasks') loadTasks();
}

function publicTree(scope) {
  const accountId = String(scope.account_id || '');
  const userId = String(scope.user_id || '');
  const agents = (scope.agents || []).map((agent) => ({
    agent_id: String(agent.agent_id || ''), owner_user_id: userId,
  }));
  return {
    accounts: accountId ? [{ account_id: accountId }] : [],
    account_id: accountId,
    users: userId ? [{ user_id: userId, role: 'user', agents }] : [],
    agents,
  };
}

function renderAccounts(accounts) {
  elements.account.innerHTML = accounts.map((account) => {
    const id = String(account.account_id || '');
    const suffix = account.user_count === undefined ? '' : ` · ${account.user_count} 用户`;
    return `<option value="${escapeHtml(id)}">${escapeHtml(id + suffix)}</option>`;
  }).join('');
  elements.account.value = state.accountId;
  elements.account.disabled = accounts.length <= 1;
}

async function loadTasks({ silent = false } = {}) {
  const requestId = ++state.tasksRequestId;
  const refresh = $('#tasks-refresh');
  if (!silent) {
    refresh.classList.add('loading');
    refresh.disabled = true;
    elements.tasksPanel.innerHTML = '<div class="tasks-empty loading"><span>↻</span><strong>正在加载任务</strong></div>';
  }
  try {
    const body = await fetchJson(apiUrl('/api/v1/memory-viewer/tasks', { accountId: state.accountId, limit: 200 }));
    if (requestId !== state.tasksRequestId) return;
    const tasks = Array.isArray(body.tasks) ? body.tasks : [];
    const retryEnabled = body.capabilities?.retry_failed === true;
    const snapshot = JSON.stringify([tasks, retryEnabled]);
    state.tasks = tasks;
    state.taskRetryEnabled = retryEnabled;
    if (!silent) state.tasksPage = 1;
    if (!silent || snapshot !== state.tasksSnapshot) renderTasks();
    state.tasksSnapshot = snapshot;
  } catch (error) {
    if (requestId !== state.tasksRequestId) return;
    if (!silent) {
      elements.tasksPanel.innerHTML = `<div class="tasks-empty error"><span>!</span><strong>任务加载失败</strong><p>${escapeHtml(error.message)}</p></div>`;
      $('#tasks-pagination').hidden = true;
    }
  } finally {
    if (requestId === state.tasksRequestId) {
      if (!silent) {
        refresh.classList.remove('loading');
        refresh.disabled = false;
      }
    }
  }
}

function filteredTasks() {
  const type = $('#tasks-type').value;
  const status = $('#tasks-status').value;
  return state.tasks.filter((task) => (!type || task.task_type === type) && (!status || task.status === status));
}

function renderTasks() {
  const tasks = filteredTasks();
  const pageSize = 20;
  const pageCount = Math.max(1, Math.ceil(tasks.length / pageSize));
  state.tasksPage = Math.min(state.tasksPage, pageCount);
  const page = tasks.slice((state.tasksPage - 1) * pageSize, state.tasksPage * pageSize);
  if (!page.length) {
    const filtered = Boolean($('#tasks-type').value || $('#tasks-status').value);
    elements.tasksPanel.innerHTML = `<div class="tasks-empty"><span>▣</span><strong>${filtered ? '没有匹配任务' : '暂无后台任务'}</strong><p>${filtered ? '调整筛选条件后重试。' : '异步索引任务启动后，会在这里显示执行状态和更新时间。'}</p></div>`;
    $('#tasks-pagination').hidden = true;
    return;
  }
  elements.tasksPanel.innerHTML = `<div class="tasks-table-wrap"><table class="tasks-table"><thead><tr><th>任务</th><th>类型</th><th>资源</th><th>状态</th><th>更新时间</th></tr></thead><tbody>${page.map((task) => `<tr tabindex="0" data-task-id="${escapeHtml(task.task_id)}"><td><code title="${escapeHtml(task.task_id)}">${escapeHtml(task.task_id)}</code></td><td>${escapeHtml(TASK_TYPE_LABELS[task.task_type] || task.task_type || '—')}</td><td><span class="task-resource" title="${escapeHtml(task.resource_id)}">${escapeHtml(task.resource_id || '—')}</span></td><td><span class="task-status" data-status="${escapeHtml(task.status)}">${escapeHtml(TASK_STATUS_LABELS[task.status] || task.status || '未知')}</span></td><td>${escapeHtml(formatTaskTime(task.updated_at || task.created_at))}</td></tr>`).join('')}</tbody></table></div>`;
  $('#tasks-pagination').hidden = false;
  $('#tasks-page-summary').textContent = `第 ${state.tasksPage} / ${pageCount} 页 · 共 ${tasks.length} 条`;
  $('#tasks-previous').disabled = state.tasksPage === 1;
  $('#tasks-next').disabled = state.tasksPage === pageCount;
}

function formatTaskTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'medium' }).format(date);
}

function formatTaskError(template, params) {
  return template.replace(/\{([a-z_]+)\}/g, (_, key) => String(params[key] ?? '—'));
}

function taskErrorText(task) {
  if (!task.error_code) return { reason: '', action: '' };
  const template = TASK_ERROR_MESSAGES[task.error_code] || TASK_ERROR_MESSAGES.TASK_UNKNOWN;
  const params = {
    detail: `任务失败（错误码：${task.error_code}）`,
    resource_id: task.resource_id || '—',
    ...(task.error_params || {}),
  };
  return {
    reason: formatTaskError(template.reason, params),
    action: formatTaskError(template.action, params),
    failedStep: String(params.failed_step || ''),
    upstream: String(params.upstream_step || ''),
  };
}

function openTaskDetail(task) {
  const error = taskErrorText(task);
  const fields = [
    ['状态', TASK_STATUS_LABELS[task.status] || task.status || '未知'],
    ['类型', TASK_TYPE_LABELS[task.task_type] || task.task_type || '—'],
    ['处理阶段', task.event_type || task.stage || '—'],
    ['重试次数', String(task.retry_count || 0)],
    ['创建时间', formatTaskTime(task.created_at)],
    ['更新时间', formatTaskTime(task.updated_at)],
  ];
  if (task.status === 'failed' && error.reason) fields.push(['失败原因', error.reason]);
  if (task.status === 'failed' && error.failedStep) fields.push(['失败步骤', error.failedStep]);
  if (task.status === 'failed' && error.upstream) fields.push(['生成链路', error.upstream]);
  if (task.status === 'failed' && error.action) fields.push(['建议处理', error.action]);
  state.selectedTaskId = task.task_id || '';
  $('#task-detail-title').textContent = task.task_id || '任务详情';
  $('#task-detail-fields').innerHTML = `${fields.map(([label, value]) => `<div class="task-detail-field${label === '失败原因' ? ' wide' : ''}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join('')}<div class="task-detail-field wide"><span>资源</span><code>${escapeHtml(task.resource_id || '—')}</code></div>`;
  $('#task-retry').hidden = task.status !== 'failed' || !state.taskRetryEnabled;
  $('#task-retry').disabled = false;
  $('#task-retry').textContent = '重新触发';
  $('#task-action-status').textContent = '';
  $('#task-detail-dialog').showModal();
}

async function retrySelectedTask() {
  const button = $('#task-retry');
  const status = $('#task-action-status');
  if (!state.selectedTaskId || button.disabled) return;
  button.disabled = true;
  button.textContent = '正在触发…';
  status.textContent = '';
  try {
    await fetchJson(apiUrl(`/api/v1/memory-viewer/tasks/${encodeURIComponent(state.selectedTaskId)}/retry`, { accountId: state.accountId }), { method: 'POST' });
    $('#task-detail-dialog').close();
    await loadTasks();
  } catch (error) {
    status.textContent = error.message;
    button.disabled = false;
    button.textContent = '重新触发';
  }
}
function renderSearchScopes() {
  const previous = elements.retrievalScope.value;
  state.searchTargets.clear();
  state.searchTargets.set('all', { accountId: state.accountId, userId: '', agentId: '' });
  const options = ['<option value="all">全部可见记忆</option>'];
  (state.tree?.users || []).forEach((user, userIndex) => {
    const userId = String(user.user_id || '');
    const userKey = `user-${userIndex}`;
    state.searchTargets.set(userKey, { accountId: state.accountId, userId, agentId: '' });
    options.push(`<option value="${userKey}">${escapeHtml(userId)} · 用户记忆</option>`);
    (user.agents || []).forEach((agent, agentIndex) => {
      const agentId = String(agent.agent_id || '');
      const agentKey = `agent-${userIndex}-${agentIndex}`;
      state.searchTargets.set(agentKey, { accountId: state.accountId, userId, agentId });
      options.push(`<option value="${agentKey}">${escapeHtml(agentId)} · Agent</option>`);
    });
  });
  elements.retrievalScope.innerHTML = options.join('') || '<option value="">无可检索范围</option>';
  elements.retrievalScope.value = state.searchTargets.has(previous) ? previous : 'all';
  renderSceneScopes();
}

function resetScenes() {
  state.sceneRequestId += 1;
  state.scenes = [];
  state.sceneOffset = 0;
  state.sceneNextOffset = null;
  state.selectedSceneId = '';
  state.sceneScopes.clear();
  elements.scenesScope.replaceChildren(new Option('加载中', ''));
  elements.scenesScope.disabled = true;
  elements.scenesList.replaceChildren();
  $('#scenes-count').textContent = '';
  $('#scenes-pagination').hidden = true;
  $('#scene-detail').hidden = true;
  $('#scenes-refresh').disabled = false;
}

function renderSceneScopes() {
  const previous = elements.scenesScope.value;
  state.sceneScopes.clear();
  const options = [];
  for (const [key, target] of state.searchTargets) {
    if (key === 'all') continue;
    const owner = target.agentId ? `agent:${target.agentId}` : `user:${target.userId}`;
    if (state.sceneScopes.has(owner)) continue;
    state.sceneScopes.set(owner, { ...target, owner_space: owner });
    options.push(new Option(target.agentId ? `${target.agentId} · Agent` : `${target.userId} · 用户`, owner));
  }
  elements.scenesScope.replaceChildren(...(options.length ? options : [new Option('无可见范围', '')]));
  if (state.sceneScopes.has(previous)) elements.scenesScope.value = previous;
  elements.scenesScope.disabled = !options.length;
  if (state.view === 'scenes') loadScenes();
}

function sceneText(tag, className, text) {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function sceneTime(cluster) {
  return [cluster.time_start, cluster.time_end].filter(Boolean).join(' — ');
}

async function loadScenes(offset = 0) {
  const requestId = ++state.sceneRequestId;
  const accountRequestId = state.accountRequestId;
  const target = state.sceneScopes.get(elements.scenesScope.value);
  const refresh = $('#scenes-refresh');
  state.scenes = [];
  $('#scene-detail').hidden = true;
  $('#scenes-pagination').hidden = true;
  $('#scenes-count').textContent = '';
  elements.scenesList.replaceChildren(sceneText('p', 'scene-status', target ? '正在加载场景…' : '请选择可见的用户或 Agent 范围。'));
  refresh.disabled = false;
  if (!target) return;
  refresh.disabled = true;
  try {
    const body = await fetchJson(apiUrl('/api/v1/scenes', {
      accountId: state.adminMode ? target.accountId : '', userId: state.adminMode ? target.userId : '',
      agentId: target.agentId, owner_space: target.owner_space, offset, limit: SCENE_PAGE_SIZE,
    }));
    if (requestId !== state.sceneRequestId || accountRequestId !== state.accountRequestId) return;
    state.scenes = body.items || [];
    state.sceneOffset = offset;
    state.sceneNextOffset = Number.isInteger(body.next_offset) ? body.next_offset : null;
    $('#scenes-count').textContent = `${body.total} 个场景`;
    renderScenes();
    $('#scenes-pagination').hidden = !(offset || state.sceneNextOffset !== null);
    $('#scenes-page-summary').textContent = `第 ${Math.floor(offset / SCENE_PAGE_SIZE) + 1} / ${Math.max(1, Math.ceil(body.total / SCENE_PAGE_SIZE))} 页`;
    $('#scenes-previous').disabled = offset === 0;
    $('#scenes-next').disabled = state.sceneNextOffset === null;
  } catch (error) {
    if (requestId !== state.sceneRequestId || accountRequestId !== state.accountRequestId) return;
    elements.scenesList.replaceChildren(sceneText('p', 'scene-status error', `场景加载失败：${error.message}`));
  } finally {
    if (requestId === state.sceneRequestId) refresh.disabled = false;
  }
}

function renderScenes() {
  if (!state.scenes.length) {
    elements.scenesList.replaceChildren(sceneText('p', 'scene-status', '当前范围暂无场景记忆。'));
    return;
  }
  const cards = state.scenes.map(scene => {
    const cluster = scene.cluster;
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'scene-card';
    card.dataset.sceneId = scene.scene_id;
    card.append(
      sceneText('strong', 'scene-card-title', cluster.label),
      sceneText('span', 'scene-card-summary', cluster.summary),
      sceneText('span', 'scene-card-count', `${cluster.members.length} 个记忆节点`),
    );
    return card;
  });
  elements.scenesList.replaceChildren(...cards);
  selectScene(state.scenes.find(scene => scene.scene_id === state.selectedSceneId) || state.scenes[0]);
}

function sceneMemberTarget(uri) {
  const match = String(uri).replace(/\/content\.md$/, '').match(/^ctx:\/\/([^/]+)\/(users|agents)\/([^/]+)\/memories\/([^/]+)(?:\/([^/]+))?\/?$/);
  const decode = value => { try { return decodeURIComponent(value); } catch { return value; } };
  if (!match || decode(match[1]) !== state.accountId) return null;
  const [, , ownerType, encodedOwner, directory, encodedId] = match;
  const owner = decode(encodedOwner);
  const spec = (ownerType === 'agents' ? AGENT_MEMORY_DIRECTORIES : USER_MEMORY_DIRECTORIES).find(item => item.directory === directory);
  if (!spec || (!spec.singleton && !encodedId)) return null;
  const id = spec.singleton ? 'self' : decode(encodedId);
  const selectedScope = state.sceneScopes.get(elements.scenesScope.value);
  const userId = ownerType === 'users' ? owner
    : (state.tree?.agents || []).find(agent => agent.agent_id === owner)?.owner_user_id || selectedScope?.userId;
  return {
    uri: `ogmemory://memories/${spec.type}/${spec.type === 'pattern' && ownerType === 'agents' ? 'agent/' : ''}${encodeURIComponent(id)}`,
    userId, agentId: ownerType === 'agents' ? owner : '', type: spec.type, id,
  };
}

function selectScene(scene) {
  state.selectedSceneId = scene.scene_id;
  elements.scenesList.querySelectorAll('[data-scene-id]').forEach(card => {
    card.setAttribute('aria-pressed', String(card.dataset.sceneId === scene.scene_id));
  });
  const cluster = scene.cluster;
  $('#scene-detail').hidden = false;
  $('#scene-detail-title').textContent = cluster.label;
  $('#scene-detail-origin').textContent = scene.origin === 'manual' ? '自定义场景' : '自动沉淀';
  $('#scene-detail-description').textContent = cluster.summary;
  $('#scene-detail-meta').replaceChildren(...[sceneTime(cluster), `版本 ${scene.version}`].filter(Boolean).map(value => sceneText('span', '', value)));
  $('#scene-detail-member-count').textContent = cluster.members.length;
  $('#scene-detail-members').replaceChildren(...cluster.members.map(member => {
    const target = sceneMemberTarget(member.context_uri);
    if (!target) return sceneText('span', 'scene-member-path', member.context_uri);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'scene-member';
    button.dataset.memberUri = target.uri;
    button.dataset.user = target.userId || '';
    button.dataset.agent = target.agentId;
    button.append(sceneText('span', 'scene-member-type', typeLabel(target.type)), sceneText('span', '', target.id));
    return button;
  }));
}
function openRootDirectories() {
  const root = contextRoot();
  state.expanded = new Set([root, `${root}/users`, `${root}/agents`]);
  if (state.adminMode) state.expanded.add('ctx://');
}
function contextRoot(accountId = state.accountId) { return accountId ? `ctx://${accountId}` : ''; }

function directoryCount(accountId, key) {
  return state.inventories.get(accountId)?.directories?.[key];
}

function skillCount(accountId, root) {
  if (!state.inventories.has(accountId)) return null;
  return (directoryCount(accountId, `${root}/generated`) || 0)
    + (directoryCount(accountId, `${root}/imported`) || 0);
}

function subtreeCount(accountId, prefix) {
  const inventory = state.inventories.get(accountId);
  return inventory ? (inventory.memoryTotals[prefix] || 0) : null;
}

function countLabel(count, unit = '记忆') { return count === null || count === undefined ? '' : `${count} ${unit}`; }

function renderContextTree() {
  const accountRoot = contextRoot();
  state.directorySpecs.clear();
  state.singletonSpecs.clear();
  elements.treeRoot.textContent = state.adminMode ? 'ctx://' : accountRoot || '无可用账户';
  elements.treeCount.textContent = state.view === 'skill'
    ? countLabel(totalInventoryCount('skills'), 'Skills')
    : countLabel(totalInventoryCount('memories'));
  elements.revealCurrent.disabled = !state.selectedUri || !state.selectedDirectoryKey;
  elements.revealCurrent.title = state.selectedDirectoryKey || '定位当前文件';
  if (!accountRoot) {
    elements.tree.innerHTML = '<div class="tree-empty">无可见 ContextFS 节点</div>';
    return;
  }
  if (state.view === 'skill') {
    renderSkillTree();
    return;
  }
  elements.tree.innerHTML = renderTreeRoot((accountId, tree) => renderMemoryAccount(accountId, tree));
}

function renderMemoryAccount(accountId, tree) {
  const accountRoot = contextRoot(accountId);
  const branchDepth = state.adminMode ? 2 : 1;
  const users = tree.users || [];
  const agents = tree.agents || [];
  const usersRoot = `${accountRoot}/users`;
  const agentsRoot = `${accountRoot}/agents`;
  return [
    folderNode(usersRoot, 'users', branchDepth, 'folder',
      users.map((user) => renderUserNode(user, usersRoot, branchDepth + 1, accountId)).join(''),
      countLabel(subtreeCount(accountId, usersRoot))),
    folderNode(agentsRoot, 'agents', branchDepth, 'folder',
      agents.map((agent) => renderAgentNode(agent, agentsRoot, branchDepth + 1, accountId)).join(''),
      countLabel(subtreeCount(accountId, agentsRoot))),
  ].join('');
}

function renderTreeRoot(renderChildren) {
  const accountRoot = contextRoot();
  const field = state.view === 'skill' ? 'skills' : 'memories';
  const unit = state.view === 'skill' ? 'Skills' : '记忆';
  if (!state.adminMode) return folderNode(accountRoot, state.accountId, 0, 'account',
    renderChildren(state.accountId, state.tree), countLabel(state.inventories.get(state.accountId)?.[field], unit));
  const accountRows = (state.tree?.accounts || []).map((account) => {
    const accountId = String(account.account_id || '');
    const inventory = state.inventories.get(accountId);
    const meta = countLabel(inventory?.[field], unit) || (account.user_count === undefined ? '' : `${account.user_count} users`);
    const tree = state.accountTrees.get(accountId);
    return accountSwitchNode(accountId, tree ? renderChildren(accountId, tree) : '<div class="tree-empty nested">加载中…</div>', meta);
  }).join('');
  return folderNode('ctx://', 'ctx://', 0, 'root', accountRows,
    state.view === 'skill' ? countLabel(totalInventoryCount('skills'), 'Skills')
      : countLabel(totalInventoryCount('memories')));
}

function totalInventoryCount(field) {
  const accounts = state.tree?.accounts || [];
  return accounts.every((account) => state.inventories.has(account.account_id))
    ? accounts.reduce((sum, account) => sum + state.inventories.get(account.account_id)[field], 0)
    : null;
}

function accountSwitchNode(accountId, children, meta) {
  const key = contextRoot(accountId);
  const expanded = state.expanded.has(key);
  return `<div class="context-node context-folder ${expanded ? 'open' : ''}" data-node-key="${escapeHtml(key)}">
    <button class="context-row" type="button" data-account-switch="${escapeHtml(accountId)}" style="--tree-depth:1">
    <span class="context-chevron">${expanded ? '⌄' : '›'}</span><span class="context-icon" data-kind="account">▱</span>
    <span class="context-name">${escapeHtml(accountId)}</span>${meta ? `<span class="context-meta">${escapeHtml(meta)}</span>` : ''}
    </button><div class="context-children">${children}</div></div>`;
}

function renderSkillTree() {
  elements.tree.innerHTML = renderTreeRoot((accountId, tree) => {
    const agents = tree.agents || [];
    const branchDepth = state.adminMode ? 2 : 1;
    const agentsRoot = `${contextRoot(accountId)}/agents`;
    const agentsHtml = agents.map((agent) => renderSkillAgentNode(agent, agentsRoot, branchDepth + 1, accountId)).join('');
    return folderNode(agentsRoot, 'agents', branchDepth, 'folder', agentsHtml,
      countLabel(state.inventories.get(accountId)?.skills, 'Skills'));
  });
}

function renderSkillAgentNode(agent, agentsRoot, depth = 2, accountId = state.accountId) {
  const agentId = String(agent.agent_id);
  const ownerUserId = String(agent.owner_user_id || '');
  const agentRoot = `${agentsRoot}/${encodeURIComponent(agentId)}`;
  const skillsRoot = `${agentRoot}/skills`;
  const skills = ['generated', 'imported'].map((source) => directoryNode(
    `${skillsRoot}/${source}`, source, depth + 2,
    { accountId, userId: ownerUserId, agentId, type: 'skill', source, itemDepth: depth + 3 },
  )).join('');
  return folderNode(agentRoot, agentId, depth, 'agent',
    folderNode(skillsRoot, 'skills', depth + 1, 'folder', skills,
      countLabel(skillCount(accountId, skillsRoot), 'Skills')),
    countLabel(skillCount(accountId, skillsRoot), 'Skills'));
}

function renderUserNode(user, usersRoot, depth = 2, accountId = state.accountId) {
  const userId = String(user.user_id);
  const userRoot = `${usersRoot}/${encodeURIComponent(userId)}`;
  const memoriesRoot = `${userRoot}/memories`;
  const categories = USER_MEMORY_DIRECTORIES.map((entry) => {
    const key = `${memoriesRoot}/${entry.directory}`;
    const spec = { accountId, userId, agentId: '', type: entry.type, itemDepth: depth + 3 };
    return entry.singleton
      ? singletonNode(key, entry.directory, depth + 2, spec)
      : directoryNode(key, entry.directory, depth + 2, spec);
  }).join('');
  const count = subtreeCount(accountId, memoriesRoot);
  const memories = folderNode(memoriesRoot, 'memories', depth + 1, 'folder', categories, countLabel(count));
  return folderNode(userRoot, userId, depth, 'user', memories, countLabel(count));
}

function renderAgentNode(agent, agentsRoot, depth = 2, accountId = state.accountId) {
  const agentId = String(agent.agent_id);
  const ownerUserId = String(agent.owner_user_id || '');
  const agentRoot = `${agentsRoot}/${encodeURIComponent(agentId)}`;
  const memoriesRoot = `${agentRoot}/memories`;
  const memories = AGENT_MEMORY_DIRECTORIES.map((entry) => directoryNode(
    `${memoriesRoot}/${entry.directory}`, entry.directory, depth + 2,
    { accountId, userId: ownerUserId, agentId, type: entry.type, itemDepth: depth + 3 },
  )).join('');
  const skillsRoot = `${agentRoot}/skills`;
  const skills = ['generated', 'imported'].map((source) => directoryNode(
    `${skillsRoot}/${source}`, source, depth + 2,
    { accountId, userId: ownerUserId, agentId, type: 'skill', source, itemDepth: depth + 3 },
  )).join('');
  const children = [
    folderNode(memoriesRoot, 'memories', depth + 1, 'folder', memories,
      countLabel(subtreeCount(accountId, memoriesRoot))),
    folderNode(skillsRoot, 'skills', depth + 1, 'folder', skills,
      countLabel(skillCount(accountId, skillsRoot), 'Skills')),
  ].join('');
  return folderNode(agentRoot, agentId, depth, 'agent', children,
    countLabel(subtreeCount(accountId, memoriesRoot)));
}

function folderNode(key, label, depth, kind, children, meta = '') {
  const expanded = state.expanded.has(key);
  return `<div class="context-node context-folder ${expanded ? 'open' : ''}" data-node-key="${escapeHtml(key)}">
    <button class="context-row" type="button" data-folder-key="${escapeHtml(key)}" style="--tree-depth:${depth}">
      <span class="context-chevron">${expanded ? '⌄' : '›'}</span><span class="context-icon" data-kind="${kind}">${kind === 'root' ? '⌂' : kind === 'account' ? '▱' : '▰'}</span>
      <span class="context-name">${escapeHtml(label)}</span>${meta ? `<span class="context-meta">${escapeHtml(meta)}</span>` : ''}
    </button><div class="context-children">${children}</div></div>`;
}
function directoryNode(key, label, depth, spec) {
  state.directorySpecs.set(key, spec);
  const record = state.directories.get(key);
  const count = directoryCount(spec.accountId, key);
  const meta = count === undefined ? (record ? `${record.items.length}${record.hasMore ? '+' : ''}` : '')
    : countLabel(count, spec.type === 'skill' ? 'Skills' : '记忆');
  return folderNode(key, label, depth, 'folder', renderDirectoryChildren(key, spec, record), meta);
}
function singletonNode(key, label, depth, spec) {
  state.singletonSpecs.set(key, spec);
  const record = state.directories.get(key);
  const selected = record?.items?.[0]?.uri === state.selectedUri;
  return `<button class="context-row context-resource ${selected ? 'active' : ''}" type="button" data-singleton-key="${escapeHtml(key)}" style="--tree-depth:${depth}">
    <span class="context-chevron"></span><span class="context-icon" data-kind="profile">◇</span>
    <span class="context-name">${escapeHtml(label)}</span><span class="context-meta">${record?.loading ? '…' : countLabel(directoryCount(spec.accountId, key))}</span>
  </button>`;
}
function renderDirectoryChildren(key, spec, record) {
  if (!state.expanded.has(key)) return '';
  if (!record) return '<div class="tree-empty nested">展开后加载</div>';
  if (record.loading && !record.items.length) return '<div class="tree-empty nested">加载中…</div>';
  if (record.error) return `<div class="tree-error">${escapeHtml(record.error)}</div>`;
  const items = record.items.filter(matchesSearch).map((memory) => renderMemoryNode(memory, spec)).join('');
  const empty = items ? '' : '<div class="tree-empty nested">没有节点</div>';
  const more = record.hasMore
    ? `<button class="tree-more" type="button" data-load-more="${escapeHtml(key)}">加载更多</button>`
    : '';
  return items + empty + more;
}
function renderMemoryNode(memory, spec) {
  const label = memory.type === 'profile' ? 'profile' : memory.id || 'unnamed';
  const selected = memory.uri === state.selectedUri;
  return `<button class="context-row context-resource ${selected ? 'active' : ''}" type="button"
    data-resource-uri="${escapeHtml(memory.uri)}" data-account="${escapeHtml(spec.accountId)}" data-user="${escapeHtml(spec.userId)}" data-agent="${escapeHtml(spec.agentId)}" style="--tree-depth:${spec.itemDepth || 5}">
    <span class="context-chevron"></span><span class="context-icon" data-kind="${escapeHtml(memory.type)}">◇</span>
    <span class="context-name">${escapeHtml(label)}</span><span class="context-meta">${escapeHtml(typeLabel(memory.type))}</span>
  </button>`;
}
function matchesSearch(memory) {
  if (!state.searchQuery) return true;
  return [memory.id, memory.abstract, memory.context_uri, memory.uri, memory.type]
    .join(' ').toLocaleLowerCase().includes(state.searchQuery);
}

async function toggleDirectory(key) {
  if (state.expanded.has(key)) {
    state.expanded.delete(key);
    renderContextTree();
    return;
  }
  state.expanded.add(key);
  renderContextTree();
  const spec = state.directorySpecs.get(key);
  if (spec && !state.directories.has(key)) await loadDirectory(key, spec, null, false);
}
async function loadDirectory(key, spec, cursor, append) {
  if (!spec.userId) {
    const record = { items: [], hasMore: false, nextCursor: null, loading: false, error: 'Agent 缺少 owner_user_id' };
    state.directories.set(key, record);
    renderContextTree();
    return record;
  }
  const previous = state.directories.get(key);
  const requestId = (previous?.requestId || 0) + 1;
  state.directories.set(key, {
    items: previous?.items || [], hasMore: previous?.hasMore || false,
    nextCursor: previous?.nextCursor || null, loading: true, error: '', requestId,
  });
  renderContextTree();
  try {
    const path = state.adminMode ? '/api/v1/memory-viewer/admin/memories' : '/api/v1/memories';
    const body = await fetchJson(apiUrl(path, {
      accountId: state.adminMode ? spec.accountId : '', userId: state.adminMode ? spec.userId : '', agentId: spec.agentId,
      type: spec.type, source: spec.source || '', cursor: cursor || '', limit: '100', sort: 'name',
    }));
    if (state.directories.get(key)?.requestId !== requestId) return state.directories.get(key);
    const items = append ? [...(previous?.items || []), ...body.memories] : body.memories;
    const record = {
      items, hasMore: body.truncated, nextCursor: body.next_cursor,
      loading: false, error: '', requestId,
    };
    state.directories.set(key, record);
    renderContextTree();
    renderDashboard();
    return record;
  } catch (error) {
    if (state.directories.get(key)?.requestId === requestId) {
      state.directories.set(key, {
        items: previous?.items || [], hasMore: previous?.hasMore || false,
        nextCursor: previous?.nextCursor || null, loading: false, error: error.message, requestId,
      });
      renderContextTree();
    }
    return state.directories.get(key);
  }
}
async function openSingleton(key) {
  const spec = state.singletonSpecs.get(key);
  if (!spec) return;
  const accountId = state.accountId;
  const requestId = ++state.detailRequestId;
  state.selectedDirectoryKey = key;
  let record = state.directories.get(key);
  if (!record) record = await loadDirectory(key, spec, null, false);
  if (accountId !== state.accountId || requestId !== state.detailRequestId) return;
  const memory = record?.items?.[0];
  if (memory) await openDetail(memory.uri, spec.userId, spec.agentId);
}

async function openDetail(uri, userId, agentId) {
  const requestId = ++state.detailRequestId;
  state.selectedUri = uri;
  renderContextTree();
  elements.detailPlaceholder.hidden = true;
  elements.detailView.hidden = false;
  $('#detail-type').textContent = '加载中';
  $('#detail-type').dataset.type = '';
  $('#detail-title').textContent = '读取节点';
  $('#detail-uri').textContent = uri;
  $('#detail-abstract').textContent = '读取中';
  $('#detail-overview').textContent = '读取中';
  $('#detail-overview').closest('section').hidden = false;
  $('#detail-content').textContent = '读取中';
  $('#detail-metadata').innerHTML = '<div class="metadata-empty">读取中</div>';
  try {
    const path = state.adminMode ? '/api/v1/memory-viewer/admin/detail' : '/api/v1/memories/detail';
    const memory = await fetchJson(apiUrl(path, {
      accountId: state.adminMode ? state.accountId : '', userId: state.adminMode ? userId : '', agentId, uri,
    }));
    if (requestId !== state.detailRequestId || uri !== state.selectedUri) return;
    $('#detail-type').textContent = typeLabel(memory.type);
    $('#detail-type').dataset.type = memory.type;
    $('#detail-title').textContent = memory.id || '未命名节点';
    $('#detail-uri').textContent = memory.context_uri || memory.uri;
    $('#detail-abstract').textContent = memory.abstract || '暂无摘要';
    $('#detail-overview').textContent = displayValue(memory.overview, '暂无概览');
    $('#detail-overview').closest('section').hidden = !String(memory.overview || '').trim();
    $('#detail-content').textContent = displayValue(memory.content, '暂无正文');
    renderMetadata(memory.metadata || {});
  } catch (error) {
    if (requestId === state.detailRequestId && uri === state.selectedUri) {
      $('#detail-title').textContent = '读取失败';
      $('#detail-content').textContent = error.message;
    }
  }
}
function clearDetail() {
  state.detailRequestId += 1;
  state.selectedUri = '';
  state.selectedDirectoryKey = '';
  elements.detailView.hidden = true;
  elements.detailPlaceholder.hidden = false;
}
function revealSelectedResource() {
  if (!state.selectedDirectoryKey || !state.selectedUri) return;
  const accountRoot = contextRoot();
  if (state.adminMode) state.expanded.add('ctx://');
  state.expanded.add(accountRoot);
  let current = accountRoot;
  state.selectedDirectoryKey.slice(accountRoot.length).split('/').filter(Boolean).forEach((segment) => {
    current = `${current}/${segment}`;
    state.expanded.add(current);
  });
  renderContextTree();
  requestAnimationFrame(() => {
    const selected = elements.tree.querySelector('.context-resource.active');
    if (!selected) return;
    selected.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'smooth' });
    selected.focus({ preventScroll: true });
  });
}
function renderMetadata(metadata) {
  renderMetadataInto($('#detail-metadata'), metadata);
}
function renderMetadataInto(container, metadata) {
  const entries = Object.entries(metadata).filter(([, value]) => value !== undefined && value !== null && value !== '');
  container.innerHTML = entries.length ? entries.map(([key, value]) => `
    <div class="metadata-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(displayValue(value, '—'))}</strong></div>`).join('') : '<div class="metadata-empty">暂无元数据</div>';
}

function renderDashboard() {
  const users = state.tree?.users || [];
  const agents = state.tree?.agents || users.flatMap((user) => user.agents || []);
  const inventory = state.inventories.get(state.accountId);
  $('#dashboard-users').textContent = users.length;
  $('#dashboard-agents').textContent = agents.length;
  $('#dashboard-memories').textContent = inventory?.memories ?? '—';
  $('#dashboard-skills').textContent = inventory?.skills ?? '—';
}
async function loadDashboardMetrics() {
  const requestId = ++state.metricsRequestId;
  const status = $('#metrics-status');
  status.hidden = true;
  $('#metrics-updated').textContent = '最近 14 天';
  $('#metrics-scope').textContent = state.accountId
    ? `${state.accountId} 的 LLM 与 Embedding Token 使用情况。`
    : '当前账户的 LLM 与 Embedding Token 使用情况。';
  $('#token-trend-chart').innerHTML = '<div class="metrics-empty">正在加载模型用量…</div>';
  setLoading(true);
  try {
    const metrics = await fetchJson(apiUrl('/api/v1/memory-viewer/metrics', {
      days: '14', accountId: state.adminMode ? state.accountId : '',
    }));
    if (requestId !== state.metricsRequestId) return;
    renderModelMetrics(metrics);
  } catch (error) {
    if (requestId !== state.metricsRequestId) return;
    status.textContent = `模型用量加载失败：${error.message}`;
    status.hidden = false;
    $('#token-trend-chart').innerHTML = '<div class="metrics-empty">暂无可用指标</div>';
  } finally {
    if (requestId === state.metricsRequestId) setLoading(false);
  }
}
function renderModelMetrics(metrics) {
  const today = metrics.today || {};
  const cumulative = metrics.cumulative || {};
  const missingUsageCalls = Number(cumulative.missing_usage_calls || 0);
  const status = $('#metrics-status');
  status.hidden = !missingUsageCalls;
  status.textContent = missingUsageCalls
    ? `${missingUsageCalls} 次成功调用未返回 Token 用量；已计入调用次数，Token 总量仅包含已报告部分。`
    : '';
  const llm = cumulative.llm || {};
  const embedding = cumulative.embedding || {};
  const todayLlm = Number(today.llm_input || 0) + Number(today.llm_output || 0);
  const embeddingCalls = Number(embedding.calls || 0);
  setMetricText('metrics-today-total', today.total);
  setMetricText('metrics-today-llm', todayLlm);
  setMetricText('metrics-today-embedding', today.embedding_input);
  setMetricText('metrics-llm-total', llm.total_tokens);
  setMetricText('metrics-llm-input', llm.input_tokens);
  setMetricText('metrics-llm-output', llm.output_tokens);
  setMetricText('metrics-llm-calls', llm.calls);
  setMetricText('metrics-embedding-total', embedding.total_tokens);
  setMetricText('metrics-embedding-calls', embeddingCalls);
  setMetricText('metrics-embedding-average', embeddingCalls ? Math.round(Number(embedding.total_tokens || 0) / embeddingCalls) : 0);
  $('#metrics-llm-model').textContent = metrics.models?.llm || '未配置模型';
  $('#metrics-embedding-model').textContent = metrics.models?.embedding || '未配置模型';
  $('#metrics-updated').textContent = `更新于 ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · 最近 14 天`;
  renderTokenTrend(metrics.series || []);
}

async function loadMonitoring() {
  const requestId = ++state.monitoringRequestId;
  const refresh = $('#monitoring-refresh');
  const status = $('#monitoring-status');
  refresh.disabled = true;
  refresh.classList.add('loading');
  status.hidden = true;
  try {
    const data = await fetchJson(apiUrl('/api/v1/memory-viewer/monitoring', {
      accountId: state.adminMode ? state.accountId : '',
    }));
    if (requestId !== state.monitoringRequestId) return;
    state.monitoringData = data;
    renderMonitoring();
  } catch (error) {
    if (requestId !== state.monitoringRequestId) return;
    status.textContent = `监控数据加载失败：${error.message}`;
    status.hidden = false;
    $('#monitoring-summary').innerHTML = '<span class="monitoring-summary-icon">!</span><div><strong>无法获取组件状态</strong><p>请检查服务连接后重试。</p></div>';
    $('#monitoring-tabs').innerHTML = '';
    elements.monitoringContent.innerHTML = '';
  } finally {
    if (requestId === state.monitoringRequestId) {
      refresh.disabled = false;
      refresh.classList.remove('loading');
    }
  }
}

function renderMonitoring() {
  const data = state.monitoringData || {};
  const components = data.components || [];
  const overall = MONITOR_STATUS[data.status] || MONITOR_STATUS.error;
  const summary = $('#monitoring-summary');
  summary.dataset.status = data.status || 'error';
  summary.innerHTML = `<span class="monitoring-summary-icon">${data.healthy ? '✓' : '!'}</span>
    <div><strong>${escapeHtml(overall.summary)}</strong><p>${Number(data.healthy_count || 0)} / ${Number(data.total_count || components.length)} 个组件正常 · ${escapeHtml(data.account_id || state.accountId)}${data.version ? ` · v${escapeHtml(data.version)}` : ''}</p></div>
    <span class="monitoring-health-badge" data-status="${escapeHtml(data.status || 'error')}">${escapeHtml(overall.label)}</span>`;
  $('#monitoring-updated').textContent = data.updated_at
    ? `更新于 ${new Date(data.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`
    : '';
  const tabs = [{ id: 'overview', name: '总览' }, ...components];
  if (!tabs.some((item) => item.id === state.activeMonitor)) state.activeMonitor = 'overview';
  $('#monitoring-tabs').innerHTML = tabs.map((item) => `<button type="button" role="tab" data-monitor="${escapeHtml(item.id)}" aria-selected="${item.id === state.activeMonitor}">
    <span>${MONITOR_ICONS[item.id] || '◇'}</span>${escapeHtml(item.name)}</button>`).join('');
  renderMonitoringContent();
}

function renderMonitoringContent() {
  const components = state.monitoringData?.components || [];
  if (state.activeMonitor === 'overview') {
    elements.monitoringContent.innerHTML = `<div class="monitoring-grid">${components.map((component) => `
      <button class="monitoring-card" type="button" data-monitor="${escapeHtml(component.id)}">
        <span class="monitoring-card-icon">${MONITOR_ICONS[component.id] || '◇'}</span>
        <span class="monitoring-card-copy"><strong>${escapeHtml(component.name)}</strong><small>${escapeHtml(component.description)}</small></span>
        <span class="monitoring-health-badge" data-status="${escapeHtml(component.status)}">${escapeHtml((MONITOR_STATUS[component.status] || MONITOR_STATUS.error).label)}</span>
      </button>`).join('')}</div>`;
    return;
  }
  const component = components.find((item) => item.id === state.activeMonitor);
  if (!component) { elements.monitoringContent.innerHTML = ''; return; }
  const rows = Object.entries(component.details || {}).filter(([, value]) => value !== '');
  elements.monitoringContent.innerHTML = `<article class="monitoring-detail">
    <header><span class="monitoring-card-icon">${MONITOR_ICONS[component.id] || '◇'}</span><div><h2>${escapeHtml(component.name)}</h2><p>${escapeHtml(component.description)}</p></div>
      <span class="monitoring-health-badge" data-status="${escapeHtml(component.status)}">${escapeHtml((MONITOR_STATUS[component.status] || MONITOR_STATUS.error).label)}</span></header>
    <dl>${rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(monitoringValue(value))}</dd></div>`).join('')}</dl>
    ${renderMonitoringTables(component.tables || [])}
  </article>`;
}

function renderMonitoringTables(tables) {
  return tables.map((table) => `<section class="monitoring-table-section">
    <h3>${escapeHtml(table.title || '运行数据')}</h3>
    <div class="monitoring-table-scroll"><table>
      <thead><tr>${(table.columns || []).map((column) => `<th>${escapeHtml(column)}</th>`).join('')}</tr></thead>
      <tbody>${table.rows?.length
        ? table.rows.map((row) => `<tr>${row.map((value) => `<td>${escapeHtml(monitoringValue(value))}</td>`).join('')}</tr>`).join('')
        : `<tr><td class="monitoring-table-empty" colspan="${Math.max(1, (table.columns || []).length)}">暂无运行数据</td></tr>`}
      </tbody>
    </table></div>
  </section>`).join('');
}

function monitoringValue(value) {
  if (value === true) return '已启用';
  if (value === false) return '未启用';
  if (typeof value === 'number') return value.toLocaleString();
  return String(value ?? '');
}
function setMetricText(id, value) {
  const number = Number(value || 0);
  $(`#${id}`).textContent = Number.isFinite(number) ? number.toLocaleString() : '0';
}
function renderTokenTrend(series) {
  const chart = $('#token-trend-chart');
  if (!series.length || series.every((item) => Number(item.total || 0) === 0)) {
    chart.innerHTML = '<div class="metrics-empty">最近 14 天暂无 Token 用量</div>';
    return;
  }
  const width = Math.min(1280, Math.max(360, chart.clientWidth || 1200));
  const height = 320;
  const padding = width < 640
    ? { top: 16, right: 14, bottom: 42, left: 54 }
    : { top: 16, right: 24, bottom: 42, left: 72 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const keys = [
    ['llm_input', 'llm-input', 'LLM 输入'],
    ['llm_output', 'llm-output', 'LLM 输出'],
    ['embedding_input', 'embedding', 'Embedding'],
  ];
  const maxValue = Math.max(1, ...series.flatMap((item) => keys.map(([key]) => Number(item[key] || 0))));
  const x = (index) => padding.left + (series.length === 1 ? plotWidth / 2 : index * plotWidth / (series.length - 1));
  const y = (value) => padding.top + plotHeight - Number(value || 0) * plotHeight / maxValue;
  const grid = Array.from({ length: 5 }, (_, index) => {
    const value = Math.round(maxValue * (4 - index) / 4);
    const yPos = padding.top + index * plotHeight / 4;
    return `<line class="token-grid-line" x1="${padding.left}" y1="${yPos}" x2="${width - padding.right}" y2="${yPos}"></line><text class="token-axis-label" x="${padding.left - 10}" y="${yPos + 3}" text-anchor="end">${value.toLocaleString()}</text>`;
  }).join('');
  const lines = keys.map(([key, className, label]) => {
    const points = series.map((item, index) => `${x(index)},${y(item[key])}`).join(' ');
    const dots = series.map((item, index) => `<circle class="token-series-point ${className}" cx="${x(index)}" cy="${y(item[key])}" r="4" data-date="${escapeHtml(String(item.date || ''))}" data-label="${label}" data-value="${Number(item[key] || 0).toLocaleString()}"></circle>`).join('');
    return `<polyline class="token-series-line ${className}" points="${points}"></polyline>${dots}`;
  }).join('');
  const labelIndexes = [...new Set([0, Math.floor((series.length - 1) / 2), series.length - 1])];
  const labels = labelIndexes.map((index) => `<text class="token-axis-label" x="${x(index)}" y="${height - 8}" text-anchor="middle">${escapeHtml(String(series[index].date || '').slice(5))}</text>`).join('');
  chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" aria-label="Token 使用趋势">${grid}${lines}${labels}</svg><div class="token-trend-tooltip" hidden></div>`;
  const tooltip = chart.querySelector('.token-trend-tooltip');
  const showTooltip = (event) => {
    const point = event.currentTarget;
    tooltip.textContent = `${point.dataset.date} · ${point.dataset.label}：${point.dataset.value} Token`;
    tooltip.hidden = false;
    const chartRect = chart.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const left = Math.min(chart.clientWidth - tooltipRect.width - 8, Math.max(8, event.clientX - chartRect.left + 12));
    const top = Math.max(8, event.clientY - chartRect.top - tooltipRect.height - 12);
    tooltip.style.transform = `translate(${left}px, ${top}px)`;
  };
  chart.querySelectorAll('.token-series-point').forEach((point) => {
    point.addEventListener('mouseenter', showTooltip);
    point.addEventListener('mousemove', showTooltip);
    point.addEventListener('mouseleave', () => { tooltip.hidden = true; });
  });
}
async function runRetrieval() {
  const query = $('#retrieval-input').value.trim();
  const target = state.searchTargets.get(elements.retrievalScope.value);
  if (!query || !target) return;
  const requestId = ++state.retrievalRequestId;
  state.lastSearchTarget = target;
  clearRetrievalDetail();
  $('#retrieval-submit').disabled = true;
  $('#retrieval-result-count').textContent = '';
  elements.retrievalResults.innerHTML = '<div class="retrieval-empty loading"><span>⌕</span><strong>正在检索</strong><p>匹配语义相关记忆…</p></div>';
  const type = $('#retrieval-type').value;
  const threshold = $('#retrieval-threshold').value;
  const payload = {
    query,
    top_k: Number($('#retrieval-count').value),
    accountId: target.accountId,
    userId: target.userId,
    agentId: target.agentId,
    types: type ? [type] : undefined,
    score_threshold: threshold === '' ? undefined : Number(threshold),
    sessionId: $('#retrieval-session').value.trim() || undefined,
  };
  try {
    const body = await fetchJson('/api/v1/memory-viewer/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (requestId !== state.retrievalRequestId) return;
    renderRetrievalResults(body.hits || []);
  } catch (error) {
    if (requestId === state.retrievalRequestId) {
      elements.retrievalResults.innerHTML = `<div class="retrieval-empty error"><span>!</span><strong>检索失败</strong><p>${escapeHtml(error.message)}</p></div>`;
    }
  } finally {
    if (requestId === state.retrievalRequestId) $('#retrieval-submit').disabled = false;
  }
}
function renderRetrievalResults(hits) {
  state.retrievalHits = hits;
  $('#retrieval-result-count').textContent = hits.length ? `Top ${hits.length}` : '';
  if (!hits.length) {
    elements.retrievalResults.innerHTML = '<div class="retrieval-empty"><span>⌕</span><strong>没有找到匹配内容</strong><p>试试其他关键词或范围。</p></div>';
    return;
  }
  elements.retrievalResults.innerHTML = hits.map((hit, index) => {
    const score = Number.isFinite(hit.score) ? hit.score.toFixed(3) : '';
    const summary = hit.abstract || hit.overview || hit.content_excerpt || '暂无摘要';
    return `<button class="retrieval-result" type="button" data-search-index="${index}">
      <span class="type-chip" data-type="${escapeHtml(hit.type)}">${escapeHtml(typeLabel(hit.type))}</span>
      <span class="retrieval-result-body"><strong>${escapeHtml(hit.id || '未命名节点')}</strong><small>${escapeHtml(hit.uri)}</small><span>${escapeHtml(summary)}</span></span>
      ${score ? `<span class="retrieval-score">${score}</span>` : ''}
    </button>`;
  }).join('');
}
function renderRetrievalDetail(hit) {
  if (!hit) return;
  $('#retrieval-detail-placeholder').hidden = true;
  $('#retrieval-detail-view').hidden = false;
  $('#retrieval-detail-type').textContent = typeLabel(hit.type);
  $('#retrieval-detail-type').dataset.type = hit.type || '';
  $('#retrieval-detail-title').textContent = hit.id || '未命名节点';
  $('#retrieval-detail-uri').textContent = hit.uri || '';
  $('#retrieval-detail-abstract').textContent = hit.abstract || '暂无摘要';
  $('#retrieval-detail-overview').textContent = displayValue(hit.overview, '暂无概览');
  $('#retrieval-detail-overview').closest('section').hidden = !String(hit.overview || '').trim();
  $('#retrieval-detail-content').textContent = displayValue(hit.content_excerpt, '暂无正文');
  renderMetadataInto($('#retrieval-detail-metadata'), {
    score: hit.score,
    when: hit.when,
    who: hit.who,
    where: hit.where,
    unit_key: hit.unit_key,
    grounding: hit.grounding,
  });
  elements.retrievalDetail.scrollIntoView({ block: 'nearest' });
}
function clearRetrievalDetail() {
  state.retrievalHits = [];
  $('#retrieval-detail-view').hidden = true;
  $('#retrieval-detail-placeholder').hidden = false;
}
function resetRetrieval() {
  state.retrievalRequestId += 1;
  state.lastSearchTarget = null;
  clearRetrievalDetail();
  $('#retrieval-submit').disabled = false;
  $('#retrieval-result-count').textContent = '';
  elements.retrievalResults.innerHTML = '<div class="retrieval-empty"><span>⌕</span><strong>输入内容后开始检索</strong><p>结果将按语义相关度排序。</p></div>';
}
function switchView(view) {
  state.view = view;
  document.querySelectorAll('[data-view]').forEach((item) => {
    item.classList.toggle('active', item.dataset.view === view);
    if (item.matches('.sidebar-link')) item.toggleAttribute('aria-current', item.dataset.view === view);
  });
  const dashboard = view === 'dashboard';
  const retrieval = view === 'retrieval';
  const monitoring = view === 'monitoring';
  const tasks = view === 'tasks';
  const scenes = view === 'scenes';
  elements.dashboardView.hidden = !dashboard;
  elements.memoryView.hidden = dashboard || retrieval || monitoring || tasks || scenes;
  elements.retrievalView.hidden = !retrieval;
  elements.monitoringView.hidden = !monitoring;
  elements.tasksView.hidden = !tasks;
  elements.scenesView.hidden = !scenes;
  elements.pageTitle.textContent = dashboard ? '控制台' : retrieval ? '检索' : monitoring ? '监控' : tasks ? '任务中心' : scenes ? '场景记忆' : view === 'skill' ? 'Agent Skill' : 'ContextFS';
  const skill = view === 'skill';
  $('#tree-label').textContent = skill ? 'Agent Skill' : 'ContextFS';
  elements.tree.setAttribute('aria-label', skill ? 'Agent Skill 目录树' : 'ContextFS 目录树');
  elements.search.placeholder = skill ? '搜索已加载 Agent Skill、路径或摘要' : '搜索已加载节点、路径或摘要';
  $('#detail-placeholder-title').textContent = skill ? '选择一个 Agent Skill' : '选择一个 ContextFS 节点';
  $('#detail-placeholder-copy').textContent = skill
    ? '展开 Agent 目录后查看已生成或导入的 Skill。'
    : '展开目录后查看摘要、概览、正文与元数据。';
  if (dashboard) { renderDashboard(); loadDashboardMetrics(); return; }
  if (retrieval) { $('#retrieval-input').focus(); return; }
  if (monitoring) { loadMonitoring(); return; }
  if (tasks) { loadTasks(); return; }
  if (scenes) { loadScenes(state.sceneOffset); return; }
  if (view === 'skill') {
    const root = contextRoot();
    state.expanded.add(root);
    state.expanded.add(`${root}/agents`);
    (state.tree?.agents || []).forEach((agent) => {
      const agentRoot = `${root}/agents/${encodeURIComponent(agent.agent_id)}`;
      state.expanded.add(agentRoot);
      state.expanded.add(`${agentRoot}/skills`);
    });
  }
  renderContextTree();
}

function showTreeError(message) {
  elements.treeRoot.textContent = '加载失败';
  elements.treeCount.textContent = '';
  elements.tree.innerHTML = `<div class="tree-error root-error">${escapeHtml(message)}</div>`;
  if (state.view === 'scenes') elements.scenesList.replaceChildren(sceneText('p', 'scene-status error', message));
}
function setLoading(loading) { elements.refresh.classList.toggle('loading', loading); }
function setConnection(online) {
  elements.connection.className = `connection sidebar-connection ${online ? 'online' : 'offline'}`;
  elements.connection.innerHTML = `<span></span>${online ? '已连接' : '连接失败'}`;
}
function displayValue(value, fallback) {
  if (value === null || value === undefined || value === '') return fallback;
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}
function typeLabel(type) { return TYPE_LABELS[type] || type; }
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

elements.account.addEventListener('change', async () => {
  try { await selectAccount(elements.account.value); }
  catch (error) { showTreeError(error.message); }
});
document.querySelector('.app-sidebar').addEventListener('click', (event) => {
  const view = event.target.closest('[data-view]');
  if (view) switchView(view.dataset.view);
});
elements.dashboardView.addEventListener('click', (event) => {
  const view = event.target.closest('[data-view]');
  if (view) switchView(view.dataset.view);
});
elements.tree.addEventListener('click', async (event) => {
  const account = event.target.closest('[data-account-switch]');
  if (account) {
    try { await selectAccount(account.dataset.accountSwitch, true); }
    catch (error) { showTreeError(error.message); }
    return;
  }
  const more = event.target.closest('[data-load-more]');
  if (more) {
    const key = more.dataset.loadMore;
    const record = state.directories.get(key);
    const spec = state.directorySpecs.get(key);
    if (record?.nextCursor && spec) await loadDirectory(key, spec, record.nextCursor, true);
    return;
  }
  const resource = event.target.closest('[data-resource-uri]');
  if (resource) {
    if (resource.dataset.account !== state.accountId) await selectAccount(resource.dataset.account);
    if (resource.dataset.account !== state.accountId) return;
    state.selectedDirectoryKey = resource.closest('[data-node-key]')?.dataset.nodeKey || '';
    await openDetail(resource.dataset.resourceUri, resource.dataset.user, resource.dataset.agent);
    return;
  }
  const singleton = event.target.closest('[data-singleton-key]');
  if (singleton) {
    const accountId = state.singletonSpecs.get(singleton.dataset.singletonKey)?.accountId;
    if (accountId && accountId !== state.accountId) await selectAccount(accountId);
    if (accountId !== state.accountId) return;
    await openSingleton(singleton.dataset.singletonKey);
    return;
  }
  const folder = event.target.closest('[data-folder-key]');
  if (folder) {
    const accountId = state.directorySpecs.get(folder.dataset.folderKey)?.accountId;
    if (accountId && accountId !== state.accountId) await selectAccount(accountId);
    if (accountId && accountId !== state.accountId) return;
    await toggleDirectory(folder.dataset.folderKey);
  }
});
elements.search.addEventListener('input', () => {
  state.searchQuery = elements.search.value.trim().toLocaleLowerCase();
  renderContextTree();
});
$('#retrieval-form').addEventListener('submit', (event) => {
  event.preventDefault();
  runRetrieval();
});
$('#retrieval-filters-toggle').addEventListener('click', () => {
  const filters = $('#retrieval-filters');
  filters.hidden = !filters.hidden;
  $('#retrieval-filters-toggle').setAttribute('aria-expanded', String(!filters.hidden));
});
elements.retrievalResults.addEventListener('click', (event) => {
  const result = event.target.closest('[data-search-index]');
  if (!result) return;
  elements.retrievalResults.querySelectorAll('.retrieval-result').forEach((item) => item.classList.remove('active'));
  result.classList.add('active');
  renderRetrievalDetail(state.retrievalHits[Number(result.dataset.searchIndex)]);
});
$('#monitoring-tabs').addEventListener('click', (event) => {
  const target = event.target.closest('[data-monitor]');
  if (!target) return;
  state.activeMonitor = target.dataset.monitor;
  renderMonitoring();
});
elements.monitoringContent.addEventListener('click', (event) => {
  const target = event.target.closest('[data-monitor]');
  if (!target) return;
  state.activeMonitor = target.dataset.monitor;
  renderMonitoring();
});
$('#monitoring-refresh').addEventListener('click', loadMonitoring);
elements.refresh.addEventListener('click', () => {
  if (state.view === 'dashboard') loadDashboardMetrics();
  else if (state.view === 'monitoring') loadMonitoring();
  else if (state.view === 'tasks') loadTasks();
  else if (state.view === 'scenes') loadScenes(state.sceneOffset);
  else loadTree();
});
$('#scenes-refresh').addEventListener('click', () => loadScenes(state.sceneOffset));
elements.scenesScope.addEventListener('change', () => {
  state.selectedSceneId = '';
  loadScenes();
});
$('#scenes-previous').addEventListener('click', () => loadScenes(Math.max(0, state.sceneOffset - SCENE_PAGE_SIZE)));
$('#scenes-next').addEventListener('click', () => {
  if (state.sceneNextOffset !== null) loadScenes(state.sceneNextOffset);
});
elements.scenesList.addEventListener('click', event => {
  const card = event.target.closest('[data-scene-id]');
  const scene = state.scenes.find(item => item.scene_id === card?.dataset.sceneId);
  if (scene) selectScene(scene);
});
$('#scene-detail-members').addEventListener('click', event => {
  const member = event.target.closest('[data-member-uri]');
  if (!member) return;
  state.selectedDirectoryKey = '';
  switchView('memory');
  openDetail(member.dataset.memberUri, member.dataset.user, member.dataset.agent);
});
$('#tasks-refresh').addEventListener('click', loadTasks);
['#tasks-type', '#tasks-status'].forEach((selector) => $(selector).addEventListener('change', () => {
  state.tasksPage = 1;
  renderTasks();
}));
$('#tasks-previous').addEventListener('click', () => { state.tasksPage -= 1; renderTasks(); });
$('#tasks-next').addEventListener('click', () => { state.tasksPage += 1; renderTasks(); });
function openSelectedTask(target) {
  const row = target.closest('[data-task-id]');
  if (!row) return;
  const task = state.tasks.find((item) => item.task_id === row.dataset.taskId);
  if (task) openTaskDetail(task);
}
elements.tasksPanel.addEventListener('click', (event) => openSelectedTask(event.target));
elements.tasksPanel.addEventListener('keydown', (event) => {
  if (!['Enter', ' '].includes(event.key)) return;
  event.preventDefault();
  openSelectedTask(event.target);
});
$('#task-detail-close').addEventListener('click', () => $('#task-detail-dialog').close());
$('#task-retry').addEventListener('click', retrySelectedTask);
$('#task-detail-dialog').addEventListener('click', (event) => {
  if (event.target === $('#task-detail-dialog')) $('#task-detail-dialog').close();
});
elements.revealCurrent.addEventListener('click', revealSelectedResource);
$('#theme-button').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('ogmemory-viewer-theme', next);
});
$('#sidebar-toggle').addEventListener('click', () => {
  elements.shell.classList.add('sidebar-collapsed');
  localStorage.setItem('ogmemory-viewer-sidebar', 'collapsed');
});
$('.workspace-icon').addEventListener('click', () => {
  if (!elements.shell.classList.contains('sidebar-collapsed')) return;
  elements.shell.classList.remove('sidebar-collapsed');
  localStorage.setItem('ogmemory-viewer-sidebar', 'expanded');
});
function openSettings() { $('#api-key-input').value = state.apiKey; elements.settings.showModal(); }
$('#settings-button').addEventListener('click', openSettings);
$('#settings-close').addEventListener('click', () => elements.settings.close());
$('#api-key-input').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' || event.isComposing) return;
  event.preventDefault();
  $('#settings-form').requestSubmit();
});
$('#settings-form').addEventListener('submit', (event) => {
  event.preventDefault();
  setApiKey($('#api-key-input').value);
  apiKeyChannel?.postMessage({ type: 'update', apiKey: state.apiKey });
  elements.settings.close();
  loadTree();
});
apiKeyChannel?.addEventListener('message', (event) => {
  const message = event.data;
  if (!message || typeof message !== 'object') return;
  if (message.type === 'request') {
    if (state.apiKey) apiKeyChannel.postMessage({ type: 'sync', apiKey: state.apiKey });
    return;
  }
  if (!['sync', 'update'].includes(message.type) || typeof message.apiKey !== 'string') return;
  const apiKey = message.apiKey.trim();
  const initialLoad = initialTreeLoadPending;
  initialTreeLoadPending = false;
  if (apiKey === state.apiKey) {
    if (initialLoad) loadTree();
    return;
  }
  setApiKey(apiKey);
  loadTree();
});
elements.settings.addEventListener('click', (event) => {
  if (event.target === elements.settings) elements.settings.close();
});

document.documentElement.dataset.theme = localStorage.getItem('ogmemory-viewer-theme') || 'light';
elements.shell.classList.toggle('sidebar-collapsed', localStorage.getItem('ogmemory-viewer-sidebar') === 'collapsed');
if (location.hash === '#scenes') switchView('scenes');
if (initialTreeLoadPending) {
  apiKeyChannel.postMessage({ type: 'request' });
  setTimeout(loadTreeAfterApiKeySync, 100);
} else loadTree();
setInterval(() => {
  if (state.view === 'monitoring' && !$('#monitoring-refresh').disabled) loadMonitoring();
}, 15_000);
setInterval(() => {
  if (state.view === 'tasks' && !document.hidden) loadTasks({ silent: true });
}, 5_000);
document.addEventListener('visibilitychange', () => {
  if (state.view === 'tasks' && !document.hidden) loadTasks({ silent: true });
});
