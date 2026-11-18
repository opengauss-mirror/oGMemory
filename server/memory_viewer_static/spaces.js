/* Shared-space configuration and system-maintained memory inspection. */
window.sharedSpaces = (() => {
  const root = $('#spaces-view');
  const body = $('#space-body');
  const pageSize = 20;
  let fieldSequence = 0;
  const model = { spaces: [], space: null, scope: {}, tab: 'content', page: {}, pagers: {}, registry: new Map(), members: [], drafts: new Map(), rule: null, epoch: 0, listSequence: 0, principalSequence: 0, search: { query: '', mode: 'hybrid' }, searchActive: false };
  const tabs = { content: '团队记忆', members: '成员管理', rules: '规则' };
  const states = { ACTIVE: '有效', PAUSED: '暂停', ARCHIVED: '已归档', EXPIRED: '已到期', SUPERSEDED: '已替代', PURGED: '已清理', RECOMPUTING: '重算中', READY: '可检索' };
  const cap = (name) => Boolean(model.space?.capabilities?.[name]);
  const endpoint = (suffix = '') => `/management/spaces/${encodeURIComponent(model.space.space_id)}${suffix}`;
  const label = (value) => escapeHtml(states[value] || value || '—');
  const button = (action, text, value = '') => `<button type="button" class="secondary-button" data-space-action="${action}" data-value="${escapeHtml(value)}">${text}</button>`;
  const deleteButton = (kind, text, value = '') => `<button type="button" class="secondary-button space-danger-button" data-space-action="delete-${kind}" data-value="${escapeHtml(value)}">${text}</button>`;
  const field = (title, control) => {
    const id = `space-field-${++fieldSequence}`;
    return `<label><span id="${id}-label">${escapeHtml(title)}</span>${control(`id="${id}" aria-labelledby="${id}-label" data-field-label="${escapeHtml(title)}"`)}</label>`;
  };
  const input = (name, title, value = '', attrs = '') => field(title, (association) => `<input name="${name}" ${association} value="${escapeHtml(value)}" ${attrs}>`);
  const select = (name, title, values, value) => field(title, (association) => `<select name="${name}" ${association}>${Object.entries(values).map(([id, text]) => `<option value="${escapeHtml(id)}" ${id === value ? 'selected' : ''}>${escapeHtml(text)}</option>`).join('')}</select>`);
  const textArea = (name, title, value, attrs = '') => field(title, (association) => `<textarea class="resize-none" name="${name}" ${association} ${attrs}>${escapeHtml(value)}</textarea>`);
  const submit = (text) => `<button class="primary-button" type="submit">${text}</button>`;
  const split = (value) => [...new Set(String(value || '').split(/[,，\s]+/).filter(Boolean))];
  const currentUser = () => model.scope.user_id || model.scope.viewer_user_id || model.space?.viewer_user_id;

  function icon(name) {
    const shapes = {
      team: '<circle cx="9" cy="8" r="3"/><path d="M3 20v-2a6 6 0 0 1 12 0v2M16 5a3 3 0 0 1 0 6M17 14a5 5 0 0 1 4 5v1"/>',
      document: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6M8 13h8M8 17h5"/>',
      rule: '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="9" cy="18" r="2"/>',
      agent: '<rect x="4" y="6" width="16" height="14" rx="4"/><path d="M12 3v3M8 15h8M8 10h.01M16 10h.01"/>',
      clear: '<path d="m6 6 12 12M18 6 6 18"/>',
    };
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${shapes[name]}</svg>`;
  }

  root.querySelector('[data-space-icon="team"]').innerHTML = icon('team');

  function memberIdentity(principal, name, detail = '') {
    const user = principal.startsWith('user:');
    const display = name || principal.slice(principal.indexOf(':') + 1);
    return `<span class="space-member-identity"><span class="space-member-avatar ${user ? '' : 'space-agent-avatar'}" aria-hidden="true">${user ? escapeHtml([...display][0]?.toLocaleUpperCase('zh-CN') || '?') : icon('agent')}</span><span class="space-identity-copy"><strong>${escapeHtml(display)}</strong><small>${escapeHtml([user ? '用户' : 'Agent', detail].filter(Boolean).join(' · '))}</small></span></span>`;
  }

  function searchField(name, title, value, maximum, accessibleName, className = '') {
    const id = `space-search-${name}`;
    return `<div class="space-search-field ${className}"><label for="${id}">${accessibleName}</label><span class="space-input-wrap"><input id="${id}" name="${name}" type="search" maxlength="${maximum}" data-field-label="${title}" value="${escapeHtml(value)}" placeholder="${title}"><button type="button" class="space-clear-search" data-space-action="clear-search" aria-label="清除${title}" ${value ? '' : 'hidden'}>${icon('clear')}</button></span></div>`;
  }

  let formSequence = 0;
  const composing = new WeakSet();

  function validateField(field) {
    if (!field.willValidate) return true;
    const title = field.dataset.fieldLabel || field.getAttribute('aria-label') || '此项';
    let error = '';
    if (field.validity.valueMissing || (field.required && !field.value.trim())) error = `请填写${title}。`;
    else if (field.validity.tooLong) error = `最多输入 ${field.maxLength} 个字符。`;
    else if (field.validity.patternMismatch) error = '以字母或数字开头，可使用字母、数字、下划线、点和短横线。';
    else if (!field.validity.valid) error = `请检查${title}的格式。`;
    if (!field.id) field.id = `space-field-${++fieldSequence}`;
    const errorId = `${field.id}-error`;
    let feedback = document.getElementById(errorId);
    if (error && !feedback) {
      feedback = document.createElement('span');
      feedback.id = errorId; feedback.className = 'space-field-error';
      feedback.setAttribute('aria-live', 'polite');
      (field.closest('label') || field.closest('.space-search-field')).append(feedback);
    }
    if (feedback) { feedback.textContent = error; feedback.hidden = !error; }
    const descriptions = (field.getAttribute('aria-describedby') || '').split(/\s+/).filter((id) => id && id !== errorId);
    if (error) { field.setAttribute('aria-invalid', 'true'); descriptions.push(errorId); }
    else field.removeAttribute('aria-invalid');
    if (descriptions.length) field.setAttribute('aria-describedby', descriptions.join(' '));
    else field.removeAttribute('aria-describedby');
    return !error;
  }

  function validateForm(form) {
    const invalid = [...form.elements].filter((field) => !validateField(field));
    invalid[0]?.focus();
    return invalid.length === 0;
  }

  function formBusy(form, busy) {
    if (busy) form.dataset.busy = String(++formSequence);
    else delete form.dataset.busy;
    form.setAttribute('aria-busy', String(busy));
    const control = form.querySelector('[type="submit"]');
    if (control) { control.disabled = busy; control.setAttribute('aria-busy', String(busy)); }
    return form.dataset.busy;
  }

  function pager(name, reset = false) {
    if (reset || !model.pagers[name]) model.pagers[name] = { cursors: [''], index: 0, next: null, total: null, busy: false };
    return model.pagers[name];
  }

  function receivePage(name, index, page) {
    const state = pager(name);
    Object.assign(state, { index, next: page.next_cursor || null, total: page.total ?? null, busy: false });
    state.cursors.length = index + 1;
    if (state.next) state.cursors.push(state.next);
  }

  function pagination(name) {
    const state = pager(name);
    const total = state.total === null ? '' : ` · 共 ${state.total} 条`;
    const control = (direction, text, disabled) => `<button type="button" class="secondary-button" data-space-action="page" data-value="${name}" data-direction="${direction}" ${disabled || state.busy ? 'disabled' : ''}>${text}</button>`;
    return `<nav class="space-pagination" ${!state.busy && state.index === 0 && !state.next ? 'hidden' : ''} aria-label="${{ space: '团队空间', content: '团队记忆', rules: '规则', members: '成员', sources: '来源', principals: '注册成员', search: '检索结果' }[name]}分页">${control('prev', '上一页', state.index === 0)}<span data-space-page="${name}">第 ${state.index + 1} 页${total}</span>${control('next', '下一页', !state.next)}</nav>`;
  }

  function localPage(name, rows) {
    const state = pager(name);
    state.index = Math.min(state.index, Math.max(0, Math.ceil(rows.length / pageSize) - 1));
    state.total = rows.length;
    state.next = (state.index + 1) * pageSize < rows.length;
    return rows.slice(state.index * pageSize, (state.index + 1) * pageSize);
  }

  function rememberMembers() {
    const previous = model.drafts.get(model.space.space_id);
    if (memberSnapshot(model.members) === memberSnapshot(model.space.members || [])) model.drafts.delete(model.space.space_id);
    else model.drafts.set(model.space.space_id, { members: model.members, revision: previous?.revision ?? model.space.authz_rev });
  }

  let messageTimer;
  function message(text, error = false) {
    clearTimeout(messageTimer);
    const target = $('#space-message');
    target.textContent = text;
    target.hidden = !text;
    target.classList.toggle('error', error);
    if (text && !error) messageTimer = setTimeout(() => { target.hidden = true; }, 6000);
  }

  function failure(error) {
    const text = {
      SPACE_NAME_EXISTS: '当前租户已有同名团队空间，请使用其他名称。',
      INVALID_CURSOR: '列表或权限已更新，请刷新列表或重新检索。',
      REVISION_MISMATCH: '配置已更新，请刷新后重试。成员编辑可先恢复已保存内容。',
      ITEM_CHANGED: '记忆已更新，请打开最新记忆重新设置权限。',
      REGISTERED_PRINCIPAL_REQUIRED: '请选择当前租户已注册的有效成员。',
      SPACE_ACCESS_DENIED: '团队成员资格已变更，请选择可访问的团队空间。',
      SPACE_NOT_FOUND: '团队空间已更新或删除，请重新选择。',
    }[error.code] || error.message;
    message(`${text}${error.requestId ? ` · 请求 ${error.requestId}` : ''}`, true);
  }

  async function read(path, values = {}) {
    return fetchJson(apiUrl(path, values), { signal: AbortSignal.timeout(10000) });
  }

  let spaceSession = new AbortController();

  async function mutate(path, method, payload, revision, create = false) {
    const session = spaceSession.signal;
    const headers = { ...requestHeaders(), 'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID() };
    if (revision !== undefined) headers['If-Match'] = `"${revision}"`;
    if (create) headers['If-None-Match'] = '*';
    const body = JSON.stringify(payload);
    for (let attempt = 0; attempt < 3; attempt += 1) {
      session.throwIfAborted();
      try {
        const result = await fetchJson(path, { method, headers, body, signal: AbortSignal.any([session, AbortSignal.timeout(10000)]) });
        session.throwIfAborted();
        return result;
      } catch (error) {
        session.throwIfAborted();
        const retryable = error.status === 429 || error.status === 503 || error instanceof TypeError || error.name === 'TimeoutError';
        if (!retryable || attempt === 2) throw error;
        await new Promise((resolve) => setTimeout(resolve, Math.min(error.retryAfter || 1, 5) * 1000));
      }
    }
  }

  const editSnapshots = new WeakMap();
  const memberSnapshot = (members) => JSON.stringify(members.map(({ principal, role }) => ({ principal, role })));
  const savingSnapshots = new WeakMap();
  const formSnapshot = (form) => form.dataset.spaceForm === 'members'
    ? memberSnapshot(model.members)
    : savingSnapshots.get(form) ?? JSON.stringify([...new FormData(form)]);

  function captureEdits() {
    for (const form of root.querySelectorAll('[data-space-form]')) {
      if (['search', 'principals'].includes(form.dataset.spaceForm) || editSnapshots.has(form)) continue;
      editSnapshots.set(form, form.dataset.spaceForm === 'members' ? memberSnapshot(model.space.members || []) : formSnapshot(form));
    }
  }

  function syncReaders(form) {
    const restricted = form.elements.acl_mode.value === 'restricted';
    form.querySelector('.space-acl-readers').hidden = !restricted;
    form.querySelector('.space-acl-readers').disabled = !restricted;
    form.querySelector('.space-acl-inherited').hidden = restricted;
  }

  function changedEdits({ except, preserveMembers = false, currentOnly = false } = {}) {
    const forms = [...root.querySelectorAll('[data-space-form]')].filter((form) => form !== except && editSnapshots.has(form)
      && !(preserveMembers && form.dataset.spaceForm === 'members') && editSnapshots.get(form) !== formSnapshot(form));
    const otherDrafts = !preserveMembers && !currentOnly && model.drafts.size > 0;
    return { forms, otherDrafts };
  }

  window.addEventListener('beforeunload', (event) => {
    const changes = changedEdits();
    if (!root.hidden && (changes.forms.length || changes.otherDrafts)) { event.preventDefault(); event.returnValue = ''; }
  });

  function discardEdits(forms, { preserveMembers = false, currentOnly = false } = {}) {
    if (!preserveMembers && !currentOnly) model.drafts.clear();
    for (const form of forms) {
      const kind = form.dataset.spaceForm;
      if (kind === 'members') {
        model.drafts.delete(model.space.space_id);
        model.members = (model.space.members || []).map((member) => ({ ...member }));
        renderMemberRows(); renderPrincipals();
      } else if (kind === 'rule') showRule(model.rule);
      else if (kind === 'create') showCreate();
      else if (kind === 'source') showSource(model.source?.binding_id);
      else { form.reset(); if (kind === 'acl') syncReaders(form); }
    }
  }

  async function guardNavigation(action, trigger, options = {}) {
    const { forms, otherDrafts } = changedEdits(options);
    const { preserveMembers = false, currentOnly = false } = options;
    if (!forms.length && !otherDrafts) { await action(); return true; }
    const names = { create: '团队空间', members: '成员', source: '来源贡献', settings: '团队配置', rule: '规则', acl: '读取权限' };
    const changed = [...new Set([...forms.map((form) => names[form.dataset.spaceForm]), ...(otherDrafts ? ['成员'] : [])])];
    return confirmAction({ title: '放弃未保存的更改？', name: changed.join('、'),
      impact: '这些更改尚未保存。放弃后将恢复已保存内容。', cancelText: '继续编辑', confirmText: '放弃更改' }, trigger, async () => {
      discardEdits(forms, { preserveMembers, currentOnly });
      await action();
    });
  }

  function confirmAction(details, trigger, action) {
    const session = spaceSession.signal;
    let completed = false;
    let resolve;
    const result = new Promise((done) => { resolve = done; });
    const confirmText = details.confirmText || '确认删除';
    const dialog = document.createElement('dialog');
    dialog.className = 'space-delete-dialog';
    dialog.setAttribute('aria-labelledby', 'space-delete-title');
    dialog.setAttribute('aria-describedby', 'space-delete-impact');
    dialog.innerHTML = `<form novalidate><h3 id="space-delete-title">${escapeHtml(details.title)}</h3><p class="space-delete-name">${escapeHtml(details.name)}</p><p id="space-delete-impact">${escapeHtml(details.impact)}</p><p class="space-delete-error" role="alert" hidden></p><div class="space-delete-actions"><button type="button" class="secondary-button" data-space-cancel-delete autofocus>${details.cancelText || '取消'}</button><button type="submit" class="primary-button space-danger-button" data-space-confirm-delete>${confirmText}</button></div></form>`;
    const form = dialog.querySelector('form');
    const controls = [...form.querySelectorAll('button')];
    const errorMessage = dialog.querySelector('[role="alert"]');
    const close = () => dialog.close();
    controls[0].addEventListener('click', close);
    session.addEventListener('abort', close, { once: true });
    dialog.addEventListener('keydown', (event) => {
      if (event.key !== 'Tab') return;
      const available = controls.filter((control) => !control.disabled);
      const first = available[0], last = available[available.length - 1];
      if (!first) { event.preventDefault(); return; }
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    });
    dialog.addEventListener('cancel', (event) => { if (form.dataset.busy) event.preventDefault(); });
    dialog.addEventListener('close', () => {
      session.removeEventListener('abort', close);
      dialog.remove();
      if (!session.aborted) {
        const focusTarget = trigger?.isConnected ? trigger : root.querySelector(details.focusAfter || '#space-tabs button');
        if (focusTarget) { focusTarget.disabled = false; focusTarget.focus(); }
      }
      resolve(completed);
    }, { once: true });
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (session.aborted || !dialog.open || form.dataset.busy) return;
      formBusy(form, true);
      controls.forEach((control) => { control.disabled = true; });
      errorMessage.hidden = true;
      try {
        await action();
        completed = true;
        dialog.close();
      } catch (error) {
        if (session.aborted) return;
        failure(error);
        errorMessage.textContent = $('#space-message').textContent;
        errorMessage.hidden = false;
      } finally {
        formBusy(form, false);
        controls.forEach((control) => { control.disabled = false; });
      }
    });
    root.append(dialog);
    dialog.showModal();
    return result;
  }

  function showDelete(kind, id, trigger) {
    const space = model.space;
    const resource = kind === 'rule' ? model.rule : kind === 'source' ? space.sources.find((source) => source.binding_id === id) : space;
    const details = {
      space: { title: '删除团队空间', name: space.name, impact: '团队成员的访问将终止，团队记忆由系统归档清理。', revision: space.authz_rev, path: endpoint() },
      rule: { title: '删除规则', name: resource.name, impact: '停止生成，已有团队记忆由系统归档清理。', revision: resource.rule_rev, path: endpoint(`/rules/${encodeURIComponent(id)}`) },
      source: { title: '删除来源', name: resource.principal, impact: '撤销来源贡献，系统重算相关团队记忆。', revision: resource.grant_rev, path: endpoint(`/sources/${encodeURIComponent(id)}`) },
    }[kind];
    const changes = kind === 'source' ? changedEdits({
      except: model.source?.binding_id === id ? body.querySelector('[data-space-form="source"]') : null,
      currentOnly: true,
    }) : { forms: [], otherDrafts: kind === 'space' && model.drafts.size > 0 };
    if (changes.forms.length || changes.otherDrafts) details.impact += ' 同时放弃其他未保存的更改。';
    confirmAction(details, trigger, async () => {
      await mutate(details.path, 'DELETE', {}, details.revision);
      if (kind === 'source') discardEdits(changes.forms, { currentOnly: true });
      if (kind === 'space') { reset(); await load(); }
      else { await open(space.space_id); pager('space', true); await loadSpaces(0); }
      message(`${details.title.replace('删除', '')}已删除。`);
    });
  }

  function reset() {
    spaceSession.abort();
    spaceSession = new AbortController();
    model.epoch += 1;
    model.listSequence += 1;
    Object.assign(model, { spaces: [], space: null, scope: {}, page: {}, pagers: {}, members: [], rule: null, search: { query: '', mode: 'hybrid' }, searchActive: false });
    model.registry.clear();
    model.drafts.clear();
    $('#space-list').replaceChildren();
    $('#space-list-pagination').replaceChildren();
    $('#space-tabs').replaceChildren();
    $('#space-header').innerHTML = '<h2>团队空间</h2>';
    body.innerHTML = '<p class="space-empty">暂无团队空间</p>';
    message('');
  }

  async function load() {
    model.epoch += 1;
    pager('space', true);
    return loadSpaces(0, true);
  }

  async function loadSpaces(index, selectSpace = false) {
    const sequence = ++model.listSequence;
    const epoch = model.epoch;
    const state = pager('space');
    state.busy = true;
    $('#space-list-pagination').innerHTML = pagination('space');
    try {
      const page = await read('/management/spaces', { limit: pageSize, cursor: state.cursors[index] });
      if (sequence !== model.listSequence) return;
      model.spaces = page.items || [];
      receivePage('space', index, page);
      setConnection(true);
      renderSpaces();
      if (selectSpace && epoch === model.epoch) {
        const id = model.space?.space_id || model.spaces[0]?.space_id;
        if (id) return await open(id);
        else { model.space = null; showCreate(); }
      }
      return true;
    } catch (error) {
      if (sequence === model.listSequence) { state.busy = false; renderSpaces(); failure(error); }
      return false;
    }
  }

  function renderSpaces() {
    const current = model.space;
    const currentOnPage = model.spaces.some((space) => space.space_id === current?.space_id);
    const placeholder = current && !currentOnPage
      ? `<option value="${escapeHtml(current.space_id)}" selected disabled>当前：${escapeHtml(current.name)}（其他页）</option>`
      : currentOnPage ? '' : '<option value="" selected disabled>选择团队空间</option>';
    const options = model.spaces.map((space) => `<option value="${escapeHtml(space.space_id)}" ${space.space_id === current?.space_id ? 'selected' : ''}>${escapeHtml(space.name)}</option>`).join('');
    $('#space-list').innerHTML = `<select id="space-picker" aria-label="当前团队" ${model.spaces.length ? '' : 'disabled'}>${placeholder}${options}</select>`;
    $('#space-list-pagination').innerHTML = pagination('space');
  }

  async function open(id) {
    message('');
    const epoch = ++model.epoch;
    body.innerHTML = '<p class="space-empty">正在读取团队空间…</p>';
    $('#space-tabs').replaceChildren();
    try {
      const [detail, permissions] = await Promise.all([
        read('/management/spaces', { space_id: id }),
        read(`/management/spaces/${encodeURIComponent(id)}/permissions`),
      ]);
      if (epoch !== model.epoch) return;
      model.space = detail.space || detail.items?.[0] || detail;
      model.scope = permissions;
      model.space.capabilities = permissions.capabilities || model.space.capabilities || {};
      if (!cap('manage_members')) model.drafts.delete(id);
      model.members = model.drafts.get(id)?.members || (model.space.members || []).map((member) => ({ ...member }));
      for (const name of ['content', 'rules', 'members', 'sources', 'principals', 'search']) pager(name, true);
      model.principalItems = [];
      model.principalSearched = false;
      model.principalQuery = { q: '', kind: '' };
      model.search = { query: '', mode: 'hybrid' };
      model.searchActive = false;
      model.spaces = model.spaces.map((space) => space.space_id === id ? { ...space, ...model.space } : space);
      renderSpaces();
      $('#space-header').innerHTML = `<h2>${escapeHtml(model.space.name)}</h2>`;
      renderTabs();
      return await showTab(model.tab);
    } catch (error) {
      if (epoch === model.epoch) {
        model.space = null;
        model.scope = {};
        model.members = [];
        if (error.status === 401 || ['SPACE_ACCESS_DENIED', 'SPACE_NOT_FOUND', 'REGISTERED_USER_REQUIRED'].includes(error.code)) model.drafts.delete(id);
        $('#space-header').innerHTML = '<h2>团队空间</h2>';
        $('#space-tabs').replaceChildren();
        body.innerHTML = '<p class="space-empty">团队空间暂不可用</p>';
        renderSpaces();
        failure(error);
      }
      return false;
    }
  }

  function renderTabs() {
    $('#space-tabs').innerHTML = Object.entries(tabs).map(([id, name]) => `<button type="button" id="space-tab-${id}" role="tab" class="secondary-button ${model.tab === id ? 'active' : ''}" data-space-action="tab" data-value="${id}" aria-controls="space-body" aria-selected="${model.tab === id}" tabindex="${model.tab === id ? 0 : -1}">${name}</button>`).join('');
    body.setAttribute('aria-labelledby', `space-tab-${model.tab}`);
  }

  function showCreate() {
    message('');
    model.epoch += 1;
    $('#space-tabs').replaceChildren();
    $('#space-header').innerHTML = '<h2>创建团队空间</h2>';
    body.removeAttribute('aria-labelledby');
    body.innerHTML = `<form novalidate class="space-form space-create-form" data-space-form="create">${input('space_id', '团队空间标识', '', 'required maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_.\\-]*" placeholder="例如 database-ops"')}${input('name', '团队空间名称', '', 'required maxlength="128" placeholder="例如 数据库运维"')}<div class="space-form-actions">${model.space ? button('cancel-create', '取消') : ''}${submit('创建团队空间')}</div></form>`;
    captureEdits();
  }

  async function showTab(tab) {
    message('');
    model.epoch += 1;
    model.tab = tab;
    model.page = {};
    model.rule = null;
    model.searchActive = false;
    pager(tab, true);
    pager('search', true);
    renderTabs();
    if (tab === 'members') {
      pager('principals').busy = false;
      renderMembers();
      if (cap('manage_members')) await findPrincipals();
      return true;
    }
    return await loadPage(0);
  }

  async function loadPage(index = 0) {
    const epoch = ++model.epoch;
    const tab = model.tab;
    const route = tab === 'content' ? 'items' : tab;
    const state = pager(tab);
    state.busy = true;
    model.page = { items: [] };
    model.searchActive = false;
    const currentList = body.querySelector(tab === 'rules' ? '.definition-grid' : '.memory-grid');
    if (!currentList) renderPage();
    else {
      body.querySelector('.space-section-head h3').textContent = tabs[tab];
      body.querySelector('[data-space-action="refresh-tab"]').textContent = '刷新';
      $('#space-page-pagination').innerHTML = pagination(tab);
    }
    const loading = body.querySelector('.space-cards');
    loading.setAttribute('aria-busy', 'true');
    loading.innerHTML = '<p class="space-empty" role="status">正在读取…</p>';
    try {
      const page = await read(endpoint(`/${route}`), { limit: pageSize, cursor: state.cursors[index], ...(tab === 'content' ? { state: 'ACTIVE' } : {}) });
      if (epoch !== model.epoch) return;
      receivePage(tab, index, page);
      model.page = page;
      model.searchActive = false;
      renderPage();
      return true;
    } catch (error) { if (epoch === model.epoch) { state.busy = false; model.page = { items: [] }; renderPage(); failure(error); } return false; }
  }

  function renderPage() {
    const items = model.page.items || [];
    const definitions = model.tab === 'rules';
    const cards = items.map((item) => {
      if (definitions) {
        return `<article class="space-card space-definition-card"><span class="space-card-icon">${icon('rule')}</span><div class="space-card-copy"><div class="space-card-heading"><h3>${escapeHtml(item.name)}</h3><span class="type-chip">${item.state === 'ACTIVE' ? '启用' : label(item.state)}</span></div><p class="space-card-description">${escapeHtml(item.spec?.description ?? item.spec?.instructions ?? '')}</p></div><div class="space-card-footer">${button('edit-rule', '查看规则', item.rule_id)}</div></article>`;
      }
      const title = item.title || item.abstract || '团队记忆';
      const summary = String(item.abstract || '').trim();
      const preview = String(model.searchActive || !summary || summary === String(title).trim() ? item.content || '' : summary)
        .replace(/^ {0,3}(?:`{3,}|~{3,}).*$/gm, '')
        .replace(/^ {0,3}(?:#{1,6}\s+|(?:[-+*]|\d+[.)])\s+|>\s?)/gm, '')
        .replace(/`([^`\n]+)`|\*\*([^*\n]+)\*\*/g, (_, code, strong) => code ?? strong)
        .replace(/\s+/g, ' ').trim();
      const previewHtml = preview ? `<p class="space-preview" data-space-item-content>${escapeHtml(preview)}</p>` : '';
      return `<article class="space-card space-memory-card"><span class="space-card-icon">${icon('document')}</span><div class="space-card-copy"><div class="space-card-heading"><h3>${escapeHtml(title)}</h3></div>${previewHtml}</div><div class="space-card-footer">${button('item', '查看详情', item.item_id)}</div></article>`;
    }).join('');
    const title = definitions ? '规则' : model.searchActive ? '检索结果' : '团队记忆';
    const count = model.page.total === undefined ? `本页 ${items.length} 项` : `${model.page.total} 项`;
    const actions = definitions
      ? `${button('refresh-tab', '刷新')}${cap('create_rule') ? button('new-rule', '新增规则') : ''}`
      : button('refresh-tab', model.searchActive ? '全部记忆' : '刷新');
    const heading = `<div class="space-section-head"><div class="space-section-copy"><h3>${title}<span class="space-count">${count}</span></h3></div><div class="space-section-actions">${actions}</div></div>`;
    const search = definitions ? '' : `<form novalidate class="space-search space-search-compact" data-space-form="search">${searchField('query', '搜索团队记忆', model.search.query, 4096, '团队记忆检索', 'space-query')}${select('mode', '方式', { hybrid: '混合检索', keyword: '关键词检索', semantic: '语义检索' }, model.search.mode)}${submit('检索')}</form>`;
    body.innerHTML = `${heading}${search}<div class="space-cards ${definitions ? 'definition-grid' : 'memory-grid'}">${cards || `<p class="space-empty">${model.searchActive ? '暂无匹配的团队记忆' : definitions ? '暂无规则' : '暂无团队记忆'}</p>`}</div><div id="space-page-pagination">${pagination(model.searchActive ? 'search' : model.tab)}</div>`;
  }

  async function searchPage(index = 0) {
    const epoch = ++model.epoch;
    const state = pager('search');
    state.busy = true;
    model.searchActive = true;
    const query = { ...model.search, kind: 'experience', space_ids: [model.space.space_id], top_k: 100, limit: pageSize, max_context_tokens: 8192 };
    if (state.cursors[index]) query.cursor = state.cursors[index];
    $('#space-page-pagination').innerHTML = pagination('search');
    body.querySelector('.space-cards').innerHTML = '<p class="space-empty">正在检索…</p>';
    try {
      const page = await fetchJson('/api/spaces/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(10000), body: JSON.stringify(query) });
      if (epoch !== model.epoch) return;
      receivePage('search', index, page);
      model.page = page;
      renderPage();
      message('');
    } catch (error) {
      if (epoch === model.epoch) { state.busy = false; model.page = { items: [] }; renderPage(); failure(error); }
    }
  }

  function settingsForm() {
    const space = model.space;
    const owner = currentUser() === space.owner_user_id || model.scope.is_admin;
    return `<details class="space-settings"><summary>团队空间配置</summary><form novalidate class="space-form" data-space-form="settings"><fieldset ${cap('manage_space') ? '' : 'disabled'}><div class="space-form-grid">${input('name', '团队空间名称', space.name, 'required maxlength="128"')}${owner ? select('state', '团队空间状态', { ACTIVE: '启用', ARCHIVED: '归档' }, space.state || 'ACTIVE') : ''}${owner ? select('owner_user_id', '团队空间所有者', Object.fromEntries((space.members || []).filter((m) => m.principal.startsWith('user:')).map((m) => [m.principal.slice(5), m.principal.slice(5)])), space.owner_user_id) : ''}</div><div class="space-form-actions">${submit('保存团队空间配置')}</div></fieldset></form>${cap('delete_space') ? `<section class="space-danger-zone"><h4>删除团队空间</h4>${deleteButton('space', '删除团队空间')}</section>` : ''}</details>`;
  }

  function renderMembers() {
    const editable = cap('manage_members');
    body.innerHTML = `<div class="space-members-grid ${editable ? '' : 'space-members-readonly'}">
      <div class="space-members-primary">
        <section class="space-panel"><div class="space-section-head"><div class="space-section-copy"><h3>当前成员<span id="space-member-count" class="space-count">${model.members.length} 位</span></h3></div></div>
          <form novalidate class="space-form space-members-form" data-space-form="members"><div id="space-members"></div>${editable ? `<div class="space-form-actions">${button('restore-members', '恢复已保存')}${submit('保存成员')}</div>` : ''}</form>
        </section>
        <section class="space-panel space-sources-panel"><div class="space-section-head"><div class="space-section-copy"><h3>来源贡献<span class="space-count">${model.space.sources?.length ?? 0} 项</span></h3></div><div class="space-section-actions">${button('new-source', '增加来源贡献')}</div></div><div id="space-sources"></div><div id="space-source-editor"></div></section>
        ${settingsForm()}
      </div>
      ${editable ? `<section class="space-panel space-members-directory"><div class="space-section-head"><div class="space-section-copy"><h3>添加已注册成员</h3></div></div>
        <form novalidate class="space-search space-principal-search" data-space-form="principals">${searchField('q', '搜索成员', model.principalQuery?.q || '', 128, '搜索成员', 'space-principal-query')}${select('kind', '身份类型', { '': '全部', user: '用户', agent: 'Agent' }, model.principalQuery?.kind || '')}${submit('查找成员')}</form><div id="space-principals"></div>
      </section>` : ''}
    </div>`;
    renderMemberRows();
    renderSources();
    if (editable) renderPrincipals();
    captureEdits();
  }

  function renderMemberRows() {
    const members = localPage('members', model.members);
    $('#space-member-count').textContent = `${model.members.length} 位`;
    const rows = members.map((member) => {
      const owner = member.principal === `user:${model.space.owner_user_id}`;
      const user = member.principal.startsWith('user:');
      return `<div class="space-member-row">${memberIdentity(member.principal, '', owner ? '所有者' : '')}${cap('manage_members') && !owner && user ? `<select data-member-role="${escapeHtml(member.principal)}" aria-label="${escapeHtml(member.principal)} 的角色"><option value="member" ${member.role === 'member' ? 'selected' : ''}>成员</option><option value="manager" ${member.role === 'manager' ? 'selected' : ''}>管理者</option></select>` : `<span class="space-member-role">${member.role === 'manager' ? '管理者' : '成员'}</span>`}${cap('manage_members') && !owner ? button('remove-member', '移出', member.principal) : ''}</div>`;
    }).join('');
    $('#space-members').innerHTML = `<div class="space-member-list">${rows || '<p class="space-empty">暂无成员</p>'}</div>${pagination('members')}`;
  }

  async function findPrincipals(form, index = 0) {
    model.principalSearched = true;
    model.principalError = false;
    if (form) {
      model.principalQuery = Object.fromEntries(new FormData(form));
      pager('principals', true);
    }
    const epoch = model.epoch;
    const sequence = ++model.principalSequence;
    const state = pager('principals');
    state.busy = true;
    model.principalItems = [];
    renderPrincipals();
    try {
      const page = await read('/management/spaces/principals', { ...model.principalQuery, space_id: model.space.space_id, limit: pageSize, cursor: state.cursors[index] });
      if (epoch !== model.epoch || sequence !== model.principalSequence || model.tab !== 'members') return;
      for (const principal of page.items || []) model.registry.set(principal.principal, principal);
      model.principalItems = page.items || [];
      receivePage('principals', index, page);
      renderPrincipals();
    } catch (error) {
      if (epoch === model.epoch && sequence === model.principalSequence) { state.busy = false; model.principalError = true; renderPrincipals(); failure(error); }
    }
  }

  function renderPrincipals() {
    const container = $('#space-principals');
    if (!container) return;
    const rows = (model.principalItems || []).map((p) => {
      const selected = model.members.some((m) => m.principal === p.principal);
      const identity = p.principal.split(':').slice(1).join(':');
      const name = p.display_name || identity;
      return `<div class="space-member-row">${memberIdentity(p.principal, name, name !== identity ? identity : '')}<button type="button" class="secondary-button" data-space-action="add-member" data-value="${escapeHtml(p.principal)}" ${selected ? 'disabled' : ''}>${selected ? '已选择' : '选择'}</button></div>`;
    }).join('');
    const empty = pager('principals').busy ? '正在加载注册成员…' : model.principalError ? '注册成员加载失败' : model.principalSearched ? '暂无匹配的注册成员' : '点击查找成员查看结果。';
    container.innerHTML = `<div class="space-member-list" aria-busy="${pager('principals').busy}">${rows || `<div class="space-empty">${empty}${model.principalError ? button('retry-principals', '重试') : ''}</div>`}</div>${pagination('principals')}`;
  }

  function renderSources() {
    const sources = localPage('sources', model.space.sources || []);
    const rows = sources.map((source) => `<div class="space-member-row">${memberIdentity(source.principal, '', source.memory_types.join('、'))}<span class="space-state-chip">${label(source.state)}</span><div class="space-source-actions">${source.can_manage || canContribute(source.principal) ? button('edit-source', '配置', source.binding_id) : ''}${source.can_delete ? deleteButton('source', '删除', source.binding_id) : ''}</div></div>`).join('');
    $('#space-sources').innerHTML = (sources.length ? `<div class="space-member-list">${rows}</div>` : '<p class="space-empty">暂无来源</p>') + pagination('sources');
  }

  function canContribute(principal) {
    const identity = model.registry.get(principal);
    const member = (model.space.members || []).find((candidate) => candidate.principal === principal);
    return model.scope.is_admin || member?.can_contribute || principal === `user:${currentUser()}` || identity?.can_contribute || (identity?.owner_user_id && identity.owner_user_id === currentUser());
  }

  function showSource(bindingId) {
    message('');
    const source = (model.space.sources || []).find((s) => s.binding_id === bindingId);
    model.source = source;
    const candidates = source ? [{ principal: source.principal }] : (model.space.members || []).filter((m) => canContribute(m.principal));
    const principal = source?.principal || candidates[0]?.principal;
    $('#space-source-editor').innerHTML = `<form novalidate class="space-form" data-space-form="source"><h4>${source ? '配置来源贡献' : '增加来源贡献'}</h4><div class="space-form-grid">${input('binding_id', '来源标识', source?.binding_id || '', `required maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_.\\-]*" ${source ? 'readonly' : ''}`)}${select('principal', '授权贡献的成员', Object.fromEntries(candidates.map((m) => [m.principal, m.principal])), principal)}${input('memory_types', '来源记忆类别（逗号分隔）', source?.memory_types.join(', ') || (principal?.startsWith('agent:') ? 'experience, case' : 'event, entity'), 'required')}${select('state', '来源贡献', { ACTIVE: '启用', PAUSED: '暂停' }, source?.state || 'ACTIVE')}</div><div class="space-form-actions">${button('cancel-source', '取消')}${submit('保存来源贡献')}</div></form>`;
    captureEdits();
    $('#space-source-editor').scrollIntoView({ block: 'nearest' });
  }

  function choices(name, selected = []) {
    const members = new Set((model.space.members || []).map((m) => m.principal));
    return [...new Set([...members, ...selected])].map((principal) => `<label class="space-check"><input type="checkbox" name="${name}" value="${escapeHtml(principal)}" ${selected.includes(principal) ? 'checked' : ''}><span>${escapeHtml(principal.split(':').slice(1).join(':'))}<small>${principal.startsWith('agent:') ? 'Agent' : '用户'}${members.has(principal) ? '' : ' · 当前未生效'}</small></span></label>`).join('');
  }

  function showRule(rule) {
    message('');
    model.epoch += 1;
    model.rule = rule;
    const spec = rule?.spec || {};
    const editable = rule ? cap('manage_rules') : cap('create_rule');
    const title = rule ? editable ? '编辑规则' : '查看规则' : '新增规则';
    const memoryTypes = spec.memory_types || [...new Set((model.space.sources || []).filter((source) => source.state === 'ACTIVE').flatMap((source) => source.memory_types || []))];
    const states = { ACTIVE: '启用', PAUSED: '暂停', ...(rule?.state === 'ARCHIVED' ? { ARCHIVED: '已归档' } : {}) };
    body.innerHTML = `<div class="space-editor-head">${button('back-list', '返回规则')}<h3>${title}</h3></div>
      <form novalidate class="space-form space-definition-form" data-space-form="rule" data-rule-id="${escapeHtml(rule?.rule_id || crypto.randomUUID())}">
        <fieldset class="space-rule-layout" ${editable ? '' : 'disabled'}>
          <section class="space-editor-section space-rule-basics"><h4>基本信息</h4>
            ${input('name', '规则名称', rule?.name || '', 'required maxlength="128" placeholder="例如 数据库排障经验"')}
            ${select('state', '状态', states, rule?.state || 'ACTIVE')}
            ${input('memory_types', '记忆类别（逗号分隔）', memoryTypes.join(', '), 'required placeholder="例如 experience, case"')}
          </section>
          <section class="space-editor-section space-rule-content">
            ${textArea('description', '规则内容', spec.description ?? spec.instructions ?? '', 'required maxlength="16384" rows="8" aria-describedby="space-rule-description-count" placeholder="描述需要沉淀的经验、适用范围和整理要求"')}
            <span id="space-rule-description-count" class="space-rule-description-count">${(spec.description ?? spec.instructions ?? '').length.toLocaleString('zh-CN')} / 16,384</span>
          </section>
          ${editable ? `<div class="space-form-actions">${submit('保存规则')}</div>` : ''}
        </fieldset>
      </form>
      ${rule && cap('manage_rules') ? `<section class="space-danger-zone"><h4>删除规则</h4>${deleteButton('rule', '删除规则', rule.rule_id)}</section>` : ''}`;
    captureEdits();
  }

  function toggleAclEditor(editing) {
    const form = body.querySelector('[data-space-form="acl"]');
    form.hidden = !editing;
    $('#space-acl-summary').hidden = editing;
    const edit = body.querySelector('[data-space-action="edit-acl"]');
    edit.hidden = editing;
    if (editing) form.elements.acl_mode.focus();
    else edit.focus();
  }

  async function showItem(id) {
    message('');
    const epoch = ++model.epoch;
    const page = await read(endpoint('/items'), { item_id: id, state: 'ACTIVE' });
    if (epoch !== model.epoch) return;
    const item = page.items?.[0] || page.item;
    if (!item) {
      await loadPage(0);
      message('记忆已更新，列表已刷新。');
      return false;
    }
    model.item = item;
    const restricted = item.acl_mode === 'restricted';
    const audience = restricted ? `指定 ${item.readers?.length || 0} 位成员及管理角色可读取` : '全部团队成员可读取';
    const readerNames = (item.readers || []).map((principal) => `<span class="space-reader-chip">${escapeHtml(principal.split(':').slice(1).join(':'))}</span>`).join('');
    body.innerHTML = `
      <div class="space-editor-head space-detail-heading">
        ${button('back-list', '← 返回')}
        <h3>${escapeHtml(item.title || item.abstract || '团队记忆')}</h3>
      </div>
      <div class="space-detail-grid">
        <div class="space-detail-main">
          <article class="space-panel space-memory-content">
            <div class="space-content space-prose">${window.renderTeamMemoryContent(item.content)}</div>
          </article>
        </div>
        <aside class="space-detail-aside space-panel">
          <section class="space-permissions-panel">
            <div class="space-section-head"><h4>读取权限</h4>${cap('manage_acl') ? button('edit-acl', '编辑') : ''}</div>
            <div id="space-acl-summary">
              <p class="space-acl-audience">${audience}</p>
              ${restricted && readerNames ? `<div class="space-reader-chips">${readerNames}</div>` : ''}
              ${button('permissions', '查看我的权限', item.item_id)}
              <p id="space-permissions" class="space-permission-result" role="status" hidden></p>
            </div>
            ${cap('manage_acl') ? `<form novalidate class="space-form space-acl-form" data-space-form="acl" hidden>
              ${select('acl_mode', '读取方式', { inherit: '继承团队权限', restricted: '指定成员读取' }, item.acl_mode || 'inherit')}
              <fieldset class="space-choices space-acl-readers" ${restricted ? '' : 'hidden disabled'}><legend class="sr-only">可读取的成员</legend>${choices('readers', item.readers || [])}</fieldset>
              <p class="space-acl-inherited" ${restricted ? 'hidden' : ''}>全部团队成员可读取</p>
              <p class="space-acl-note">空间所有者、管理者及租户管理员保留读取权限。</p>
              <div class="space-form-actions">${button('cancel-acl', '取消')}${submit('保存')}</div>
            </form>` : ''}
          </section>
        </aside>
      </div>`;
    captureEdits();
    return true;
  }

  root.addEventListener('change', async (event) => {
    if (event.target.getAttribute('aria-invalid') === 'true') validateField(event.target);
    if (event.target.id === 'space-picker') {
      event.target.disabled = true;
      const id = event.target.value;
      await guardNavigation(() => open(id), event.target, { preserveMembers: true });
      renderSpaces();
      $('#space-picker')?.focus();
      return;
    }
    const principal = event.target.dataset.memberRole;
    if (principal) {
      model.members.find((member) => member.principal === principal).role = event.target.value;
      rememberMembers();
    }
    if (event.target.name === 'principal' && event.target.form?.dataset.spaceForm === 'source' && !model.source) {
      event.target.form.elements.memory_types.value = event.target.value.startsWith('agent:') ? 'experience, case' : 'event, entity';
    }
    if (event.target.name === 'acl_mode' && event.target.form?.dataset.spaceForm === 'acl') syncReaders(event.target.form);
  });

  function inputChanged(field) {
    const clear = field.closest('.space-input-wrap')?.querySelector('.space-clear-search');
    if (clear) clear.hidden = !field.value;
    if (composing.has(field)) return;
    if (field.getAttribute('aria-invalid') === 'true') validateField(field);
    if (field.name === 'description' && field.form?.dataset.spaceForm === 'rule') {
      $('#space-rule-description-count').textContent = `${field.value.length.toLocaleString('zh-CN')} / 16,384`;
    }
    const form = field.closest('[data-space-form="search"], [data-space-form="principals"]');
    if (!form) return;
    if (form.dataset.spaceForm === 'principals') {
      model.principalSequence += 1;
      model.principalQuery = Object.fromEntries(new FormData(form));
      model.principalItems = [];
      model.principalError = false;
      model.principalSearched = false;
      pager('principals', true);
      renderPrincipals();
      formBusy(form, false);
      return;
    }
    model.epoch += 1;
    model.search = Object.fromEntries(new FormData(form));
    model.searchActive = false;
    pager('search', true);
    body.querySelector('.space-cards').innerHTML = '<p class="space-empty">点击检索查看结果。</p>';
    $('#space-page-pagination').replaceChildren();
    formBusy(form, false);
    message('');
  }

  root.addEventListener('input', (event) => inputChanged(event.target));
  root.addEventListener('compositionstart', (event) => composing.add(event.target));
  root.addEventListener('compositionend', (event) => { composing.delete(event.target); inputChanged(event.target); });
  root.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.isComposing || event.keyCode === 229 || composing.has(event.target))) {
      event.preventDefault();
      return;
    }
    const tab = event.target.closest('#space-tabs [role="tab"]');
    if (!tab || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const buttons = [...$('#space-tabs').querySelectorAll('[role="tab"]')];
    const index = buttons.indexOf(tab);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    buttons.forEach((control, position) => { control.tabIndex = position === next ? 0 : -1; });
    buttons[next].focus();
  });

  root.addEventListener('click', async (event) => {
    const target = event.target.closest('[data-space-action]');
    if (!target || target.disabled) return;
    const { spaceAction: action, value } = target.dataset;
    target.disabled = true;
    try {
      if (action === 'create') await guardNavigation(showCreate, target, { preserveMembers: true });
      else if (action === 'tab') {
        if (await guardNavigation(() => showTab(value), target, { preserveMembers: true })) $(`#space-tab-${model.tab}`)?.focus();
      }
      else if (action === 'back-list') await guardNavigation(() => { message(''); model.epoch += 1; renderPage(); }, target);
      else if (action === 'cancel-create') await guardNavigation(() => open(model.space.space_id), target, { preserveMembers: true });
      else if (action === 'cancel-source') await guardNavigation(() => { $('#space-source-editor').replaceChildren(); message(''); }, target, { preserveMembers: true, except: body.querySelector('[data-space-form="settings"]') });
      else if (action === 'restore-members') {
        await guardNavigation(() => open(model.space.space_id), target, { currentOnly: true });
      }
      else if (action === 'clear-search') {
        const form = target.closest('[data-space-form]');
        const field = form.querySelector('input[type="search"]');
        const kind = form.dataset.spaceForm;
        field.value = '';
        inputChanged(field);
        field.focus();
        if (kind !== 'principals') pager('content', true);
        const refresh = kind === 'principals' ? findPrincipals(form) : loadPage(0);
        root.querySelector(`[data-space-form="${kind}"] input[type="search"]`)?.focus();
        const epoch = model.epoch;
        await refresh;
        if (epoch === model.epoch) root.querySelector(`[data-space-form="${kind}"] input[type="search"]`)?.focus();
      }
      else if (action === 'refresh-tab') {
        await guardNavigation(async () => {
          message(''); model.searchActive = false; pager(model.tab, true);
          if (await loadPage(0)) message('列表已刷新。');
        }, target, { preserveMembers: true });
      } else if (action === 'page') {
        const state = pager(value);
        if (state.busy) return;
        const index = state.index + (target.dataset.direction === 'next' ? 1 : -1);
        if (index < 0 || (index > state.index && !state.next)) return;
        if (value === 'space') await loadSpaces(index);
        else if (value === 'principals') await findPrincipals(null, index);
        else if (value === 'members') { state.index = index; renderMemberRows(); }
        else if (value === 'sources') { state.index = index; renderSources(); }
        else if (value === 'search') await searchPage(index);
        else await loadPage(index);
      }
      else if (action === 'add-member') {
        const registered = model.principalItems?.find((principal) => principal.principal === value);
        if (!registered || pager('principals').busy) return;
        if (!model.members.some((member) => member.principal === value)) model.members.push({ principal: value, role: 'member' });
        rememberMembers(); renderMemberRows(); renderPrincipals(); message('成员选择已更新，保存后生效。');
      } else if (action === 'retry-principals') {
        await findPrincipals();
      } else if (action === 'remove-member') {
        const name = model.registry.get(value)?.display_name || value.split(':').slice(1).join(':');
        confirmAction({ title: '移出成员', name, confirmText: '确认移出',
          impact: '保存成员后，撤销该身份的团队成员访问权限并暂停其来源贡献；相关团队记忆由系统处理。',
          focusAfter: '[data-space-form="members"] [type="submit"]' }, target, () => {
          model.members = model.members.filter((member) => member.principal !== value);
          rememberMembers(); renderMemberRows(); renderPrincipals(); message('成员已移出草稿，待保存成员后生效。');
        });
      } else if (action === 'new-source' || action === 'edit-source') await guardNavigation(() => showSource(value), target, { preserveMembers: true, except: body.querySelector('[data-space-form="settings"]') });
      else if (action === 'new-rule') await guardNavigation(() => showRule(null), target, { preserveMembers: true });
      else if (action.startsWith('delete-')) showDelete(action.slice(7), value, target);
      else if (action === 'edit-rule') await guardNavigation(() => showRule(model.page.items.find((rule) => rule.rule_id === value)), target, { preserveMembers: true });
      else if (action === 'item') await guardNavigation(() => showItem(value), target, { preserveMembers: true });
      else if (action === 'edit-acl') { message(''); toggleAclEditor(true); }
      else if (action === 'cancel-acl') await guardNavigation(() => { message(''); toggleAclEditor(false); }, target, { currentOnly: true });
      else if (action === 'permissions') {
        const epoch = ++model.epoch;
        const permissions = await read(endpoint('/permissions'), { item_id: value });
        if (epoch !== model.epoch) return;
        const readable = permissions.can_read ?? permissions.decisions?.find((decision) => decision.action === 'resource.read')?.allowed;
        $('#space-permissions').textContent = `${permissions.user_id || currentUser()}：${readable ? '可以读取此记忆' : '当前读取受限'}`;
        $('#space-permissions').hidden = false;
      }
    } catch (error) { if (target.isConnected) failure(error); }
    finally { target.disabled = false; }
  });

  root.addEventListener('submit', async (event) => {
    const form = event.target.closest('[data-space-form]');
    if (!form) return;
    event.preventDefault();
    if (form.dataset.busy || [...form.elements].some((field) => composing.has(field)) || !validateForm(form)) return;
    const save = async () => {
      const data = new FormData(form);
      const values = Object.fromEntries(data);
      const kind = form.dataset.spaceForm;
      const epoch = model.epoch;
      const spaceId = model.space?.space_id;
      const saving = kind !== 'principals' && kind !== 'search';
      const workspace = saving ? form.closest('.space-members-grid') : null;
      if (workspace?.inert) return;
      if (workspace) workspace.inert = true;
      const controls = saving ? [...form.elements].filter((field) =>
        field.matches('input, select, textarea, button:not([type="submit"])') && !field.matches(':disabled')) : [];
      if (saving) savingSnapshots.set(form, JSON.stringify([...data]));
      controls.forEach((field) => { field.disabled = true; });
      const busyId = formBusy(form, true);
      message('');
      try {
        let result;
        if (kind === 'principals') { await findPrincipals(form); return; }
        if (kind === 'search') {
          model.search = values;
          pager('search', true);
          if (values.query.trim()) await searchPage(0);
          else { pager('content', true); await loadPage(0); }
          return;
        }
        if (kind === 'create') {
          result = await mutate('/management/spaces', 'POST', values);
          if (epoch !== model.epoch) return;
          model.space = result.space || result;
          model.tab = 'members';
          if (await load()) message('团队空间已创建。'); return;
        }
        if (kind === 'settings') result = await mutate(endpoint(), 'PATCH', values, model.space.authz_rev);
        if (kind === 'members') {
          const members = model.members.map(({ principal, role }) => ({ principal, role }));
          result = await mutate(endpoint('/members'), 'PUT', { members }, model.drafts.get(spaceId)?.revision ?? model.space.authz_rev);
          const draft = model.drafts.get(spaceId);
          if (draft && memberSnapshot(draft.members) === memberSnapshot(members)) model.drafts.delete(spaceId);
          else if (draft) draft.revision = result.authz_rev;
        }
        if (kind === 'source') result = await mutate(endpoint(`/sources/${encodeURIComponent(values.binding_id)}`), 'PUT', { principal: values.principal, memory_types: split(values.memory_types), state: values.state }, model.source?.grant_rev, !model.source);
        if (kind === 'acl') result = await mutate(endpoint('/acl'), 'PUT', { item_id: model.item.item_id, acl_mode: values.acl_mode, readers: values.acl_mode === 'restricted' ? data.getAll('readers') : [] }, model.space.authz_rev);
        if (kind === 'rule') {
          const spec = { ...model.rule?.spec, description: values.description.trim(),
            output_kind: 'experience', memory_types: split(values.memory_types) };
          for (const key of ['instructions', 'output_schema', 'fields']) delete spec[key];
          result = await mutate(endpoint(`/rules/${encodeURIComponent(form.dataset.ruleId)}`), 'PUT', { name: values.name, state: values.state, spec }, model.rule?.rule_rev, !model.rule);
        }
        if (epoch !== model.epoch) return;
        if (kind === 'acl') {
          model.space.authz_rev = result.authz_rev;
          if (await showItem(model.item.item_id)) message('读取权限已保存。');
          return;
        }
        if (await open(spaceId)) message(kind === 'rule' ? '规则已保存。' : kind === 'source' ? '来源贡献已保存。' : '配置已保存。');
      } catch (error) {
        if (epoch === model.epoch) {
          if (kind === 'acl' && error.code === 'ITEM_CHANGED') await open(spaceId);
          failure(error);
        }
      }
      finally {
        savingSnapshots.delete(form);
        if (workspace) workspace.inert = false;
        controls.forEach((field) => { field.disabled = false; });
        if (form.dataset.busy === busyId) formBusy(form, false);
      }
    };
    if (['search', 'principals'].includes(form.dataset.spaceForm)) await save();
    else await guardNavigation(save, event.submitter, { except: form, currentOnly: true });
  });

  function initialize() {
    if (!model.pagers.space && !body.querySelector('[data-space-form]')) return load();
  }

  return { initialize, load, reset, guardNavigation, notify: message };
})();
