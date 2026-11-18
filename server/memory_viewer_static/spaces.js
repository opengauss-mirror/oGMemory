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

  async function mutate(path, method, payload, revision, create = false) {
    const headers = { 'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID() };
    if (revision !== undefined) headers['If-Match'] = `"${revision}"`;
    if (create) headers['If-None-Match'] = '*';
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        return await fetchJson(path, { method, headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(10000) });
      } catch (error) {
        const retryable = error.status === 429 || error.status === 503 || error instanceof TypeError || error.name === 'TimeoutError';
        if (!retryable || attempt === 2) throw error;
        await new Promise((resolve) => setTimeout(resolve, Math.min(error.retryAfter || 1, 5) * 1000));
      }
    }
  }

  const editSnapshots = new WeakMap();
  const memberSnapshot = (members) => JSON.stringify(members.map(({ principal, role }) => ({ principal, role })));
  const formSnapshot = (form) => form.dataset.spaceForm === 'members' ? memberSnapshot(model.members) : JSON.stringify([...new FormData(form)]);

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
    controls[0].addEventListener('click', () => dialog.close());
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
      dialog.remove();
      const focusTarget = trigger?.isConnected ? trigger : root.querySelector(details.focusAfter || '#space-tabs button');
      if (focusTarget) { focusTarget.disabled = false; focusTarget.focus(); }
      resolve(completed);
    }, { once: true });
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (form.dataset.busy) return;
      formBusy(form, true);
      controls.forEach((control) => { control.disabled = true; });
      errorMessage.hidden = true;
      try {
        await action();
        completed = true;
        dialog.close();
      } catch (error) {
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

})();
