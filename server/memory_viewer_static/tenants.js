window.tenantAdmin = (() => {
  const directory = $('#tenant-directory');
  const catalog = $('#tenant-catalog');
  const catalogRows = $('#tenant-catalog-rows');
  const rows = $('#tenant-rows');
  const query = $('#tenant-query');
  const dialog = $('#tenant-dialog');
  const fields = $('#tenant-dialog-fields');
  const form = $('#tenant-form');
  const cancel = $('#tenant-cancel');
  const save = $('#tenant-save');
  const feedback = $('#tenant-dialog-error');
  const description = $('#tenant-dialog-description');
  const roleNames = { ROOT: '部署管理员', ADMIN: '租户管理员', MEMBER: '普通用户' };
  const base = '/api/v1/admin/accounts';
  const pageSize = 20;
  const duplicateIdentities = {
    account: ['account_id', '租户标识已存在，请换一个。'],
    user: ['user_id', '用户标识已存在，请换一个。'],
    agent: ['agent_id', 'Agent 标识已存在，请换一个。'],
  };
  let section = 'users';
  let deploymentDetail = false;
  let catalogPage = 0;
  let catalogCursors = [''];
  let createdTenant;
  let account;
  let page = 0;
  let cursors = [''];
  let search = '';
  let request;
  let action;
  let trigger;
  let disposePicker;

  function errorText(error) {
    if (error.name === 'TimeoutError') return '请求超时，请刷新目录后重试。';
    if (error.status === 401) return 'Key 已失效，请在「身份与 API Key」中更新。';
    if (error.status === 403) return '当前身份无法管理此租户。';
    if (error instanceof TypeError || error.status >= 500) return '服务暂不可用，请稍后重试。';
    if (error.status === 404) return '记录已不存在，请刷新目录。';
    if (error.message === 'Keep at least one tenant administrator') return '请先设置另一位租户管理员。';
    if (error.message === "Reassign the user's Agents before removing this user") return '请先在 Agent 列表中变更归属。';
    if (error.message.startsWith('owner user not found') || error.message === 'Agent owner is no longer registered') return '请选择当前租户中已注册的用户。';
    return error.message;
  }

  function setStatus(status, text, error = false) {
    status.textContent = text;
    status.classList.toggle('error', error);
  }

  function message(text, error = false) {
    setStatus($('#tenant-message'), text, error);
  }

  function fieldProblem(field, text) {
    field.setAttribute('aria-describedby', feedback.id);
    showFieldProblem(field, feedback, text);
  }

  function api(path, method, data) {
    return fetchJson(apiUrl(path), {
      method,
      signal: AbortSignal.timeout(VIEWER_REQUEST_TIMEOUT_MS),
      ...(data && { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
    });
  }

  function reset() {
    request?.abort();
    if (dialog.open) dialog.close();
    disposePicker?.();
    fields.replaceChildren();
    action = null;
    trigger = null;
    deploymentDetail = false;
    catalogPage = 0;
    catalogCursors = [''];
    createdTenant = null;
    catalog.hidden = directory.hidden = true;
    catalogRows.replaceChildren();
    rows.replaceChildren();
    account = undefined;
    resetPages();
  }

  function resetPages(value = '') {
    search = value;
    query.value = value;
    $('#tenant-clear').hidden = !value;
    page = 0;
    cursors = [''];
  }

  function reflectScope() {
    const scope = state.tenantScope;
    const root = scope?.role === 'ROOT';
    const showCatalog = root && !deploymentDetail;
    catalog.hidden = !showCatalog;
    $('#tenant-back').hidden = !root || showCatalog;
    $('#tenant-account').textContent = showCatalog ? '租户目录'
      : scope?.admin_mode && state.accountId ? state.accountId : '租户管理';
    $('#tenant-identity').textContent = scope?.admin_mode
      ? [roleNames[scope.role] || roleNames.ADMIN, scope.user_id].filter(Boolean).join(' · ') : '';
    directory.hidden = showCatalog || !scope?.admin_mode || !state.accountId;
    return !directory.hidden;
  }

  function renderCatalog(accounts, nextAfter) {
    catalogRows.innerHTML = accounts.map((entry) => {
      const id = escapeHtml(entry.account_id);
      return '<li class="tenant-row"><strong>' + id + '</strong><span class="tenant-row-detail">'
        + '<span class="tenant-mobile-label">用户数</span>' + escapeHtml(entry.user_count ?? '—')
        + '</span><div class="tenant-actions"><button type="button" class="secondary-button" data-enter-tenant="'
        + id + '" aria-label="进入租户 ' + id + '">进入租户</button></div></li>';
    }).join('') || '<li class="tenant-empty">暂无租户。创建后即可添加用户和 Agent。</li>';
    $('#tenant-catalog-pages').hidden = !catalogPage && !nextAfter;
    $('#tenant-catalog-page').textContent = '第 ' + (catalogPage + 1) + ' 页';
    $('#tenant-catalog-prev').disabled = catalogPage === 0;
    $('#tenant-catalog-next').disabled = !nextAfter;
  }

  async function loadCatalog(targetPage = catalogPage, notice = '') {
    request?.abort();
    const controller = request = new AbortController();
    catalogRows.inert = true;
    $('#tenant-catalog-prev').disabled = true;
    $('#tenant-catalog-next').disabled = true;
    try {
      const data = await fetchJson(apiUrl(base, { limit: pageSize, after: catalogCursors[targetPage] }), {
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(VIEWER_REQUEST_TIMEOUT_MS)]),
      });
      if (controller.signal.aborted) return;
      if (targetPage && !data.accounts.length) return loadCatalog(targetPage - 1, notice);
      catalogPage = targetPage;
      catalogCursors.length = catalogPage + 1;
      if (data.next_after) catalogCursors.push(data.next_after);
      renderCatalog(data.accounts, data.next_after);
      message(notice);
    } catch (error) {
      if (!controller.signal.aborted) {
        catalogRows.replaceChildren();
        message(errorText(error), true);
      }
    } finally {
      if (request === controller) catalogRows.inert = false;
    }
  }

  async function load(targetPage = page, notice = '') {
    request?.abort();
    if (!reflectScope()) {
      if (!catalog.hidden) return loadCatalog(catalogPage, notice);
      const scope = state.tenantScope;
      message(scope?.admin_mode ? '创建租户后即可添加用户。' : scope ? '此页面仅供租户管理员使用。' : '连接后可查看租户管理权限。');
      return;
    }
    if (account !== state.accountId) {
      account = state.accountId;
      rows.replaceChildren();
      resetPages();
      targetPage = 0;
    }
    const controller = request = new AbortController();
    const kind = section;
    message('正在读取…');
    directory.setAttribute('aria-busy', 'true');
    rows.inert = true;
    $('#tenant-prev').disabled = true;
    $('#tenant-next').disabled = true;
    try {
      const path = base + '/' + encodeURIComponent(account) + '/' + kind;
      const body = await fetchJson(apiUrl(path, { limit: pageSize, after: cursors[targetPage], q: search }), {
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(VIEWER_REQUEST_TIMEOUT_MS)]),
      });
      if (controller.signal.aborted) return;
      const records = body[kind];
      if (targetPage && !records.length) return load(targetPage - 1, notice);
      page = targetPage;
      cursors.length = page + 1;
      if (body.next_after) cursors.push(body.next_after);
      render(records);
      $('#tenant-page').textContent = '第 ' + (page + 1) + ' 页';
      $('#tenant-pagination').hidden = !page && !body.next_after;
      $('#tenant-prev').disabled = !page;
      $('#tenant-next').disabled = !body.next_after;
      message(notice);
    } catch (error) {
      if (!controller.signal.aborted) {
        rows.replaceChildren();
        $('#tenant-columns').hidden = true;
        $('#tenant-pagination').hidden = true;
        message(errorText(error), true);
      }
    } finally {
      if (request === controller) {
        directory.removeAttribute('aria-busy');
        rows.inert = false;
      }
    }
  }

  function roleName(role) {
    return role?.toLowerCase() === 'admin' ? roleNames.ADMIN : roleNames.MEMBER;
  }

  function actionButton(name, label, id, className = 'secondary-button') {
    return '<button class="' + className + '" type="button" data-tenant-action="' + name
      + '" aria-label="' + label + ' ' + id + '">' + label + '</button>';
  }

  function render(records) {
    const users = section === 'users';
    $('#tenant-id-heading').textContent = users ? '用户' : 'Agent';
    $('#tenant-role-heading').textContent = users ? '租户角色' : '归属用户';
    $('#tenant-columns').hidden = !records.length;
    if (!records.length) {
      rows.innerHTML = '<li class="tenant-empty">' + (search ? '没有匹配的记录。' : users ? '暂无用户。' : '暂无 Agent。可从用户列表绑定 Agent。') + '</li>';
      return;
    }
    rows.innerHTML = records.map((record) => {
      const id = escapeHtml(users ? record.user_id : record.agent_id);
      const detail = users ? roleName(record.role) : escapeHtml(record.owner_user_id || '未绑定用户');
      const actions = users
        ? actionButton('agent', '绑定 Agent', id)
          + '<details class="tenant-menu" name="tenant-row-actions"><summary aria-label="' + id + ' 的更多操作">更多 <span aria-hidden="true">⌄</span></summary>'
          + '<div class="tenant-menu-panel">' + actionButton('role', '调整角色', id, 'tenant-menu-item')
          + (detail === roleNames.MEMBER ? actionButton('reveal', '查看 API Key', id, 'tenant-menu-item') : '')
          + actionButton('key', '轮换 Key', id, 'tenant-menu-item')
          + actionButton('delete', '移除用户', id, 'tenant-menu-item') + '</div></details>'
        : actionButton('owner', '变更归属', id);
      return '<li class="tenant-row" data-id="' + id + '" data-role="' + escapeHtml(record.role) + '">'
        + '<div class="tenant-row-identity"><strong>' + id + '</strong>'
        + (users ? actionButton('agents', '查看 Agent', id, 'tenant-link-button') : '')
        + '</div><span class="tenant-row-detail"><span class="tenant-mobile-label">'
        + (users ? '租户角色' : '归属用户') + '</span><span' + (users ? ' class="tenant-role-tag"' : '') + '>' + detail + '</span></span>'
        + '<div class="tenant-actions">' + actions + '</div></li>';
    }).join('');
  }

  function describe(text = '') {
    description.textContent = text;
    description.hidden = !text;
  }

  function input(name, label) {
    return '<label>' + label + '<input name="' + name + '" required maxlength="128" pattern="[A-Za-z0-9][A-Za-z0-9_.\\-]{0,127}" autocomplete="off"></label>';
  }

  function roleInput(admin = false) {
    return '<label>租户角色<select name="role"><option value="MEMBER"' + (admin ? '' : ' selected')
      + '>普通用户</option><option value="ADMIN"' + (admin ? ' selected' : '') + '>租户管理员</option></select></label>';
  }

  function open(name, button, id = '', role = '') {
    if (dialog.open || !state.tenantScope?.admin_mode) return;
    if (name === 'account' && state.tenantScope.role !== 'ROOT') return;
    const tenant = state.accountId;
    const path = base + '/' + encodeURIComponent(tenant);
    const user = path + '/users/' + encodeURIComponent(id);
    const options = {
      account: ['创建租户', base, 'POST', input('account_id', '租户标识') + input('admin_user_id', '首位管理员用户名')],
      user: ['创建用户', path + '/users', 'POST', input('user_id', '用户标识') + roleInput()],
      agent: ['绑定 Agent', path + '/agents', 'PATCH', '', 'agents'],
      register: ['注册 Agent', path + '/agents', 'POST', input('agent_id', 'Agent 标识'), 'users'],
      owner: ['变更归属', path + '/agents/' + encodeURIComponent(id), 'PATCH', '', 'users'],
      agents: ['查看 Agent', path + '/agents', 'GET', ''],
      reveal: ['查看 API Key', user + '/key/reveal', 'POST', ''],
      role: ['调整角色', user + '/role', 'PATCH', roleInput(roleName(role) === roleNames.ADMIN)],
      key: ['轮换 Key', user + '/key', 'POST', ''],
      delete: ['移除用户', user, 'DELETE', ''],
    };
    const [label, url, method, content, picker] = options[name];
    action = method === 'GET' ? null : { name, url, method, picker, label, id, role };
    trigger = button;
    $('#tenant-dialog-title').textContent = id ? label + ' · ' + id : label;
    const explanations = {
      account: '创建租户时会同时生成首位管理员的 API Key。',
      user: '创建后即可复制该用户的 API Key。',
      agent: '选择要转交给此用户的 Agent。',
      register: '归属用户可留空，注册后也可以绑定。',
      owner: '选择新的归属用户，也可以解除当前绑定。',
      reveal: '查看会记录审计。请仅向该用户提供 Key。',
      role: '租户管理员可管理本租户的用户和 Agent。',
      key: '旧 Key 会立即失效。使用旧 Key 的客户端需要更新配置。',
      delete: '移除后，此用户将失去租户访问权限。请先转交其 Agent。',
    };
    describe(explanations[name]);
    fields.innerHTML = content;
    feedback.textContent = '';
    cancel.textContent = action ? '取消' : '完成';
    save.textContent = name === 'reveal' ? '获取 API Key' : label;
    save.hidden = !action;
    save.disabled = Boolean(picker) && name !== 'register';
    if (picker || name === 'agents') {
      mountPicker(path, picker || 'agents', name === 'agents' ? id : '', action);
    }
    dialog.showModal();
    (form.querySelector('input, select') || cancel).focus();
  }

  function mountPicker(path, kind, owner, pending) {
    const users = kind === 'users';
    const label = users ? '归属用户' : 'Agent';
    const optionalOwner = pending?.name === 'register';
    fields.insertAdjacentHTML('beforeend', '<div class="tenant-picker">'
      + '<label>' + (pending ? '选择' : '搜索') + label + (optionalOwner ? '（可选）' : '')
      + '<input type="search" maxlength="128" placeholder="输入部分标识搜索"></label>'
      + '<div class="tenant-picker-results"></div><p class="tenant-picker-status" role="status"></p>'
      + '<div class="tenant-picker-pages"><button class="secondary-button" type="button" data-picker-step="-1">上一页</button>'
      + '<button class="secondary-button" type="button" data-picker-step="1">下一页</button></div></div>');
    const picker = fields.querySelector('.tenant-picker');
    const filter = picker.querySelector('input');
    const results = picker.querySelector('.tenant-picker-results');
    const status = picker.querySelector('.tenant-picker-status');
    const [prev, next] = picker.querySelectorAll('[data-picker-step]');
    let index = 0;
    let offsets = [''];
    let timer;
    let controller;
    let composing = false;

    function select(record) {
      pending.selection = record;
      save.disabled = form.hasAttribute('aria-busy') || (!record && !optionalOwner) || (!users && record?.owner_user_id === pending.id);
      status.classList.remove('error');
      if (!record) status.textContent = '';
      else if (pending.name === 'agent') {
        status.textContent = record.owner_user_id === pending.id ? '该 Agent 已归属此用户。'
          : record.owner_user_id ? record.agent_id + ' 将从 ' + record.owner_user_id + ' 转交给 ' + pending.id + '。'
            : record.agent_id + ' 将绑定给 ' + pending.id + '。';
      } else {
        status.textContent = record.user_id ? '已选择 ' + record.user_id + '。' : '保存后，此 Agent 将处于未绑定状态。';
      }
    }

    async function loadOptions() {
      controller?.abort();
      const request = controller = new AbortController();
      prev.disabled = next.disabled = true;
      results.setAttribute('aria-busy', 'true');
      results.inert = true;
      setStatus(status, '正在读取…');
      try {
        const data = await fetchJson(apiUrl(path + '/' + kind, {
          limit: 8, q: filter.value.trim(), after: offsets[index], owner_user_id: owner,
        }), { signal: AbortSignal.any([request.signal, AbortSignal.timeout(VIEWER_REQUEST_TIMEOUT_MS)]) });
        if (request.signal.aborted) return;
        const records = data[kind];
        offsets.length = index + 1;
        if (data.next_after) offsets.push(data.next_after);
        prev.disabled = index === 0;
        next.disabled = !data.next_after;
        status.textContent = records.length ? '第 ' + (index + 1) + ' 页' : '没有匹配的' + label + '。';
        if (!records.length && !(users && pending)) {
          results.replaceChildren();
          return;
        }
        const idField = users ? 'user_id' : 'agent_id';
        const options = records.map((record, position) => {
          const detail = users ? roleName(record.role) : record.owner_user_id ? '归属 ' + record.owner_user_id : '未绑定用户';
          const text = escapeHtml(record[idField]) + ' · ' + escapeHtml(detail);
          return pending ? '<option value="' + position + '">' + text + '</option>' : '<li>' + text + '</li>';
        }).join('');
        const unassigned = users && pending ? '<option value="unassigned">不绑定用户</option>' : '';
        results.innerHTML = pending ? '<select size="5" aria-label="匹配的' + label + '">' + unassigned + options + '</select>'
          : '<ul class="tenant-owner-list" aria-label="归属 Agent">' + options + '</ul>';
        if (!pending) return;
        const list = results.querySelector('select');
        const match = records.findIndex((record) => record[idField] === pending.selection?.[idField]);
        list.value = match < 0 ? (pending.selection?.user_id === '' ? 'unassigned' : '') : String(match);
        list.addEventListener('change', () => select(list.value === 'unassigned' ? { user_id: '' } : records[Number(list.value)]));
        if (list.selectedIndex >= 0) select(pending.selection);
      } catch (error) {
        if (request.signal.aborted) return;
        results.replaceChildren();
        setStatus(status, errorText(error), true);
      } finally {
        if (controller === request) {
          results.removeAttribute('aria-busy');
          results.inert = false;
        }
      }
    }

    function search(immediate = false) {
      clearTimeout(timer);
      controller?.abort();
      index = 0;
      offsets = [''];
      results.replaceChildren();
      prev.disabled = next.disabled = true;
      if (pending) select(null);
      if (immediate) loadOptions();
      else timer = setTimeout(loadOptions, 250);
    }
    filter.addEventListener('compositionstart', () => { composing = true; });
    filter.addEventListener('compositionend', () => { composing = false; search(); });
    filter.addEventListener('input', () => { if (!composing) search(); });
    filter.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' || event.isComposing || composing) return;
      event.preventDefault();
      search(true);
    });
    for (const button of [prev, next]) {
      button.addEventListener('click', () => {
        index += Number(button.dataset.pickerStep);
        if (pending) select(null);
        loadOptions();
      });
    }
    disposePicker = () => { clearTimeout(timer); controller?.abort(); disposePicker = null; };
    loadOptions();
  }

  async function copyCredential(value) {
    if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(value);
    const field = document.createElement('textarea');
    field.className = 'sr-only';
    field.value = value;
    const active = document.activeElement;
    dialog.append(field);
    try {
      field.select();
      if (!document.execCommand('copy')) throw new Error('Clipboard unavailable');
    } finally {
      field.remove();
      active?.focus();
    }
  }

  function showCredential(result, pending, data) {
    action = null;
    const identity = escapeHtml(result.account_id + ' / ' + result.user_id);
    const viewed = pending.name === 'reveal';
    const member = pending.name !== 'account' && roleName(data.role || pending.role) === roleNames.MEMBER;
    const createdAccount = pending.name === 'account';
    $('#tenant-dialog-title').textContent = createdAccount ? '租户已创建' : viewed ? '用户 API Key' : 'API Key 已发放';
    describe(createdAccount ? '用此 Key 以首位管理员身份登录新租户。' : '');
    fields.innerHTML = '<p class="tenant-issued-identity">' + identity + '</p>'
      + '<div class="tenant-credential"><label>服务地址<input readonly value="' + escapeHtml(location.origin) + '"></label>'
      + '<label>API Key<input id="tenant-issued-key" type="password" readonly autocomplete="off"></label>'
      + '<div class="tenant-actions"><button class="primary-button" type="button" id="tenant-copy-key">复制 API Key</button>'
      + '<button class="secondary-button" type="button" id="tenant-copy">复制接入配置</button>'
      + (createdAccount ? '<button class="secondary-button" type="button" id="tenant-open-created">进入租户</button>' : '')
      + '<button class="secondary-button" type="button" id="tenant-reveal" aria-controls="tenant-issued-key" aria-pressed="false">显示 Key</button></div></div>'
      + '<p class="tenant-key-help">' + (member ? '普通用户的 Key 可在「更多」中查看。' : '请保存此 Key。')
      + 'HTTP 请求使用 <code>X-API-Key</code> 请求头。</p>'
      + '<p class="tenant-clipboard-status" role="status"></p>';
    const key = $('#tenant-issued-key');
    key.value = result.user_key;
    $('#tenant-reveal').addEventListener('click', (event) => {
      const reveal = key.type === 'password';
      key.type = reveal ? 'text' : 'password';
      event.currentTarget.textContent = reveal ? '隐藏 Key' : '显示 Key';
      event.currentTarget.setAttribute('aria-pressed', String(reveal));
    });
    $('#tenant-open-created')?.addEventListener('click', () => {
      dialog.close();
      enterTenant(result.account_id);
    });
    const status = fields.querySelector('.tenant-clipboard-status');
    for (const [selector, value, done] of [
      ['#tenant-copy-key', key.value, '已复制 API Key。'],
      ['#tenant-copy', JSON.stringify({ url: location.origin, api_key: key.value }, null, 2), '已复制接入配置。'],
    ]) {
      $(selector).addEventListener('click', async () => {
        try {
          await copyCredential(value);
          setStatus(status, done);
        } catch {
          setStatus(status, '复制失败，请显示 Key 后手动复制。', true);
        }
      });
    }
    cancel.textContent = '完成';
    save.hidden = true;
    $('#tenant-copy-key').focus();
  }

  fields.addEventListener('input', (event) => {
    const field = event.target;
    if (!field.hasAttribute('aria-invalid')) return;
    field.removeAttribute('aria-describedby');
    clearFieldProblem(field, feedback);
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!action || save.disabled || form.hasAttribute('aria-busy')) return;
    feedback.textContent = '';
    const invalid = [...fields.querySelectorAll('input')].find((field) => !field.validity.valid);
    if (invalid) {
      fieldProblem(invalid, '请输入以字母或数字开头的标识，可使用字母、数字、点、下划线和连字符。');
      return;
    }
    const pending = action;
    const data = Object.fromEntries(new FormData(form));
    let url = pending.url;
    if (pending.picker) {
      if (!pending.selection && pending.name !== 'register') return;
      if (pending.name === 'agent') {
        url += '/' + encodeURIComponent(pending.selection.agent_id);
        data.owner_user_id = pending.id;
      } else {
        data.owner_user_id = pending.selection?.user_id || null;
      }
    }
    save.disabled = cancel.disabled = fields.inert = true;
    form.setAttribute('aria-busy', 'true');
    try {
      const result = await api(url, pending.method, data).finally(() => { fields.inert = false; });
      if (action !== pending || !dialog.open) return;
      if (result.updated === false || result.deleted === false) throw new Error('记录已不存在，请刷新目录。');
      if (result.user_key) showCredential(result, pending, data);
      else dialog.close();
      if (pending.name === 'reveal') return;
      if (pending.name === 'account') {
        createdTenant = result.account_id;
        state.tenantDirectoryChanged = true;
        await loadCatalog();
      } else {
        state.accountTrees.clear();
        state.tenantDirectoryChanged = true;
        const affectsCurrentUser = pending.id === state.tenantScope?.user_id;
        if (affectsCurrentUser && ['role', 'delete'].includes(pending.name)) await loadTree();
        else if (!affectsCurrentUser || pending.name !== 'key') await load(page, pending.label + '已完成。');
      }
    } catch (error) {
      if (action !== pending) return;
      const duplicate = error.message.match(/^(account|user|agent) already exists:/)?.[1];
      if (duplicate) {
        const [name, text] = duplicate === 'user' && pending.name === 'account'
          ? ['admin_user_id', '首位管理员标识已存在，请换一个。']
          : duplicateIdentities[duplicate];
        const field = form.elements.namedItem(name);
        if (field && field.value === data[name]) fieldProblem(field, text);
        else feedback.textContent = text;
      } else {
        feedback.textContent = errorText(error);
      }
    } finally {
      save.disabled = cancel.disabled = false;
      form.removeAttribute('aria-busy');
    }
  });

  cancel.addEventListener('click', () => dialog.close());
  dialog.addEventListener('cancel', (event) => { if (form.hasAttribute('aria-busy')) event.preventDefault(); });
  dialog.addEventListener('close', () => {
    disposePicker?.();
    fields.replaceChildren();
    describe();
    feedback.textContent = '';
    const newTenantButton = createdTenant && [...catalogRows.querySelectorAll('[data-enter-tenant]')]
      .find((button) => button.dataset.enterTenant === createdTenant);
    createdTenant = null;
    const restoreFocus = newTenantButton || trigger;
    action = trigger = null;
    (restoreFocus?.isConnected ? restoreFocus : $('#tenant-query')).focus();
  });

  $('#tenant-bootstrap').addEventListener('click', (event) => open('account', event.currentTarget));
  $('#tenant-back').addEventListener('click', () => {
    deploymentDetail = false;
    load();
    $('#tenant-bootstrap').focus();
  });
  for (const [button, step] of [['#tenant-catalog-prev', -1], ['#tenant-catalog-next', 1]]) {
    $(button).addEventListener('click', () => loadCatalog(catalogPage + step));
  }
  async function enterTenant(accountId) {
    deploymentDetail = true;
    state.accountRequestId += 1;
    if (accountId !== state.accountId) clearSearchScopes();
    state.accountId = accountId;
    state.tenantDirectoryChanged = true;
    localStorage.setItem('ogmemory-viewer-account-id', accountId);
    await load();
    $('#tenant-back').focus();
  }

  catalogRows.addEventListener('click', (event) => {
    const button = event.target.closest('[data-enter-tenant]');
    if (button && !catalogRows.inert) enterTenant(button.dataset.enterTenant);
  });
  $('#tenant-create-user').addEventListener('click', (event) => open('user', event.currentTarget));
  $('#tenant-create-agent').addEventListener('click', (event) => open('register', event.currentTarget));
  directory.addEventListener('click', (event) => {
    const tab = event.target.closest('[data-tenant-section]');
    if (tab && tab.dataset.tenantSection !== section) {
      section = tab.dataset.tenantSection;
      directory.querySelectorAll('[data-tenant-section]').forEach((button) => {
        button.setAttribute('aria-pressed', String(button === tab));
      });
      $('#tenant-create-user').hidden = section !== 'users';
      $('#tenant-create-agent').hidden = section !== 'agents';
      query.placeholder = section === 'users' ? '搜索用户标识' : '搜索 Agent 标识';
      resetPages();
      rows.replaceChildren();
      load();
    }
    const button = event.target.closest('[data-tenant-action]');
    if (button) {
      const { id, role } = button.closest('.tenant-row').dataset;
      const menu = button.closest('.tenant-menu');
      if (menu) menu.open = false;
      open(button.dataset.tenantAction, menu?.querySelector('summary') || button, id, role);
    }
  });

  document.addEventListener('click', (event) => {
    const menu = directory.querySelector('.tenant-menu[open]');
    if (menu && !menu.contains(event.target)) menu.open = false;
  });
  directory.addEventListener('toggle', (event) => {
    const menu = event.target;
    if (!menu.matches?.('.tenant-menu[open]')) return;
    menu.classList.remove('tenant-menu-up');
    const panel = menu.querySelector('.tenant-menu-panel');
    menu.classList.toggle('tenant-menu-up', panel.getBoundingClientRect().bottom > innerHeight - 8);
  }, true);
  directory.addEventListener('keydown', (event) => {
    const menu = event.target.closest('.tenant-menu[open]');
    if (event.key !== 'Escape' || !menu) return;
    menu.open = false;
    menu.querySelector('summary').focus();
  });

  query.addEventListener('input', () => { $('#tenant-clear').hidden = !query.value; });
  $('#tenant-search').addEventListener('submit', (event) => {
    event.preventDefault();
    resetPages(query.value.trim());
    load();
  });
  $('#tenant-clear').addEventListener('click', () => { resetPages(); load(); query.focus(); });
  $('#tenant-prev').addEventListener('click', () => load(page - 1));
  $('#tenant-next').addEventListener('click', () => load(page + 1));

  return { load, reset };
})();
