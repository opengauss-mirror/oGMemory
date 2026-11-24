(() => {
  const root = document.querySelector('#knowledge-view');
  if (!root) return;
  const element = (name) => document.getElementById('knowledge-' + name);
  const base = '/api/v1/resources/documents';
  const libraryBase = '/api/v1/knowledge-bases';
  const pageSize = 20;
  const indexStates = {
    pending: '等待解析', indexing: '解析中', ready: '可检索',
    failed: '解析失败', canceled: '解析已停止', unknown: '状态未知',
  };
  const dialog = element('dialog');
  const file = element('file');
  const queryInput = element('query');
  const list = element('list');
  const actions = ['download', 'replace', 'delete'].map(element);
  const librarySelect = element('base-select');
  const libraryDialog = element('base-dialog');
  const libraryForm = element('base-form');
  const libraryName = element('base-name');
  const libraryVisibility = element('base-visibility');
  const membersDialog = element('members-dialog');
  const memberQuery = element('member-query');
  let accountId = '';
  let active = false;
  let generation = 0;
  let page = 1;
  let query = '';
  let selected = null;
  let listRequest = null;
  let detailRequest = null;
  let poll = null;
  let dialogAction = null;
  let dialogTrigger = null;
  let mutating = false;
  let downloading = false;
  let composing = false;
  let names = new Map();
  let library = null;
  let libraryItems = [];
  let libraryPermissions = {};
  let libraryRequest = null;
  let libraryAction = null;
  let libraryTrigger = null;
  let libraryBusy = false;
  let memberSession = null;

  function headers() {
    return { ...requestHeaders(), ...(accountId && { 'X-Account-ID': accountId }) };
  }

  function documentUrl(suffix = '', params = {}) {
    const url = new URL(base + suffix, location.origin);
    url.searchParams.set('knowledge_base_id', library.id);
    for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
    return url;
  }

  function feedback(message, error = false) {
    const region = element('feedback');
    region.textContent = message;
    region.classList.toggle('is-error', error);
  }

  function errorText(error, fallback) {
    if (error.status === 401) return '请在连接设置中更新 API Key。';
    if (error.status === 403) return '当前账户无法访问知识库。请检查账户权限和知识库配置。';
    if (error.status === 404) return '文档已移除。请刷新文档列表。';
    if (error.status === 409) return '文档或知识库配置已变更。请刷新后重试。';
    if (error.status === 413) return '文件过大。请选择 31 MiB 以内的文件。';
    if (error.status === 503) return '知识库暂不可用。请核对服务配置后刷新。';
    if (error.name === 'TimeoutError') return '请求超时。请先刷新确认结果，再重试。';
    return fallback;
  }

  function getJson(path, options = {}) {
    const timeout = AbortSignal.timeout(options.timeout || 12000);
    const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
    return fetchJson(new URL(path, location.origin), { ...options, headers: {
      ...headers(), ...options.headers,
    }, signal });
  }

  function renderLibraryControls() {
    const permissions = library?.permissions || {};
    const create = Boolean(libraryPermissions.create || libraryPermissions.create_shared);
    librarySelect.disabled = Boolean(libraryRequest) || libraryItems.length === 0;
    element('base-create').hidden = !create;
    element('base-create').disabled = Boolean(libraryRequest);
    element('base-edit').hidden = !permissions.manage;
    element('base-delete').hidden = !permissions.manage || library?.is_default;
    element('base-manage-members').hidden = !permissions.manage || library?.visibility !== 'shared';
    for (const action of ['base-edit', 'base-delete', 'base-manage-members']) {
      element(action).disabled = Boolean(libraryRequest);
    }
    element('base-scope').textContent = library
      ? `${library.visibility === 'private' ? '个人' : '账户共享'}${permissions.write ? '' : ' · 只读'}` : '';
    element('upload').hidden = Boolean(library && !permissions.write);
    element('upload').disabled = !permissions.write;
    element('replace').hidden = element('delete').hidden = !permissions.write;
    queryInput.disabled = !permissions.read;
    element('search').querySelector('[type="submit"]').disabled = !permissions.read;
  }

  function renderLibraryOptions() {
    librarySelect.replaceChildren();
    for (const [visibility, label] of [['private', '个人'], ['shared', '账户共享']]) {
      const group = document.createElement('optgroup');
      group.label = label;
      for (const item of libraryItems.filter((item) => item.visibility === visibility)) {
        group.append(new Option(item.name + (item.is_default ? '（默认）' : ''), item.id));
      }
      if (group.childElementCount) librarySelect.append(group);
    }
    if (!libraryItems.length) librarySelect.append(new Option('暂无知识库', ''));
    librarySelect.value = library?.id || '';
  }

  async function loadLibraries({ preferred = library?.id, message = '', focus = false } = {}) {
    libraryRequest?.abort();
    const request = new AbortController();
    libraryRequest = request;
    renderLibraryControls();
    if (!library) feedback('正在读取知识库…');
    try {
      const result = await getJson(libraryBase, { signal: request.signal });
      if (request.signal.aborted || request !== libraryRequest || !active) return;
      libraryItems = result.items;
      libraryPermissions = result.permissions || {};
      const next = libraryItems.find((item) => item.id === preferred)
        || libraryItems.find((item) => item.id === result.default_knowledge_base_id)
        || libraryItems[0] || null;
      if (next?.id !== library?.id) {
        closeMembers();
        reset();
      } else if (!next?.permissions?.manage) closeMembers();
      library = next;
      renderLibraryOptions();
    } catch (error) {
      if (request.signal.aborted || request !== libraryRequest || !active) return;
      reset();
      library = null;
      libraryItems = [];
      libraryPermissions = {};
      renderLibraryOptions();
      renderEmpty('暂时无法显示知识库。可使用顶部的刷新按钮。');
      feedback(errorText(error, '读取知识库失败，请刷新后重试。'), true);
      return;
    } finally {
      if (request === libraryRequest) {
        libraryRequest = null;
        renderLibraryControls();
      }
    }
    if (focus) (libraryItems.length ? librarySelect : element('base-create')).focus();
    loadList({ message });
    if (selected) loadStatus(Date.now() + 120000);
  }

  function clearDetail() {
    detailRequest?.abort();
    detailRequest = null;
    clearTimeout(poll);
    selected = null;
    downloading = false;
    element('detail').hidden = true;
    element('detail-placeholder').hidden = false;
    element('evidence').hidden = true;
    element('excerpt').textContent = '';
    element('document-title').textContent = '';
    element('document-uri').textContent = '';
    element('document-type').textContent = '';
    element('document-state').textContent = '';
    element('document-chunks').textContent = '';
    updateSelection();
  }

  function updateSelection() {
    for (const row of list.querySelectorAll('[data-document-key]')) {
      if (row.dataset.documentKey === selected?.key) row.setAttribute('aria-current', 'true');
      else row.removeAttribute('aria-current');
    }
  }

  function renderRows(rows, searching) {
    list.replaceChildren();
    for (const [index, record] of rows.entries()) {
      const id = searching ? resourceId(record.uri) : record.id;
      if (!id) continue;
      const key = searching ? id + ':' + index : id;
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'knowledge-row';
      row.dataset.documentKey = key;
      const name = document.createElement('span');
      name.className = 'knowledge-row-name';
      name.textContent = searching ? (names.get(id) || '文档片段') : record.filename;
      const metadata = document.createElement('span');
      metadata.className = 'knowledge-row-meta';
      metadata.textContent = searching ? record.content : record.media_type;
      row.append(name, metadata);
      row.addEventListener('click', () => openDocument({
        id, key, uri: record.uri, filename: names.get(id) || record.filename,
        media_type: record.media_type, excerpt: searching ? record.content : '',
      }, true));
      const item = document.createElement('li');
      item.append(row);
      list.append(item);
    }
    updateSelection();
  }

  function resourceId(uri) {
    const match = /^ctx:\/\/[^/]+\/resources\/doc-([0-9a-f]{32})$/i.exec(uri || '');
    return match?.[1].replace(/^(.{8})(.{4})(.{4})(.{4})(.{12})$/, '$1-$2-$3-$4-$5');
  }

  function renderEmpty(message) {
    element('empty').hidden = !message;
    element('empty').textContent = message;
  }

  async function loadList({ message = '' } = {}) {
    listRequest?.abort();
    const request = new AbortController();
    listRequest = request;
    const current = generation;
    const searching = Boolean(query);
    element('list-title').textContent = searching ? '检索结果' : '文档';
    element('count').textContent = '';
    element('previous').disabled = true;
    element('next').disabled = true;
    element('pagination').hidden = searching;
    list.replaceChildren();
    renderEmpty('');
    if (!library?.permissions?.read) {
      listRequest = null;
      list.removeAttribute('aria-busy');
      renderEmpty(state.apiKey ? '当前没有可查看的知识库。' : '请先在连接设置中连接账户。');
      feedback(message);
      return;
    }
    list.setAttribute('aria-busy', 'true');
    feedback(searching ? '正在检索知识库…' : '正在读取文档…');
    try {
      const payload = searching ? await getJson(documentUrl('/search'), {
        method: 'POST', body: JSON.stringify({ query, limit: pageSize }),
        headers: { 'Content-Type': 'application/json' }, signal: request.signal,
      }) : await getJson(documentUrl('', { page, page_size: pageSize }), {
        signal: request.signal,
      });
      if (request.signal.aborted || current !== generation) return;
      const rows = searching ? payload.hits : payload.items;
      const total = searching ? rows.length : payload.total;
      if (!searching) {
        const pages = Math.max(1, Math.ceil(total / pageSize));
        if (page > pages) {
          page = pages;
          return loadList({ message });
        }
        names = new Map(rows.map((item) => [item.id, item.filename]));
        element('page').textContent = '第 ' + page + ' / ' + pages + ' 页';
        element('pagination').hidden = pages === 1;
        element('previous').disabled = page <= 1;
        element('next').disabled = page >= pages;
      }
      element('count').textContent = total + (searching ? ' 个片段' : ' 份文档');
      renderRows(rows, searching);
      if (!list.childElementCount) {
        renderEmpty(searching ? '暂无相关片段。可以换个关键词再试。'
          : library.permissions.write ? '暂无文档。上传文档后即可开始检索。' : '此知识库暂无文档。');
      }
      feedback(message);
    } catch (error) {
      if (request.signal.aborted || current !== generation) return;
      renderEmpty('暂时无法显示文档。可使用顶部的刷新按钮。');
      feedback(errorText(error, '读取知识库失败，请刷新后重试。'), true);
    } finally {
      if (listRequest === request) {
        listRequest = null;
        list.removeAttribute('aria-busy');
      }
    }
  }

  function openDocument(record, focus = false) {
    clearDetail();
    selected = record;
    element('detail-placeholder').hidden = true;
    element('detail').hidden = false;
    element('document-title').textContent = record.filename || '文档详情';
    element('document-uri').textContent = record.uri;
    element('document-type').textContent = record.media_type || '—';
    element('document-state').textContent = '查询中…';
    element('document-chunks').textContent = '—';
    element('evidence').hidden = !record.excerpt;
    element('excerpt').textContent = record.excerpt || '';
    actions.forEach((action) => { action.disabled = false; });
    updateSelection();
    if (focus && window.matchMedia('(max-width: 760px)').matches) {
      element('document-title').focus();
    }
    loadStatus(Date.now() + 120000);
  }

  async function loadStatus(deadline) {
    if (!selected) return;
    clearTimeout(poll);
    detailRequest?.abort();
    const request = new AbortController();
    detailRequest = request;
    const record = selected;
    element('status-refresh').disabled = true;
    try {
      const info = await getJson(documentUrl('/' + record.id), { signal: request.signal });
      if (request.signal.aborted || record !== selected) return;
      Object.assign(record, {
        filename: info.filename, uri: info.uri, media_type: info.media_type,
      });
      names.set(record.id, record.filename);
      element('document-title').textContent = record.filename;
      element('document-uri').textContent = record.uri;
      element('document-type').textContent = record.media_type;
      const status = element('document-state');
      const label = indexStates[info.index_state] || indexStates.unknown;
      if (status.textContent !== label) status.textContent = label;
      element('document-chunks').textContent = String(info.chunk_count);
      if (['pending', 'indexing'].includes(info.index_state) && Date.now() < deadline && active) {
        poll = setTimeout(() => loadStatus(deadline), 2500);
      }
    } catch (error) {
      if (request.signal.aborted || record !== selected) return;
      element('document-state').textContent = '无法读取';
      if (error.status === 404 || error.status === 403 || error.status === 401) {
        actions.forEach((action) => { action.disabled = true; });
      }
      feedback(errorText(error, '无法读取解析状态。请重试。'), true);
    } finally {
      if (detailRequest === request) {
        detailRequest = null;
        element('status-refresh').disabled = false;
      }
    }
  }

  function openDialog(action, trigger) {
    if (!library?.permissions?.write) return;
    const record = selected;
    if (action !== 'upload' && !record) return;
    dialogTrigger = trigger;
    dialogAction = { action, record };
    file.value = '';
    file.removeAttribute('aria-invalid');
    element('dialog-error').textContent = '';
    const deleting = action === 'delete';
    const titles = { upload: '上传文档', replace: '替换文档', delete: '删除文档' };
    const description = {
      upload: '选择本地文件。解析完成后可从知识库检索。',
      replace: '将「' + (record?.filename || '当前文档') + '」替换为新文件。文档地址保持不变。',
      delete: '将从知识库「' + library.name + '」删除「' + (record?.filename || '当前文档') + '」及其索引片段。',
    };
    element('dialog-title').textContent = titles[action];
    element('dialog-description').textContent = description[action];
    element('dialog-submit').textContent = titles[action];
    element('dialog-submit').classList.toggle('space-danger-button', deleting);
    element('file-field').hidden = deleting;
    dialog.showModal();
    (deleting ? element('dialog-cancel') : file).focus();
  }

  async function submitDialog(event) {
    event.preventDefault();
    if (!dialogAction || mutating) return;
    const pending = dialogAction;
    const { action, record } = pending;
    const errorRegion = element('dialog-error');
    errorRegion.textContent = '';
    const chosen = file.files[0];
    if (action !== 'delete' && (!chosen || !chosen.size || chosen.size > 31 * 1024 * 1024)) {
      errorRegion.textContent = '请选择一个 31 MiB 以内的非空文件。';
      file.setAttribute('aria-invalid', 'true');
      file.focus();
      return;
    }
    const current = generation;
    const submit = element('dialog-submit');
    mutating = true;
    submit.disabled = true;
    element('dialog-cancel').disabled = true;
    file.disabled = true;
    element('form').setAttribute('aria-busy', 'true');
    submit.textContent = '处理中…';
    try {
      const path = documentUrl(record && action !== 'upload' ? '/' + record.id : '');
      let result;
      if (action === 'delete') {
        const response = await fetch(path, {
          method: 'DELETE', headers: headers(), signal: AbortSignal.timeout(45000),
        });
        if (!response.ok) throw Object.assign(new Error(), { status: response.status });
      } else {
        const body = new FormData();
        body.append('file', chosen);
        result = await getJson(path, { method: action === 'upload' ? 'POST' : 'PUT', body, timeout: 45000 });
      }
      if (current !== generation) return;
      if (action !== 'delete') {
        query = '';
        queryInput.value = '';
        element('clear').hidden = true;
        if (action === 'upload') page = 1;
        openDocument({ ...result, key: result.id });
      } else clearDetail();
      dialog.close();
      const message = action === 'delete' ? '文档已删除。'
        : result.cleanup_pending ? '已提交替换。旧版文件等待清理。'
        : action === 'replace' ? '已提交替换。' : '已上传并提交解析。';
      loadList({ message });
    } catch (error) {
      if (current === generation) {
        errorRegion.textContent = errorText(error, '操作未完成，请刷新确认状态后重试。');
      }
    } finally {
      if (dialogAction === pending) {
        mutating = false;
        submit.disabled = false;
        element('dialog-cancel').disabled = false;
        file.disabled = false;
        element('form').removeAttribute('aria-busy');
        submit.textContent = element('dialog-title').textContent;
      }
    }
  }

  async function download() {
    if (!selected || downloading || !library?.permissions?.read) return;
    const record = selected;
    const current = generation;
    const button = element('download');
    downloading = true;
    button.disabled = true;
    try {
      const response = await fetch(documentUrl('/' + record.id + '/content'), {
        headers: headers(), signal: AbortSignal.timeout(60000),
      });
      if (!response.ok) throw Object.assign(new Error(), { status: response.status });
      const blob = await response.blob();
      if (current !== generation || record !== selected || !active) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = record.filename || 'document';
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      feedback('已开始下载原文。');
    } catch (error) {
      if (current === generation && record === selected) {
        feedback(errorText(error, '下载失败，请重试。'), true);
      }
    } finally {
      if (current === generation && record === selected) {
        downloading = false;
        button.disabled = false;
      }
    }
  }

  function openLibraryDialog(action, trigger) {
    if (trigger.disabled || trigger.hidden) return;
    libraryAction = { action, item: library };
    libraryTrigger = trigger;
    const creating = action === 'create';
    const deleting = action === 'delete';
    const titles = { create: '新建知识库', edit: '重命名知识库', delete: '删除空库' };
    const submitLabels = { create: '新建知识库', edit: '保存名称', delete: '删除知识库' };
    element('base-dialog-title').textContent = titles[action];
    element('base-dialog-description').textContent = deleting
      ? `将删除知识库「${library.name}」。仅可删除空知识库。`
      : creating ? '选择知识库的可见范围。' : `当前名称为「${library.name}」。`;
    element('base-dialog-error').textContent = '';
    element('base-dialog-submit').textContent = submitLabels[action];
    element('base-dialog-submit').classList.toggle('space-danger-button', deleting);
    element('base-name-field').hidden = deleting;
    element('base-visibility-field').hidden = !creating;
    libraryName.value = creating ? '' : library.name;
    libraryName.removeAttribute('aria-invalid');
    libraryVisibility.replaceChildren();
    if (libraryPermissions.create_private) libraryVisibility.append(new Option('个人', 'private'));
    if (libraryPermissions.create_shared) libraryVisibility.append(new Option('账户共享', 'shared'));
    libraryDialog.showModal();
    (deleting ? element('base-dialog-cancel') : libraryName).focus();
  }

  function libraryFormValues(action) {
    const error = element('base-dialog-error');
    const invalid = (field, message) => {
      field.setAttribute('aria-invalid', 'true');
      error.textContent = message;
      field.focus();
      return null;
    };
    const name = libraryName.value.trim();
    if (!name) return invalid(libraryName, '请输入知识库名称。');
    const body = { name };
    if (action === 'create') body.visibility = libraryVisibility.value;
    return body;
  }

  function setLibraryBusy(busy) {
    libraryBusy = busy;
    for (const control of libraryForm.elements) {
      control.disabled = busy;
    }
    if (busy) libraryForm.setAttribute('aria-busy', 'true');
    else libraryForm.removeAttribute('aria-busy');
  }

  async function submitLibraryDialog(event) {
    event.preventDefault();
    if (!libraryAction || libraryBusy) return;
    const pending = libraryAction;
    const { action, item } = pending;
    const errorRegion = element('base-dialog-error');
    errorRegion.textContent = '';
    const deleting = action === 'delete';
    const body = deleting ? null : libraryFormValues(action);
    if (!deleting && !body) return;
    const currentAccount = accountId;
    const submit = element('base-dialog-submit');
    const idleLabel = submit.textContent;
    setLibraryBusy(true);
    submit.textContent = '处理中…';
    const current = () => libraryAction === pending && currentAccount === accountId && active;
    try {
      let result;
      if (deleting) {
        const response = await fetch(`${libraryBase}/${item.id}`, {
          method: 'DELETE', headers: headers(), signal: AbortSignal.timeout(45000),
        });
        if (!response.ok) throw Object.assign(new Error(), { status: response.status });
      } else {
        const creating = action === 'create';
        result = await getJson(creating ? libraryBase : `${libraryBase}/${item.id}`, {
          method: creating ? 'POST' : 'PATCH', body: JSON.stringify(body),
          headers: { 'Content-Type': 'application/json' }, timeout: 45000,
        });
      }
      if (!current()) return;
      libraryDialog.close();
      if (deleting) {
        library = null;
        reset();
      }
      const messages = { create: '知识库已新建。', edit: '知识库名称已更新。', delete: '知识库已删除。' };
      loadLibraries({ preferred: result?.id, message: messages[action], focus: true });
    } catch (error) {
      if (!current()) return;
      errorRegion.textContent = deleting && error.status === 409
        ? '此知识库仍有文档。请先清理文档，再删除知识库。'
        : error.status === 404 || error.status === 409 ? '知识库已变更，请刷新后重试。'
          : error.status === 400 ? '请输入有效的知识库名称。'
            : errorText(error, '知识库操作未完成，请重试。');
    } finally {
      if (current()) {
        setLibraryBusy(false);
        submit.textContent = idleLabel;
      }
    }
  }

  function memberPath(session, user) {
    const path = `${libraryBase}/${session.id}/members`;
    return user ? `${path}/${encodeURIComponent(user)}` : path;
  }

  function memberFeedback(message, error = false) {
    const region = element('member-feedback');
    region.textContent = message;
    region.classList.toggle('is-error', error);
  }

  function chooseMember(user = '') {
    if (!memberSession) return;
    memberSession.selected = user;
    element('member-selected').textContent = user;
    element('member-grant').hidden = !user;
    for (const button of element('member-results').querySelectorAll('button')) {
      button.setAttribute('aria-pressed', String(button.dataset.user === user));
    }
    if (user) {
      element('member-permission').value = 'read';
      element('member-permission').focus();
    }
  }

  function memberButton(label, destructive = false) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = destructive ? 'secondary-button knowledge-base-danger' : 'secondary-button';
    button.textContent = label;
    return button;
  }

  function renderGrantedMembers(session, members) {
    const version = JSON.stringify(members);
    if (version === session.membersVersion) return;
    session.membersVersion = version;
    const list = element('member-list');
    list.replaceChildren();
    element('member-count').textContent = `${members.length} 位`;
    element('member-empty').hidden = Boolean(members.length);
    for (const { user_id: user, permission: saved } of members) {
      const row = document.createElement('li');
      row.className = 'knowledge-member-row';
      const name = document.createElement('strong');
      name.textContent = user;
      const controls = document.createElement('div');
      controls.className = 'knowledge-member-row-actions';
      const permission = document.createElement('select');
      permission.dataset.user = user;
      permission.setAttribute('aria-label', `${user} 的权限`);
      permission.append(new Option('只读', 'read'), new Option('可编辑', 'write'));
      permission.value = saved;
      const save = memberButton('保存权限');
      save.disabled = true;
      save.setAttribute('aria-label', `保存 ${user} 的权限`);
      const revoke = memberButton('撤销', true);
      revoke.setAttribute('aria-label', `撤销 ${user} 的权限`);
      permission.addEventListener('change', () => { save.disabled = permission.value === saved; });
      save.addEventListener('click', () => changeMember(session, user, permission.value, '权限已更新。', 'role'));
      revoke.addEventListener('click', () => {
        session.revoke = user;
        session.revokeTrigger = revoke;
        element('member-confirm-label').textContent = `撤销「${user}」的知识库授权？`;
        element('member-confirm').hidden = false;
        element('member-confirm-cancel').focus();
      });
      controls.append(permission, save, revoke);
      row.append(name, controls);
      list.append(row);
    }
  }

  function renderMembers(session, payload) {
    renderGrantedMembers(session, payload.members);
    const results = element('member-results');
    results.replaceChildren();
    for (const { user_id: user } of payload.candidates) {
      const row = document.createElement('li');
      const select = document.createElement('button');
      select.type = 'button';
      select.className = 'knowledge-member-candidate';
      const name = document.createElement('strong');
      name.textContent = user;
      const arrow = document.createElement('span');
      arrow.setAttribute('aria-hidden', 'true');
      arrow.textContent = '›';
      select.dataset.user = user;
      select.setAttribute('aria-label', `选择 ${user}`);
      select.setAttribute('aria-pressed', String(user === session.selected));
      select.addEventListener('click', () => chooseMember(user));
      select.append(name, arrow);
      row.append(select);
      results.append(row);
    }
    const empty = element('member-search-empty');
    empty.textContent = !payload.candidates.length
      ? session.query ? '没有符合该关键词的可授权成员。' : '当前账户暂无可授权成员。'
      : payload.candidate_count > payload.candidates.length ? `显示前 ${payload.candidates.length} 位，可缩小搜索范围。` : '';
    empty.hidden = !empty.textContent;
    if (session.selected && !payload.candidates.some(({ user_id }) => user_id === session.selected)) {
      chooseMember();
    }
  }

  async function loadMembers(session, message = '') {
    if (session !== memberSession) return;
    session.request?.abort();
    const request = new AbortController();
    session.request = request;
    const url = new URL(memberPath(session), location.origin);
    if (session.query) url.searchParams.set('q', session.query);
    memberFeedback(message || '正在读取成员…');
    try {
      const payload = await getJson(url, { signal: request.signal });
      if (session !== memberSession || request.signal.aborted) return;
      renderMembers(session, payload);
      memberFeedback(message);
    } catch (error) {
      if (session === memberSession && !request.signal.aborted) {
        memberFeedback(errorText(error, '成员暂时无法读取，请重新查找。'), true);
      }
    }
  }

  function hideRevoke(session, restoreFocus = false) {
    if (restoreFocus) session.revokeTrigger?.focus();
    session.revoke = null;
    session.revokeTrigger = null;
    element('member-confirm').hidden = true;
  }

  async function changeMember(session, user, permission, message, focus = '') {
    if (session !== memberSession || session.mutating) return;
    session.mutating = true;
    clearTimeout(session.timer);
    session.request?.abort();
    const controls = [...membersDialog.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled)')];
    controls.forEach((control) => { control.disabled = true; });
    memberFeedback('正在保存权限…');
    try {
      if (permission) {
        await getJson(memberPath(session, user), {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ permission }),
        });
      } else {
        const response = await fetch(memberPath(session, user), {
          method: 'DELETE', headers: headers(), signal: AbortSignal.timeout(12000),
        });
        if (!response.ok) throw Object.assign(new Error(), { status: response.status });
      }
      if (session !== memberSession) return;
      hideRevoke(session);
      chooseMember();
      await loadMembers(session, message);
    } catch (error) {
      if (session !== memberSession) return;
      const explanation = error.status === 404 ? '知识库或成员已变更，请关闭后重新打开成员管理。'
        : errorText(error, '权限未确认。可重新查找核对后重试。');
      memberFeedback(explanation, true);
    } finally {
      if (session === memberSession) {
        session.mutating = false;
        controls.forEach((control) => { control.disabled = false; });
        const role = [...element('member-list').querySelectorAll('select')].find((control) => control.dataset.user === user);
        (focus === 'role' && role ? role : memberQuery).focus();
      }
    }
  }

  function openMembers(event) {
    if (!library?.permissions?.manage || library.visibility !== 'shared') return;
    memberSession = { id: library.id, trigger: event.currentTarget, query: '', selected: '' };
    element('members-context').textContent = library.name;
    element('member-count').textContent = '';
    element('member-list').replaceChildren();
    element('member-results').replaceChildren();
    for (const name of ['member-empty', 'member-search-empty', 'member-grant', 'member-confirm', 'member-clear']) {
      element(name).hidden = true;
    }
    memberQuery.value = '';
    membersDialog.showModal();
    memberQuery.focus();
    loadMembers(memberSession);
  }

  function closeMembers(restoreFocus = false) {
    const session = memberSession;
    if (!session) return;
    clearTimeout(session.timer);
    session.request?.abort();
    memberSession = null;
    if (membersDialog.open) membersDialog.close();
    for (const control of membersDialog.querySelectorAll(':disabled')) control.disabled = false;
    if (restoreFocus && !session.trigger.hidden && !session.trigger.disabled) session.trigger.focus();
  }

  function searchMembers(delay = 0) {
    const session = memberSession;
    if (!session || session.mutating || session.composing) return;
    clearTimeout(session.timer);
    session.request?.abort();
    chooseMember();
    element('member-results').replaceChildren();
    element('member-search-empty').hidden = true;
    session.query = memberQuery.value.trim();
    if (delay) session.timer = setTimeout(() => loadMembers(session), delay);
    else loadMembers(session);
  }

  function reset() {
    generation += 1;
    listRequest?.abort();
    listRequest = null;
    list.removeAttribute('aria-busy');
    page = 1;
    query = '';
    composing = false;
    queryInput.value = '';
    element('clear').hidden = true;
    names.clear();
    list.replaceChildren();
    element('count').textContent = '';
    element('pagination').hidden = true;
    renderEmpty('');
    clearDetail();
    feedback('');
    dialogAction = null;
    dialogTrigger = null;
    file.value = '';
    file.disabled = false;
    mutating = false;
    element('dialog-submit').disabled = false;
    element('dialog-cancel').disabled = false;
    element('form').removeAttribute('aria-busy');
    if (dialog.open) dialog.close();
  }

  function stopLibraries() {
    libraryRequest?.abort();
    libraryRequest = null;
    if (libraryDialog.open) libraryDialog.close();
    libraryAction = null;
    libraryTrigger = null;
    setLibraryBusy(false);
    closeMembers();
    reset();
  }

  function clearLibrary() {
    library = null;
    libraryItems = [];
    libraryPermissions = {};
    stopLibraries();
    renderLibraryOptions();
    renderLibraryControls();
  }

  librarySelect.addEventListener('change', () => {
    const next = libraryItems.find((item) => item.id === librarySelect.value);
    if (!next || next.id === library?.id) return;
    closeMembers();
    reset();
    library = next;
    renderLibraryControls();
    loadList();
  });
  for (const action of ['create', 'edit', 'delete']) {
    element('base-' + action).addEventListener('click', (event) => openLibraryDialog(action, event.currentTarget));
  }
  element('base-manage-members').addEventListener('click', openMembers);
  element('members-close').addEventListener('click', () => closeMembers(true));
  membersDialog.addEventListener('close', () => { if (!membersDialog.open) closeMembers(true); });
  membersDialog.addEventListener('cancel', (event) => { if (memberSession?.mutating) event.preventDefault(); });
  memberQuery.addEventListener('compositionstart', () => { if (memberSession) memberSession.composing = true; });
  memberQuery.addEventListener('compositionend', () => {
    if (memberSession) memberSession.composing = false;
    searchMembers(300);
  });
  memberQuery.addEventListener('input', () => {
    element('member-clear').hidden = !memberQuery.value;
    searchMembers(300);
  });
  element('member-search').addEventListener('submit', (event) => {
    event.preventDefault();
    searchMembers();
  });
  element('member-clear').addEventListener('click', () => {
    memberQuery.value = '';
    element('member-clear').hidden = true;
    memberQuery.focus();
    searchMembers();
  });
  element('member-grant').addEventListener('submit', (event) => {
    event.preventDefault();
    if (memberSession?.selected) changeMember(memberSession, memberSession.selected, element('member-permission').value, '成员权限已授予。');
  });
  element('member-confirm-cancel').addEventListener('click', () => { if (memberSession) hideRevoke(memberSession, true); });
  element('member-confirm-submit').addEventListener('click', () => {
    if (memberSession?.revoke) changeMember(memberSession, memberSession.revoke, '', '成员权限已撤销。');
  });
  element('base-dialog-cancel').addEventListener('click', () => libraryDialog.close());
  libraryForm.addEventListener('submit', submitLibraryDialog);
  libraryForm.addEventListener('input', (event) => {
    event.target.removeAttribute('aria-invalid');
    element('base-dialog-error').textContent = '';
  });
  libraryDialog.addEventListener('cancel', (event) => { if (libraryBusy) event.preventDefault(); });
  libraryDialog.addEventListener('close', () => {
    if (libraryTrigger?.isConnected && !libraryTrigger.hidden && !libraryTrigger.disabled) libraryTrigger.focus();
    else if (active && !librarySelect.disabled) librarySelect.focus();
    libraryAction = null;
    libraryTrigger = null;
    setLibraryBusy(false);
  });

  element('search').addEventListener('submit', (event) => {
    event.preventDefault();
    if (composing || !library?.permissions?.read) return;
    query = queryInput.value.trim();
    page = 1;
    clearDetail();
    loadList();
  });
  queryInput.addEventListener('compositionstart', () => { composing = true; });
  queryInput.addEventListener('compositionend', () => { composing = false; });
  queryInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.isComposing || composing)) event.preventDefault();
  });
  queryInput.addEventListener('input', () => { element('clear').hidden = !queryInput.value; });
  element('clear').addEventListener('click', () => {
    queryInput.value = '';
    queryInput.focus();
    element('clear').hidden = true;
    query = '';
    page = 1;
    clearDetail();
    loadList();
  });
  for (const [name, delta] of [['previous', -1], ['next', 1]]) {
    element(name).addEventListener('click', () => {
      page += delta;
      loadList();
    });
  }
  for (const action of ['upload', 'replace', 'delete']) {
    element(action).addEventListener('click', (event) => openDialog(action, event.currentTarget));
  }
  element('status-refresh').addEventListener('click', () => loadStatus(Date.now() + 120000));
  element('download').addEventListener('click', download);
  element('dialog-cancel').addEventListener('click', () => dialog.close());
  element('form').addEventListener('submit', submitDialog);
  file.addEventListener('change', () => {
    file.removeAttribute('aria-invalid');
    element('dialog-error').textContent = '';
  });
  dialog.addEventListener('cancel', (event) => { if (mutating) event.preventDefault(); });
  dialog.addEventListener('close', () => {
    if (dialogTrigger?.isConnected && !dialogTrigger.closest('[hidden]')) dialogTrigger.focus();
    else if (active) element('upload').focus();
    dialogAction = null;
    dialogTrigger = null;
    file.value = '';
  });

  window.knowledgeLibrary = {
    reset() { accountId = ''; active = false; clearLibrary(); },
    show(account) {
      if (account !== accountId) clearLibrary();
      accountId = account;
      active = true;
      loadLibraries();
    },
    leave() {
      active = false;
      stopLibraries();
    },
    connection(message, error = false) {
      clearLibrary();
      feedback(message, error);
    },
  };
})();
