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
  const queryInput = element('query');
  const list = element('list');
  const downloadButton = element('download');
  const librarySelect = element('base-select');
  let accountId = '';
  let active = false;
  let generation = 0;
  let page = 1;
  let query = '';
  let selected = null;
  let listRequest = null;
  let detailRequest = null;
  let poll = null;
  let downloading = false;
  let composing = false;
  let names = new Map();
  let library = null;
  let libraryItems = [];
  let libraryRequest = null;

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
    librarySelect.disabled = Boolean(libraryRequest) || libraryItems.length === 0;
    element('base-scope').textContent = library
      ? `${library.visibility === 'private' ? '个人' : '账户共享'}${permissions.write ? '' : ' · 只读'}` : '';
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

  async function loadLibraries() {
    const preferred = library?.id;
    libraryRequest?.abort();
    const request = new AbortController();
    libraryRequest = request;
    renderLibraryControls();
    if (!library) feedback('正在读取知识库…');
    try {
      const result = await getJson(libraryBase, { signal: request.signal });
      if (request.signal.aborted || request !== libraryRequest || !active) return;
      libraryItems = result.items;
      const next = libraryItems.find((item) => item.id === preferred)
        || libraryItems.find((item) => item.id === result.default_knowledge_base_id)
        || libraryItems[0] || null;
      if (next?.id !== library?.id) {
        reset();
      }
      library = next;
      renderLibraryOptions();
    } catch (error) {
      if (request.signal.aborted || request !== libraryRequest || !active) return;
      reset();
      library = null;
      libraryItems = [];
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
    loadList();
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

  async function loadList() {
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
      feedback('');
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
          return loadList();
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
          : '此知识库暂无文档。');
      }
      feedback('');
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
    downloadButton.disabled = false;
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
        downloadButton.disabled = true;
      }
      feedback(errorText(error, '无法读取解析状态。请重试。'), true);
    } finally {
      if (detailRequest === request) {
        detailRequest = null;
        element('status-refresh').disabled = false;
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
  }

  function stopLibraries() {
    libraryRequest?.abort();
    libraryRequest = null;
    reset();
  }

  function clearLibrary() {
    library = null;
    libraryItems = [];
    stopLibraries();
    renderLibraryOptions();
    renderLibraryControls();
  }

  librarySelect.addEventListener('change', () => {
    const next = libraryItems.find((item) => item.id === librarySelect.value);
    if (!next || next.id === library?.id) return;
    reset();
    library = next;
    renderLibraryControls();
    loadList();
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
  element('status-refresh').addEventListener('click', () => loadStatus(Date.now() + 120000));
  element('download').addEventListener('click', download);
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
