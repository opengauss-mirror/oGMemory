"use strict";

(() => {
  const API = "/api/v1/bot";
  const embedded = new URLSearchParams(location.search).get("embedded") === "1" && window.parent !== window;
  document.documentElement.classList.toggle("embedded", embedded);
  if (embedded) document.documentElement.dataset.theme = localStorage.getItem("ogmemory-viewer-theme") || "light";
  const $ = (selector, root = document) => root.querySelector(selector);
  const labels = {
    queued: "已接收", running: "执行中", retry_wait: "等待重试", waiting_confirmation: "待核对",
    succeeded: "已完成", cancelled: "已停止", failed: "执行失败", processing: "处理中",
    ready: "可用", active: "已启用", paused: "已暂停", completed: "已结束", pending: "待处理",
    stored: "已保存，建立检索中",
    searchable: "记忆可用于后续问答", unchanged: "本次记忆无变化", not_requested: "未触发记忆整理",
  };
  const pageTitles = {chat: "对话", files: "资料与成果", plans: "计划", memories: "记忆"};
  const memoryCategoryLabels = {
    profile: "用户画像", preference: "偏好", entity: "实体", event: "事件", case: "案例",
    pattern: "模式", tool: "工具记忆", tool_memory: "工具记忆", tool_evidence: "工具证据",
    experience: "经验", session_summary: "会话摘要", session_archive: "会话归档", document: "资料", skill: "技能",
  };
  const terminal = new Set(["succeeded", "cancelled", "failed"]);
  const state = {
    key: sessionStorage.getItem(embedded ? "ogmemory-viewer-api-key" : "memorybot.apiKey") || "", page: "chat", config: {}, session: null,
    sessions: [], files: [], plans: [], memories: [], templates: [], agents: [], draftAgentId: "", cursors: {}, collectionRequests: {},
    tasks: new Map(), drafts: new Map(), progress: new Map(), streams: new Map(),
    attachments: new Map(), historyCursor: null, previewTask: null, sendRequest: null, sessionRequest: null,
    draftTitle: "", draftScope: null, sending: false, pendingSend: null, sessionListRequest: 0,
    workspaceVersion: 0, connectionVersion: 0, renderScheduled: false, followBottom: true,
  };
  const iconPaths = {
    plus: "M12 5v14M5 12h14", close: "m6 6 12 12M6 18 18 6", up: "M12 19V5m-6 6 6-6 6 6",
    down: "M12 5v14m-6-6 6 6 6-6", stop: "M6 6h12v12H6z",
    chat: "M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z",
    grid: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
    send: "m22 2-7 20-4-9-9-4 20-7ZM22 2 11 13",
    search: "M21 21l-4.3-4.3M19 11a8 8 0 1 1-16 0 8 8 0 0 1 16 0Z",
    refresh: "M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 11.5-2L20 8M4 16l2.5 3A7 7 0 0 0 18 17",
    settings: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2",
    panel: "M3 3h18v18H3zM9 3v18", more: "M5 12h.01M12 12h.01M19 12h.01",
    file: "M14 2H5v20h14V7l-5-5Zm0 0v6h5M8 12h8m-8 4h8",
    clock: "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0ZM12 6v6l4 2",
    memory: "M12 3a4 4 0 0 0-7 3 4 4 0 0 0-2 7 4 4 0 0 0 4 6 3 3 0 0 0 5 1V3Zm0 0a4 4 0 0 1 7 3 4 4 0 0 1 2 7 4 4 0 0 1-4 6 3 3 0 0 1-5 1M7 7l2 2m8-2-2 2M7 15l2-2m8 2-2-2",
    bot: "M7 6h10a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V9a3 3 0 0 1 3-3ZM12 3v3M8 11v2m8-2v2M9 17h6M1 10v6m22-6v6",
    sliders: "M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6m2-6h6m2 8h6",
    copy: "M8 8h13v13H8zM16 8V3H3v13h5", archive: "M3 3h18v5H3zM5 8v13h14V8M10 12h4",
    edit: "m15 5 4 4M4 20l4-1L20 7a2.8 2.8 0 0 0-4-4L4 15l-1 6Z",
  };
  function icon(name) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    node.setAttribute("viewBox", "0 0 24 24"); node.setAttribute("fill", "none");
    node.setAttribute("stroke", "currentColor"); node.setAttribute("stroke-width", name === "more" ? "3.5" : "1.7");
    node.setAttribute("stroke-linecap", "round"); node.setAttribute("stroke-linejoin", "round");
    node.setAttribute("aria-hidden", "true"); node.classList.add("icon");
    const path = document.createElementNS(node.namespaceURI, "path");
    path.setAttribute("d", iconPaths[name] || iconPaths.chat); node.append(path); return node;
  }
  function iconButton(name, label, action, value, className = "icon-button") {
    const node = button("", action, value, className);
    node.append(icon(name)); node.setAttribute("aria-label", label); node.title = label; return node;
  }
  function setSidebar(open) {
    const sidebar = $("#session-sidebar"), backdrop = $("#sidebar-backdrop");
    const toggle = $('[data-action="open-sidebar"]');
    const restoreFocus = sidebar.contains(document.activeElement) || document.activeElement === backdrop;
    document.body.classList.toggle("sidebar-open", open);
    backdrop.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) $('[data-action="close-sidebar"]', sidebar).focus({preventScroll: true});
    else if (restoreFocus && toggle.getClientRects().length) toggle.focus({preventScroll: true});
  }
  function renderMarkdown(node, text) {
    if (node.dataset.text === text) return;
    const selection = window.getSelection();
    if (selection && !selection.isCollapsed && node.contains(selection.anchorNode)) return;
    node.dataset.text = text;
    if (!window.marked || !window.DOMPurify) { node.textContent = text; return; }
    node.innerHTML = DOMPurify.sanitize(marked.parse(text, {gfm: true, breaks: true}), {
      ALLOWED_TAGS: ["p", "br", "strong", "em", "del", "a", "code", "pre", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td", "hr"],
      ALLOWED_ATTR: ["href", "title", "class", "start", "align"],
    });
    for (const link of node.querySelectorAll("a")) {
      try {
        const url = new URL(link.getAttribute("href") || "", location.origin);
        if (["https:", "http:", "mailto:"].includes(url.protocol)) { link.target = "_blank"; link.rel = "noopener noreferrer"; }
        else link.removeAttribute("href");
      } catch { link.removeAttribute("href"); }
    }
    for (const pre of node.querySelectorAll("pre")) {
      const code = $("code", pre);
      if (code) { const copy = iconButton("copy", "复制代码", "copy-code"); pre.append(copy); }
    }
    for (const table of node.querySelectorAll("table")) { const wrap = el("div", null, "table-scroll"); table.before(wrap); wrap.append(table); }
  }

  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  function button(text, action, id, className = "secondary") {
    const node = el("button", text, className);
    node.type = "button";
    node.dataset.action = action;
    if (id !== undefined) node.dataset.id = id;
    return node;
  }
  function badge(value, override) {
    return el("span", override || labels[value] || value, `status ${Object.hasOwn(labels, value) ? value : ""}`);
  }
  function formatTime(value, timezone) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", {timeZone: timezone || undefined, hour12: false});
  }
  function empty(container, text) {
    container.replaceChildren(el("div", text, "empty"));
  }
  function toast(message) {
    const node = $("#toast");
    node.textContent = message;
    node.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { node.hidden = true; }, 4500);
  }
  function report(error) {
    if (error.name === "AbortError") return;
    const message = error.message || String(error);
    const banner = $("#status-banner");
    banner.replaceChildren(el("span", message));
    if (embedded) banner.append(button("连接设置", "connect"));
    banner.hidden = false;
  }
  function requestId() {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
  }
  function headers(key) {
    const result = new Headers();
    if (state.key) result.set("Authorization", `Bearer ${state.key}`);
    if (key) result.set("Idempotency-Key", key);
    return result;
  }
  async function decodeError(response) {
    let body;
    try { body = await response.json(); } catch { body = {}; }
    const error = new Error(body.error?.message || `请求未完成（HTTP ${response.status}）`);
    error.status = response.status;
    error.details = body.error?.details || {};
    error.code = body.error?.code;
    if (body.request_id) error.message += ` · 请求 ${body.request_id}`;
    return error;
  }
  async function api(path, {method = "GET", body, key, signal, timeout = 60000} = {}) {
    const connection = state.connectionVersion;
    const controller = signal ? null : new AbortController();
    const timer = controller ? setTimeout(() => controller.abort(), timeout) : null;
    const auth = headers(method === "GET" ? null : key || requestId());
    const form = body instanceof FormData;
    if (body !== undefined && !form) auth.set("Content-Type", "application/json");
    try {
      const response = await fetch(API + path, {method, headers: auth, signal: signal || controller.signal,
        body: body === undefined ? undefined : form ? body : JSON.stringify(body), credentials: "same-origin"});
      if (!response.ok) throw await decodeError(response);
      const result = await response.json();
      if (connection !== state.connectionVersion) throw new DOMException("连接已更新", "AbortError");
      return result.data;
    } catch (error) {
      if (connection !== state.connectionVersion) throw new DOMException("连接已更新", "AbortError");
      throw error;
    } finally { if (timer) clearTimeout(timer); }
  }
  function query(values) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(values)) {
      if (Array.isArray(value)) { for (const item of value) params.append(key, item); }
      else if (value !== null && value !== undefined && value !== "") params.set(key, value);
    }
    return params.toString() ? `?${params}` : "";
  }
  const id = value => encodeURIComponent(value);
  const defaultScope = () => ({memory_scope: "user", resource_ids: []});

  function dialog(title, content) {
    state.previewTask = null;
    $("#dialog-title").textContent = title;
    $("#dialog-body").replaceChildren(content);
    if (!$("#dialog").open) $("#dialog").showModal();
  }
  function closeDialog() {
    state.previewTask = null;
    $("#dialog").close();
  }
  function formDialog(title, markup, submitLabel, submit) {
    const form = el("form");
    form.noValidate = true;
    const fields = el("div", null, "form-grid");
    fields.innerHTML = markup;
    const error = el("p", "", "form-error");
    error.id = `form-error-${requestId()}`;
    error.setAttribute("role", "alert");
    error.hidden = true;
    const actions = el("div", null, "form-actions");
    actions.append(button("取消", "close-dialog"));
    const save = el("button", submitLabel, "primary");
    save.type = "submit";
    actions.append(save);
    form.append(fields, error, actions);
    for (const input of fields.querySelectorAll("input,select,textarea")) {
      input.setAttribute("aria-describedby", error.id);
      input.addEventListener("input", () => input.removeAttribute("aria-invalid"));
      if (input.matches("textarea")) input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = `${Math.min(280, Math.max(95, input.scrollHeight))}px`;
      });
      if (input.type === "password") {
        const reveal = el("button", "显示密钥", "subtle secret-reveal");
        reveal.type = "button";
        reveal.setAttribute("aria-pressed", "false");
        reveal.addEventListener("click", () => {
          const show = input.type === "password";
          input.type = show ? "text" : "password";
          reveal.textContent = show ? "隐藏密钥" : "显示密钥";
          reveal.setAttribute("aria-pressed", String(show));
        });
        input.after(reveal);
      }
    }
    form.addEventListener("input", () => { delete form.dataset.requestId; });
    form.addEventListener("submit", async event => {
      event.preventDefault();
      if (save.disabled) return;
      const invalid = [...fields.querySelectorAll("input,select,textarea")].find(input => !input.validity.valid);
      if (invalid) {
        invalid.setAttribute("aria-invalid", "true");
        error.textContent = invalid.validity.valueMissing ? "请填写必填信息。" : invalid.type === "url" ? "请输入完整网页地址。" : "请检查输入格式和取值范围。";
        error.hidden = false;
        invalid.focus();
        return;
      }
      save.disabled = true;
      save.setAttribute("aria-busy", "true");
      error.hidden = true;
      form.dataset.requestId ||= requestId();
      try { await submit(form, form.dataset.requestId); }
      catch (failure) { error.textContent = failure.message || String(failure); error.hidden = false; }
      finally { save.disabled = false; save.removeAttribute("aria-busy"); }
    });
    dialog(title, form);
    return form;
  }
  function option(select, value, text, selected = false) {
    const item = el("option", text);
    item.value = value;
    item.selected = selected;
    select.append(item);
  }
  function populateTemplates(select, selected = "") {
    select.replaceChildren();
    option(select, "", "自由对话", !selected);
    for (const item of state.templates) option(select, item.id, item.name || item.id, item.id === selected);
  }
  function populateFormats(select, templateId, preferred = "markdown") {
    const formats = state.templates.find(item => item.id === templateId)?.output?.formats || ["markdown"];
    select.closest("label").hidden = !templateId;
    select.replaceChildren();
    for (const format of formats) option(select, format, format.toUpperCase(), format === preferred);
  }
  function connectDialog() {
    if (embedded) {
      window.parent.postMessage({type: "memorybot:open-settings"}, location.origin);
      return;
    }
    const form = formDialog("连接工作空间", '<label class="full">用户 API Key<input name="api_key" type="password" autocomplete="off" required placeholder="输入你的用户 API Key"></label><p class="form-help full">密钥保存在当前浏览器标签页，关闭标签页后清除。</p>', "连接", async form => {
      stopStreams();
      const nextKey = form.elements.api_key.value.trim();
      if (nextKey !== state.key) clearWorkspace();
      state.key = nextKey;
      sessionStorage.setItem("memorybot.apiKey", state.key);
      await connect();
      closeDialog();
      toast("已连接工作空间");
    });
    form.elements.api_key.value = state.key;
    if (state.key) $(".form-actions", form).prepend(button("清除连接", "disconnect"));
  }
  async function connect() {
    const version = state.workspaceVersion;
    state.config = await api("/config");
    const agents = await api("/agents");
    state.agents = agents.items;
    state.draftAgentId = agents.default_agent_id || state.agents[0]?.agent_id || "";
    renderAgentPicker();
    await loadSessions();
    await Promise.all([loadCollection("templates"), loadCollection("files")]);
    if (version !== state.workspaceVersion) return;
    $("#connection-dot").classList.add("connected");
    $("#connection-label").textContent = "工作空间已连接";
    $("#status-banner").hidden = true;
    if (state.page !== "chat") await loadPage();
    const selected = sessionStorage.getItem("memorybot.session");
    if (selected) await openSession(selected);
    else renderConversation();
  }
  function disconnect() {
    stopStreams();
    sessionStorage.removeItem("memorybot.apiKey");
    state.key = "";
    clearWorkspace();
    $("#connection-dot").classList.remove("connected");
    $("#connection-label").textContent = "连接你的工作空间";
    closeDialog();
    toast("已清除连接");
  }
  function clearWorkspace() {
    state.workspaceVersion += 1;
    state.connectionVersion += 1;
    sessionStorage.removeItem("memorybot.session");
    state.session = null;
    state.tasks.clear();
    state.drafts.clear();
    state.progress.clear();
    state.cursors = {};
    state.historyCursor = null;
    state.sendRequest = null;
    state.sessionRequest = null;
    state.sessions = [];
    state.files = [];
    state.plans = [];
    state.memories = [];
    state.templates = [];
    state.agents = []; state.draftAgentId = ""; renderAgentPicker();
    state.attachments.clear();
    state.pendingSend = null; state.sending = false; state.draftTitle = ""; state.draftScope = null;
    $("#conversation").replaceChildren();
    renderSessions();
    renderConversation();
    for (const name of ["files", "plans", "memories", "templates"]) renderCollection(name);
  }
  async function loadSessions(more = false) {
    const request = ++state.sessionListRequest;
    const result = await api("/sessions" + query({limit: 30, q: $("#session-search").value.trim(), archived: $("#session-filter").value, cursor: more ? state.cursors.sessions : null}));
    if (request !== state.sessionListRequest) return;
    state.sessions = more ? [...new Map([...state.sessions, ...result.items].map(item => [item.session_id, item])).values()] : result.items;
    state.cursors.sessions = result.next_cursor;
    renderSessions();
  }
  function renderSessions() {
    const list = $("#session-list");
    list.replaceChildren();
    let previousGroup = "";
    for (const session of state.sessions) {
      const date = new Date(session.updated_at || session.created_at);
      const today = new Date(); today.setHours(0, 0, 0, 0);
      const age = Math.floor((today - date) / 86400000);
      const group = age < 0 ? "今天" : age === 0 ? "昨天" : age < 7 ? "过去 7 天" : "更早";
      if (group !== previousGroup) { list.append(el("p", group, "session-group")); previousGroup = group; }
      const row = el("div", null, "session-row");
      row.classList.toggle("selected", state.session?.session_id === session.session_id);
      const item = button("", "open-session", session.session_id, "session-item");
      const title = el("span", session.title, "session-title");
      const time = el("time", date.toLocaleString("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false}), "session-time");
      time.dateTime = date.toISOString();
      item.append(icon("chat"), title, time); item.title = session.title;
      if (state.session?.session_id === session.session_id) item.setAttribute("aria-current", "true");
      const menu = el("details", null, "session-menu");
      const toggle = el("summary"); toggle.append(icon("more")); toggle.setAttribute("aria-label", `${session.title}，更多操作`);
      const actions = el("div", null, "session-menu-items");
      actions.append(button("重命名", "rename-session", session.session_id), button(session.archived ? "恢复会话" : "归档会话", "archive-session", session.session_id));
      menu.append(toggle, actions); row.append(item, menu); list.append(row);
    }
    if (!state.sessions.length) empty(list, $("#session-search").value.trim() ? "暂无匹配会话" : $("#session-filter").value === "true" ? "暂无归档会话" : "从一条消息开始");
    $("#session-more").hidden = !state.cursors.sessions;
  }
  async function openSession(sessionId, more = false) {
    const sameSession = state.session?.session_id === sessionId;
    const version = more ? state.workspaceVersion : ++state.workspaceVersion;
    const scroll = $("#chat-scroll"), beforeHeight = scroll.scrollHeight, beforeTop = scroll.scrollTop;
    const result = await api(`/sessions/${id(sessionId)}` + query({limit: 100, history_order: "desc", cursor: more ? state.historyCursor : null}));
    if (version !== state.workspaceVersion) return;
    if (!more) {
      stopStreams(); state.tasks.clear(); state.drafts.clear(); state.progress.clear();
      state.pendingSend = null; state.sending = false; state.followBottom = true;
      $("#conversation").replaceChildren();
      if (!sameSession) { $("#message").value = ""; state.attachments.clear(); resizeComposer(); }
    }
    state.session = result.session;
    renderAgentPicker();
    await loadCollection("templates");
    if (version !== state.workspaceVersion) return;
    state.historyCursor = result.next_cursor;
    sessionStorage.setItem("memorybot.session", sessionId);
    for (const task of result.messages) state.tasks.set(task.task_id, task);
    if (version !== state.workspaceVersion) return;
    showPage("chat");
    setSidebar(false);
    renderSessions();
    renderConversation();
    if (more) { scroll.scrollTop = beforeTop + scroll.scrollHeight - beforeHeight; state.followBottom = nearBottom(); }
    else scrollBottom();
    for (const task of result.messages) if (!settled(task)) watchTask(task);
  }
  function newSession() {
    const agentId = currentAgentId();
    state.workspaceVersion += 1;
    stopStreams(); state.session = null; state.tasks.clear(); state.drafts.clear(); state.progress.clear();
    state.draftAgentId = state.agents.some(agent => agent.agent_id === agentId) ? agentId : state.agents[0]?.agent_id || "";
    renderAgentPicker();
    state.attachments.clear(); state.pendingSend = null; state.sending = false; state.historyCursor = null;
    state.sendRequest = null; state.sessionRequest = null; state.draftTitle = ""; state.draftScope = null; state.followBottom = true;
    sessionStorage.removeItem("memorybot.session");
    $("#message").value = ""; $("#message").style.height = ""; $("#conversation").replaceChildren();
    $("#status-banner").hidden = true; showPage("chat"); renderSessions(); renderConversation(); setSidebar(false);
    state.templates = []; renderCollection("templates");
    if (state.key) loadCollection("templates").catch(report);
    $("#message").focus();
  }
  function sessionSettings(sessionId = null) {
    const selected = sessionId ? state.sessions.find(item => item.session_id === sessionId) : state.session;
    const scope = selected?.scope || state.draftScope || defaultScope();
    const form = formDialog(selected ? "会话设置" : "新会话设置", '<label class="full">名称<input name="title" maxlength="128" placeholder="默认使用第一条消息"></label><label class="full">记忆范围<select name="memory_scope"><option value="user">我的记忆</option><option value="authorized">全部已授权记忆</option></select></label><label class="full">资料 URI（每行一项）<textarea name="resources" rows="2" placeholder="ctx://…"></textarea></label>', "保存", async (form, key) => {
      const title = form.elements.title.value.trim();
      if (selected) {
        if (!title) throw new Error("请填写会话名称");
        const updated = await api(`/sessions/${id(selected.session_id)}`, {method: "PATCH", key, body: {expected_revision: selected.revision, title}});
        if (state.session?.session_id === updated.session_id) state.session = updated;
        await loadSessions();
      } else {
        state.draftTitle = title;
        state.draftScope = {memory_scope: form.elements.memory_scope.value, resource_ids: form.elements.resources.value.split("\n").map(item => item.trim()).filter(Boolean)};
        await loadCollection("templates");
      }
      closeDialog(); showPage(state.page);
    });
    form.elements.title.value = selected?.title || state.draftTitle;
    form.elements.memory_scope.value = scope.memory_scope;
    form.elements.resources.value = (scope.resource_ids || []).join("\n");
    for (const name of ["memory_scope", "resources"]) form.elements[name].disabled = !!selected;
    if (selected) form.elements.title.required = true;
  }
  async function archiveSession(sessionId) {
    const session = state.sessions.find(item => item.session_id === sessionId) || state.session;
    const updated = await api(`/sessions/${id(sessionId)}`, {method: "PATCH", body: {expected_revision: session.revision, archived: !session.archived}});
    if (state.session?.session_id === sessionId) { state.session = updated; if (updated.archived) newSession(); }
    await loadSessions(); renderSessions(); showPage(state.page); toast(updated.archived ? "会话已归档" : "会话已恢复");
  }
  function showPage(name) {
    state.page = name;
    for (const page of document.querySelectorAll(".page")) page.hidden = page.id !== `page-${name}`;
    for (const nav of document.querySelectorAll("[data-page]")) {
      nav.classList.toggle("active", nav.dataset.page === name);
      if (nav.dataset.page === name) nav.setAttribute("aria-current", "page");
      else nav.removeAttribute("aria-current");
    }
    $("#session-settings").hidden = name !== "chat";
    $("#agent-picker").hidden = name !== "chat";
    $("#page-title").textContent = name === "chat" ? state.session?.title || state.draftTitle || "新会话" : pageTitles[name];
    document.title = `${$("#page-title").textContent} · memoryBot`;
  }
  async function navigate(name) {
    showPage(name);
    setSidebar(false);
    if (state.key && name !== "chat") await loadPage();
  }
  async function loadPage() {
    if (state.page === "chat") {
      if (state.session) await openSession(state.session.session_id);
      else await loadSessions();
    } else await loadCollection(state.page);
  }
  function renderAttachments() {
    const list = $("#attachments");
    list.replaceChildren();
    for (const [fileId, name] of state.attachments) {
      const item = el("span", name, "attachment");
      const remove = button("×", "remove-attachment", fileId, "");
      remove.setAttribute("aria-label", `移除 ${name}`);
      item.append(remove);
      list.append(item);
    }
  }
  function taskNode(task, includeInput = true) {
    const card = el("article", null, "task-card"); card.dataset.taskId = task.task_id;
    if (includeInput && task.input) {
      const user = el("div", null, "user-turn"), bubble = el("div", null, "user-message");
      bubble.append(el("div", task.input.text || "处理指定资料", "user-text"));
      if (task.input.file_ids?.length) bubble.append(el("small", `${task.input.file_ids.length} 份资料`, "message-files"));
      user.append(bubble, iconButton("copy", "复制提问", "copy-input", task.task_id)); card.append(user);
    }
    const turn = el("div", null, "assistant-turn"), avatar = el("span", null, "assistant-avatar"); avatar.append(icon("bot"));
    const answer = el("div", null, "assistant-message");
    answer.append(el("div", null, "answer-text markdown-body"), el("div", null, "answer-status"), el("div", null, "answer-resources"), el("div", null, "answer-meta"));
    turn.append(avatar, answer); card.append(turn); patchTaskNode(card, task); return card;
  }
  function patchTaskNode(card, task) {
    const draft = state.drafts.get(task.task_id)?.text || "", text = task.result?.text || draft;
    const body = $(".answer-text", card), streaming = !terminal.has(task.state);
    if (text) { body.classList.remove("thinking"); renderMarkdown(body, text); }
    else if (!streaming) { body.classList.remove("thinking"); body.textContent = task.confirmation?.reason || task.error?.message || labels[task.state] || "已结束"; delete body.dataset.text; }
    else if (!body.classList.contains("thinking")) {
      body.replaceChildren(el("span"), el("span"), el("span")); body.classList.add("thinking"); body.setAttribute("aria-label", "正在回复"); delete body.dataset.text;
    }
    const progress = state.progress.get(task.task_id);
    const statusKey = JSON.stringify([task.state, task.cancel_requested, task.error, progress, task.memory_status, task.steps]);
    const status = $(".answer-status", card);
    if (status.dataset.signature !== statusKey) {
      const wasOpen = $("details", status)?.open; status.replaceChildren(); status.dataset.signature = statusKey;
      const steps = (task.steps || []).slice().sort((a, b) => a.step_no - b.step_no);
      const running = steps.find(step => step.state === "running"), latest = steps.at(-1);
      const finalPhase = {succeeded: task.kind === "chat" ? "回复已完成" : "已完成", failed: "执行失败", cancelled: "已停止"}[task.state];
      const phase = finalPhase || (task.cancel_requested ? "正在停止…" : running ? running.label : progress || (latest ? latest.label : "已接收，等待开始"));
      const detail = el("details", null, "task-details"); detail.open = Boolean(wasOpen);
      const summary = el("summary");
      summary.append(el("span", "执行过程", "execution-title"), el("span", `${steps.length ? steps.length + " 步 · " : ""}${phase}`, "execution-phase"));
      const path = steps.length > 5 ? [steps[0].label, steps[1].label, "…", steps.at(-2).label, latest.label] : steps.map(step => step.label);
      if (path.length) summary.append(el("span", path.join(" → "), "execution-path"));
      if (task.memory_status && task.memory_status !== "not_requested") summary.append(el("span", task.memory_status === "pending" ? "记忆整理中" : labels[task.memory_status] || task.memory_status, "execution-memory"));
      const list = el("ol", null, "execution-steps");
      for (const step of steps) {
        const item = el("li", null, "execution-step"); item.dataset.stepId = step.step_id; item.dataset.state = step.state;
        const head = el("div", null, "step-heading");
        head.append(el("span", step.label, "step-label"), badge(step.state, step.state === "running" ? "进行中" : null));
        const duration = el("time", null, "step-duration");
        if (step.state === "running" && step.started_at) duration.dataset.startedAt = step.started_at;
        duration.textContent = formatDuration(step.elapsed_ms ?? (step.finished_at && step.started_at ? new Date(step.finished_at) - new Date(step.started_at) : null));
        head.append(duration); item.append(head);
        const text = stepSummary(step.summary || {});
        if (text) item.append(el("p", text, "step-result"));
        if (step.summary?.sources?.length) {
          const sources = step.summary.sources, sourceText = el("p", null, "step-sources");
          for (const source of sources.slice(0, 2)) {
            const title = Array.from((source.title || source.uri || "来源").replace(/\s+/g, " ").trim());
            if (sourceText.childNodes.length) sourceText.append(" · ");
            sourceText.append(el("span", title.length > 40 ? title.slice(0, 39).join("") + "…" : title.join(""), "step-source-title"));
          }
          const total = step.summary.source_count ?? sources.length;
          if (total > 2) sourceText.append(` 等 ${total} 项来源`);
          item.append(sourceText);
        }
        list.append(item);
      }
      const info = el("div", null, "task-detail-content");
      if (task.error) info.append(el("p", task.error.message, "step-result"));
      if (task.confirmation) {
        info.append(button("核对结果", "confirm-task", task.task_id), button("停止任务", "cancel-confirmation", task.task_id));
      } else if (streaming && task.task_id !== "pending" && task.kind !== "chat") {
        const stop = button(task.cancel_requested ? "正在停止…" : "停止任务", "cancel-task", task.task_id);
        stop.disabled = Boolean(task.cancel_requested); info.append(stop);
      }
      if (task.task_id !== "pending") info.append(button("刷新进度", "refresh-task", task.task_id, "subtle"));
      detail.append(summary, list, info); status.append(detail); updateExecutionTimes();
    }
    const resources = $(".answer-resources", card), resourceKey = JSON.stringify([task.result?.citations, task.result?.file_ids]);
    if (resources.dataset.signature !== resourceKey) {
      const opened = $(".citations", resources)?.open; resources.replaceChildren(); resources.dataset.signature = resourceKey;
      if (task.result?.citations?.length) { const citations = citationNode(task.result.citations); citations.open = Boolean(opened); resources.append(citations); }
      if (task.result?.file_ids?.length) {
        const files = el("div", null, "result-files");
        for (const fileId of task.result.file_ids) { const file = state.files.find(item => item.file_id === fileId); const link = button(file?.name || "下载成果", "download", fileId); link.prepend(icon("file")); files.append(link); }
        resources.append(files);
      }
    }
    const meta = $(".answer-meta", card), metaKey = JSON.stringify([task.state, Boolean(text), task.created_at]);
    if (meta.dataset.signature !== metaKey) {
      meta.dataset.signature = metaKey; meta.replaceChildren();
      if (text && task.task_id !== "pending") meta.append(iconButton("copy", "复制回复", "copy-answer", task.task_id));
      const time = el("time", task.created_at ? new Date(task.created_at).toLocaleTimeString("zh-CN", {hour: "2-digit", minute: "2-digit", hour12: false}) : ""); time.title = formatTime(task.created_at); meta.append(time);
      if (["failed", "cancelled"].includes(task.state)) meta.append(button("重试", "retry-task", task.task_id, "subtle"));
    }
  }
  function formatDuration(milliseconds) {
    if (milliseconds === null || milliseconds === undefined || !Number.isFinite(Number(milliseconds))) return "";
    const seconds = Math.max(0, Number(milliseconds)) / 1000;
    return seconds < 60 ? `${seconds.toFixed(1)} 秒` : `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒`;
  }
  function updateExecutionTimes() {
    for (const node of document.querySelectorAll(".step-duration[data-started-at]")) node.textContent = formatDuration(Date.now() - new Date(node.dataset.startedAt));
  }
  function stepSummary(summary) {
    const parts = [];
    if (summary.tool_name) parts.push(`工具：${summary.tool_name}`);
    if (summary.source_count !== undefined || summary.sources?.length) parts.push(`${summary.source_count ?? summary.sources.length} 项来源`);
    if (summary.schedule_count !== undefined) parts.push(`找到 ${summary.schedule_count} 项计划`);
    if (summary.file_name) parts.push(summary.file_name);
    if (summary.file_count !== undefined) parts.push(`${summary.file_count} 份资料`);
    if (summary.character_count !== undefined) parts.push(`${summary.character_count} 字`);
    if (summary.tool_call_count !== undefined) parts.push(`${summary.tool_call_count} 次工具调用`);
    if (summary.completed !== undefined && summary.total !== undefined) parts.push(`已处理 ${summary.completed} / ${summary.total} ${summary.unit === "file" ? "份资料" : "段"}`);
    if (summary.memory_status) parts.push(labels[summary.memory_status] || summary.memory_status);
    if (summary.confirmed) parts.push("已核对结果");
    return parts.join(" · ");
  }
  function mergeSteps(previous = [], incoming = []) {
    const steps = new Map(previous.map(step => [step.step_id, step]));
    for (const step of incoming) {
      const current = steps.get(step.step_id);
      if (!current || (step.event_seq || 0) >= (current.event_seq || 0)) steps.set(step.step_id, step);
    }
    return [...steps.values()].sort((a, b) => a.step_no - b.step_no);
  }
  function citationNode(citations) {
    const section = el("details", null, "citations");
    section.append(el("summary", `${citations.length} 项来源`));
    for (const citation of citations) {
      const item = el("div", null, "citation");
      item.append(el("strong", `${citation.id ? "[" + citation.id + "] " : ""}${citation.title || "来源"}`), el("blockquote", citation.quote || ""));
      if (citation.source_time) item.append(el("small", formatTime(citation.source_time)));
      item.append(el("code", citation.uri));
      if (/^https?:\/\//i.test(citation.uri || "")) {
        const link = el("a", "打开原文");
        link.href = citation.uri;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        item.append(link);
      } else if (citation.file_id) item.append(button("下载来源", "download", citation.file_id, "subtle"));
      else if (citation.task_id) item.append(button("查看来源任务", "open-task", citation.task_id, "subtle"));
      else item.append(button("查看记忆来源", "source-memory", citation.uri, "subtle"));
      section.append(item);
    }
    return section;
  }
  function nearBottom() { const scroll = $("#chat-scroll"); return scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 100; }
  function scrollBottom() { const scroll = $("#chat-scroll"); scroll.scrollTop = scroll.scrollHeight; state.followBottom = true; $("#jump-latest").hidden = true; }
  function activeTask() { return [...state.tasks.values()].find(task => task.kind === "chat" && task.session_id === state.session?.session_id && !terminal.has(task.state)); }
  function updateComposer() {
    const active = activeTask();
    $("#stop-button").hidden = !active; $("#stop-button").disabled = Boolean(active?.cancel_requested);
    $("#send-button").hidden = Boolean(active); $("#send-button").disabled = state.sending || !currentAgentId() || !($("#message").value.trim() || state.attachments.size || $("#chat-template").value);
    $("#chat-agent").disabled = Boolean(state.session) || state.sending;
    $("#composer-mode").textContent = $("#chat-template").value ? $("#chat-template").selectedOptions[0]?.textContent || "" : "";
  }
  function currentAgentId() { return state.session?.agent_id || state.draftAgentId; }
  function populateAgents(select, selected) {
    select.replaceChildren();
    for (const agent of state.agents) option(select, agent.agent_id, agent.agent_id, agent.agent_id === selected);
    if (selected && !state.agents.some(agent => agent.agent_id === selected)) option(select, selected, selected, true);
    if (!select.options.length) option(select, "", "暂无可用 Agent");
  }
  function renderAgentPicker() { populateAgents($("#chat-agent"), currentAgentId()); $("#chat-agent").disabled = Boolean(state.session) || state.sending; }
  function resizeComposer() { const input = $("#message"); input.style.height = "auto"; input.style.height = `${Math.min(200, Math.max(44, input.scrollHeight))}px`; updateComposer(); }
  function scheduleRender() {
    if (state.renderScheduled) return;
    state.renderScheduled = true; requestAnimationFrame(() => { state.renderScheduled = false; renderConversation(); });
  }
  document.addEventListener("selectionchange", () => { if (window.getSelection()?.isCollapsed) scheduleRender(); });
  function renderConversation() {
    const tasks = [...state.tasks.values()].filter(task => task.kind === "chat" && task.session_id === state.session?.session_id).sort((a, b) => a.seq - b.seq);
    if (state.pendingSend) tasks.push(state.pendingSend);
    const list = $("#conversation"), wanted = new Set(tasks.map(task => task.task_id));
    for (const node of [...list.children]) if (!wanted.has(node.dataset.taskId)) node.remove();
    let preceding = null;
    for (const task of tasks) {
      let node = [...list.children].find(item => item.dataset.taskId === task.task_id);
      if (!node) node = taskNode(task); else patchTaskNode(node, task);
      const position = preceding ? preceding.nextElementSibling : list.firstElementChild;
      if (position !== node) list.insertBefore(node, position);
      preceding = node;
    }
    $("#welcome").hidden = tasks.length > 0;
    $("#history-more").hidden = !state.historyCursor;
    renderAttachments(); showPage(state.page); renderPreview(); updateComposer();
    if (state.followBottom) scrollBottom(); else $("#jump-latest").hidden = nearBottom();
  }
  function updateTask(task) {
    const current = state.tasks.get(task.task_id);
    if (Number.isInteger(task.last_event_seq) && task.last_event_seq < current?.last_event_seq) return current;
    const updated = {...current, ...task, steps: mergeSteps(current?.steps, task.steps)};
    state.tasks.set(task.task_id, updated);
    renderConversation();
    return updated;
  }
  async function getTask(taskId) {
    const version = state.workspaceVersion;
    const task = await api(`/tasks/${id(taskId)}`);
    return version === state.workspaceVersion ? updateTask(task) : task;
  }
  function renderPreview() {
    if (state.previewTask && $("#dialog").open) {
      const task = state.tasks.get(state.previewTask);
      if (task) {
        const existing = $(".task-card", $("#dialog-body"));
        if (existing?.dataset.taskId === task.task_id) patchTaskNode(existing, task);
        else $("#dialog-body").replaceChildren(taskNode(task, !!task.input));
      }
    }
  }
  async function openTask(taskId) {
    const task = await getTask(taskId);
    if (task.session_id && task.kind === "chat") {
      closeDialog();
      if (state.session?.session_id !== task.session_id) await openSession(task.session_id);
      updateTask(task); showPage("chat");
      const card = [...$("#conversation").children].find(node => node.dataset.taskId === taskId);
      card?.scrollIntoView({block: "center", behavior: "smooth"});
      if (!settled(task)) watchTask(task);
      return;
    }
    dialog("任务进度与结果", taskNode(task, !!task.input));
    state.previewTask = taskId;
    if (!settled(task)) watchTask(task);
  }
  function stopStreams() {
    for (const controller of state.streams.values()) controller.abort();
    state.streams.clear();
  }
  function wait(ms, signal) {
    return new Promise(resolve => {
      if (signal.aborted) return resolve();
      const timer = setTimeout(done, ms);
      function done() { clearTimeout(timer); signal.removeEventListener("abort", done); resolve(); }
      signal.addEventListener("abort", done, {once: true});
    });
  }
  function settled(task) {
    return terminal.has(task.state) && !["pending", "stored"].includes(task.memory_status);
  }
  async function handleEvent(taskId, block) {
    let type = "message", data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) type = line.slice(6).trim();
      if (line.startsWith("data:")) data += (data ? "\n" : "") + line.slice(5).trimStart();
    }
    if (!data) return null;
    const event = JSON.parse(data), payload = event.payload || {};
    if (type.startsWith("step.") && payload.step_id) {
      const task = state.tasks.get(taskId);
      if (task) task.steps = mergeSteps(task.steps, [{...payload, event_seq: payload.event_seq || event.event_seq}]);
    }
    if (type === "answer.reset") state.drafts.set(taskId, {attempt: payload.attempt, text: ""});
    if (type === "answer.delta") {
      let draft = state.drafts.get(taskId);
      if (!draft || draft.attempt < payload.attempt) draft = {attempt: payload.attempt, text: ""};
      if (draft.attempt === payload.attempt) draft.text += payload.text || "";
      state.drafts.set(taskId, draft);
    }
    if (type === "progress") state.progress.set(taskId, payload.message + (payload.total ? ` (${payload.completed || 0}/${payload.total})` : ""));
    if (type === "memory.updated") {
      const task = state.tasks.get(taskId);
      if (task) updateTask({
        ...task, last_event_seq: event.event_seq, memory_status: payload.memory_status,
        memory_operation_id: payload.memory_operation_id,
      });
    }
    if (type === "task.completed" || type === "task.updated") await getTask(taskId);
    scheduleRender();
    return event;
  }
  async function watchTask(task) {
    if (state.streams.has(task.task_id)) return;
    const controller = new AbortController();
    state.streams.set(task.task_id, controller);
    let after = 0, backoff = 1000;
    try {
      while (!controller.signal.aborted) {
        try {
          const auth = headers();
          auth.set("Accept", "text/event-stream");
          if (after) auth.set("Last-Event-ID", `${task.task_id}:${after}`);
          const response = await fetch(`${API}/tasks/${id(task.task_id)}/events?after=${after}`, {headers: auth, signal: controller.signal, credentials: "same-origin"});
          if (response.status === 410) {
            const snapshot = await getTask(task.task_id);
            after = snapshot.last_event_seq;
            if (settled(snapshot)) return;
            continue;
          }
          if (!response.ok) throw await decodeError(response);
          const reader = response.body.getReader(), decoder = new TextDecoder();
          let buffer = "";
          try {
            while (!controller.signal.aborted) {
              const chunk = await reader.read();
              if (chunk.done) break;
              buffer += decoder.decode(chunk.value, {stream: true});
              buffer = buffer.replace(/\r\n/g, "\n");
              let boundary;
              while ((boundary = buffer.indexOf("\n\n")) >= 0) {
                const block = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);
                const raw = block.split("\n").find(line => line.startsWith("id:"));
                const seq = raw ? Number(raw.slice(raw.lastIndexOf(":") + 1)) : null;
                if (seq && seq <= after) continue;
                const event = await handleEvent(task.task_id, block);
                if (event) { after = event.event_seq; backoff = 1000; }
                if (state.tasks.get(task.task_id)?.state === "waiting_confirmation") return;
                if (event && settled(state.tasks.get(task.task_id))) return;
              }
            }
          } finally { await reader.cancel().catch(() => {}); }
          if (controller.signal.aborted) return;
          const snapshot = await getTask(task.task_id);
          if (settled(snapshot) || snapshot.state === "waiting_confirmation") return;
          await wait(1500, controller.signal);
        } catch (error) {
          if (controller.signal.aborted) return;
          if ([400, 401, 403, 404].includes(error.status)) { report(error); return; }
          state.progress.set(task.task_id, "正在重新连接，已保存的进度会继续显示…");
          renderConversation();
          await wait(backoff, controller.signal);
          backoff = Math.min(backoff * 2, 15000);
        }
      }
    } finally {
      if (state.streams.get(task.task_id) === controller) state.streams.delete(task.task_id);
      if (state.page === "files") loadCollection("files").catch(report);
    }
  }
})();
