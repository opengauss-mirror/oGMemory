// Run with node --test; provide puppeteer through NODE_PATH and Chrome through CHROME_PATH.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const puppeteer = require('puppeteer');

test('team text rules, backend privacy preservation, pagination, member drafts, retrieval and deletion', { timeout: 120000 }, async (t) => {
  const directory = process.env.SPACE_UI_SCREENSHOT ? path.dirname(process.env.SPACE_UI_SCREENSHOT) : null;
  if (directory) await fs.mkdir(directory, { recursive: true });
  const capabilities = { manage_space: true, delete_space: true, manage_members: true, manage_acl: true, manage_rules: true, create_rule: true, inspect: true, search: true };
  const numbered = (count, make) => Array.from({ length: count }, (_, index) => make(String(index + 1).padStart(2, '0')));
  const principals = [
    { principal: 'user:alice', display_name: 'Alice', status: 'active', can_contribute: true },
    { principal: 'user:bob', display_name: 'Bob', status: 'active', can_contribute: false },
    ...numbered(18, (id) => ({ principal: `user:member-${id}`, display_name: `Member ${id}`, status: 'active', can_contribute: false })),
    { principal: 'agent:ops-agent', display_name: 'Ops Agent', owner_user_id: 'alice', status: 'active', can_contribute: true },
    { principal: 'user:carol', display_name: 'Carol', status: 'active', can_contribute: false },
  ];
  const writes = [];
  const reads = [];
  let space;
  let rule;
  const deletedRules = new Set();
  const item = { item_id: 'experience-1', title: '连接池排障', kind: 'experience', state: 'ACTIVE', rule_id: 'ops-rule', rule_rev: 1, content_rev: 1, content: '## 适用场景\n\n连接池等待时间持续上升。\n\n## 处理步骤\n\n- 检查**长事务**。\n- 确认 `connection.close()` 已执行。\n\n## 验证\n\n```sql\nSELECT count(*) FROM pg_stat_activity;\n```', index_state: 'READY', source_count: 1, sources: [{ uri: 'ctx://acme/agents/ops-agent/memories/case/pool' }], acl_mode: 'inherit', readers: [] };
  const spaceRows = numbered(22, (id) => ({ space_id: `team-${id}`, name: `团队 ${id}`, owner_user_id: 'alice', state: 'ACTIVE', authz_rev: 1, capabilities, member_count: 1, memory_count: 0, members: [{ principal: 'user:alice', role: 'manager', can_contribute: true }], sources: [] }));
  const itemRows = () => rule ? [item, ...numbered(21, (id) => ({ ...item, item_id: `experience-${id}-extra`, title: `连接池检查 ${id}`, content: `检查连接池步骤 ${id}。` }))] : [];
  const ruleRows = () => rule ? [rule, ...numbered(21, (id) => ({ ...rule, rule_id: `rule-${id}`, name: `检查规则 ${id}` }))].filter((entry) => !deletedRules.has(entry.rule_id)) : [];
  const paginate = (rows, values, kind) => {
    const offset = values.cursor ? Number(String(values.cursor).split(':')[1]) : 0;
    const limit = Number(values.limit || 20);
    assert.equal(limit, 20);
    return { items: rows.slice(offset, offset + limit), total: rows.length, next_cursor: offset + limit < rows.length ? `${kind}:${offset + limit}` : null };
  };
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const json = (body, status = 200) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); };
    try {
      if (url.pathname.startsWith('/memory-viewer/')) {
        const name = url.pathname.split('/').pop() || 'index.html';
        const content = await fs.readFile(path.join(__dirname, '../../server/memory_viewer_static', name));
        res.writeHead(200, { 'Content-Type': name.endsWith('.js') ? 'application/javascript' : name.endsWith('.css') ? 'text/css' : 'text/html' });
        res.end(content); return;
      }
      if (url.pathname.endsWith('/admin/tree')) return json({ account_id: 'acme', accounts: [{ account_id: 'acme' }], users: [{ user_id: 'alice', agents: [{ agent_id: 'ops-agent' }] }] });
      if (url.pathname.endsWith('/scope')) return json({ account_id: 'acme', user_id: 'alice', admin_mode: false, agents: [{ agent_id: 'ops-agent' }] });
      if (url.pathname.endsWith('/inventory')) return json({ directories: {} });
      if (url.pathname.endsWith('/admin/memories')) return json({ memories: [], count: 0, page: 1, page_count: 1, type_counts: {} });
      if (req.method === 'GET') {
        const values = Object.fromEntries(url.searchParams);
        reads.push({ path: url.pathname, values });
        if (url.pathname.endsWith('/principals')) {
          const members = (values.space_id === space?.space_id ? space : spaceRows.find((row) => row.space_id === values.space_id))?.members || [];
          return json(paginate(principals.filter((p) => !members.some((member) => member.principal === p.principal)
            && (!values.kind || p.principal.startsWith(`${values.kind}:`))
            && `${p.display_name} ${p.principal}`.toLowerCase().includes((values.q || '').toLowerCase())), values, 'principals'));
        }
        if (url.pathname === '/management/spaces') return json(values.space_id ? { space: values.space_id === space?.space_id ? space : spaceRows.find((entry) => entry.space_id === values.space_id) } : paginate(space ? [space, ...spaceRows] : spaceRows, values, 'space'));
        if (url.pathname.endsWith('/permissions')) return json({ user_id: 'alice', principal: 'user:alice', is_admin: false, capabilities, can_read: true });
        if (url.pathname.endsWith('/rules')) return json(paginate(ruleRows(), values, 'rules'));
        if (url.pathname.endsWith('/items')) {
          if (values.item_id) {
            const selected = itemRows().find((entry) => entry.item_id === values.item_id);
            const detail = selected;
            return json({ items: [detail], next_cursor: null });
          }
          return json(paginate(itemRows(), values, 'content'));
        }
      }
      let raw = '';
      for await (const chunk of req) raw += chunk;
      const body = JSON.parse(raw || '{}');
      writes.push({ method: req.method, path: url.pathname, body, headers: req.headers });
      if (req.method === 'DELETE') {
        await new Promise((resolve) => setTimeout(resolve, 150));
        const id = url.pathname.split('/').pop();
        if (url.pathname.includes('/rules/')) {
          const selected = ruleRows().find((entry) => entry.rule_id === id);
          assert.equal(req.headers['if-match'], `"${selected.rule_rev}"`);
          deletedRules.add(id);
          return json({ deleted: true, rule_id: id, rule_rev: selected.rule_rev + 1 });
        }
        if (url.pathname.includes('/sources/')) {
          const selected = space.sources.find((entry) => entry.binding_id === id);
          assert.equal(req.headers['if-match'], `"${selected.grant_rev}"`);
          space.sources = space.sources.filter((entry) => entry.binding_id !== id);
          space.authz_rev += 1;
          return json({ deleted: true, binding_id: id, grant_rev: selected.grant_rev + 1, authz_rev: space.authz_rev });
        }
        assert.equal(req.headers['if-match'], `"${space.authz_rev}"`);
        const result = { deleted: true, space_id: id, authz_rev: space.authz_rev + 1 };
        space = undefined;
        return json(result);
      }
      if (req.method === 'POST' && url.pathname === '/management/spaces') {
        space = { ...body, owner_user_id: 'alice', state: 'ACTIVE', authz_rev: 1, capabilities, member_count: 1, memory_count: 0, members: [{ principal: 'user:alice', role: 'manager', can_contribute: true }], sources: [] };
        return json(space, 201);
      }
      if (url.pathname.endsWith('/members')) {
        space.members = body.members.map((member) => ({ ...member, can_contribute: principals.find((p) => p.principal === member.principal)?.can_contribute }));
        space.member_count = space.members.length;
        space.authz_rev += 1;
        return json({ authz_rev: space.authz_rev });
      }
      if (url.pathname.includes('/sources/')) {
        const binding_id = url.pathname.split('/').pop();
        const source = { ...body, binding_id, grant_rev: 1, can_manage: true, can_delete: true };
        space.sources = [source, ...numbered(21, (id) => ({ ...source, binding_id: `source-${id}` }))];
        space.authz_rev += 1;
        return json(source, 201);
      }
      if (url.pathname.includes('/rules/')) {
        const rule_id = url.pathname.split('/').pop();
        rule = { ...body, rule_id, rule_rev: rule_id === rule?.rule_id ? rule.rule_rev + 1 : 1, spec_hash: 'stable-hash' };
        item.rule_id = rule.rule_id;
        return json(rule, 201);
      }
      if (url.pathname.endsWith('/acl')) { Object.assign(item, body); space.authz_rev += 1; return json({ authz_rev: space.authz_rev }); }
      if (url.pathname === '/api/spaces/search') {
        assert.equal(body.top_k, 100);
        assert.ok(['hybrid', 'keyword', 'semantic'].includes(body.mode));
        return json({ ...paginate(itemRows(), body, 'search'), partial: false });
      }
      if (req.method === 'PATCH') { Object.assign(space, body); space.authz_rev += 1; return json(space); }
      json({ error: `Unexpected fixture route: ${req.method} ${url.pathname}` }, 500);
    } catch (error) { json({ error: error.message }, 500); }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await puppeteer.launch({ headless: true, executablePath: process.env.CHROME_PATH, args: ['--no-sandbox'] });
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 1000 });
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    await page.evaluateOnNewDocument(() => sessionStorage.setItem('ogmemory-viewer-api-key', 'alice-test-key'));
    await page.goto(`http://127.0.0.1:${server.address().port}/memory-viewer/`);
    const click = async (selector) => {
      await page.waitForSelector(selector, { visible: true });
      await page.$eval(selector, (element) => element.scrollIntoView({ block: 'center' }));
      await page.locator(selector).click();
    };
    const selectTab = async (name) => {
      const selector = `[data-space-action="tab"][data-value="${name}"]`;
      await click(selector);
      await page.waitForSelector(`${selector}[aria-selected="true"]`);
    };
    const fill = async (selector, value) => {
      await page.waitForSelector(selector, { visible: true });
      await page.$eval(selector, (element, text) => { element.value = text; element.dispatchEvent(new Event('input', { bubbles: true })); }, value);
    };
    const inspectConfirmation = async (name, impact, screenshot) => {
      await page.waitForSelector('.space-delete-dialog[open]');
      assert.equal(await page.$eval('.space-delete-name', (element) => element.textContent), name);
      assert.match(await page.$eval('#space-delete-impact', (element) => element.textContent), impact);
      assert.equal(await page.evaluate(() => document.activeElement.hasAttribute('data-space-cancel-delete')), true);
      if (directory && screenshot) await page.screenshot({ path: path.join(directory, `${screenshot}.png`) });
    };
    const layoutMetrics = [];
    let layoutChecks = 0;
    const captureLayout = async (name, dark = true) => {

      for (const theme of dark ? ['light', 'dark'] : ['light']) {
        if (await page.$eval('html', (element) => element.dataset.theme) !== theme) await click('#theme-button');
        for (const [size, width, height] of [['desktop', 1440, 1000], ['laptop', 1280, 900], ['sidebar-wide', 1101, 900], ['sidebar-compact', 1100, 900], ['narrow', 1024, 1000], ['columns', 900, 900], ['stacked', 720, 900], ['panel', 615, 900], ['mobile', 390, 844]]) {
          await page.setViewport({ width, height });
          await page.evaluate(() => { document.activeElement.blur(); window.scrollTo(0, 0); });
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name}: ${size} fits viewport`);
          const metrics = await page.evaluate(() => {
            const box = (element) => {
              if (!element || !element.getClientRects().length) return null;
              const rect = element.getBoundingClientRect();
              return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: rect.width, height: rect.height, clientHeight: element.clientHeight, scrollHeight: element.scrollHeight };
            };
            const controls = [...document.querySelectorAll('#spaces-view button, #spaces-view input, #spaces-view select, #spaces-view textarea')].map((element) => {
              const bounds = element.getBoundingClientRect();
              const rect = { left: bounds.left, right: bounds.right, top: Math.max(0, bounds.top), bottom: Math.min(innerHeight, bounds.bottom) };
              for (let ancestor = element.parentElement; ancestor; ancestor = ancestor.parentElement) {
                const style = getComputedStyle(ancestor);
                const parent = ancestor.getBoundingClientRect();
                if (/(auto|scroll|hidden|clip)/.test(style.overflowX)) { rect.left = Math.max(rect.left, parent.left); rect.right = Math.min(rect.right, parent.right); }
                if (/(auto|scroll|hidden|clip)/.test(style.overflowY)) { rect.top = Math.max(rect.top, parent.top); rect.bottom = Math.min(rect.bottom, parent.bottom); }
              }
              return { name: element.getAttribute('aria-label') || element.textContent.trim().slice(0, 30) || element.name, width: bounds.width, height: bounds.height, ...rect };
            }).filter((rect) => rect.width > 0 && rect.height > 0 && rect.right > rect.left && rect.bottom > rect.top);
            const overlaps = controls.flatMap((control, index) => controls.slice(index + 1).filter((other) => Math.min(control.right, other.right) - Math.max(control.left, other.left) > 1 && Math.min(control.bottom, other.bottom) - Math.max(control.top, other.top) > 1).map((other) => [control.name, other.name]));
            return { viewportWidth: innerWidth, scrollWidth: document.documentElement.scrollWidth, body: box(document.querySelector('#space-body')), teamList: box(document.querySelector('#space-list')), memberLists: [...document.querySelectorAll('.space-member-list')].map(box), sections: { detail: [document.querySelector('.space-detail-main'), document.querySelector('.space-detail-aside')].map(box).filter(Boolean), members: [...document.querySelectorAll('.space-members-grid > .space-panel')].map(box), memberSettings: [...document.querySelectorAll('.space-members-bottom > *')].map(box), rule: [document.querySelector('.space-rule-basics'), document.querySelector('.space-rule-content')].map(box).filter(Boolean), ruleTitles: [document.querySelector('.space-rule-basics > h4'), document.querySelector('.space-rule-content > h4')].map(box).filter(Boolean), cards: [...document.querySelectorAll('.space-cards > .space-card')].slice(0, 2).map(box), picker: box(document.querySelector('#space-picker')), create: box(document.querySelector('.space-heading [data-space-action="create"]')) }, controls, overlaps };
          });
          assert.ok(metrics.controls.every((control) => control.left >= -1 && control.right <= metrics.viewportWidth + 1), `${name}: ${size} controls fit viewport`);
          assert.equal(metrics.overlaps.length, 0, `${name}: ${size} controls remain separately selectable: ${JSON.stringify(metrics.overlaps)}`);
          for (const [group, sections] of Object.entries(metrics.sections)) {
            if (!Array.isArray(sections) || sections.length !== 2) continue;
            const [left, right] = sections;
            if (width >= 900) {
              assert.ok(left.right <= right.left + 1 && Math.abs(left.top - right.top) <= 3, `${name}: ${width}px ${group} uses aligned left and right sections: ${JSON.stringify(sections)}`);
            } else {
              assert.ok(Math.abs(left.left - right.left) <= 1 && right.top >= left.bottom - 1, `${name}: ${width}px ${group} uses one column: ${JSON.stringify(sections)}`);
            }
          }
          if (width >= 900) {
            const { picker, create } = metrics.sections;
            assert.ok(picker.right <= create.left && Math.abs((picker.top + picker.bottom) - (create.top + create.bottom)) <= 6, `${name}: ${width}px team picker and create share a row`);
          }
          layoutChecks += 1;
          if (!directory) continue;
          layoutMetrics.push({ surface: name, size, theme, ...metrics });
          if (['desktop', 'narrow', 'panel', 'mobile'].includes(size)) await page.screenshot({ path: path.join(directory, `${name}-${size}${theme === 'dark' ? '-dark' : ''}.png`) });
          if (size === 'desktop' && (name === 'members' || name.startsWith('rule-text'))) {
            await page.screenshot({ path: path.join(directory, `${name}-desktop-full${theme === 'dark' ? '-dark' : ''}.png`), fullPage: true });
            await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
            await page.screenshot({ path: path.join(directory, `${name}-desktop-bottom${theme === 'dark' ? '-dark' : ''}.png`) });
          }
          if (size === 'mobile') {
            const selector = name.startsWith('rule-text') ? '.space-rule-content' : '#space-body';
            await page.$eval(selector, (element) => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 80));
            await page.screenshot({ path: path.join(directory, `${name}-mobile-content${theme === 'dark' ? '-dark' : ''}.png`) });
          }
        }
        await page.setViewport({ width: 1440, height: 1000 });
      }
      if (await page.$eval('html', (element) => element.dataset.theme) === 'dark') await click('#theme-button');
      await page.evaluate(() => window.scrollTo(0, 0));

    };
    const saved = async (text = '配置已保存') => page.waitForFunction((expected) => document.querySelector('#space-message').textContent.includes(expected), {}, text);
    const pageNumber = async (name, number) => page.waitForFunction((kind, expected) => document.querySelector(`[data-space-page="${kind}"]`)?.textContent.includes(`第 ${expected} 页`), {}, name, number);
    const turnPage = async (name, direction, number) => {
      await click(`[data-space-action="page"][data-value="${name}"][data-direction="${direction}"]`);
      await pageNumber(name, number);
    };
    const values = (selector) => page.$$eval(selector, (elements) => elements.map((element) => element.dataset.value ?? element.value));
    const roundTrip = async (name, selector, expected) => {
      await page.waitForSelector(name === 'space' ? '#space-picker' : selector, { visible: true });
      await pageNumber(name, 1);
      assert.deepEqual(await values(selector), expected.slice(0, 20));
      await turnPage(name, 'next', 2);
      assert.deepEqual(await values(selector), expected.slice(20));
      if (name === 'space') {
        await page.select('#space-picker', expected.at(-1));
        await page.waitForFunction((id) => document.querySelector('#space-picker')?.value === id && document.querySelector('#space-header h2')?.textContent === '团队 22', {}, expected.at(-1));
      }
      if (name === 'rules' || name === 'content') {
        await click(selector);
        await click('[data-space-action="back-list"]');
        await pageNumber(name, 2);
        assert.deepEqual(await values(selector), expected.slice(20));
      }
      await turnPage(name, 'prev', 1);
      assert.deepEqual(await values(selector), expected.slice(0, 20));
    };
    const tabs = async () => assert.deepEqual(await page.$$eval('#space-tabs [data-space-action="tab"]', (elements) => elements.map((element) => [element.dataset.value, element.textContent])), [['content', '团队记忆'], ['members', '成员管理'], ['rules', '规则']]);
    await click('[data-view="spaces"]');
    await page.waitForSelector('#space-tabs [data-space-action="tab"]');
    assert.equal(await page.$eval('#page-title', (element) => element.textContent), '团队空间');
    await tabs();
    await roundTrip('space', '#space-picker option:not(:disabled)', spaceRows.map((entry) => entry.space_id));
    await click('[data-space-action="create"]');
    await fill('[name="space_id"]', 'draft-team');
    await click('[data-space-action="cancel-create"]');
    await inspectConfirmation('团队空间', /尚未保存/);
    await click('[data-space-cancel-delete]');
    assert.equal(await page.$eval('[name="space_id"]', (element) => element.value), 'draft-team');
    await click('[data-space-action="cancel-create"]');
    await click('[data-space-confirm-delete]');
    await page.waitForSelector('#space-tabs [data-space-action="tab"]');
    await click('[data-space-action="create"]');
    await fill('[name="space_id"]', 'ops');
    await fill('[data-space-form="create"] [name="name"]', '数据库运维');
    await click('[data-space-form="create"] [type="submit"]');
    await saved('空间已创建');
    await tabs();
    await page.waitForSelector('#space-principals [data-space-action="add-member"]');
    assert.ok(reads.some((read) => read.path.endsWith('/principals') && read.values.space_id === 'ops' && !read.values.q));
    await roundTrip('principals', '#space-principals [data-space-action="add-member"]', principals.slice(1).map((entry) => entry.principal));
    for (const principal of principals.slice(1, 21)) await click(`[data-space-action="add-member"][data-value="${principal.principal}"]`);
    await turnPage('principals', 'next', 2);
    for (const principal of principals.slice(21)) await click(`[data-space-action="add-member"][data-value="${principal.principal}"]`);
    await pageNumber('members', 1);
    await page.select('[data-member-role="user:bob"]', 'manager');
    await turnPage('members', 'next', 2);
    assert.deepEqual(await values('#space-members [data-space-action="remove-member"]'), ['agent:ops-agent', 'user:carol']);
    await page.select('[data-member-role="user:carol"]', 'manager');
    await turnPage('members', 'prev', 1);
    assert.equal(await page.$eval('[data-member-role="user:bob"]', (element) => element.value), 'manager');
    await click('[data-space-form="members"] [type="submit"]');
    await saved();
    assert.equal(space.member_count, 22);
    assert.deepEqual(space.members.filter((member) => ['user:bob', 'user:carol'].includes(member.principal)).map(({ principal, role }) => [principal, role]), [['user:bob', 'manager'], ['user:carol', 'manager']]);
    await turnPage('members', 'next', 2);
    assert.equal(await page.$eval('[data-member-role="user:carol"]', (element) => element.value), 'manager');
    await turnPage('members', 'prev', 1);
    await page.select('[data-member-role="user:bob"]', 'member');
    await page.select('#space-picker', 'team-01');
    await page.waitForFunction(() => document.querySelector('#space-header h2')?.textContent === '团队 01');
    await page.select('#space-picker', 'ops');
    await page.waitForFunction(() => document.querySelector('[data-member-role="user:bob"]')?.value === 'member');
    await click('[data-space-action="restore-members"]');
    await page.waitForSelector('.space-delete-dialog[open]');
    await click('[data-space-confirm-delete]');
    await page.waitForFunction(() => document.querySelector('[data-member-role="user:bob"]')?.value === 'manager');
    await click('[data-space-action="new-source"]');
    await fill('[name="binding_id"]', 'draft-source');
    await click('[data-space-action="cancel-source"]');
    await inspectConfirmation('来源贡献', /尚未保存/);
    await click('[data-space-cancel-delete]');
    assert.equal(await page.$eval('[name="binding_id"]', (element) => element.value), 'draft-source');
    await click('[data-space-action="cancel-source"]');
    await click('[data-space-confirm-delete]');
    await page.waitForSelector('[data-space-form="source"]', { hidden: true });
    await click('[data-space-action="new-source"]');
    await fill('[name="binding_id"]', 'ops-source');
    await page.select('[data-space-form="source"] [name="principal"]', 'agent:ops-agent');
    await fill('[data-space-form="source"] [name="memory_types"]', 'experience, case');
    await page.$eval('.space-settings', (element) => { element.open = true; });
    await fill('[data-space-form="settings"] [name="name"]', '待保存团队名称');
    await click('[data-space-form="source"] [type="submit"]');
    await inspectConfirmation('团队配置', /尚未保存/, 'multiple-form-guard');
    await click('[data-space-cancel-delete]');
    assert.equal(await page.$eval('[data-space-form="source"] [name="binding_id"]', (element) => element.value), 'ops-source');
    assert.equal(await page.$eval('[data-space-form="settings"] [name="name"]', (element) => element.value), '待保存团队名称');
    await fill('[data-space-form="settings"] [name="name"]', space.name);
    await click('[data-space-form="source"] [type="submit"]');
    await saved('来源贡献已保存');
    await roundTrip('sources', '#space-sources [data-space-action="edit-source"]', space.sources.map((source) => source.binding_id));
    await captureLayout('members');
    await selectTab('rules');
    await click('[data-space-action="new-rule"]');
    const ruleId = await page.$eval('[data-space-form="rule"]', (form) => form.dataset.ruleId);
    assert.match(ruleId, /^[0-9a-f-]{36}$/);
    await fill('[data-space-form="rule"] [name="name"]', '数据库排障');
    await fill('[name="description"]', '团队数据库故障的适用场景、处理步骤与验证方法。');
    await captureLayout('rule-text');
    await click('[data-space-form="rule"] [type="submit"]');
    await saved('规则已保存');
    assert.deepEqual(rule.spec, {
      description: '团队数据库故障的适用场景、处理步骤与验证方法。', output_kind: 'experience', memory_types: ['experience', 'case'],
    });
    await roundTrip('rules', '[data-space-action="edit-rule"]', ruleRows().map((entry) => entry.rule_id));
    await captureLayout('rules-list', false);
    const originalSpec = JSON.stringify(rule.spec);
    await selectTab('members');
    await page.select('[data-member-role="user:bob"]', 'member');
    await click('[data-space-action="remove-member"][data-value="user:bob"]');
    await inspectConfirmation('Bob', /保存成员后.*访问权限.*来源贡献/, 'member-removal-confirmation');
    await click('[data-space-cancel-delete]');
    await page.waitForSelector('.space-delete-dialog', { hidden: true });
    assert.equal(await page.evaluate(() => document.activeElement.dataset.value), 'user:bob');
    assert.equal(await page.$eval('[data-member-role="user:bob"]', (element) => element.value), 'member');
    assert.equal(space.members.find((member) => member.principal === 'user:bob').role, 'manager');
    await click('[data-space-action="remove-member"][data-value="user:bob"]');
    await inspectConfirmation('Bob', /保存成员后/);
    await click('[data-space-confirm-delete]');
    await saved('待保存成员');
    assert.equal(space.members.length, 22);
    assert.equal(await page.$eval('#space-member-count', (element) => element.textContent), '21 位');
    await click('[data-space-form="members"] [type="submit"]');
    await saved();
    assert.deepEqual(space.members.map((member) => member.principal), principals.filter((member) => member.principal !== 'user:bob').map((member) => member.principal));
    await selectTab('rules');
    await click(`[data-space-action="edit-rule"][data-value="${ruleId}"]`);
    assert.equal(await page.$eval('[name="description"]', (element) => element.value), '团队数据库故障的适用场景、处理步骤与验证方法。');
    assert.equal(JSON.stringify(rule.spec), originalSpec);
    assert.equal(rule.rule_rev, 1);
    await fill('[name="description"]', '尚未保存的规则改动');
    await click('[data-view="dashboard"]');
    await inspectConfirmation('规则', /尚未保存/, 'navigation-guard');
    assert.equal(await page.$eval('[data-space-cancel-delete]', (element) => element.textContent), '继续编辑');
    await click('[data-space-cancel-delete]');
    assert.equal(await page.evaluate(() => document.activeElement.dataset.view), 'dashboard');
    assert.equal(await page.$eval('[name="description"]', (element) => element.value), '尚未保存的规则改动');
    await click('#refresh-button');
    await inspectConfirmation('规则', /尚未保存/);
    await click('[data-space-cancel-delete]');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'refresh-button');
    assert.equal(await page.$eval('[name="description"]', (element) => element.value), '尚未保存的规则改动');
    await click('[data-space-action="back-list"]');
    await inspectConfirmation('规则', /尚未保存/);
    await click('[data-space-confirm-delete]');
    await click(`[data-space-action="edit-rule"][data-value="${ruleId}"]`);
    assert.equal(await page.$eval('[name="description"]', (element) => element.value), rule.spec.description);
    await fill('[name="description"]', '离开页面前的规则草稿');
    await click('[data-view="dashboard"]');
    await inspectConfirmation('规则', /尚未保存/);
    await click('[data-space-confirm-delete]');
    await page.waitForSelector('#dashboard-view', { visible: true });
    await click('[data-view="spaces"]');
    await page.waitForSelector('[data-space-action="edit-rule"]', { visible: true });
    await click(`[data-space-action="edit-rule"][data-value="${ruleId}"]`);
    assert.equal(await page.$eval('[name="description"]', (element) => element.value), rule.spec.description);
    const maintenance = { retention_days: 90, archive_keep_days: 30, half_life_days: 90, min_age_days: 30, archive_threshold: 0.2 };
    rule.spec = { instructions: '数据库排障经验定义', source_subjects: ['agent:ops-agent'], output_kind: 'experience', memory_types: ['case'],
      output_schema: { fields: [{ name: 'scenario', type: 'string', required: true }, { name: 'steps', type: 'string_array', required: true }] }, maintenance_policy: maintenance, privacy_policy: { redact_terms: ['张三', '客户A'] } };
    await click('[data-space-action="back-list"]');
    await click('[data-space-action="refresh-tab"]');
    await click(`[data-space-action="edit-rule"][data-value="${ruleId}"]`);
    assert.equal(await page.$eval('[name="description"]', (element) => element.value), '数据库排障经验定义');
    await click('[data-space-form="rule"] [type="submit"]');
    await saved('规则已保存');
    assert.deepEqual(rule.spec, { description: '数据库排障经验定义', source_subjects: ['agent:ops-agent'], output_kind: 'experience', memory_types: ['case'], maintenance_policy: maintenance,
      privacy_policy: { redact_terms: ['张三', '客户A'] } });
    assert.equal(rule.rule_rev, 2);
    await selectTab('content');
    await roundTrip('content', '[data-space-action="item"]', itemRows().map((entry) => entry.item_id));
    await captureLayout('team-memory', true);
    const preview = (entry) => entry.item_id === 'experience-1' ? '适用场景 连接池等待时间持续上升。 处理步骤 检查长事务。 确认 connection.close() 已执行。 验证 SELECT count(*) FROM pg_stat_activity;' : entry.content;
    for (const mode of ['hybrid', 'keyword', 'semantic']) {
      await fill('[name="query"]', '连接池排障');
      await page.select('[data-space-form="search"] [name="mode"]', mode);
      await click('[data-space-form="search"] [type="submit"]');
      await page.waitForSelector('.space-cards [data-space-item-content]');
      await pageNumber('search', 1);
      assert.deepEqual(await page.$$eval('[data-space-item-content]', (elements) => elements.map((element) => element.textContent)), itemRows().slice(0, 20).map(preview));
      await turnPage('search', 'next', 2);
      assert.deepEqual(await page.$$eval('[data-space-item-content]', (elements) => elements.map((element) => element.textContent)), itemRows().slice(20).map(preview));
      await click('[data-space-action="item"]');
      await click('[data-space-action="back-list"]');
      await pageNumber('search', 2);
      assert.equal(await page.$eval('[data-space-form="search"] [name="query"]', (element) => element.value), '连接池排障');
      assert.equal(await page.$eval('[data-space-form="search"] [name="mode"]', (element) => element.value), mode);
      assert.deepEqual(await page.$$eval('[data-space-item-content]', (elements) => elements.map((element) => element.textContent)), itemRows().slice(20).map(preview));
      await turnPage('search', 'prev', 1);
      assert.equal(await page.$eval('[data-space-form="search"] [name="mode"]', (element) => element.value), mode);
      assert.deepEqual(await page.$$eval('[data-space-item-content]', (elements) => elements.map((element) => element.textContent)), itemRows().slice(0, 20).map(preview));
      assert.deepEqual(writes.filter((entry) => entry.path === '/api/spaces/search').slice(-3).map((entry) => ({ mode: entry.body.mode, cursor: entry.body.cursor || '' })), [{ mode, cursor: '' }, { mode, cursor: 'search:20' }, { mode, cursor: '' }]);
    }
    await click('[data-space-action="refresh-tab"]');
    await click('[data-space-action="item"][data-value="experience-1"]');
    await page.waitForSelector('[data-space-action="edit-acl"]', { visible: true });
    assert.deepEqual(await page.$$eval('.space-memory-content .space-prose :is(h2, h3, h4, h5, h6)', (elements) => elements.map((element) => element.textContent)), ['适用场景', '处理步骤', '验证']);
    assert.equal(await page.$eval('.space-memory-content .space-prose strong', (element) => element.textContent), '长事务');
    assert.equal(await page.$$eval('.space-memory-content .space-prose li', (elements) => elements.length), 2);
    assert.equal(await page.$eval('.space-memory-content .space-prose pre code', (element) => element.textContent.trim()), 'SELECT count(*) FROM pg_stat_activity;');
    await captureLayout('memory-detail');
    await click('[data-space-action="permissions"]');
    await page.waitForFunction(() => document.querySelector('#space-permissions')?.textContent.includes('可以读取此记忆'));
    await click('[data-space-action="edit-acl"]');
    await page.waitForSelector('[data-space-form="acl"] [name="acl_mode"]', { visible: true });
    await page.select('[data-space-form="acl"] [name="acl_mode"]', 'restricted');
    await click('[data-space-form="acl"] [name="readers"][value="agent:ops-agent"]');
    await click('[data-space-action="cancel-acl"]');
    await inspectConfirmation('读取权限', /尚未保存/);
    await click('[data-space-cancel-delete]');
    assert.equal(await page.$eval('[name="readers"][value="agent:ops-agent"]', (element) => element.checked), true);
    await click('[data-space-action="cancel-acl"]');
    await click('[data-space-confirm-delete]');
    await page.waitForSelector('[data-space-action="edit-acl"]', { visible: true });
    await click('[data-space-action="edit-acl"]');
    await page.select('[data-space-form="acl"] [name="acl_mode"]', 'restricted');
    await click('[data-space-form="acl"] [name="readers"][value="agent:ops-agent"]');
    await page.select('[data-space-form="acl"] [name="acl_mode"]', 'inherit');
    assert.equal(await page.$eval('.space-acl-readers', (element) => element.hidden && element.disabled), true);
    await page.select('[data-space-form="acl"] [name="acl_mode"]', 'restricted');
    assert.equal(await page.$eval('[name="readers"][value="agent:ops-agent"]', (element) => element.checked), true);
    const aclActions = await page.$$eval('[data-space-form="acl"] .space-form-actions button', (elements) => elements.map((element) => ({ text: element.textContent, top: element.getBoundingClientRect().top })));
    assert.equal(aclActions.length, 2);
    assert.ok(Math.abs(aclActions[0].top - aclActions[1].top) <= 1, 'ACL cancel and save share one action row');
    await captureLayout('memory-permissions-edit', false);
    await click('[data-space-form="acl"] [type="submit"]');
    await saved('读取权限已保存');
    await page.waitForSelector('[data-space-action="edit-acl"]', { visible: true });
    assert.match(await page.$eval('#space-body', (element) => element.textContent), /连接池排障/);
    assert.deepEqual(item.readers, ['agent:ops-agent']);
    await click('[data-space-action="edit-acl"]');
    assert.equal(await page.$eval('[data-space-form="acl"] [name="acl_mode"]', (element) => element.value), 'restricted');
    assert.equal(await page.$eval('[name="readers"][value="agent:ops-agent"]', (element) => element.checked), true);
    await click('[data-space-action="cancel-acl"]');
    await selectTab('members');
    await page.$eval('[data-space-form="settings"]', (form) => { form.closest('details').open = true; });
    await fill('[data-space-form="settings"] [name="name"]', '数据库运维经验库');
    await click('[data-space-form="settings"] [type="submit"]');
    await saved();
    assert.equal(space.name, '数据库运维经验库');
    await selectTab('rules');
    await click('[data-space-action="new-rule"]');
    await fill('[data-space-form="rule"] [name="name"]', '连接池故障复盘');
    await fill('[name="description"]', '整理连接池排障中可复用的判断依据与经验，保留自然段和必要的操作清单。');
    await fill('[data-space-form="rule"] [name="memory_types"]', 'case');
    await click('[data-space-form="rule"] [type="submit"]');
    await saved('规则已保存');
    assert.deepEqual(rule.spec, { description: '整理连接池排障中可复用的判断依据与经验，保留自然段和必要的操作清单。', output_kind: 'experience', memory_types: ['case'] });
    await page.setViewport({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const confirmDelete = async (name, impact, savedText) => {
      await page.waitForSelector('.space-delete-dialog[open]');
      assert.equal(await page.$eval('.space-delete-name', (element) => element.textContent), name);
      assert.match(await page.$eval('#space-delete-impact', (element) => element.textContent), impact);
      assert.equal(await page.$eval('[data-space-cancel-delete]', (element) => element.textContent), '取消');
      assert.ok(await page.$eval('.space-delete-dialog', (element) => element.getBoundingClientRect().width <= innerWidth));
      if (directory) await page.screenshot({ path: path.join(directory, `delete-${writes.filter((entry) => entry.method === 'DELETE').length + 1}.png`) });
      await click('[data-space-confirm-delete]');
      await page.waitForFunction(() => document.querySelector('.space-delete-dialog form')?.getAttribute('aria-busy') === 'true');
      assert.equal(await page.$eval('[data-space-confirm-delete]', (element) => element.disabled), true);
      await saved(savedText);
    };
    await turnPage('rules', 'next', 2);
    const removedRule = ruleRows().at(-1);
    await click(`[data-space-action="edit-rule"][data-value="${removedRule.rule_id}"]`);
    await click('[data-space-action="delete-rule"]');
    await confirmDelete(removedRule.name, /停止生成，已有团队记忆由系统归档清理/, '规则已删除');
    await pageNumber('rules', 1);
    assert.deepEqual(await values('[data-space-action="edit-rule"]'), ruleRows().slice(0, 20).map((entry) => entry.rule_id));
    assert.match(await page.$eval('[data-space-page="rules"]', (element) => element.textContent), /共 21 条/);
    await selectTab('members');
    await turnPage('sources', 'next', 2);
    const removedSource = space.sources.at(-1);
    await page.$eval('.space-settings', (element) => { element.open = true; });
    await fill('[data-space-form="settings"] [name="name"]', '来源删除前的团队配置草稿');
    await click(`[data-space-action="delete-source"][data-value="${removedSource.binding_id}"]`);
    await inspectConfirmation(removedSource.principal, /同时放弃其他未保存的更改/, 'source-delete-with-draft');
    await click('[data-space-cancel-delete]');
    assert.equal(await page.$eval('[data-space-form="settings"] [name="name"]', (element) => element.value), '来源删除前的团队配置草稿');
    await click(`[data-space-action="delete-source"][data-value="${removedSource.binding_id}"]`);
    await confirmDelete(removedSource.principal, /撤销来源贡献，系统重算相关团队记忆/, '来源已删除');
    assert.equal(await page.$eval('[data-space-form="settings"] [name="name"]', (element) => element.value), space.name);
    await pageNumber('sources', 1);
    assert.deepEqual(await values('#space-sources [data-space-action="edit-source"]'), space.sources.slice(0, 20).map((entry) => entry.binding_id));
    assert.match(await page.$eval('[data-space-page="sources"]', (element) => element.textContent), /共 21 条/);
    await page.$eval('.space-settings', (element) => { element.open = true; });
    await click('[data-space-action="delete-space"]');
    await confirmDelete(space.name, /团队成员的访问将终止，团队记忆由系统归档清理/, '团队空间已删除');
    await pageNumber('space', 1);
    assert.deepEqual(await values('#space-picker option:not(:disabled)'), spaceRows.slice(0, 20).map((entry) => entry.space_id));
    assert.equal(await page.$eval('#space-picker', (element) => element.value), 'team-01');
    assert.equal(await page.$eval('#space-header h2', (element) => element.textContent), '团队 01');
    t.diagnostic(`Layout checks passed: ${layoutChecks}`);
    if (directory) await fs.writeFile(path.join(directory, 'layout-metrics.json'), JSON.stringify(layoutMetrics, null, 2));
    assert.equal(pageErrors.length, 0, pageErrors.join('\n'));
    for (const write of writes.filter((entry) => entry.path.startsWith('/management/'))) {
      assert.equal(write.headers['x-api-key'], 'alice-test-key');
      assert.match(write.headers['idempotency-key'], /^[0-9a-f-]{36}$/);
    }
    for (const route of ['/management/spaces', '/management/spaces/principals', '/management/spaces/ops/items', '/management/spaces/ops/rules']) assert.ok(reads.some((entry) => entry.path === route && entry.values.cursor));
    const ruleWrite = writes.find((entry) => entry.path.includes('/rules/'));
    assert.equal(ruleWrite.headers['if-none-match'], '*');
    assert.equal(writes.filter((entry) => entry.path.includes('/rules/') && entry.method === 'PUT').length, 3);
    assert.deepEqual(writes.filter((entry) => entry.method === 'DELETE').map((entry) => entry.path), ['/management/spaces/ops/rules/rule-21', '/management/spaces/ops/sources/source-21', '/management/spaces/ops']);
  } finally {
    try {
      await browser?.close();
    } finally {
      await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    }
  }
});
