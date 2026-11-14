// SPACE_QA_CONFIG supplies an isolated openGauss + AGFS deployment and registered identities.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright-core');

test('registered members, team permissions and lifecycle through real browser sessions', {
  skip: !process.env.SPACE_QA_CONFIG, timeout: 420000,
}, async (t) => {
  const config = JSON.parse(await fs.readFile(process.env.SPACE_QA_CONFIG, 'utf8'));
  const origin = new URL(config.url).origin;
  const directory = process.env.SPACE_QA_REPORT || path.resolve('reports/team-fix-20261114/permissions-ui');
  await fs.mkdir(directory, { recursive: true });
  const spaceId = `permissions-${Date.now()}`;
  const route = `/management/spaces/${spaceId}`;
  const evidence = { space_id: spaceId, phases: [], calls: [], browser_calls: [], layouts: [], errors: [] };
  const keys = { alice: config.alice_key, bob: config.bob_key, carol: config.carol_key, other: config.other_key };
  const api = async (actor, endpoint, options = {}) => {
    const started = performance.now();
    const response = await fetch(`${origin}${endpoint}`, {
      ...options, headers: { 'X-API-Key': keys[actor], 'Content-Type': 'application/json', ...options.headers },
    });
    const result = await response.json();
    evidence.calls.push({ actor, endpoint, method: options.method || 'GET', status: response.status, ms: +(performance.now() - started).toFixed(1) });
    assert.equal(response.ok, true, `${actor} ${endpoint}: ${response.status} ${JSON.stringify(result)}`);
    return result;
  };
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH, headless: true });
  t.after(() => browser.close());
  const pages = {};
  const phase = (name, details = {}) => { evidence.phases.push({ name, ...details }); console.log(name); };
  const action = (page, name) => page.locator(`[data-space-action="${name}"]`);
  const save = async (page, form, text = '配置已保存') => {
    await page.locator(`[data-space-form="${form}"] [type="submit"]`).click();
    await page.waitForFunction(expected => {
      const message = document.querySelector('#space-message');
      return message.classList.contains('error') || message.textContent.includes(expected);
    }, text);
    assert.equal(await page.locator('#space-message').getAttribute('class'), 'space-message', await page.locator('#space-message').textContent());
  };
  const tab = (page, name) => page.locator(`[data-space-action="tab"][data-value="${name}"]`).click();
  const openTeam = async page => {
    const target = `${origin}/memory-viewer/#spaces`;
    if (page.url() === target) await page.reload();
    else await page.goto(target);
    await page.locator(`#space-picker option[value="${spaceId}"]`).waitFor({ state: 'attached' });
    await page.locator('#space-picker').selectOption(spaceId);
    await page.waitForFunction(id => document.querySelector('#space-header h2')?.textContent === `权限验收 ${id}`, spaceId);
    await tab(page, 'content');
    await page.locator('.space-cards').waitFor();
  };
  const newPage = async actor => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    await context.addInitScript(key => sessionStorage.setItem('ogmemory-viewer-api-key', key), keys[actor]);
    const page = await context.newPage();
    page.on('pageerror', error => evidence.errors.push({ actor, message: error.message }));
    page.on('response', response => {
      const endpoint = new URL(response.url()).pathname;
      if (endpoint.startsWith('/management/spaces') || endpoint.startsWith('/api/spaces'))
        evidence.browser_calls.push({ actor, endpoint, method: response.request().method(), status: response.status() });
    });
    pages[actor] = page;
    return page;
  };
  const capture = async (page, name) => {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('DOM.enable');
    await cdp.send('CSS.enable');
    const { root } = await cdp.send('DOM.getDocument');
    const fonts = [];
    for (const selector of ['#space-header h2', '#space-tabs button', '.space-member-identity strong', '.space-prose']) {
      const { nodeId } = await cdp.send('DOM.querySelector', { nodeId: root.nodeId, selector });
      if (nodeId) fonts.push({ selector, ...(await cdp.send('CSS.getPlatformFontsForNode', { nodeId })) });
    }
    await cdp.detach();
    evidence.fonts = [...(evidence.fonts || []), { name, fonts }];
    for (const theme of ['light', 'dark']) {
      if (await page.locator('html').getAttribute('data-theme') !== theme) await page.locator('#theme-button').click();
      for (const width of [1440, 1024, 390]) {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
        await page.evaluate(() => { document.activeElement.blur(); window.scrollTo(0, 0); });
        const layout = await page.evaluate(() => ({ width: innerWidth, pageWidth: document.documentElement.scrollWidth }));
        assert.ok(layout.pageWidth <= width, `${name}: fits ${width}px`);
        evidence.layouts.push({ name, theme, ...layout });
        await page.screenshot({ path: path.join(directory, `${name}-${theme}-${width}.png`), fullPage: true });
      }
      await page.setViewportSize({ width: 1440, height: 1000 });
    }
    await page.locator('#theme-button').click();
  };
  const privacy = text => { assert.match(text, /\[邮箱\]/); assert.match(text, /\[手机号\]/); };
  let item;
  const checkApiReads = async (actor, expected, stage) => {
    const results = [];
    for (const mode of ['keyword', 'semantic', 'hybrid']) {
      const result = await api(actor, '/api/spaces/search', { method: 'POST', body: JSON.stringify({
        query: '连接池排障', space_ids: [spaceId], mode, top_k: 100, limit: 20, max_context_tokens: 8192,
      }) });
      assert.equal(result.items.some(row => row.item_id === item.item_id), expected, `${stage}: ${actor} ${mode}`);
      for (const row of result.items) privacy(row.content);
      results.push({ mode, count: result.items.length });
    }
    phase(stage, { actor, results });
  };
  const checkUiSearch = async (page, expected) => {
    await tab(page, 'content');
    for (const mode of ['keyword', 'semantic', 'hybrid']) {
      await page.getByLabel('团队记忆检索').fill('连接池排障');
      await page.locator('[data-space-form="search"] [name="mode"]').selectOption(mode);
      const response = page.waitForResponse(result => new URL(result.url()).pathname === '/api/spaces/search' && result.request().method() === 'POST');
      await page.locator('[data-space-form="search"] [type="submit"]').click();
      const result = await (await response).json();
      assert.equal(result.items.some(row => row.item_id === item.item_id), expected);
      if (expected) await page.locator(`[data-space-action="item"][data-value="${item.item_id}"]`).waitFor();
      else await page.getByText('暂无匹配的团队记忆', { exact: true }).waitFor();
    }
  };
  const setReaders = async (page, readers) => {
    await tab(page, 'content');
    await page.locator(`[data-space-action="item"][data-value="${item.item_id}"]`).click();
    await action(page, 'edit-acl').click();
    await page.locator('[name="acl_mode"]').selectOption('restricted');
    const controls = page.locator('[data-space-form="acl"] [name="readers"]');
    for (const input of await controls.all()) await input.setChecked(readers.includes(await input.getAttribute('value')));
    await save(page, 'acl', '读取权限已保存');
  };
  const confirm = async (page, name) => {
    const dialog = page.locator('.space-delete-dialog[open]');
    await dialog.waitFor();
    assert.equal(await dialog.locator('.space-delete-name').textContent(), name);
    await dialog.locator('[data-space-confirm-delete]').click();
    await dialog.waitFor({ state: 'detached' });
  };
  try {
    const alice = await newPage('alice');
    await alice.goto(`${origin}/memory-viewer/#spaces`);
    await action(alice, 'create').click();
    await alice.locator('[data-space-form="create"] [name="space_id"]').fill(spaceId);
    await alice.locator('[data-space-form="create"] [name="name"]').fill(`权限验收 ${spaceId}`);
    await save(alice, 'create', '空间已创建');
    for (const principal of ['user:bob', 'user:carol', 'agent:ops-agent']) {
      await alice.locator(`[data-space-action="add-member"][data-value="${principal}"]`).click();
    }
    await capture(alice, 'registered-member-selection');
    await save(alice, 'members');
    const members = (await api('alice', `/management/spaces?space_id=${spaceId}`)).space.members;
    assert.deepEqual(new Set(members.map(member => member.principal)), new Set(['user:alice', 'user:bob', 'user:carol', 'agent:ops-agent']));
    phase('registered_members_selected_and_saved');
    await action(alice, 'new-source').click();
    await alice.locator('[name="binding_id"]').fill('ops-source');
    await alice.locator('[data-space-form="source"] [name="principal"]').selectOption('agent:ops-agent');
    await alice.locator('[data-space-form="source"] [name="memory_types"]').fill('case');
    await save(alice, 'source', '来源贡献已保存');
    await tab(alice, 'rules');
    await action(alice, 'new-rule').click();
    const ruleId = await alice.locator('[data-space-form="rule"]').getAttribute('data-rule-id');
    await alice.locator('[data-space-form="rule"] [name="name"]').fill('连接池排障经验');
    await alice.locator('[data-space-form="rule"] [name="memory_types"]').fill('case');
    await alice.getByLabel('规则内容', { exact: true }).fill('沉淀连接池排障的适用场景、处理经验与恢复验证。');
    assert.equal(await alice.getByText(/隐私|脱敏|遮蔽/).count(), 0);
    await capture(alice, 'text-rule');
    await save(alice, 'rule', '规则已保存');
    const rule = (await api('alice', `${route}/rules?rule_id=${ruleId}`)).items[0];
    assert.equal(rule.spec.output_kind, 'experience');
    assert.deepEqual(rule.spec.privacy_policy, { redact_terms: [] });
    const deadline = Date.now() + 180000;
    while (Date.now() < deadline) {
      item = (await api('alice', `${route}/items?state=ACTIVE`)).items.find(row => row.index_state === 'READY' && row.rule_id === ruleId);
      if (item) break;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    assert.ok(item, 'system publishes an indexed team memory');
    privacy(item.content);
    phase('generation_and_index_ready', { item_id: item.item_id, content: item.content });
    for (const actor of ['bob', 'carol']) {
      const page = await newPage(actor);
      await openTeam(page);
      await page.locator(`[data-space-action="item"][data-value="${item.item_id}"]`).click();
      privacy(await page.locator('.space-prose').textContent());
      assert.equal(await action(page, 'edit-acl').count(), 0);
      await tab(page, 'members');
      assert.equal(await page.locator('[data-space-form="members"] [type="submit"]').count(), 0);
      const permissions = await api(actor, `${route}/permissions?item_id=${item.item_id}`);
      assert.equal(permissions.can_read, true);
      assert.equal(permissions.capabilities.manage_members, false);
      assert.equal(permissions.capabilities.manage_acl, false);
      await tab(page, 'rules');
      await action(page, 'edit-rule').click();
      assert.equal(await page.getByLabel('规则内容', { exact: true }).isDisabled(), true);
      assert.equal(await action(page, 'delete-rule').count(), 0);
    }
    phase('inherited_read_and_member_ui');
    await setReaders(alice, ['user:bob']);
    await capture(alice, 'restricted-memory');
    await action(alice, 'edit-acl').click();
    await capture(alice, 'reader-editor');
    await action(alice, 'cancel-acl').click();
    await checkApiReads('bob', true, 'restricted_reader');
    await checkUiSearch(pages.bob, true);
    await checkApiReads('carol', false, 'restricted_membership');
    await checkUiSearch(pages.carol, false);
    assert.equal((await api('carol', `${route}/permissions?item_id=${item.item_id}`)).can_read, false);
    assert.equal((await api('carol', `${route}/items?state=ACTIVE`)).items.length, 0);
    await checkApiReads('other', false, 'tenant_boundary');
    await setReaders(alice, ['user:bob', 'user:carol']);
    await checkApiReads('carol', true, 'expanded_readers');
    await checkUiSearch(pages.carol, true);
    await tab(alice, 'members');
    await alice.locator('[data-space-action="remove-member"][data-value="user:bob"]').click();
    await confirm(alice, 'bob');
    await save(alice, 'members');
    await checkApiReads('bob', false, 'member_removed');
    await pages.bob.locator('#refresh-button').click();
    await pages.bob.getByText('团队空间暂不可用', { exact: true }).waitFor();
    assert.equal(await action(pages.bob, 'item').count(), 0);
    await alice.locator('[data-space-action="add-member"][data-value="user:bob"]').click();
    await save(alice, 'members');
    await setReaders(alice, ['user:bob', 'user:carol']);
    await openTeam(pages.bob);
    await checkUiSearch(pages.bob, true);
    await checkApiReads('bob', true, 'member_rejoined');
    const currentRule = (await api('alice', `${route}/rules?rule_id=${ruleId}`)).items[0];
    assert.equal(currentRule.spec_hash, rule.spec_hash);
    assert.equal(currentRule.rule_rev, rule.rule_rev);
    await tab(alice, 'rules');
    await alice.locator(`[data-space-action="edit-rule"][data-value="${ruleId}"]`).click();
    await action(alice, 'delete-rule').click();
    await confirm(alice, '连接池排障经验');
    await alice.getByText('规则已删除。', { exact: true }).waitFor();
    assert.equal((await api('alice', `${route}/rules`)).items.length, 0);
    await tab(alice, 'members');
    await alice.locator('.space-settings > summary').click();
    await action(alice, 'delete-space').click();
    await confirm(alice, `权限验收 ${spaceId}`);
    await alice.getByText('团队空间已删除。', { exact: true }).waitFor();
    await checkApiReads('bob', false, 'team_deleted');
    assert.deepEqual(evidence.errors, []);
    phase('completed', { rule_id: ruleId, layouts: evidence.layouts.length });
  } catch (error) {
    evidence.failure = error.message;
    for (const [actor, page] of Object.entries(pages)) await page.screenshot({ path: path.join(directory, `failure-${actor}.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    try {
      await fs.writeFile(path.join(directory, 'evidence.json'), JSON.stringify(evidence, null, 2));
    } catch (error) {
      if (!Object.hasOwn(evidence, 'failure')) throw error;
      t.diagnostic(`Could not write evidence.json: ${error.message}`);
    }
  }
});
