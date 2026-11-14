// Supply a QA deployment through SPACE_QA_CONFIG: {url, alice_key, bob_key}.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs/promises');
const nodePath = require('node:path');
const puppeteer = require('puppeteer');

test('team-space browser flow with openGauss, AGFS, member pages, retrieval and deletion', { skip: !process.env.SPACE_QA_CONFIG, timeout: 420000 }, async () => {
  const config = JSON.parse(await fs.readFile(process.env.SPACE_QA_CONFIG, 'utf8'));
  const origin = new URL(config.url).origin;
  const spaceId = `qa-ops-${Date.now()}`;
  const path = `/management/spaces/${spaceId}`;
  const api = async (route, key = config.alice_key, options = {}) => {
    const response = await fetch(`${origin}${route}`, { ...options, headers: { 'X-API-Key': key, 'Content-Type': 'application/json', ...options.headers } });
    const result = await response.json();
    assert.equal(response.ok, true, `${route}: ${response.status} ${JSON.stringify(result)}`);
    return result;
  };
  const browser = await puppeteer.launch({ headless: true, executablePath: process.env.CHROME_PATH, args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 1000 });
    const errors = [];
    const searches = [];
    const deletions = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('request', (request) => {
      if (new URL(request.url()).pathname === '/api/spaces/search' && request.method() === 'POST') searches.push(JSON.parse(request.postData()));
      if (request.method() === 'DELETE') deletions.push({ path: new URL(request.url()).pathname, revision: request.headers()['if-match'], idempotencyKey: request.headers()['idempotency-key'] });
    });
    await page.evaluateOnNewDocument((key) => sessionStorage.setItem('ogmemory-viewer-api-key', key), config.alice_key);
    await page.goto(`${origin}/memory-viewer/#spaces`);
    const click = async (selector) => { await page.waitForSelector(selector, { visible: true }); await page.click(selector); };
    const fill = async (selector, value) => {
      await page.waitForSelector(selector, { visible: true });
      await page.$eval(selector, (element, text) => { element.value = text; element.dispatchEvent(new Event('input', { bubbles: true })); }, value);
    };
    const captureLayout = async (name, dark = false) => {
      if (!process.env.SPACE_UI_SCREENSHOT) return;
      const directory = nodePath.dirname(process.env.SPACE_UI_SCREENSHOT);
      await fs.mkdir(directory, { recursive: true });
      for (const theme of dark ? ['light', 'dark'] : ['light']) {
        if (await page.$eval('html', (element) => element.dataset.theme) !== theme) await click('#theme-button');
        for (const [size, width, height] of [['desktop', 1440, 1000], ['narrow', 1024, 1000], ['mobile', 390, 844]]) {
          await page.setViewport({ width, height });
          await page.evaluate(() => { document.activeElement.blur(); window.scrollTo(0, 0); });
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name}: ${size} fits viewport`);
          await page.screenshot({ path: nodePath.join(directory, `live-${name}-${size}${theme === 'dark' ? '-dark' : ''}.png`) });
          if (size === 'mobile') {
            await page.$eval('#space-body', (element) => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 80));
            await page.screenshot({ path: nodePath.join(directory, `live-${name}-mobile-content${theme === 'dark' ? '-dark' : ''}.png`) });
          }
        }
        await page.setViewport({ width: 1440, height: 1000 });
      }
      if (await page.$eval('html', (element) => element.dataset.theme) === 'dark') await click('#theme-button');
      await page.evaluate(() => { document.activeElement.blur(); window.scrollTo(0, 0); });
    };
    const saved = async (text = '配置已保存') => {
      await page.waitForFunction((expected) => {
        const message = document.querySelector('#space-message');
        return message.classList.contains('error') || message.textContent.includes(expected);
      }, {}, text);
      assert.equal(await page.$eval('#space-message', (element) => element.classList.contains('error')), false, await page.$eval('#space-message', (element) => element.textContent));
    };
    const turnPage = async (name, direction, number) => {
      await click(`[data-space-action="page"][data-value="${name}"][data-direction="${direction}"]`);
      await page.waitForFunction((kind, expected) => document.querySelector(`[data-space-page="${kind}"]`)?.textContent.includes(`第 ${expected} 页`), {}, name, number);
    };
    const directory = async (query) => {
      await fill('[data-space-form="principals"] [name="q"]', query);
      await click('[data-space-form="principals"] [type="submit"]');
      await page.waitForFunction((text) => document.querySelector('#space-principals').textContent.includes(text), {}, query);
    };
    await page.waitForSelector('#spaces-view', { visible: true });
    assert.equal(await page.$eval('#page-title', (element) => element.textContent), '团队空间');
    await click('[data-space-action="create"]');
    await fill('[data-space-form="create"] [name="space_id"]', spaceId);
    await fill('[data-space-form="create"] [name="name"]', `团队空间验收 ${spaceId}`);
    await click('[data-space-form="create"] [type="submit"]');
    await saved('空间已创建');
    assert.deepEqual(await page.$$eval('#space-tabs [data-space-action="tab"]', (elements) => elements.map((element) => [element.dataset.value, element.textContent])), [['content', '团队记忆'], ['members', '成员管理'], ['rules', '规则']]);
    await directory('bob');
    await click('[data-space-action="add-member"][data-value="user:bob"]');
    await directory('ops-agent');
    await click('[data-space-action="add-member"][data-value="agent:ops-agent"]');
    await directory('qa-page-');
    const registered = Array.from({ length: 21 }, (_, index) => `user:qa-page-${String(index + 1).padStart(2, '0')}`);
    const selectedPrincipals = () => page.$$eval('#space-principals [data-space-action="add-member"]', (elements) => elements.map((element) => element.dataset.value));
    assert.deepEqual(await selectedPrincipals(), registered.slice(0, 20));
    for (const principal of registered.slice(0, 20)) await click(`[data-space-action="add-member"][data-value="${principal}"]`);
    await turnPage('principals', 'next', 2);
    assert.deepEqual(await selectedPrincipals(), registered.slice(20));
    await click('[data-space-action="add-member"][data-value="user:qa-page-21"]');
    await turnPage('principals', 'prev', 1);
    assert.deepEqual(await selectedPrincipals(), registered.slice(0, 20));
    await page.select('[data-member-role="user:qa-page-01"]', 'manager');
    await turnPage('members', 'next', 2);
    assert.equal(await page.$$eval('#space-members .space-member-row', (elements) => elements.length), 4);
    await page.select('[data-member-role="user:qa-page-21"]', 'manager');
    await turnPage('members', 'prev', 1);
    assert.equal(await page.$$eval('#space-members .space-member-row', (elements) => elements.length), 20);
    assert.equal(await page.$eval('[data-member-role="user:qa-page-01"]', (element) => element.value), 'manager');
    await click('[data-space-form="members"] [type="submit"]');
    await saved();
    const members = (await api(`/management/spaces?space_id=${spaceId}`)).space.members;
    assert.equal(members.length, 24);
    assert.deepEqual(members.filter((member) => ['user:qa-page-01', 'user:qa-page-21'].includes(member.principal)).map(({ principal, role }) => [principal, role]), [['user:qa-page-01', 'manager'], ['user:qa-page-21', 'manager']]);
    await turnPage('members', 'next', 2);
    assert.equal(await page.$eval('[data-member-role="user:qa-page-21"]', (element) => element.value), 'manager');
    await turnPage('members', 'prev', 1);
    await page.select('[data-member-role="user:qa-page-01"]', 'member');
    await click('[data-space-action="restore-members"]');
    await page.waitForSelector('.space-delete-dialog[open]');
    await click('[data-space-confirm-delete]');
    await page.waitForSelector('.space-delete-dialog[open]', { hidden: true });
    await page.waitForFunction(() => document.querySelector('[data-member-role="user:qa-page-01"]')?.value === 'manager');
    console.log('Registered member pages and saved roles verified.');
    await click('[data-space-action="new-source"]');
    await fill('[name="binding_id"]', 'ops-source');
    await page.select('[data-space-form="source"] [name="principal"]', 'agent:ops-agent');
    await fill('[data-space-form="source"] [name="memory_types"]', 'case');
    await click('[data-space-form="source"] [type="submit"]');
    await saved('来源贡献已保存');
    await captureLayout('members');
    await click('[data-space-action="tab"][data-value="rules"]');
    await click('[data-space-action="new-rule"]');
    const ruleId = await page.$eval('[data-space-form="rule"]', (form) => form.dataset.ruleId);
    assert.match(ruleId, /^[0-9a-f-]{36}$/);
    await fill('[data-space-form="rule"] [name="name"]', '连接池排障经验');
    await fill('[data-space-form="rule"] [name="memory_types"]', 'case');
    await fill('[name="description"]', '连接池故障的适用场景、处理步骤与验证方法。');
    await captureLayout('rule-text', true);
    await click('[data-space-form="rule"] [type="submit"]');
    await saved('规则已保存');
    const created = (await api(`${path}/rules?rule_id=${ruleId}`)).items[0];
    assert.deepEqual(created.spec.privacy_policy, { redact_terms: [] });
    await api(`${path}/rules/${ruleId}`, config.alice_key, {
      method: 'PUT', headers: { 'If-Match': `"${created.rule_rev}"`, 'Idempotency-Key': `${spaceId}-privacy` },
      body: JSON.stringify({ name: created.name, state: created.state,
        spec: { ...created.spec, privacy_policy: { redact_terms: ['张三', '客户A'] } } }),
    });
    await click('[data-space-action="refresh-tab"]');
    await click(`[data-space-action="edit-rule"][data-value="${ruleId}"]`);
    await fill('[name="description"]', '连接池故障的适用场景、排查经验与恢复验证。');
    await click('[data-space-form="rule"] [type="submit"]');
    await saved('规则已保存');
    const original = (await api(`${path}/rules?rule_id=${ruleId}`)).items[0];
    assert.equal(original.spec.description, '连接池故障的适用场景、排查经验与恢复验证。');
    assert.equal(original.spec.output_kind, 'experience');
    assert.deepEqual(new Set(original.spec.privacy_policy.redact_terms), new Set(['张三', '客户A']));
    const bobDefinition = (await api(`${path}/rules?rule_id=${ruleId}`, config.bob_key)).items[0];
    assert.equal(bobDefinition.rule_id, ruleId);
    assert.equal(bobDefinition.name, original.name);
    assert.deepEqual(bobDefinition.spec, original.spec);
    console.log('Team definition saved. Waiting for system generation and index.');
    let item;
    const deadline = Date.now() + 300000;
    while (Date.now() < deadline) {
      const items = (await api(`${path}/items`)).items;
      item = items.find((entry) => entry.state === 'ACTIVE' && entry.index_state === 'READY');
      if (item) break;
      const jobs = (await api(`${path}/jobs?rule_id=${ruleId}`)).items;
      const failed = jobs.find((job) => job.state === 'FAILED');
      if (failed) throw new Error(`System task failed: ${failed.last_error}`);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    assert.ok(item, 'Team memory becomes ACTIVE and READY');
    const jobs = (await api(`${path}/jobs?rule_id=${ruleId}`)).items;
    assert.ok(jobs.some((job) => job.state === 'SUCCEEDED'));
    console.log('System generation and index READY.');
    await click('[data-space-action="tab"][data-value="content"]');
    await click(`[data-space-action="item"][data-value="${item.item_id}"]`);
    await page.waitForSelector('[data-space-action="edit-acl"]', { visible: true });
    assert.match(await page.$eval('.space-memory-content .space-prose', (element) => element.textContent), /连接|排障/);
    await captureLayout('memory-detail');
    await click('[data-space-action="edit-acl"]');
    await page.waitForSelector('[data-space-form="acl"]', { visible: true });
    await page.select('[data-space-form="acl"] [name="acl_mode"]', 'restricted');
    await click('[data-space-form="acl"] [name="readers"][value="user:bob"]');
    await click('[data-space-form="acl"] [type="submit"]');
    await saved('读取权限已保存');
    await page.waitForSelector('[data-space-action="edit-acl"]', { visible: true });
    assert.match(await page.$eval('#space-body', (element) => element.textContent), /连接池排障经验/);
    const query = { query: '连接池排障', space_ids: [spaceId], mode: 'hybrid', top_k: 100, limit: 20, max_context_tokens: 4096 };
    const found = await api('/api/spaces/search', config.bob_key, { method: 'POST', body: JSON.stringify(query) });
    assert.ok(found.items.some((result) => result.item_id === item.item_id && result.content));
    assert.ok(found.total >= found.items.length);
    assert.ok(Object.hasOwn(found, 'next_cursor'));
    const permissions = await api(`${path}/permissions?item_id=${item.item_id}`, config.bob_key);
    assert.equal(permissions.can_read, true);
    await click('[data-space-action="tab"][data-value="members"]');
    await page.select('[data-member-role="user:bob"]', 'manager');
    await click('[data-space-form="members"] [type="submit"]');
    await saved();
    const latest = (await api(`${path}/rules?rule_id=${ruleId}`)).items[0];
    assert.equal(latest.rule_rev, original.rule_rev);
    assert.equal(latest.spec_hash, original.spec_hash);
    assert.deepEqual(latest.spec, original.spec);
    await click('[data-space-action="tab"][data-value="content"]');
    for (const mode of ['hybrid', 'keyword', 'semantic']) {
      await fill('[name="query"]', query.query);
      await page.select('[data-space-form="search"] [name="mode"]', mode);
      await click('[data-space-form="search"] [type="submit"]');
      await page.waitForSelector('.space-cards [data-space-item-content]');
      assert.match(await page.$eval('[data-space-item-content]', (element) => element.textContent), /连接|排障/);
      assert.match(await page.$eval('[data-space-page="search"]', (element) => element.textContent), /第 1 页/);
      assert.equal(searches.at(-1).mode, mode);
      assert.equal(searches.at(-1).top_k, 100);
      assert.equal(searches.at(-1).limit, 20);
      assert.equal(searches.at(-1).cursor || '', '');
      await click('[data-space-action="item"]');
      await click('[data-space-action="back-list"]');
      assert.match(await page.$eval('[data-space-page="search"]', (element) => element.textContent), /第 1 页/);
      assert.equal(await page.$eval('[data-space-form="search"] [name="query"]', (element) => element.value), query.query);
      assert.equal(await page.$eval('[data-space-form="search"] [name="mode"]', (element) => element.value), mode);
    }
    await captureLayout('team-memory', true);
    if (process.env.SPACE_UI_SCREENSHOT) await page.screenshot({ path: process.env.SPACE_UI_SCREENSHOT });
    const confirmDelete = async (name, route, revision, savedText) => {
      await page.waitForSelector('.space-delete-dialog[open]');
      assert.equal(await page.$eval('.space-delete-name', (element) => element.textContent), name);
      if (process.env.SPACE_UI_SCREENSHOT) await page.screenshot({ path: nodePath.join(nodePath.dirname(process.env.SPACE_UI_SCREENSHOT), `live-delete-${deletions.length + 1}.png`) });
      const response = page.waitForResponse((result) => new URL(result.url()).pathname === route && result.request().method() === 'DELETE');
      await click('[data-space-confirm-delete]');
      const result = await (await response).json();
      assert.equal(result.deleted, true);
      await saved(savedText);
      assert.equal(deletions.at(-1).revision, `"${revision}"`);
      assert.match(deletions.at(-1).idempotencyKey, /^[0-9a-f-]{36}$/);
      return result;
    };
    await click('[data-space-action="tab"][data-value="rules"]');
    await click(`[data-space-action="edit-rule"][data-value="${ruleId}"]`);
    await click('[data-space-action="delete-rule"]');
    const ruleDeleted = await confirmDelete(latest.name, `${path}/rules/${ruleId}`, latest.rule_rev, '规则已删除');
    assert.equal(ruleDeleted.rule_rev, latest.rule_rev + 1);
    assert.equal((await api(`${path}/rules?limit=20`)).items.length, 0);
    assert.match(await page.$eval('[data-space-page="rules"]', (element) => element.textContent), /第 1 页/);
    await click('[data-space-action="tab"][data-value="members"]');
    const beforeSourceDelete = (await api(`/management/spaces?space_id=${spaceId}`)).space;
    const source = beforeSourceDelete.sources.find((entry) => entry.binding_id === 'ops-source');
    await click('[data-space-action="delete-source"][data-value="ops-source"]');
    const sourceDeleted = await confirmDelete(source.principal, `${path}/sources/ops-source`, source.grant_rev, '来源已删除');
    assert.equal(sourceDeleted.grant_rev, source.grant_rev + 1);
    const beforeSpaceDelete = (await api(`/management/spaces?space_id=${spaceId}`)).space;
    assert.deepEqual(beforeSpaceDelete.sources, []);
    assert.match(await page.$eval('[data-space-page="sources"]', (element) => element.textContent), /第 1 页/);
    await page.$eval('.space-settings', (element) => { element.open = true; });
    await click('[data-space-action="delete-space"]');
    const spaceDeleted = await confirmDelete(beforeSpaceDelete.name, path, beforeSpaceDelete.authz_rev, '团队空间已删除');
    assert.equal(spaceDeleted.authz_rev, beforeSpaceDelete.authz_rev + 1);
    const remainingIds = (await api('/management/spaces?limit=20')).items.map((entry) => entry.space_id);
    const picker = await page.$eval('#space-picker', (element) => ({
      ids: [...element.options].filter((option) => !option.disabled).map((option) => option.value),
      value: element.value,
      disabled: element.disabled,
    }));
    assert.deepEqual(picker, { ids: remainingIds, value: remainingIds[0] ?? '', disabled: remainingIds.length === 0 });
    if (remainingIds.length === 0) await page.waitForSelector('[data-space-form="create"]', { visible: true });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ space_id: spaceId, rule_id: ruleId, item_id: item.item_id, job_state: 'SUCCEEDED', index_state: 'READY', member_count: members.length, member_pages: 2, modes: searches.map((entry) => entry.mode), bob_rule_read: bobDefinition.rule_id === ruleId, bob_result_count: found.items.length, stable_rule_rev: latest.rule_rev, deleted: ['rule', 'source', 'space'] }));
  } finally { await browser.close(); }
});
