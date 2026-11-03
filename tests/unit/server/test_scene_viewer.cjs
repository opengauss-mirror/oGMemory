const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const script = readFileSync(path.join(__dirname, '../../../server/memory_viewer_static/app.js'), 'utf8');
const constants = script.slice(0, script.indexOf('const MONITOR_ICONS'));
const sceneCode = script.slice(script.indexOf('function resetScenes()'), script.indexOf('function openRootDirectories()'));

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.value = '';
    this.textContent = '';
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) {
    this.children = children;
    if (this.tagName === 'select') this.value = children[0]?.value || '';
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  querySelectorAll() { return this.children; }
}

function viewer() {
  const nodes = new Map();
  const $ = selector => {
    if (!nodes.has(selector)) nodes.set(selector, new Element(selector === '#scenes-scope' ? 'select' : 'div'));
    return nodes.get(selector);
  };
  const requests = [];
  const state = {
    accountId: 'acme', accountRequestId: 1, adminMode: true, view: 'memory',
    sceneRequestId: 0, sceneScopes: new Map(), scenes: [], sceneOffset: 0,
    searchTargets: new Map([
      ['all', { accountId: 'acme' }],
      ['user', { accountId: 'acme', userId: 'alice', agentId: '' }],
      ['agent', { accountId: 'acme', userId: 'alice', agentId: 'helper' }],
    ]),
    tree: { agents: [{ agent_id: 'helper', owner_user_id: 'alice' }] },
  };
  const context = vm.createContext({
    state, $, SCENE_PAGE_SIZE: 6, typeLabel: value => value,
    elements: { scenesScope: $('#scenes-scope'), scenesList: $('#scenes-list') },
    document: { createElement: tag => new Element(tag) },
    Option: function (label, value) { this.textContent = label; this.value = value; },
    apiUrl: (url, params) => ({ url, params }),
    fetchJson: request => new Promise((resolve, reject) => requests.push({ request, resolve, reject })),
  });
  vm.runInContext(constants + sceneCode, context);
  context.renderSceneScopes();
  return { context, state, requests, $ };
}

function scene(id, account = 'acme') {
  return {
    scene_id: id, origin: 'automatic', version: 2,
    cluster: {
      label: '<release>', summary: 'Friday release', time_start: '2026-01-01',
      members: [{ context_uri: `ctx://${account}/users/alice/memories/events/release` }],
    },
  };
}

test('scene list renders selected detail, canonical member navigation and pagination', async () => {
  const { context, requests, state, $ } = viewer();
  const loading = context.loadScenes(6);
  assert.equal(requests[0].request.params.owner_space, 'user:alice');
  assert.equal(requests[0].request.params.offset, 6);
  requests[0].resolve({ items: [scene('release')], total: 13, next_offset: 12 });
  await loading;
  assert.equal($('#scenes-count').textContent, '13 个场景');
  assert.equal(state.sceneOffset, 6);
  assert.equal(state.sceneNextOffset, 12);
  assert.equal($('#scenes-page-summary').textContent, '第 2 / 3 页');
  assert.equal($('#scene-detail-title').textContent, '<release>');
  assert.equal($('#scene-detail-members').children[0].dataset.memberUri, 'ogmemory://memories/event/release');
  assert.equal($('#scenes-list').children[0].attributes['aria-pressed'], 'true');
});

test('account switch renders latest account after the previous request completes', async () => {
  const { context, requests, state, $ } = viewer();
  const previous = context.loadScenes();
  context.resetScenes();
  state.accountId = 'new-account';
  state.accountRequestId += 1;
  state.searchTargets = new Map([['user', { accountId: 'new-account', userId: 'alice', agentId: '' }]]);
  context.renderSceneScopes();
  const current = context.loadScenes();
  requests[1].resolve({ items: [scene('current', 'new-account')], total: 1, next_offset: null });
  await current;
  requests[0].resolve({ items: [scene('previous')], total: 1, next_offset: null });
  await previous;
  assert.equal(state.selectedSceneId, 'current');
  assert.equal($('#scenes-list').children[0].dataset.sceneId, 'current');
  assert.equal($('#scenes-refresh').disabled, false);
});

test('scene refresh recovers after a temporary service failure', async () => {
  const { context, requests, state, $ } = viewer();
  const first = context.loadScenes();
  requests[0].reject(new Error('temporary failure'));
  await first;
  assert.equal($('#scenes-refresh').disabled, false);
  const recovered = context.loadScenes();
  requests[1].resolve({ items: [scene('recovered')], total: 1, next_offset: null });
  await recovered;
  assert.equal(state.selectedSceneId, 'recovered');
  assert.equal($('#scene-detail').hidden, false);
});

test('agent member links preserve agent ownership and encoded memory identifiers', () => {
  const { context, $ } = viewer();
  $('#scenes-scope').value = 'agent:helper';
  const target = context.sceneMemberTarget('ctx://acme/agents/helper/memories/patterns/release%20plan/content.md');
  assert.equal(target.uri, 'ogmemory://memories/pattern/agent/release%20plan');
  assert.equal(target.userId, 'alice');
  assert.equal(target.agentId, 'helper');
});
