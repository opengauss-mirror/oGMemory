# 场景记忆运行时

## 启用

在已有 SQL ContextFS 配置中增加：

```yaml
scene:
  enabled: true
  options:
    analysis_trigger_tokens: 4096
    max_pending_nodes: 32
    max_llm_calls_per_batch: 3
    lexical_enabled: false
```

`OGMEM_SCENE_ENABLED=true` 可覆盖总开关。运行时复用 ContextFS 的 SQL 连接池，持久化场景快照、任务、通知去重键与配置历史。`lexical_enabled: true` 使用已配置的原生 openGauss 词法 provider，独立表名为 `scene_lexical_v1`；连接参数沿用 `retrieval_lexical_options`。该开关可按账户与 owner 热更新。

SQL 持久化支持 PostgreSQL 和 PG 兼容模式的 openGauss，连接初始化沿用 SQL ContextFS 的方言检查。

## 写入、恢复与检索

1. 普通记忆完成向量索引后，Outbox 投递幂等场景通知，再确认任务完成。
2. 后台按 owner 聚合有界证据窗口，领取租约并续租，执行预检、决策和质量审核。
3. 预检可补查同一范围的节点、场景卡片与原始来源。补查节点进入同一规范来源校验。
4. 发布同时检查任务租约、场景 revision 和记忆来源版本；冲突按任务预算恢复。
5. `WAIT` 保留证据，新增证据触发下一轮；可恢复失败使用有界退避。429 记录失败并停止当前 worker。
6. 检索结合语义、词法及直接命中成员入口，补充同场景节点并保留直接证据配额；返回节点经过当前规范存储与索引校验。

`wait_until_idle` 传入 `waitScenes: true` 可等待当前用户及 Agent 的场景队列，响应包含 `scenes` 状态。`WAITING` 表示等待新证据。

## 管理接口与前台

复用服务现有 API-Key 鉴权及账户/owner 授权：

| 接口 | 用途 |
| --- | --- |
| `GET /api/v1/scenes?owner_space=user:alice&offset=0&limit=20` | 当前场景与 revision |
| `POST /api/v1/scenes` | 创建自定义场景 |
| `PUT /api/v1/scenes/{scene_id}` | 修改标题、描述或成员 |
| `DELETE /api/v1/scenes/{scene_id}` | 归档场景 |
| `GET/PUT /api/v1/scenes/configuration` | 查询或热更新 owner 配置 |
| `GET /api/v1/scenes/status` | 队列、运行与限流停止状态 |
| `POST /api/v1/scenes/flush` | 立即触发待处理证据，后台领取任务 |

创建、修改及归档携带最新 `revision`；配置修改携带配置自己的 `revision`。冲突返回 409，客户端重新读取后提交。修改配置示例：

```json
{"owner_space":"user:alice","revision":0,"values":{"analysis_trigger_tokens":2048}}
```

列表响应包含 `revision`、`total`、`items` 和 `next_offset`；`limit` 范围为 1..100。写操作使用 JSON 对象请求体；`flush` 可发送 `{}` 操作当前用户范围，或携带 `owner_space` 选择已授权范围。

场景写入示例：

```json
{
  "owner_space": "user:alice",
  "revision": 0,
  "title": "产品发布准备",
  "description": "发布评审、回滚准备与验收节点。",
  "member_uris": ["ctx://acme/users/alice/memories/events/release-review"]
}
```

成员为同一账户、owner 范围内的有效已索引记忆。保存完整来源引用与成员版本，发布前再次校验规范记忆；默认可选最多 64 个成员。修改标题、描述或成员后更新场景向量，保留已有场景边界和时间。

前台入口：`/memory-viewer/#scenes`。可切换用户或 Agent，分页查看场景标题、描述、时间、版本及成员，并打开对应记忆详情。

## 验收

- 启用场景记忆后写入同一活动、不同来源的记忆；等待索引完成并调用 `flush`。
- 核对队列成功、场景 revision 增加、来源引用及成员版本完整。
- 服务重新创建后核对任务、通知去重与配置仍可恢复。
- 修改规范记忆后再处理，核对新的场景成员版本与检索节点一致。
- 在前台核对分页、owner 切换和成员详情。
- 单元测试：`pytest -q tests/unit/scene tests/unit/server/test_scene_runtime.py tests/unit/server/test_scene_management.py`。
- 前台交互测试：`node --test tests/unit/server/test_scene_viewer.cjs`。
- SQL 验收：配置隔离测试连接 `OGMEM_SCENE_SQL_DSN` 和 `OGMEM_SCENE_SQL_DIALECT`，运行 `pytest -q tests/integration/test_scene_sql_persistence.py`。测试使用唯一 schema，结束后清理。
