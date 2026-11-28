# ZCode 原生 Hook 接入

oGMemory 插件在 ZCode 中自动召回记忆，在 Stop 时写回对话，并记录工具、权限请求和失败等 Trace。Agent Memory 需要服务端显式启用，参见 [部署说明](../../deploy/agent-memory.md)。

1. 部署 [oGMemory HTTP 服务](http-rest.md)，准备授权的租户、用户和 Agent。启用角色控制时，使用有权访问该 Agent 的成员密钥。远程地址使用 HTTPS，本地 SSH 转发可用 loopback HTTP。
2. 通过 `ogmem onboard --mode plugin --agent zcode` 导出配置。将 `zcode-plugin.json` 放在 `~/.ogmemory/`，或通过 `OG_MEMORY_ZCODE_CONFIG` 指定私有配置文件。
3. 确认 ZCode 能运行 `python`（Python 3.10 及以上，带 SQLite）。如解释器使用其他名称，将插件 `hooks/hooks.json` 的 `command` 改为该解释器的绝对路径。
4. 将以下设置合并到 ZCode 用户配置 `~/.zcode/cli/config.json`。保留已有插件目录和设置。启动新任务使设置生效。

```json
{
  "plugins": {"enabled": true, "dirs": ["/absolute/path/to/oGMemory/plugins/zcode"]},
  "hooks": {"enabled": true}
}
```

| 开关 | 默认值 |
| --- | --- |
| `OGMEM_AUTO_RECALL` | 开启记忆与 Skill 召回 |
| `OGMEM_AUTO_CAPTURE` | 开启对话写回 |
| `OGMEM_AUTO_TRACE` | 开启 Trace 采集 |
| `OGMEM_TRACE_CAPTURE_CONTENT` | 开启 Trace 正文 |
| `OGMEM_FORCE_EXTRACT_ON_STOP` | 关闭每次 Stop 强制抽取。显式记忆请求仍可触发抽取 |
| `OGMEM_ZCODE_QUEUE_LIMIT` | 每个服务身份最多约 10,000 条待送记录 |
| `OGMEM_ZCODE_RETENTION_DAYS` | 本地已完成记录和轮次关联保留 7 天 |

写回与 Trace 在插件的数据目录分别排队。同一 ZCode 轮次如触发多次 Stop，分别归档 Trace 片段；用户消息只随首次写回提交。连接故障后，后续 Hook 会重新尝试发送。对于配置或权限错误，修正问题后可以使用同一配置和 `ZCODE_PLUGIN_DATA` 运行 `python scripts/ogmemory_zcode_hook.py --resume-delivery`。队列诊断位于插件数据目录的 `delivery.log`。

验收时，在一个会话中让 ZCode 使用工具并记住一条可辨认的事实。检查工具事件和轮次归档、对话抽取状态，等待索引后在独立会话里查询。再核对已完成 Trace 对应的 Agent Memory 处理和可复用经验。
