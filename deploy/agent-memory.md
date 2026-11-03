# Agent 记忆部署说明

镜像包含 Agent 记忆抽取代码。部署模板默认 `agent_memory.enabled: false`；启用后，从完成的 Trace 中提取 Tools / Experiences。

## 开启

编辑当前实例挂载的配置，将现有开关改为：

```yaml
agent_memory:
  enabled: true
```

配置位置：

| 部署方式 | 修改位置 | 应用配置 |
| --- | --- | --- |
| Compose | 仓库内 `deploy/compose/ogmemory.yaml` | `docker compose up -d --force-recreate ogmemory` |
| 一键部署 | 部署输出“数据目录”中的 `ogmemory.yaml` | `docker restart "${INSTANCE_NAME}_ogmemory"` |
| `deploy/deploy.sh` | 部署使用的 `ogmemory.yaml` | 重启挂载该文件的 oGMemory 容器 |

一键部署的 `INSTANCE_NAME` 可通过 `source "$HOME/ogmemory-runtime.env"` 加载。再次执行一键部署脚本会重新生成 YAML；需要持续开启时，同时修改脚本中的 YAML 模板。

也可在创建容器时显式传入 `-e OGMEM_AGENT_MEMORY_ENABLED=true`，并移除 YAML 中的 `agent_memory.enabled`，让配置加载器使用环境变量。推荐直接修改 YAML。

## 验证

用实际容器名替换 `<ogmemory容器名>`，检查启动时生成的配置：

```sh
docker exec <ogmemory容器名> python3 -c 'from providers.unified_config import OgMemConfig; print(OgMemConfig.load("/tmp/ogmem.resolved.yaml").agent_memory_enabled)'
```

输出应为 `True`。通过已启用 Trace 上报的客户端完成一次包含工具使用的任务，等待后台抽取与索引完成。核对同一 account / user / agent 的 Trace、抽取日志及记忆页面中的 Tools / Experiences。

Agent 记忆依赖完成的 Trace；身份注册和共享配置分别控制身份及可见范围。普通对话可能没有可复用的工具或经验，零产出符合预期；抽取报错需要单独排查。开关为 `True`、服务健康或队列空闲，只能证明对应状态，验收还应检查实际 Trace 处理结果。
