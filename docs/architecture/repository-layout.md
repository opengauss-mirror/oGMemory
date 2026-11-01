# 仓库目录说明

## 核心运行时

| 目录 | 职责 |
| --- | --- |
| `core/` | 领域模型、接口、错误和通用运行时协议 |
| `fs/` | AGFS、SQL 和访问控制适配 |
| `extraction/` | Schema、提示词和记忆候选抽取 |
| `commit/` | 候选合并、原子写入和 Outbox |
| `index/` | 索引记录和异步索引 |
| `retrieval/` | 查询分类、候选召回和排序 |
| `providers/` | LLM、Embedding、向量索引和关系存储适配 |
| `service/` | 记忆 API、代码资源和生命周期服务 |
| `server/` | HTTP、MCP、认证和服务编排入口 |
| `session/` | 会话状态、压缩和归档 |
| `agfs/` | AGFS 服务、文件系统插件和 Go SDK |
| `cli/` | `ogmem` 命令和本地服务管理 |

## 功能扩展

| 目录 | 职责 |
| --- | --- |
| `features/` | 可选功能模块，例如工具结果外化 |
| `scene/` | 场景记忆契约、投影和检索 |
| `skills/` | Skill 管理和沉淀 |

## 客户端与 Agent 集成

| 目录 | 入口 |
| --- | --- |
| `plugins/agent-plugins/` | 通用 Agent Plugin 包 |
| `plugins/ogmemory/` | Codex 插件 |
| `plugins/claude-code/` | Claude Code 插件 |
| `plugins/openclaw/` | OpenClaw Context Engine |
| `plugins/opencode/` | OpenCode 插件 |
| `plugins/deepseek-harness/` | DeepSeek Harness 插件 |
| `plugins/hermes/` | Hermes Memory Provider |
| `integrations/deepagents/` | LangChain Deep Agents Python 中间件 |

`plugins/` 提供可独立安装的 Agent 插件；`integrations/` 提供进程内 Python 集成。
Deep Agents 中间件的导入路径为 `integrations.deepagents`。各客户端安装与配置见
[接入总览](../integrations/README.md)。

## 部署、工具与验证

| 目录 | 职责 |
| --- | --- |
| `config/` | 配置模板和环境变量参考 |
| `deploy/` | Compose、单机和用户快速部署入口 |
| `docker/` | Dockerfile 和容器入口脚本 |
| `scripts/` | 开发、启动、检查和维护脚本 |
| `examples/` | 可运行示例 |
| `tests/` | contract、unit、integration、e2e 和 benchmark |
| `perf/` | 性能脚本和结果处理 |

## 文档

| 目录 | 内容 |
| --- | --- |
| `docs/architecture/` | 仓库布局、存储格式、抽取流程和异步索引设计 |
| `docs/integrations/` | HTTP、Python、MCP 和 Agent 客户端接入 |
| `docs/deployment/` | 镜像构建、源码部署和运维操作 |
| `docs/testing/` | LoCoMO 评测和 Jenkins 测试方案 |

[文档总览](../README.md)提供快速开始、配置和各专题入口。
