# oGMemory 文档

## 入门

- [特性能力与使用指南](features.md)：记忆写入、检索、管理、学习复用等对外特性的介绍、操作步骤与效果验证。
- [快速开始](QUICKSTART.md)：本地 CLI 启动，以及 Compose、ARM64 和 OpenClaw 部署入口。
- [客户端接入总览](integrations/README.md)：HTTP、Python、MCP 和 Agent 插件的选择规则。

## 配置

- [oGMemory 配置参考](OGMEMORY_ENV.md)：字段说明、环境变量映射和 SQL 直连示例。
- [配置模板](../config/ogmem.reference.yaml)：所有部署模式共用的参考 YAML。

## 部署与运维

- [Compose 部署](../deploy/compose/README_zh.md)：openGauss、oGMemory、HTTP 和 MCP 的容器部署。
- [部署运维指南](deployment/deployment_operations_guide.md)：OpenClaw 容器部署、日志、健康检查和故障处理。
- [Docker 部署说明](deployment/docker_deployment_guide.md)：镜像构建和容器运行参考。
- [OpenClaw Context Engine 部署](deployment/openclaw_context_engine_quickstart_zh.md)：OpenClaw 单机接入。
- [部署方式选择](deployment.md)：Compose、ARM64、源码和镜像部署。
- [Memory Viewer 使用指南](memory_viewer_guide.md)：连接账户、浏览记忆和查看运行状态。

## 集成

- [集成总览](integrations/README.md)
- [Agent 接入配置生成](integrations/plugin-config.md)
- [HTTP REST](integrations/http-rest.md)
- [Python API](integrations/python-api.md)
- [MCP](integrations/mcp.md)
- [MCP 协议兼容性](integrations/mcp-protocol-compatibility.md)
- [通用 Agent Plugin](integrations/agent-plugins.md)
- [Claude Code](integrations/claude-code.md)
- [Codex](integrations/codex.md)
- [OpenCode](integrations/opencode.md)
- [DeepSeek Harness](integrations/deepseek-harness.md)
- [Hermes Agent](integrations/hermes.md)
- [Deep Agents](integrations/deepagents.md)
- [OpenClaw](integrations/openclaw.md)

## 架构

- [仓库目录说明](architecture/repository-layout.md)
- [AGFS 目录规范](architecture/agfs-directory-spec.md)
- [记忆卡片格式](architecture/memory-card-format.md)
- [目录摘要流程](architecture/directory-summarizer-flow.md)
- [节点处理流程](architecture/node-processing.md)
- [异步索引设计](architecture/async-indexing.md)

## 开发与测试

- [脚本与测试指南](../scripts/README.md)
- [记忆评测教程](testing/memory-evaluation-guide-zh.md)：六个 Agent 宿主、MCP、LoCoMO 与自备数据的预检、四阶段命令、全部参数和结果口径。
- [Jenkins 测试方案](testing/jenkins-test-strategy.md)：测试分层、触发矩阵和流水线设计。
