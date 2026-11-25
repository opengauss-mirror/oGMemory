# oGMemory 文档

## 入门

- [特性能力与使用指南](/docs/features.md)：记忆写入、检索、管理、关系网络、学习复用等对外特性的介绍、操作步骤与效果验证。
- [快速开始](/docs/QUICKSTART.md)：本地 CLI 启动，以及 Compose、ARM64 和 OpenClaw 部署入口。
- [客户端接入总览](/docs/integrations/README.md)：HTTP、Python、MCP 和 Agent 插件的选择规则。
- [memoryBot 特性说明与使用指南](/docs/memorybot.md)：能力介绍、单机与多实例部署，以及前台、HTTP 和 CLI 操作。

## 配置

- [oGMemory 配置参考](/docs/OGMEMORY_ENV.md)：字段说明、环境变量映射和 SQL 直连示例。
- [配置模板](/config/ogmem.reference.yaml)：所有部署模式共用的参考 YAML。

## 部署与运维

- [Compose 部署](/deploy/compose/README_zh.md)：openGauss、oGMemory、HTTP 和 MCP 的容器部署。
- [部署运维指南](/docs/deployment/deployment_operations_guide.md)：OpenClaw 容器部署、日志、健康检查和故障处理。
- [Docker 部署说明](/docs/deployment/docker_deployment_guide.md)：镜像构建和容器运行参考。
- [OpenClaw Context Engine 部署](/docs/deployment/openclaw_context_engine_quickstart_zh.md)：OpenClaw 单机接入。
- [部署方式选择](/docs/deployment.md)：Compose、ARM64、源码和镜像部署。
- [记忆导入导出](/docs/deployment/memory-migration.md)：维护窗口内的全量迁移、附件恢复与索引重建。
- [Memory Viewer 使用指南](/docs/memory_viewer_guide.md)：连接账户、浏览记忆和查看运行状态。
- [团队空间使用指南](/docs/shared_spaces.md)：成员管理、来源贡献、规则、团队记忆与读取权限，含前台截图和后台接口示例。

## 集成

- [集成总览](/docs/integrations/README.md)
- [Agent 接入配置生成](/docs/integrations/plugin-config.md)
- [HTTP REST](/docs/integrations/http-rest.md)
- [Python API](/docs/integrations/python-api.md)
- [MCP](/docs/integrations/mcp.md)
- [MCP 协议兼容性](/docs/integrations/mcp-protocol-compatibility.md)
- [通用 Agent Plugin](/docs/integrations/agent-plugins.md)
- [Claude Code](/docs/integrations/claude-code.md)
- [Codex](/docs/integrations/codex.md)
- [OpenCode](/docs/integrations/opencode.md)
- [DeepSeek Harness](/docs/integrations/deepseek-harness.md)
- [Hermes Agent](/docs/integrations/hermes.md)
- [Deep Agents](/docs/integrations/deepagents.md)
- [OpenClaw](/docs/integrations/openclaw.md)

## 架构

- [memoryBot 设计](/docs/memorybot-design.md)：功能、数据模型、接口报文、多实例执行与验收。
- [仓库目录说明](/docs/architecture/repository-layout.md)
- [AGFS 目录规范](/docs/architecture/agfs-directory-spec.md)
- [记忆卡片格式](/docs/architecture/memory-card-format.md)
- [目录摘要流程](/docs/architecture/directory-summarizer-flow.md)
- [节点处理流程](/docs/architecture/node-processing.md)
- [异步索引设计](/docs/architecture/async-indexing.md)
- [记忆导入导出设计](/docs/architecture/memory-migration.md)

## 开发与测试

- [脚本与测试指南](/scripts/README.md)
- [记忆评测教程](/docs/testing/memory-evaluation-guide-zh.md)：六个 Agent 宿主、MCP、LoCoMO 与自备数据的预检、四阶段命令、全部参数和结果口径。
- [Jenkins 测试方案](/docs/testing/jenkins-test-strategy.md)：测试分层、触发矩阵和流水线设计。
