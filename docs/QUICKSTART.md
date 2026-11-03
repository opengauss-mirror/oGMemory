# 快速开始

按部署方式选择入口。配置字段和环境变量见[配置参考](OGMEMORY_ENV.md)。

## 本地开发

本地默认使用 SQL 存储。先准备可连接的 PostgreSQL 或 openGauss 数据库；数据库不会由本地向导创建。配置示例见 [SQL 直连最小本地启动](OGMEMORY_ENV.md#sql-直连最小本地启动)。

```bash
python -m pip install -e ".[dev]"
ogmem onboard --mode local
ogmem start local -d
ogmem status
```

停止本地服务：

```bash
ogmem stop local
```

默认 SQL 模式只启动 ContextEngine，不管理 AGFS 或 IndexService。需要 AGFS 时按
[AGFS README](../agfs/README.md) 启动；需要索引服务时单独执行 `ogmem start index -d`。
配置文件生成在 `config/ogmem.yaml`，运行数据和日志默认位于 `.ogmem_data/`。

## Docker Compose

适合启动 openGauss、oGMemory HTTP/MCP 和 AGFS：

- [Compose 部署指南](../deploy/compose/README_zh.md)
- [配置目录说明](../config/README.md)

## ARM64 一键部署

适合在 ARM64 Linux 服务器创建独立实例：

- [用户一键部署与体验指南](../deploy/user-quickstart/README_zh.md)
- [记忆评测教程](testing/memory-evaluation-guide-zh.md#10-一键部署实例)

## OpenClaw 源码部署

需要直接运行 OpenClaw 和 oGMemory 时，按[源码部署教程](deployment/openclaw_context_engine_quickstart_zh.md)执行。

## Docker 镜像部署

需要构建或运行镜像时，按[Docker 部署说明](deployment/docker_deployment_guide.md)执行。

## 客户端接入

服务启动后，按客户端类型选择[集成总览](integrations/README.md)：

- HTTP REST
- Python API
- MCP
- Claude Code、Codex、OpenCode、OpenClaw、DeepSeek Harness、Hermes
- LangChain Deep Agents

## 验证

基础单元测试：

```bash
pytest tests/unit -q
```

Agent 插件、MCP、LoCoMO 和自备数据评测见[记忆评测教程](testing/memory-evaluation-guide-zh.md)。测试分层和流水线设计见
[Jenkins 测试方案](testing/jenkins-test-strategy.md)。
