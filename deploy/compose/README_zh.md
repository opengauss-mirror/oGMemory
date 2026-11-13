# oGMemory Docker Compose 部署指南

oGMemory 是基于 openGauss 构建的语义记忆引擎，为 OpenClaw、Codex 等 Agent 提供跨会话的长期记忆能力。

本指南介绍如何使用 Docker Compose 部署 oGMemory、配置模型服务，以及连接用户自己的客户端。

## 1. 部署内容

默认启动两个容器：

| 服务 | 职责 | 容器内部端口 |
| --- | --- | --- |
| `opengauss` | 保存记忆向量索引 | 5432 |
| `ogmemory` | 提供 HTTP、MCP，运行 AGFS 和索引进程 | HTTP 8090、MCP 8091、AGFS 1833 |

LLM 和 Embedding 服务由用户提供，可以使用云端 API，也可以使用容器能够访问的宿主机模型服务。LLM 用于提取记忆，Embedding 用于生成检索向量。

```text
用户自己的客户端
  ├─ HTTP → 宿主机 HTTP 端口 → ogmemory:8090
  └─ MCP  → 宿主机 MCP 端口  → ogmemory:8091/mcp
                                  ├─ AGFS → 记忆正文与状态
                                  ├─ opengauss:5432 → 向量索引
                                  └─ 用户配置的模型服务
```

Compose 不启动 OpenClaw，也不下载 OpenClaw 镜像。OpenClaw 的 Context Engine 与 MCP 两种接入方式由用户按需配置，见第 6 节。

数据库通过项目独立的 bridge 网络提供给 oGMemory。默认只向宿主机发布 HTTP、MCP 端口，数据库与 AGFS 不发布；需要数据库外部访问时另行显式配置。

### 1.1 服务镜像

| 服务 | 默认镜像 |
| --- | --- |
| openGauss | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328` |
| oGMemory | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:1013` |

镜像地址通过 `.env` 配置，需要部署其他版本时修改对应镜像引用。模型和服务访问密钥在部署时配置。

## 2. 环境准备

需要：

- ARM64 Linux 主机，或 Apple Silicon Mac 上的 Docker Desktop。当前默认数据库镜像要求 ARM64，本方案暂不承诺 x86 兼容。
- 正在运行的 Docker Engine / Docker Desktop，以及 Docker Compose 插件；建议 Compose 2.20 或更新版本。
- Bash，供一键部署入口使用。
- Python 3（部署入口仅使用标准库解析结构化配置）。
- 可访问 SWR 镜像仓库，以及 LLM、Embedding 服务的网络。
- LLM、Embedding 地址、模型名、密钥和正确的向量维度。

### 2.1 检查 Docker 和架构

```sh
docker info --format '{{.Architecture}}'
docker compose version
```

架构通常显示 `aarch64` 或 `arm64`。如果无法连接 Docker，先启动 Docker 服务或 Docker Desktop。

选择部署目录后，检查该目录所在分区和 Docker 数据分区的空间：

```sh
df -h "$PWD"
docker info --format '{{.DockerRootDir}}'
# Linux：将上一条命令输出的实际目录传给 df -h。
```

确保部署数据分区至少有 5 GB 空间，Docker 数据分区有镜像下载空间。根分区已满时，改用有空间的数据分区；例如服务器有 `/data0` 时可以在那里建立部署目录。不要为此清理其他项目的数据。

### 2.2 选择空闲端口

本指南默认 HTTP 为 18090，MCP 为 18091。已有服务使用这些端口时，为新实例选择其他端口，例如 18110、18111。

```sh
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

Linux：

```sh
ss -ltn '( sport = :18090 or sport = :18091 )'
```

macOS：

```sh
lsof -nP -iTCP:18090 -iTCP:18091 -sTCP:LISTEN
```

更换端口后按新端口重新检查。修改的是宿主机端口，容器内部仍使用 8090、8091；不同实例还应使用不同项目名和数据目录。

## 3. 部署配置

### 3.1 获取部署文件

将维护者提供的 `ogmemory-compose.tar.gz` 复制到部署主机，选择有足够空间的目录解压。例如：

```sh
mkdir -p /path/to/ogmemory-deploy
tar -xzf /path/to/ogmemory-compose.tar.gz -C /path/to/ogmemory-deploy
cd /path/to/ogmemory-deploy
```

`/path/to/` 必须替换为实际目录；不要直接复制占位路径。维护者可在仓库根目录运行 `bash deploy/compose/package.sh /path/to/output` 生成部署文件包，包内不包含私有配置、凭据和实例数据。


部署目录包含以下文件。`.env` 由用户填写，`data/` 在首次启动时生成：

```text
ogmemory-deploy/
├── compose.yaml
├── .env.example
├── .env
├── deploy/
│   └── compose/
│       ├── README_zh.md
│       ├── deploy.sh            # 启动和管理入口
│       ├── ogmemory.yaml
│       ├── compose.network.yaml
│       ├── compose.model-network.yaml
│       └── scripts/             # 服务就绪检查、身份初始化与连接信息
└── data/
    └── ogmemory/                # 对应 COMPOSE_PROJECT_NAME
        ├── opengauss/
        ├── agfs/
        └── secrets.env
```

获取部署文件后，进入包含 `compose.yaml` 的部署目录。后续命令均在此目录执行，请保持文件的相对目录结构。

```sh
cd /path/to/ogmemory-deploy
```

### 3.2 配置模型与实例

首次创建 `.env`；已有文件时保留，不覆盖：

```sh
if [ ! -e .env ]; then
  cp .env.example .env
fi
chmod 600 .env
```

每个部署目录使用自己的 `.env`。启动入口隔离终端中已有的模型、身份等环境变量；配置值及变量引用以本目录的 `.env` 为准。Docker 连接、登录凭据目录和代理等运行环境仍保留。

编辑环境变量文件：

```sh
vi .env
```

修改以下配置：

```dotenv
# 实例与数据：创建后保持稳定；同机多实例不能共用项目名或数据目录。
COMPOSE_PROJECT_NAME=ogmemory
DATA_ROOT=./data

# 默认只允许本机访问；远程客户端需要另外调整监听地址与网络访问配置。
PUBLISH_IP=127.0.0.1
OGMEM_HTTP_HOST_PORT=18090
OGMEM_MCP_HOST_PORT=18091

# 服务镜像。
OPENGAUSS_IMAGE=swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328
OGMEM_IMAGE=swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:1013

# LLM：填写供应商提供的 OpenAI 兼容地址、模型名和密钥。
LLM_PROVIDER=openai
LLM_BASE_URL=
LLM_MODEL=
LLM_API_KEY=

# Embedding：Base URL 不附加 /embeddings，维度必须与模型输出一致。
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=
EMBEDDING_API_KEY=
EMBEDDING_DIMENSION=1024

# 默认身份；可以调整，但已有数据时不随意更换。
OG_ACCOUNT_ID=ogmemory-account
OG_USER_ID=ogmemory-user
OG_AGENT_ID=ogmemory-api-agent

# 自动提取的待处理 token 阈值；短对话实验可显式设置为 1。
OGMEM_AFTER_TURN_THRESHOLD=400
OGMEM_MCP_ENABLED=true
OGMEM_MCP_ALLOWED_HOSTS=localhost,127.0.0.1,ogmemory,host.docker.internal
```

保存并退出 `vi`：按 `Esc`，输入 `:wq`，按回车。

必填的模型项不能留空。`1024` 只是示例，须使用模型真实输出维度。宿主机上的模型通常填写 `http://host.docker.internal:端口/v1`；容器中的 `127.0.0.1` 指向容器自身。

`.env` 使用 Compose 环境文件格式，不是 shell 脚本，不执行 `source .env`。特殊字符按 Compose 格式引用；配置解析不向终端打印密钥。

首次部署自动生成数据库密码、服务根密钥和账户管理密钥，保存到实例目录的私有 `secrets.env`，以后复用。已有实例缺失 `secrets.env` 时会拒绝启动，请从备份恢复该文件，避免更换已初始化数据使用的凭据。模型供应商密钥与 oGMemory 客户端密钥用途不同，不能互换。

默认 MCP 配置为：

```yaml
mcp:
  default_toolset: read-write
  enable_write_tools: true
  enable_delete_tools: false
```

`read-write` 提供记忆读取和写入能力，不开放删除工具。需要调整时修改 `deploy/compose/ogmemory.yaml`；`all` 和 `full` 工具集会开放删除工具。HTTP 生命周期写入接口仍可使用。

### 3.3 模型访问代理

需要代理访问云端 LLM 或 Embedding 时，在本实例 `.env` 中填写：

```dotenv
HTTP_PROXY=http://host.docker.internal:7897
HTTPS_PROXY=http://host.docker.internal:7897
NO_PROXY=local-model,model.example.internal
```

端口按实际代理服务修改；容器中的 `127.0.0.1` 指向容器本身，不能用它访问宿主机代理。Compose 同时传递大小写代理变量，并始终保留 localhost、127.0.0.1、opengauss、ogmemory、host.docker.internal 等内部服务的绕过项。

`.env` 中显式配置的代理（包括空值）优先于终端代理。未在文件中声明代理时可沿用 Docker 客户端运行环境中的代理。应用的模型调用使用这些代理；本地服务健康检查始终直连内部服务。代理含认证信息时也不要提交到仓库。

## 4. 执行部署

配置完成后，在部署目录执行：

```sh
bash deploy/compose/deploy.sh start
```

部署脚本按以下顺序执行：

1. 检查 Docker、Compose、`.env` 是否存在、端口和已有实例的存储归属及挂载位置。
2. 拉取 `.env` 指定的数据库与 oGMemory 镜像。
3. 使用已拉取的 oGMemory 镜像校验必填模型配置、准备持久化目录和配置，首次生成凭据；已有实例保留数据和密钥。准备工具复用镜像中的 Python 依赖，因此镜像下载失败时尚未执行此步骤。
4. 启动 openGauss，等待 SQL 检查通过。
5. 启动 oGMemory，检查 HTTP、AGFS、数据库连接、向量扩展和可选 MCP 监听端口。
6. 创建缺少的用户及归属于该用户的默认 Agent；重复启动不重复创建或改变归属。
7. 显示服务状态、客户端地址、身份及私有凭据文件位置。

只有所有必要步骤通过，才报告启动成功。失败时显示阶段和排查入口；不自动删数据，也不自动更换模型或凭据。配置填写错误时，先修正 `.env` 再重新执行 `start`。准备过程中的文件写入中断也可重试，已生成的实例凭据会继续使用；没有所有权记录的既有数据目录仍会被拒绝，避免接管其他实例。

日常健康检查不调用真实模型、不写测试记忆。服务健康并不证明模型可用或记忆链路已完成，实际效果由用户通过客户端使用确认。

## 5. 部署完成后的使用

启动成功表示容器及必要依赖已就绪，不代表模型访问、记忆提取或召回效果已经验证。请按第 6 节选择客户端，在自己的环境中真实使用，确认写入和读取符合预期。

部署入口不提供自动验收命令，不调用模型、不写入测试记忆。

## 6. 客户端接入

默认配置下：

| 配置项 | 示例 / 来源 |
| --- | --- |
| HTTP Base URL | `http://127.0.0.1:18090` |
| MCP URL | `http://127.0.0.1:18091/mcp` |
| API Key | 实例凭据中的 `OG_TENANT_API_KEY` |
| Account ID | `.env` 的 `OG_ACCOUNT_ID` |
| User ID | `.env` 的 `OG_USER_ID` |
| Agent ID | `.env` 的 `OG_AGENT_ID` |

Account、User、Agent 是不同字段。根密钥用于初始化与管理，普通客户端使用账户管理密钥；多用户场景应进一步通过管理接口分发各用户密钥。

HTTP 接入使用 `compose`、`prefetch`、`after_turn` 等生命周期接口；逐条查询和读取通过 `memories/query`、`memories/read` 接口或 MCP 工具完成。所用接口应与服务镜像的支持范围一致。

HTTP 接入使用上述 Base URL，参考 [HTTP REST 接入说明](../../docs/integrations/http-rest.md)。服务已经由 Compose 启动，不重复执行接入文档中的服务安装或启动步骤。

### 6.1 OpenClaw Context Engine

用户自行安装兼容的 OpenClaw 和 `og-memory-context-engine` 插件，参考 [OpenClaw 接入说明](../../docs/integrations/openclaw.md)，使用 **remote 模式**连接已部署服务。

填写 `memoryApiBaseUrl`、认证密钥、Account、User 和 Agent。Base URL 不带 `/api/v1`。如果使用独立 Agent，先通过管理接口登记并确认属于对应用户；部署自动初始化的是默认 Agent。

### 6.2 OpenClaw MCP

使用支持 MCP 的 OpenClaw 客户端，将 MCP 服务地址设为上述 URL，传入：

```text
X-API-Key: 账户管理密钥
X-Account-ID: Account ID
X-User-ID: User ID
X-Agent-ID: Agent ID
```

参考 [MCP 接入说明](../../docs/integrations/mcp.md) 的鉴权与 Streamable HTTP 部分。客户端不需要安装 oGMemory Python 服务；本指南不要求下载集成 OpenClaw 镜像。确认客户端能完成 MCP 初始化、发现工具，再验证实际写入和检索。

用户可以选择 Context Engine、MCP 或按客户端能力分别配置两种方式；同一轮对话不要无意中重复提交记忆。

### 6.3 其他机器或容器上的客户端

默认 `PUBLISH_IP=127.0.0.1` 只允许本机访问。远程访问需要调整服务发布地址，开放所选 HTTP/MCP 端口，并使用部署主机的实际地址；MCP 同时在 `OGMEM_MCP_ALLOWED_HOSTS` 中加入访问域名或 IP。

另一个容器里的 `127.0.0.1` 不是部署主机，应使用容器能访问的宿主机地址或同网络服务名。客户端配置须与实际访问路径一致。

## 7. 常用操作与数据维护

在部署目录执行以下命令，查看状态、查看日志、停止或恢复服务：

```sh
bash deploy/compose/deploy.sh status
bash deploy/compose/deploy.sh logs
bash deploy/compose/deploy.sh stop
bash deploy/compose/deploy.sh start
# 删除本实例容器和网络，保留宿主机数据：
bash deploy/compose/deploy.sh down
```

修改 `.env` 或服务配置后执行 `start`，由入口重新创建 oGMemory 容器并重新检查，数据库和宿主机数据保留。普通 `docker compose restart` 不会更新容器环境变量。

数据默认保存在 `DATA_ROOT/COMPOSE_PROJECT_NAME/`：

- `opengauss/`：数据库和向量索引。
- `agfs/`：记忆正文、摘要、会话、控制数据和索引任务。
- `secrets.env`：本实例凭据。

停止或删除容器不会自动删除这些宿主机目录。备份时先停止本实例，同时保存两个数据目录和私有配置，保持正文与索引为同一时间点。不要只备份数据库。

改变项目名或数据位置会指向另一套数据；改变身份可能使原有记忆不再出现在当前查询中。已有数据库的密码不会因改配置自动变更。不要在已有索引上直接更换 Embedding 模型或维度，应另建实例或执行经过验证的迁移。

使用本方案创建的实例，保持原项目名、数据位置和配置时，日常启动、停止及兼容镜像版本更新保留实例数据。部署入口不接管其他部署目录的实例。需要清空数据时，先停止对应实例，核对要删除的目录并确认数据无需保留。

启动、停止和删除操作通过部署目录锁及 Docker 项目锁互斥，防止并发操作改变同一实例。正常退出自动释放锁；异常断电等情况遗留的项目锁，需确认没有操作仍在执行后，按锁容器的确切 ID 清理。`stop` 和 `down` 使用本地缓存镜像建立互斥，不拉取当前配置的目标镜像。

实例目录自动生成 `.gitignore`，保护自定义数据目录中的运行时内容；不要提交凭据、模型配置或实例数据。

## 8. 相关文档

- [HTTP REST 接入](../../docs/integrations/http-rest.md)
- [OpenClaw 接入](../../docs/integrations/openclaw.md)
- [MCP 接入](../../docs/integrations/mcp.md)
- [客户端接入目录](../../docs/integrations/README.md)

客户端版本、插件版本与服务镜像版本应匹配；使用接入文档中的其他功能时，确认当前镜像支持相应接口。
