# OpenClaw + oGMemory 一键安装部署指南

## 概述

本部署方案采用 Docker 容器化部署，包含以下组件：

| 组件 | 容器名称 | 端口 | 说明 |
|------|---------|------|------|
| openGauss | opengauss | 15432 | 向量数据库（可选） |
| PostgreSQL | ogmem-postgres | 54321 | oGMemory 的 SQL 存储后端（context files、events、grants、outbox 等） |
| oGMemory | ogmem | 8090 | 记忆服务 HTTP API |
| OpenClaw Gateway | openclaw_ogmem | 18789 | Web 界面入口 |

> **⚠️ 架构说明：** 本项目提供的 Docker 镜像仅支持 **ARM64 (鲲鹏)** 架构，不支持 x86_64。请确保部署环境为 ARM 服务器（如华为云鲲鹏实例）。

## 部署方式

### 已有 standalone 部署升级

请先阅读 [Standalone 升级兼容与向量表迁移](README.standalone-upgrade.md)：
保留原 AGFS 数据卷及挂载前缀，旧向量表的回填改为维护窗口内显式执行的一次性迁移。

### 多分支部署与测试隔离

- 使用不同的 Compose project name（`docker compose -p <project> ...`）。
  默认构建标签为 `ogmemory:<project>-local` 和
  `openclaw-ogmemory:<project>-local`。已有 `.env` 中的 `OGMEM_IMAGE` /
  `OPENCLAW_IMAGE` 会覆盖默认值；不要让不同分支继续共用 `:local` 标签。
- 容器名、监听端口、数据库和持久卷也必须独立。仅修改 project name
  不会隔离显式配置的外部数据库连接。
- 检查旧版 `docker run` 容器是否仍连接相同数据库。SQL outbox worker
  会消费共享队列中的任务；不同版本的 worker 混用会导致同一次测试的
  索引由不同代码生成。重跑基准前，应由服务所有者停掉旧 worker，
  或为测试配置独立数据库。
- 部署代码修复后使用新的测试 account/user/session，重新执行完整入库。
  恢复旧索引的可见性不会自动补回历史提取失败时缺失的记忆。
- 推理模型的输出预算和供应商参数需要单独验证。不要假定所有模型
  都支持 `thinking.type=disabled`，也不要把超时或截断当作成功的空提取。

只更新应用容器、不重建数据库的示例：

```bash
docker compose -p my-project --profile with-db --profile with-openclaw build ogmem openclaw
docker compose -p my-project --profile with-db --profile with-openclaw up -d --no-deps --no-build ogmem openclaw
```

M3-Bench 的独立人脸检测服务也支持 Compose 管理（可选 `with-cv`
profile，或不需要 DB/LLM 配置的 CV-only 启动入口），见
[CV 服务部署说明](README.cv.md)。

本目录同时提供两种部署方式：

| 方式 | 入口 | 状态 |
|------|------|------|
| **docker compose** (推荐) | `docker compose --profile with-db up -d` | 本目录主推方式，详见下文 |
| **docker run** (legacy) | `bash deploy.sh -password <密码>` | 旧版一键脚本，保留以保持向后兼容 |

两份脚本**共用同一份配置文件** `deploy/.env`（从 `deploy/.env.example`
复制得到，已 gitignore）。`deploy.sh` 仍支持读取旧版 `deploy/deploy.env`
作为兜底，方便老用户迁移；新部署应统一使用 `.env`。

下文先讲推荐的 compose 方式，再讲 legacy `deploy.sh`。

## docker compose 快速开始（推荐）

```bash
cd deploy

# 1. 创建 .env（从模板复制；不要提交）
cp .env.example .env
# 编辑 .env，至少填入 OG_PASSWORD / LLM_PROVIDER / LLM_API_KEY / LLM_BASE_URL / LLM_MODEL

# 2. 创建 ogmemory.yaml（从模板复制；不要提交）
cp ogmemory.example.yaml ogmemory.yaml
# 一般不需要改；如需自定义 LLM/Embedding/向量库等，编辑此文件

# 3. 启动 stack
docker compose --profile with-db up -d          # opengauss + ogmem
docker compose --profile with-db --profile with-openclaw up -d   # + openclaw gateway

# 4. 日常使用
docker compose --profile with-db ps            # 查看容器状态
docker compose --profile with-db logs -f ogmem # 跟踪日志
docker compose --profile with-db down          # 停止（保留数据）
docker compose --profile with-db down -v       # 停止并删除 volumes
```

### Profiles

| Profile     | Services added          | When to use                              |
|-------------|-------------------------|------------------------------------------|
| (none)      | ogmem only              | 外部 DB；自行设置 `VECTOR_DB_HOST` 和 `STORAGE_DB_HOST` |
| `with-db`   | + opengauss, postgres   | 本地 openGauss + 本地 Postgres（典型 dev 场景） |
| `with-openclaw`  | + openclaw (gateway)    | 需要 Web UI 时启用                       |

`with-db` 和 `with-openclaw` 互相独立，按需叠加。

### 数据库环境变量命名

数据库相关变量只有两套，按**角色**命名而非产品名：

| 变量族 | 指向 | 说明 |
|--------|------|------|
| `STORAGE_DB_*` | SQL 存储库（Postgres） | context files / events / grants / outbox / M3 graph store |
| `VECTOR_DB_*` | 向量库 | 生产推荐 openGauss；dev / 单库部署可指向同一个 pgvector Postgres |

两套变量默认指向两个独立数据库；单库部署时把 `VECTOR_DB_HOST/PORT/NAME/USER`
改成 `STORAGE_DB_*` 的值即可。旧的 `OG_*`（数据库连接部分）、`POSTGRES_*`、
`OPENGAUSS_HOST_IP` / `OPENGAUSS_HOST` 命名已废弃。容器镜像自身要求的变量
（postgres 镜像的 `POSTGRES_PASSWORD`、openGauss 镜像的 `GS_*`）只在
docker-compose.yml 内部映射，`.env` 里不出现。

> **端口语义**：ogmem 进程读取 `STORAGE_DB_HOST_PORT` 和 `VECTOR_DB_HOST_PORT`
> 作为拨号端口。两者都是宿主机端口——compose 把它们转发到 DB 容器内的
> 端口（postgres 硬编码 5432，openGauss 由 `VECTOR_DB_PORT` 决定）。因为
> ogmem 跑在 host 网络命名空间里，必须 dial loopback 才能命中 bridge 端口
> 转发，所以端口必须用宿主端口。指向外部 DB 时直接写 DB 的真实端口即可。
>
> `VECTOR_DB_PORT`（仅 vector 族有）是 openGauss 容器**内部**的监听端口
>（`GS_PORT` / compose `ports:` 右值），不会被 ogmem 看到。

### 与 `deploy.sh` 的差异

| 维度                | `deploy.sh` (legacy)         | `docker compose` (推荐)       |
|---------------------|------------------------------|-------------------------------|
| 容器生命周期         | 命令式 `docker run`          | 声明式 `docker-compose.yml`   |
| 网络                 | `--network host`             | ogmem `network_mode: host`，DB 用 Bridge + 端口映射 |
| openGauss / ogmem URL | `VECTOR_DB_HOST=127.0.0.1`  | `VECTOR_DB_HOST=127.0.0.1`（默认；ogmem 是 host-net，只能 dial loopback） |
| 配置文件             | `deploy/.env` (相同)         | `deploy/.env` (相同)          |
| 健康检查             | 自定义 bash 循环             | 原生 `healthcheck:` 块        |
| 多租户               | `OPENCLAW_INSTANCES`         | 同上，v2 处理                 |
| 数据持久化           | 宿主机 bind mount            | 命名 volumes                  |

### 网络坑

`OGMEM_URL` 默认 `http://127.0.0.1:${OGMEM_HTTP_PORT}`，因为 ogmem 和 openclaw 都
使用 `network_mode: host`（共享宿主机 loopback）。Compose 的 `ogmem` 服务名**不会**
从 host 网络容器内解析——`http://ogmem:<port>` 在这里会失败。

如果你的 ogmem 跑在另一台机器或用 bridge 网络启动，可以在 `.env` 里显式覆盖：

```bash
OGMEM_URL=http://ogmem-host.local:8090
```

### 数据落点

三个命名 volume（由 compose 管理）：

| Volume                    | Mounted as                  | 用途                                  |
|---------------------------|-----------------------------|---------------------------------------|
| `ogmem_ogdb`              | `/var/lib/opengauss`        | openGauss 数据（向量索引）             |
| `ogmem_postgres-data`     | `/var/lib/postgresql/data`  | Postgres SQL 存储（context files、events、grants、outbox 等） |
| `ogmem_openclaw-home`     | `/home/node/.openclaw`      | OpenClaw 状态（auth、agents 等）       |

`down` 保留这些；`down -v` 删除。

### 从源码构建 ogmem 镜像

```bash
docker build -f ../docker/Dockerfile.standalone -t ogmemory:local .
OGMEM_IMAGE=ogmemory:local docker compose --profile with-db up -d
```

### 从 `deploy.sh` 迁移

1. `cp .env.example .env`（从已提交模板创建 gitignored 运行时配置）
2. `cp ogmemory.example.yaml ogmemory.yaml`
3. 编辑 `.env`（如果之前用 `deploy.env`，把值搬过来）
4. 停掉旧容器：`docker compose --profile with-db down`（如已用 compose 启动过）
5. `docker compose --profile with-db up -d`

## 环境要求

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+, CentOS 7+, openEuler)
- **CPU 架构**: **ARM64 (鲲鹏)** — 镜像仅支持 ARM，不支持 x86_64
- **内存**: 至少 8GB (推荐 16GB+)
- **磁盘**: 至少 20GB 可用空间

### 软件要求

- **Docker**: 20.10+
- **curl**: 用于下载脚本

检查 Docker 是否安装：

```bash
docker --version
```

确认当前系统为 ARM64 架构：

```bash
uname -m
# 输出应为 aarch64
```

如未安装 Docker，请参考 [Docker 官方文档](https://docs.docker.com/engine/install/) 安装。

### 网络要求

- 确保能访问以下地址：
  - 华为云 SWR 镜像仓库: `swr.cn-north-4.myhuaweicloud.com`
  - LLM API 服务地址（如华为云、火山引擎、阿里云等）

---

## 快速开始

多用户服务器中，每位用户使用独立实例、模型和端口时，参见[用户一键部署与体验指南](user-quickstart/README_zh.md)。

### 远程一键安装

```bash
curl -fsSL https://raw.gitcode.com/opengauss/oGMemory/raw/dev/deploy/install.sh | bash
```

执行后会下载以下文件到 `./ogmem-deploy/` 目录：
- `.env`: 配置文件（OpenClaw + 基础设施配置）
- `ogmemory.example.yaml`: oGMemory 服务配置模板
- `deploy.sh`: 部署脚本

---

## 配置说明

### 配置文件结构

本部署方案使用 **两个配置文件**，各司其职：

| 文件 | 职责 | 内容 |
|------|------|------|
| `.env` | OpenClaw + 基础设施 | LLM 共享配置、openGauss、容器名/镜像、部署参数 |
| `ogmemory.yaml` | oGMemory 业务配置 | LLM、Embedding、向量数据库、身份认证、缓存等 |

**关键机制**：`ogmemory.yaml` 支持 `${ENV_VAR}` 语法引用 `.env` 中的变量，避免重复配置。

**职责边界更新：**

- `deploy.sh` 不再从 `ogmemory.yaml` 反向推导 OpenClaw 默认认证参数。
- 单实例模式下，OpenClaw 认证参数只按 `OG_AUTH_API_KEY -> OG_ROOT_API_KEY`、`OG_AUTH_ACCOUNT_ID -> OG_ACCOUNT_ID` 回退。
- `openclaw.json` 由 OpenClaw 容器入口脚本托管：首次启动使用 JSON 模板生成配置，`gateway.port` 会转换为整数。已有配置不会因环境变量变化而整体重建；入口脚本只替换已知占位符、补齐缺失的 `gateway.mode=local`、端口及已有记忆插件的认证字段，并在修改前创建权限为 `0600` 的 `openclaw.json.bak.*` 备份，保留其他用户设置。旧版未加引号的 `__GATEWAY_PORT__` 也会迁移。
- 容器托管的配置须为严格 JSON（不支持 JSON5 注释或尾随逗号）。JSON 损坏、端口非法或显式设置了非 `local` 的 gateway mode 时，入口脚本会报错退出且不覆盖原配置；请修复或从确认有效的备份恢复，不要删除整个状态卷或绕过 gateway 启动检查。有效配置及凭据不会自动从新的 `.env` 值覆盖；需要变更时请明确修改对应配置字段。

**推荐做法：**

- 把共享配置统一放在 `.env`，例如 `OG_ROOT_API_KEY`、`OG_ACCOUNT_ID`、`OG_USER_ID`、`OG_AGENT_ID`。
- `ogmemory.yaml` 默认通过 `${ENV_VAR}` 引用这些值，不再手工重复填写相同内容。
- 只有在 oGMemory 需要和 OpenClaw 使用不同身份或不同认证策略时，才在 `ogmemory.yaml` 里单独覆盖。

```
.env                    ogmemory.yaml
┌──────────────────┐          ┌────────────────────────┐
│ LLM_API_KEY=sk-xxx│──────→  │ api_key: "${LLM_API_KEY}" │
│ LLM_BASE_URL=... │──────→  │ base_url: "${LLM_BASE_URL}"│
│ LLM_MODEL=qwen3  │──────→  │ model: "${LLM_MODEL}"      │
│ OG_ROOT_API_KEY= │──────→  │ root_api_key: "${OG_ROOT...│
│ OG_ACCOUNT_ID=   │──────→  │ account_id: "${OG_ACCOUNT..│
│ ...              │          │ ...                      │
└──────────────────┘          └────────────────────────┘
```

- **共享配置**：`ogmemory.yaml` 中写 `${LLM_API_KEY}` 等占位符，自动使用 `.env` 的值
- **oGMemory 专用**：将 `${...}` 替换为硬编码值即可覆盖

### 第一步：编辑 .env

```bash
cd ogmem-deploy  # 或 cd oGMemory/deploy
vim .env
```

### 第二步：创建 ogmemory.yaml

```bash
cp ogmemory.example.yaml ogmemory.yaml
vim ogmemory.yaml
```

> **💡 提示：** 默认模板中 LLM/Embedding 配置已使用 `${...}` 引用，大多数情况下只需修改 Embedding 模型名称和向量维度即可。

### .env 环境变量列表

> **💡 重要提示：**
> 1. `LLM_PROVIDER`、`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 这四个变量同时供 OpenClaw 和 oGMemory 使用（通过 `ogmemory.yaml` 中的 `${...}` 引用）。如果 oGMemory 需要不同的配置，请在 `ogmemory.yaml` 中将对应字段改为硬编码值。
> 2. `OPENCLAW_HOST_IP` 建议填写宿主机实际 IP 地址（而非 `127.0.0.1`），以便其他机器或容器能够访问服务。

#### 必填环境变量

| 变量名 | 说明 | 示例值 | 适用服务 |
|--------|------|--------|---------|
| `LLM_PROVIDER` | LLM 提供商标识 | `dashscope` | OpenClaw + oGMemory 共用 |
| `LLM_API_KEY` | LLM API 密钥 | `sk-xxxxx` | OpenClaw + oGMemory 共用 |
| `LLM_BASE_URL` | LLM API 基础地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenClaw + oGMemory 共用 |
| `LLM_MODEL` | LLM 模型名称 | `qwen3.6-plus` | OpenClaw + oGMemory 共用 |
| `ENABLE_OPENGAUSS` | 是否启用 openGauss（仅 deploy.sh） | `true` / `false` | openGauss |
| `VECTOR_DB_PASSWORD` | 向量库密码（启用本地 openGauss 时必填） | `OpenGauss@2024` | openGauss / oGMemory |
| `VECTOR_DB_HOST` | 向量库宿主机可达地址（deploy.sh 必填；compose 默认 `opengauss`） | `127.0.0.1` | oGMemory |
| `STORAGE_DB_PASSWORD` | 存储库密码（启用 with-db profile 时必填） | `postgres-strong-pw` | postgres / oGMemory |
| `OPENCLAW_HOST_IP` | OpenClaw 宿主机 IP 地址（建议填写宿主机实际 IP） | `192.168.1.100` | OpenClaw + oGMemory |

#### 可选环境变量 — OpenClaw 连接配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `GATEWAY_PORT` | OpenClaw Gateway 端口 | `18789` | OpenClaw |
| `OPENCLAW_GATEWAY_TOKEN` | 前端配对密码 | `ogmem-default-token` | OpenClaw |
| `OPENCLAW_BIND_MODE` | 绑定模式（`lan` / `localhost`） | `lan` | OpenClaw |
| `OGMEM_URL` | oGMemory API 地址（供 OpenClaw 连接） | `http://127.0.0.1:8090` | OpenClaw |
| `OG_AUTH_API_KEY` | OpenClaw 连接 oGMemory 的认证 API Key | 回退至 `OG_ROOT_API_KEY` | OpenClaw |
| `OG_AUTH_ACCOUNT_ID` | OpenClaw 连接 oGMemory 的账户 ID | 回退至 `OG_ACCOUNT_ID` | OpenClaw |

> **说明：** 单实例模式默认不需要额外配置 `OG_AUTH_API_KEY` / `OG_AUTH_ACCOUNT_ID`，脚本会分别回退到 `OG_ROOT_API_KEY` / `OG_ACCOUNT_ID`。只有当 OpenClaw 需要使用不同于默认值的认证身份时，才建议单独覆盖。

#### 可选环境变量 — 多租认证配置

> 启用多租认证后，所有数据路由和管理路由都需要 API Key 认证。启用后需要同时配置 `OG_ROOT_API_KEY` 或 `OG_ADMIN_API_KEYS`。

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `OG_ROOT_API_KEY` | Root API Key（超级管理员密钥，拥有全局最高权限） | 无 | oGMemory |
| `OG_ADMIN_API_KEYS` | Admin API Keys（账户管理员密钥，逗号分隔） | 无 | oGMemory |
| `OG_ACCOUNT_ID` | 默认租户标识，同时也是单实例模式下 OpenClaw 默认访问账户 | `acct-demo` | oGMemory / OpenClaw |
| `OG_USER_ID` | 默认用户标识 | `u-alice` | oGMemory |
| `OG_AGENT_ID` | 默认 Agent 标识 | `main` | oGMemory |

> **推荐：** 单实例或共享默认身份场景下，只需要在 `.env` 里配置一次 `OG_ROOT_API_KEY`、`OG_ACCOUNT_ID`、`OG_USER_ID`、`OG_AGENT_ID`；默认模板中的 `ogmemory.yaml` 会直接复用这些值。

**`OG_ADMIN_API_KEYS` 格式说明：**

- 绑定账户格式：`account_id:key`（只能管理指定账户）
- 全局格式：`key`（绑定到 `OG_ACCOUNT_ID` 对应的默认账户）
- 多个 key 用逗号分隔

```bash
# 示例：两个账户各一个 Admin Key
OG_ADMIN_API_KEYS="acct-company-a:admin-key-for-a,acct-company-b:admin-key-for-b"

# 示例：全局 Admin Key（绑定到 OG_ACCOUNT_ID）
OG_ADMIN_API_KEYS="global-admin-key"
```

**`OG_AUTH_API_KEY` 说明：**

- 默认使用 `OG_ROOT_API_KEY`（ROOT 权限，可访问所有账户数据）
- 也可设置为 Admin Key（限定账户权限），此时需填写 `account_id:key` 中的 **key 部分**
- 当 `OG_AUTH_API_KEY` 是 Admin Key 时，`OG_AUTH_ACCOUNT_ID` 必须与 Key 绑定的账户一致

#### 可选环境变量 — 数据持久化配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `AGFS_DATA_DIR` | AGFS 数据存储目录（持久化记忆文件） | 无（不挂载） | oGMemory |
| `OPENCLAW_HOME_DIR` | OpenClaw 状态目录（避免重复配对） | 无（不挂载） | OpenClaw |

> **💡 推荐：** 设置 `AGFS_DATA_DIR` 以持久化记忆数据，设置 `OPENCLAW_HOME_DIR` 以避免重复配对。

#### 可选环境变量 — 容器配置

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `OGMEM_CONTAINER_NAME` | ogmemory 容器名称 | `ogmem` | 容器 |
| `OPENCLAW_CONTAINER_NAME` | openclaw-ogmemory 容器名称 | `openclaw_ogmem` | 容器 |
| `OGMEM_IMAGE` | ogmemory 镜像地址 | 见 .env | 容器 |
| `OPENCLAW_IMAGE` | openclaw-ogmemory 镜像地址 | 见 .env | 容器 |

#### 可选环境变量 — 向量库（VECTOR_DB_*）

> 向量库默认是 openGauss 容器（生产推荐目标）；单库部署时把
> `VECTOR_DB_HOST` 改为 `postgres` 即可让向量索引落在 pgvector Postgres
> 里，M3 的 `OpenGaussVectorIndex` 代码路径无需改动。

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `VECTOR_DB_HOST` | oGMemory 看到的向量库主机。默认 `127.0.0.1`：ogmem 是 host-net，loopback 是唯一能命中 bridge 端口转发的地址（compose 服务名 `opengauss` 从 host-net 解析不了）。 | `127.0.0.1` | oGMemory |
| `VECTOR_DB_HOST_PORT` | oGMemory 拨号时命中的端口（宿主机端口）。Compose `ports:` 把它转发到容器内 `VECTOR_DB_PORT`。外部 DB 路径下直接写 DB 的真实端口即可。 | `5432` | oGMemory |
| `VECTOR_DB_PORT` | openGauss 容器**内部**监听端口（`GS_PORT` / compose `ports:` 右值）。ogmem 看不到这个变量。 | `5432` | openGauss |
| `VECTOR_DB_NAME` | 向量库数据库名 | `postgres` | openGauss |
| `VECTOR_DB_USER` | 向量库用户名 | `gaussdb` | openGauss |
| `VECTOR_DB_CONTAINER_NAME` | 向量库容器名称 | `opengauss` | openGauss |
| `VECTOR_DB_NODE_NAME` | openGauss 节点名称 | `gaussdb` | openGauss |
| `OG_IMAGE_REPO` / `OG_IMAGE_TAG` | openGauss 镜像仓库 / 标签 | 见 .env | openGauss |
| `OG_CPUSET_CPUS` | CPU 亲和性 | 无 | openGauss |
| `OG_WAIT_TIMEOUT` | 启动超时时间（秒，仅 deploy.sh） | `120` | openGauss |

#### 可选环境变量 — 存储库（STORAGE_DB_*）

> oGMemory 的 SQL 存储后端（`storage.backend: sql`）通过 Postgres 服务持久化
> context files、events、grants、outbox、subscriptions 等业务数据。只有
> `with-db` profile 启动 postgres 服务时才需要这些变量；如果使用外部
> Postgres，设 `STORAGE_DB_HOST` 即可跳过本地容器。

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|----------|
| `STORAGE_DB_HOST` | oGMemory 看到的存储库主机。默认 `127.0.0.1`，对齐 `VECTOR_DB_HOST`：ogmem 是 host-net，loopback 是唯一能命中 bridge 端口转发的地址。 | `127.0.0.1` | oGMemory |
| `STORAGE_DB_HOST_PORT` | oGMemory 拨号时命中的端口（宿主机端口）。Compose `ports:` 把它转发到 postgres 容器内的 5432。外部 DB 路径下直接写 DB 的真实端口即可。 | `54321` | oGMemory |
| `STORAGE_DB_PORT` | **不存在**——postgres 容器内部端口固定为 5432，没有可调变量。`STORAGE_DB_*` 族只有 host / host_port / name / user / password / container_name。 | — | — |
| `STORAGE_DB_NAME` | 存储库数据库名 | `ogmem` | postgres / oGMemory |
| `STORAGE_DB_USER` | 存储库用户名 | `ogmem` | postgres / oGMemory |
| `STORAGE_DB_CONTAINER_NAME` | 存储库容器名称 | `ogmem-postgres` | 容器 |

> **deploy.sh 专用补充**：legacy docker run 路径还需要 `VECTOR_DB_HOST`
> （向量库宿主机可达地址，默认 `127.0.0.1`）和 `ENABLE_OPENGAUSS`
> （是否启动本地 openGauss 容器，默认 `true`）。

> **说明：** `STORAGE_DB_*` 一组变量由 entrypoint 在容器启动时替换到
> `ogmemory.yaml` 的 `storage.connection_string`，无需手工编辑 yaml。

#### 可选环境变量 — 部署控制

| 变量名 | 说明 | 默认值 | 适用服务 |
|--------|------|--------|---------|
| `AUTO_HEALTH_CHECK` | 是否自动健康检查 | `true` | 部署控制 |
| `SKIP_PULL` | 是否跳过镜像拉取 | `false` | 部署控制 |

### ogmemory.yaml 配置字段

> 完整字段说明请参考 `ogmemory.example.yaml` 中的注释。

#### YAML 字段与环境变量的对应关系

下面这张表结合了 [OGMEMORY_ENV.md](../docs/OGMEMORY_ENV.md) 和当前代码默认值，方便直接查询字段映射、默认值和可选值。

| YAML 字段 | 对应环境变量 | 默认值 | 常见可选值 / 格式 | 说明 |
|-----------|-------------|--------|------------------|------|
| `llm.provider` | `CONTEXTENGINE_PROVIDER` | 代码默认 `mock`；部署时通常直接复用 `LLM_PROVIDER` | `mock` / `openai` / `openai-cached`，以及 `dashscope`、`volcengine`、`zhipu` 等兼容 OpenAI 协议的提供商标识 | LLM 提供商 |
| `llm.api_key` | `LLM_API_KEY` | 空 | API Key 字符串 | 部署模板通常复用 OpenClaw 的 `LLM_API_KEY` |
| `llm.base_url` | `LLM_BASE_URL` | 空 | 兼容接口地址 | 部署模板通常复用 OpenClaw 的 `LLM_BASE_URL` |
| `llm.model` | `OGMEM_LLM_MODEL` 或模板中的 `LLM_MODEL` | `gpt-4o-mini` | 任意可用模型名，如 `qwen3-max`、`gpt-4o-mini`、`glm-4.5` | 部署模板默认直接引用 `LLM_MODEL` |
| `llm.temperature` | `LLM_TEMPERATURE` | `0.7` | 通常 `0.0` 到 `2.0` | 数值越低越稳定 |
| `llm.max_tokens` | `LLM_MAX_TOKENS` | `4096` | 正整数 | 单次生成最大 token 数 |
| `embedding.provider` | `EMBEDDING_PROVIDER` | 空，回退到 `llm.provider` | `openai` / `st` / `volcengine` / `mock` | Embedding 提供商 |
| `embedding.model` | `OGMEM_EMBEDDING_MODEL` | `text-embedding-ada-002` | 你的 Embedding 模型名，如 `text-embedding-v4` | 需要和向量维度匹配 |
| `embedding.base_url` | `OGMEM_EMBEDDING_BASE_URL` | 空，回退到 `llm.base_url` | 兼容接口地址 | Embedding API 地址 |
| `embedding.api_key` | `OGMEM_EMBEDDING_API_KEY` | 空，回退到 `llm.api_key` | API Key 字符串 | Embedding API Key |
| `embedding.multimodal` | `OGMEM_EMBEDDING_MULTIMODAL` | `false` | `true` / `false` | 仅多模态 Embedding 端点时设为 `true` |
| `embedding.st_model` | `ST_MODEL` | `BAAI/bge-large-en-v1.5` | 任意 Sentence Transformer 模型名 | 仅 `embedding.provider=st` 时通常有意义 |
| `vector_db.type` | `VECTOR_DB_TYPE` | 代码默认 `chroma`；部署模板默认写的是 `opengauss` | `memory` / `opengauss` / `chroma` | 向量数据库类型 |
| `vector_db.connection_string` | — | 空 | `host=... port=... dbname=... user=... password=...` | 启用 openGauss 时由 `deploy.sh` 自动替换 |
| `vector_db.dimension` | `OPENGAUSS_DIMENSION` | `1024` | 正整数，如 `1024` / `1536` / `3072` | 必须与 Embedding 模型维度一致 |
| `vector_db.table_name` | `OPENGAUSS_TABLE_NAME` | `vector_index` | 任意合法表名 | openGauss 向量表名 |
| `vector_db.pool_size` | `OPENGAUSS_POOL_SIZE` | `5` | 正整数 | openGauss 连接池大小 |
| `vector_db.chroma_persist_dir` | `CHROMA_PERSIST_DIR` | `.chroma_data` | 目录路径 | 使用 Chroma 持久化时生效 |
| `vector_db.chroma_collection` | `CHROMA_COLLECTION` | `contextengine` | 任意集合名 | 使用 Chroma 时生效 |
| `service.http_port` | `OGMEM_HTTP_PORT` | `8090` | 合法端口号 | oGMemory HTTP 服务端口 |
| `service.workers` | `OGMEM_WORKERS` | `2` | 正整数 | Gunicorn worker 数量 |
| `service.http_ip_allowlist` | `OG_HTTP_IP_ALLOWLIST` | 空 | 逗号分隔 IP / CIDR，例如 `127.0.0.1,10.0.0.0/8` | HTTP 白名单 |
| `service.http_ip_allowlist_trust_proxy` | `OG_HTTP_IP_ALLOWLIST_TRUST_PROXY` | `false` | `true` / `false` | 只有在反向代理前置时才建议开启 |
| `service.http_trusted_proxies` | `OG_HTTP_TRUSTED_PROXIES` | 空 | 逗号分隔 IP / CIDR | 受信任代理列表 |
| `agfs.base_url` | `AGFS_BASE_URL` | `http://127.0.0.1:1833` | URL | AGFS 服务地址 |
| `agfs.mount_prefix` | `AGFS_MOUNT_PREFIX` | `/local` | 路径前缀 | AGFS 挂载前缀 |
| `index.interval` | `INDEX_INTERVAL` | `30` | 正整数，单位秒 | 索引轮询间隔 |
| `index.workers` | `INDEX_WORKERS` | `1` | 正整数 | 索引 worker 数量 |
| `identity.account_id` | `OG_ACCOUNT_ID` | `acct-demo` | 任意账户标识，如 `acct-company-a` | 租户标识 |
| `identity.user_id` | `OG_USER_ID` | `u-alice` | 任意用户标识，如 `u-bob` | 用户标识 |
| `identity.agent_id` | `OG_AGENT_ID` | `main` | 任意 Agent 标识，如 `shared-agent` | Agent 标识 |
| `auth.role_control_enabled` | `OG_ROLE_CONTROL_ENABLED` | `false` | `true` / `false` | 是否启用多租认证 |
| `auth.root_api_key` | `OG_ROOT_API_KEY` | 空 | API Key 字符串 | Root API Key |
| `auth.admin_api_keys` | `OG_ADMIN_API_KEYS` | 空 | `account_id:key` 或 `key`，多个值逗号分隔 | Admin API Keys 列表 |
| `sharing.agent_shared_mode` | `OG_AGENT_SHARED_MODE` | `off` | `off` / `user` | Agent 共享模式 |
| `sharing.agent_shared_list` | `OG_AGENT_SHARED_LIST` | 空 | 逗号分隔 Agent ID，例如 `shared-agent,agent-handbook` | 共享 Agent ID 列表 |
| `memory.after_turn_threshold` | `OGMEM_AFTER_TURN_THRESHOLD` | `400` | 正整数 | 达到阈值后触发记忆提取 |
| `memory.rolling_compress_enabled` | `OGMEM_ROLLING_COMPRESS_ENABLED` | `true` | `true` / `false` | 是否启用滚动压缩 |
| `memory.compact_prepare_token_ttl` | `OGMEM_COMPACT_PREPARE_TOKEN_TTL` | `300` | 正整数，单位秒 | prepare_compaction token 有效期 |
| `memory.directory_summary_enabled` | `OGMEM_DIRECTORY_SUMMARY_ENABLED` | `false` | `true` / `false` | 是否启用目录摘要 |
| `memory.prefetch_enabled` | `OGMEM_PREFETCH_ENABLED` | `false` | `true` / `false` | 是否启用 compose 前预取并填充 SessionTopicBuffer |
| `memory.prefetch_top_k` | `OGMEM_PREFETCH_TOP_K` | `5` | 正整数 | 预取写入 pending_injection 的最大命中数 |
| `memory.session_state_bridge_enabled` | `OGMEM_SESSION_STATE_BRIDGE_ENABLED` | `true` | `true` / `false` | 是否将 SessionState 同步到 SessionWindowState |
| `memory.session_state_sync_interval_turns` | `OGMEM_SESSION_STATE_SYNC_INTERVAL_TURNS` | `1` | 正整数 | SessionState 版本变化后至少间隔多少用户轮次同步一次 |
| `memory.topic_detection_enabled` | `OGMEM_TOPIC_DETECTION_ENABLED` | `false` | `true` / `false` | 是否启用 P2 话题检测 |
| `memory.compression_quality_enabled` | `OGMEM_COMPRESSION_QUALITY_ENABLED` | `false` | `true` / `false` | 是否计算 P2 archive 压缩质量指标 |
| `memory.compression_quality_persist_metadata` | `OGMEM_COMPRESSION_QUALITY_PERSIST_METADATA` | `false` | `true` / `false` | 是否将压缩质量指标持久化到 archive metadata |
| `cache.enabled` | `OGMEM_CACHE_ENABLED` | `true` | `true` / `false` | 是否启用缓存 |
| `cache.max_size` | `OGMEM_CACHE_MAX_SIZE` | `1000` | 正整数 | 缓存最大条目数 |

> **说明：**
> 1. 布尔值建议统一写成 `true` / `false`。
> 2. 列表型变量通常使用逗号分隔。
> 3. 如果某个字段已经在 `ogmemory.yaml` 中写死为硬编码值，则该字段优先级高于环境变量。

#### `${ENV_VAR}` 引用规则

- `${VAR_NAME}` → 如果环境变量存在，替换为实际值
- `${VAR_NAME}` → 如果环境变量不存在，保持原样（方便排查缺失配置）
- 非 `${...}` 的硬编码值 → 不受影响，优先级最高

#### 推荐的去冗余写法

默认模板已经按下面的方式处理：

```yaml
identity:
  account_id: "${OG_ACCOUNT_ID}"
  user_id: "${OG_USER_ID}"
  agent_id: "${OG_AGENT_ID}"

auth:
  root_api_key: "${OG_ROOT_API_KEY}"
```

对应地，你只需要在 `.env` 中维护一份共享默认值：

```bash
OG_ROOT_API_KEY="root-secret-key-change-me"
OG_ACCOUNT_ID="acct-demo"
OG_USER_ID="u-alice"
OG_AGENT_ID="main"
```

这样：

- OpenClaw 默认通过 `OG_ROOT_API_KEY` / `OG_ACCOUNT_ID` 连接 oGMemory
- oGMemory 默认也通过同一组 `OG_*` 变量确定身份与认证
- 不需要在 `.env` 和 `ogmemory.yaml` 中手工写两遍相同的账户和 Root Key

只有在以下场景下，才建议在 `ogmemory.yaml` 中单独覆盖：

- oGMemory 需要与 OpenClaw 使用不同的默认 `account_id`
- oGMemory 需要不同的 `user_id` 或 `agent_id`
- oGMemory 不想使用 Root Key，而是使用单独的认证配置

---

## 部署方式

### 方式一：完整部署（openGauss + ogmemory + openclaw-ogmemory）

适用于首次部署，需要向量数据库的场景。

```bash
bash deploy.sh -password <密码>
```

**密码要求：**
- 长度至少 8 个字符
- 包含大写字母、小写字母、数字、特殊符号(#?!@$%^&*)中的至少三种

**示例：**

```bash
bash deploy.sh -password "OpenGauss@2024"
```

### 方式二：仅部署服务（不使用 openGauss）

适用于已有外部数据库，或使用内存模式的场景。

1. 修改配置文件：

```bash
ENABLE_OPENGAUSS="false"
```

2. 执行部署：

```bash
bash deploy.sh
```

### 方式三：多租户部署

适用于多个租户共享同一套 oGMemory 服务，但各自拥有独立 OpenClaw Gateway 的场景。

1. 在 `ogmemory.yaml` 中启用多租认证：

```yaml
auth:
  role_control_enabled: true
  root_api_key: "${OG_ROOT_API_KEY}"
  admin_api_keys:
    - "acct-company-a:admin-key-for-a"
    - "acct-company-b:admin-key-for-b"
```

2. 在 `.env` 中配置认证 Key 和多租户 OpenClaw 实例：

```bash
OG_ROOT_API_KEY="root-secret-key-change-me"
OG_ADMIN_API_KEYS="acct-company-a:admin-key-for-a,acct-company-b:admin-key-for-b"

OPENCLAW_INSTANCES="openclaw_tenant_a|18789|tenant-a-web-token|admin-key-for-a|acct-company-a|/data/openclaw-a;openclaw_tenant_b|18790|tenant-b-web-token|admin-key-for-b|acct-company-b|/data/openclaw-b"
```

**`OPENCLAW_INSTANCES` 格式说明：**

```
container_name|gateway_port|gateway_token|auth_api_key|auth_account_id|openclaw_home_dir
```

各字段含义：

| 字段 | 说明 |
|------|------|
| `container_name` | OpenClaw 容器名称 |
| `gateway_port` | Gateway 端口 |
| `gateway_token` | Web 界面配对密码 |
| `auth_api_key` | 连接 oGMemory 的 API Key（Admin Key 的 key 部分，或 Root Key） |
| `auth_account_id` | 绑定的账户 ID |
| `openclaw_home_dir` | OpenClaw 状态持久化目录（可选，留空则不挂载） |

多个实例用分号 `;` 分隔。

> **说明：** 配置了 `OPENCLAW_INSTANCES` 后，每个实例中的 `auth_api_key` / `auth_account_id` 优先级最高；单实例模式才会使用 `OG_AUTH_API_KEY` / `OG_AUTH_ACCOUNT_ID` 的默认回退逻辑。

3. 执行部署：

```bash
bash deploy.sh -password <密码>
```

---

## 多租认证与 Agent 共享场景详解

### 核心概念

| 概念 | 说明 |
|------|------|
| **Account（账户）** | 数据隔离的最小单位，不同 Account 的记忆数据完全隔离 |
| **User（用户）** | 同一 Account 下的用户，数据共享 |
| **Agent（智能体）** | 同一 Account 下的 Agent，数据共享 |
| **Agent 共享** | 将指定 Agent 的记忆对同 Account 下所有 User 可见 |
| **Root API Key** | 超级管理员密钥，可访问所有 Account 的数据 |
| **Admin API Key** | 账户管理员密钥，只能访问绑定的 Account 数据 |

### 场景一：同 Account + 不同 User + Agent 共享

**需求**：一个公司（`acct-company-a`）有多个员工（`u-alice`、`u-bob`），他们共享同一个 Agent（`shared-agent`）的记忆，同时各自也有私有 Agent。

**ogmemory.yaml 配置：**

```yaml
identity:
  account_id: "${OG_ACCOUNT_ID}"    # acct-company-a
  user_id: "${OG_USER_ID}"          # 默认用户，OpenClaw 可覆盖
  agent_id: "${OG_AGENT_ID}"        # 默认 Agent

auth:
  role_control_enabled: true
  root_api_key: "${OG_ROOT_API_KEY}"
  admin_api_keys:
    - "acct-company-a:admin-key-for-a"

sharing:
  agent_shared_mode: "user"          # 启用用户级共享
  agent_shared_list:
    - "shared-agent"                 # shared-agent 对所有 MEMBER 用户可见
```

**.env 配置：**

```bash
# 共享 LLM 配置
LLM_PROVIDER="dashscope"
LLM_API_KEY="sk-xxxxx"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL="qwen3-max"

# 多租认证
OG_ROOT_API_KEY="root-secret-key-change-me"
OG_ACCOUNT_ID="acct-company-a"
OG_USER_ID="u-alice"
OG_AGENT_ID="main"

# 两个员工各一个 OpenClaw Gateway，共享同一个 Account
OPENCLAW_INSTANCES="openclaw_alice|18789|alice-web-token|admin-key-for-a|acct-company-a|/data/openclaw-alice;openclaw_bob|18790|bob-web-token|admin-key-for-a|acct-company-a|/data/openclaw-bob"
```

**数据隔离效果：**

| 用户 | 可见的 Agent | 可见的记忆 |
|------|-------------|-----------|
| u-alice（通过 openclaw_alice） | `main`（私有）+ `shared-agent`（共享） | 自己的 + 共享的 |
| u-bob（通过 openclaw_bob） | `main`（私有）+ `shared-agent`（共享） | 自己的 + 共享的 |

> **说明：** 两个 OpenClaw Gateway 使用同一个 Admin Key（`admin-key-for-a`）和同一个 Account（`acct-company-a`），但各自以不同 User 身份访问 oGMemory。`shared-agent` 的记忆对两个用户都可见，而各自的 `main` Agent 记忆互相隔离。

### 场景二：不同 Account + Agent 共享

**需求**：两个公司（`acct-company-a`、`acct-company-b`）各自独立，数据完全隔离，每个公司有自己的 Admin Key。

**ogmemory.yaml 配置：**

```yaml
identity:
  account_id: "${OG_ACCOUNT_ID}"
  user_id: "${OG_USER_ID}"
  agent_id: "${OG_AGENT_ID}"

auth:
  role_control_enabled: true
  root_api_key: "${OG_ROOT_API_KEY}"
  admin_api_keys:
    - "acct-company-a:admin-key-for-a"
    - "acct-company-b:admin-key-for-b"

sharing:
  agent_shared_mode: "user"
  agent_shared_list:
    - "shared-agent"
```

**.env 配置：**

```bash
LLM_PROVIDER="dashscope"
LLM_API_KEY="sk-xxxxx"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL="qwen3-max"

OG_ROOT_API_KEY="root-secret-key-change-me"
OG_ACCOUNT_ID="acct-company-a"
OG_USER_ID="u-alice"
OG_AGENT_ID="main"

# 两个公司各一个 OpenClaw Gateway，各自使用自己的 Admin Key
OPENCLAW_INSTANCES="openclaw_company_a|18789|company-a-web-token|admin-key-for-a|acct-company-a|/data/openclaw-a;openclaw_company_b|18790|company-b-web-token|admin-key-for-b|acct-company-b|/data/openclaw-b"
```

**数据隔离效果：**

| 公司 | OpenClaw | 认证 | 可见数据 |
|------|---------|------|---------|
| acct-company-a | openclaw_company_a:18789 | admin-key-for-a | 仅 company-a 的记忆 |
| acct-company-b | openclaw_company_b:18790 | admin-key-for-b | 仅 company-b 的记忆 |

> **说明：** 两个公司使用不同的 Admin Key 和不同的 Account，数据完全隔离。每个公司内部的 `shared-agent` 仅对本公司的用户可见。

### 场景三：单租户（无认证，最简配置）

**需求**：个人使用或内网环境，不需要多租隔离。

**ogmemory.yaml 配置：**

```yaml
identity:
  account_id: "acct-demo"
  user_id: "u-alice"
  agent_id: "main"

auth:
  role_control_enabled: false

sharing:
  agent_shared_mode: "off"
```

**.env 配置：**

```bash
LLM_PROVIDER="dashscope"
LLM_API_KEY="sk-xxxxx"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL="qwen3-max"
```

> **说明：** 无认证模式下，所有请求默认使用 `identity` 中配置的账户，无需 API Key。

### 关于 `openclaw.json` 的自动迁移

如果设置了 `OPENCLAW_HOME_DIR`，`openclaw.json` 会持久化保存。当前版本的容器入口脚本会这样处理：

1. 如果 `openclaw.json` 不存在，按当前模板直接生成。
2. 如果文件中仍残留模板占位符，也会自动重建。
3. 如果文件缺少关键字段，例如 `authApiKey`、`authAccountId`，会先备份旧文件，再按当前模板重建。
4. 自动备份最多保留最近 `3` 份，超过后会清理更旧的备份。

备份文件名示例：

```text
openclaw.json.bak.20260420-101530
```

如果你曾手工修改过 `openclaw.json`，升级后请优先对比备份文件，再决定是否手工迁移你的自定义内容。

---

## 配置修改与生效

### 修改 oGMemory 配置

1. 编辑 `deploy/ogmemory.yaml`
2. 重启容器：

```bash
docker restart ogmem
```

> **💡 提示：** `ogmemory.yaml` 直接挂载到容器中（只读），容器启动时 entrypoint 会自动完成 `${...}` 替换和 openGauss 连接信息替换。修改后只需 `docker restart` 即可生效，无需重新运行 `deploy.sh`。

### 修改 OpenClaw 或基础设施配置

1. 编辑 `deploy/.env`
2. 重新部署：

```bash
bash deploy.sh
```

> **说明：** 如果只修改了 `OG_AUTH_API_KEY`、`OG_AUTH_ACCOUNT_ID`、`OG_ROOT_API_KEY`、`OG_ACCOUNT_ID` 这类 OpenClaw 连接参数，也建议重新执行 `bash deploy.sh`，让容器以新的环境变量启动，并在需要时触发 `openclaw.json` 自动迁移。

### 配置生效流程

```
ogmemory.yaml (用户编辑)
    │
    ▼ docker run -v ogmemory.yaml:/etc/ogmem/config.yaml:ro
    │
    ▼ 容器 entrypoint 自动执行：
    │   1. 读取 /etc/ogmem/config.yaml
    │   2. 替换 ${LLM_API_KEY} 等环境变量引用
    │   3. 替换 vector_db.connection_string（如有 OPENGAUSS_* 环境变量）
    │   4. 写入 /tmp/ogmem.resolved.yaml
    │   5. 设置 OGMEM_CONFIG=/tmp/ogmem.resolved.yaml
    │   6. 启动应用
    │
    ▼ 应用读取 /tmp/ogmem.resolved.yaml
```

---

## 部署过程说明

部署脚本会依次执行以下步骤：

1. **前置检查**: 检查 Docker 环境
2. **拉取镜像**: 从华为云 SWR 拉取所需镜像
3. **部署 openGauss**: 启动向量数据库容器（如启用）
4. **部署 Postgres**: 启动 SQL 存储后端容器（如启用 `with-db` profile）
5. **等待数据库就绪**: 最长等待 120 秒
6. **部署 oGMemory**: 挂载 `ogmemory.yaml`，传入环境变量，启动容器
7. **部署 OpenClaw Gateway**: 启动 Web 服务（单实例或多实例）
8. **健康检查**: 验证服务是否正常

---

## 常用命令

### 查看部署状态

```bash
bash deploy.sh --status
```

> `--status` 只查看当前目录下这份 `.env` 解析出来的容器名，不会再按全局标签扫描同机上的其它部署实例。

### 清理部署

```bash
bash deploy.sh --cleanup
```

> `--cleanup` 只会删除当前目录下这份 `.env` 中声明的容器名，例如 `OG_CONTAINER_NAME`、`OGMEM_CONTAINER_NAME` 和 `OPENCLAW_INSTANCES` / `OPENCLAW_CONTAINER_NAME` 对应的容器。
> 这样同一台机器上可以并存多套部署；在某一套目录里执行 cleanup，不会误删另一套环境的容器。数据仍会保留在你配置的持久化目录中。
> 这里说的“所有容器”是指当前这份 `.env` 对应的全部容器，不是宿主机上所有带 `ogmem.managed=true` 标签的容器。

> 注意：此命令会停止并删除所有容器，数据不会丢失（如果配置了持久化目录）

### 查看日志

```bash
# 查看 oGMemory 日志
docker logs ogmem -f

# 查看 OpenClaw 日志
docker logs openclaw_ogmem -f

# 查看 openGauss 日志
docker logs opengauss -f

# 查看 Postgres 日志
docker logs ogmem-postgres -f
```

### 重启服务

```bash
# 重启所有服务
docker restart ogmem openclaw_ogmem

# 重启单个服务
docker restart ogmem
```

### 进入容器

```bash
# 进入 oGMemory 容器
docker exec -it ogmem bash

# 进入 openGauss 容器
docker exec -it opengauss bash

# 进入 Postgres 容器
docker exec -it ogmem-postgres psql -U ogmem -d ogmem
```

### 健康检查

```bash
# 检查 oGMemory 服务
curl http://<OPENCLAW_HOST_IP>:8090/api/v1/health

# 检查 OpenClaw 服务
curl http://<OPENCLAW_HOST_IP>:18789/health

# 检查 Postgres 连通性（容器内 psql）
docker exec ogmem-postgres psql -U ogmem -d ogmem -c "SELECT 1;"
```

### 查看 OpenClaw 当前配置及备份

```bash
ls -lt <OPENCLAW_HOME_DIR>/openclaw.json*
```

---

## 访问服务

部署成功后，通过以下地址访问：

### 单租户模式

| 服务 | 地址 | 说明 |
|------|------|------|
| OpenClaw Web | http://\<OPENCLAW_HOST_IP\>:18789 | Web 界面入口 |
| oGMemory API | http://\<OPENCLAW_HOST_IP\>:8090 | HTTP API |
| openGauss (VECTOR_DB) | \<VECTOR_DB_HOST\>:15432 | 向量数据库连接 |
| Postgres (STORAGE_DB) | \<OPENCLAW_HOST_IP\>:54321 | SQL 存储后端连接 |

### 多租户模式

每个租户拥有独立的 OpenClaw Gateway 端口：

| 租户 | OpenClaw Web | 绑定账户 |
|------|-------------|---------|
| 租户 A | http://\<OPENCLAW_HOST_IP\>:18789 | acct-company-a |
| 租户 B | http://\<OPENCLAW_HOST_IP\>:18790 | acct-company-b |

### 首次访问

1. 进入 openclaw 容器内
2. 输入 `openclaw tui` 进行交互对话

---

## 文件说明

### Reasoning-model ingestion and repeatable experiments

`llm.max_tokens` includes reasoning tokens on compatible reasoning endpoints.
An output ending with `finish_reason=length` is an ingestion error, not an
empty extraction. The provider now fails immediately rather than parsing
partial tool arguments or repeatedly retrying the same exhausted budget.
Set `llm.request_timeout_seconds` alongside the output budget when using a
slow reasoning model. The defaults remain 4096 tokens / 120 seconds; the
tested GLM deployment needed 16384 / 300 for a complete LoCoMo session.
Each extraction may make multiple requests, so the caller's pipeline wait
deadline must exceed a single request timeout. Validate full sessions before
starting a benchmark.

For isolated benchmark reruns, set `OGMEM_ACCOUNT_ID`, `OGMEM_USER_ID` and
`OGMEM_AGENT_ID` in `.env` when the gateway does not supply identity.
Explicit plugin configuration and trusted host identity retain precedence.
Align `identity` in `ogmemory.yaml` for direct API requests that omit identity.
If `OG_AUTH_ACCOUNT_ID` is configured, align it as well: it supplies the
gateway's `X-Account-ID` header independently of the request-body identity.
Also select a **fresh** `OPENCLAW_HOME_DIR`: changing only a runner label or
backend identity does not clear cached gateway session UUIDs/transcripts.
Retain the previous state directory/volume and database contents for rollback.

If your client sends `X-OpenClaw-Agent-ID`, register that ID using
`OPENCLAW_AGENT_IDS` (comma-separated) in `.env`. A fresh gateway only knows
`main`; the memory identity setting `OGMEM_AGENT_ID` does not register gateway
agents. An unregistered ID returns HTTP 400 `Unknown agent` before ingestion.
Registration adds missing agents without replacing existing agent settings.
The shipped plugin template also enables
`hooks.allowConversationAccess` for the bundled oGMemory integration; existing
configs are preserved and need an explicit opt-in to enable these hooks.

```
deploy/
├── README.md                # 本文档
├── install.sh               # 一键安装脚本（下载部署文件）
├── deploy.sh                # 部署执行脚本
├── .env               # OpenClaw + 基础设施配置
├── ogmemory.example.yaml    # oGMemory 配置模板（复制为 ogmemory.yaml 后编辑）
└── ogmemory.yaml            # oGMemory 实际配置（用户创建，不纳入版本控制）
```

## Agent 工具与经验记忆

部署模板默认关闭 Agent 记忆抽取。需要启用时，按 [Agent 记忆部署说明](agent-memory.md) 修改挂载 YAML、重启服务并验证配置和 Trace 抽取结果。
