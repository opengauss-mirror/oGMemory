# SWE-bench Lite 测试指导

本目录提供 baseline 与 oGMemory SQL 两组的代码修复测试参考实现。使用同一题集、模型接口、模型、代码起点和官方评分方法，比较修复结果、解题 Token 和工具调用。单题与完整 300 题使用同一套脚本，仅配置范围不同。

## 1. 范围与执行原理

SWE-bench Lite test 集包含 300 道 Python 项目问题，主要涉及功能错误、边界条件、配置行为和兼容性。每题给出问题描述及起始提交，模型实际修改文件形成补丁，随后由官方测试判定是否修复。

| 方案 | 工具与存储 |
| --- | --- |
| baseline | Claude Code 原生 Bash、Read、Edit、Write、Glob、Grep，不加载 oGMemory |
| sql | 使用与 baseline 相同的编程助手和模型，额外启用本仓库 Claude 插件及代码 MCP 工具。每次执行自动创建独立的 openGauss 数据库容器，保存 oGMemory 的用户、Agent、资源及工作区注册信息，执行结束后清理容器。 |

SQL 配置使用当前 dev 的 `storage.backend=sql`、`storage.dialect=opengauss`、`features.code_retrieval.mode=candidate_local` 和工作区注册接口。设置 `mcp.default_toolset=code`，仅提供代码任务相关的 MCP 工具；同时设置 `mcp.enable_code_edit_tools=true`，允许通过 `ogm_edit_file` 修改源代码。openGauss 保存 oGMemory 的资源、身份及工作区注册等服务状态；题集、补丁、Token 统计和评分报告仍保存在文件中。代码搜索在请求内生成候选代码向量，不依赖数据库中的持久化代码向量索引。脚本自动部署 openGauss，不部署 AGFS 或另一套 LLM 服务；代码检索需要用户提供真实 Embedding API。通用会话写入关闭，未使用的记忆提取 LLM 配置为 mock，解题模型仍是真实模型。

因此，本参考实现测量代码检索、编辑与可选输出过滤的组合效果，不是 openGauss 向量索引性能或通用长期记忆效果。AGFS 暂不提供方案选项。

MCP 工具描述指导模型用 `ogm_search_code` 定位代码、复用片段和 `edit_target.locator`，用文件读取工具补读缺失行；编辑时保留缩进和换行，完成后检查 diff，失败时按反馈修正参数。这些指导通过工具列表提供，无需额外配置 `CLAUDE.md`，不限制搜索次数或替代任务验证。对比版本时应记录工具描述变化，避免重复添加指导。

代码搜索与编辑遵循以下约定：

- `scope` 仅接受 `paths`、`glob_patterns`、`grep_terms`。搜索路径为仓库相对路径：定位符 `ROOT/src/module.py` 对应 `paths=["src/module.py"]`；编辑则复制完整定位符，不附加行号或符号说明。省略 `paths` 可搜索已注册工作区。
- `grep_terms` 是不区分大小写的字面 AND：所有词须出现在同一文件的完整源码中，可位于注释或跨越分块边界；每个返回块仍须包含至少一个完整过滤词。仅命中未分块的注释时可能无结果。路径、元数据和摘要不计入，也不跨文件匹配。它不支持正则或备选词；语义描述写入 `query`。
- `coverage` 中的 `truncated` 表示省略了源码，`implementation_complete` 在边界可靠时说明 Python 函数实现是否完整，`read_lines` 给出可补读的闭区间。符号或实现查询优先返回实现，即使同时提到文档；纯文档查询和字面过滤保留相应命中。跨块、超长行和响应截断均保守标记完整性并提示补读；实现完整时 `read_lines` 可为空，即使声明或文档被省略。无需每次读取所有提示行。
- `related_files` 每次响应最多提供三个去重的关联路径，带工作区前缀，不展开源码。先读取或搜索关联文件再编辑；用于 `scope.paths` 时去掉前缀。
- 代码工具集的 `ogm_read_memory` 用原始 URI 恢复被过滤的工具输出，不读取源码文件。参数仅有 `uri`、`max_chars`（1–50000）和固定为 `true` 的 `recover_original`；通用记忆工具集不变。

两种检索模式和融合方式均最多优先保留两个候选：匹配明确标识符的符号命中，以及同时位于符号或 BM25 前三名的语义首位命中。去重后其余候选保持融合顺序，禁用或零权重通道不参与保护；数量、长度限制和原始融合分数不变，分数不表示保护后的输出顺序。明确标识符包括下划线名称、非首位含大写字母的名称、反引号名称、点号限定名称及单独查询的名称；普通单词仅首字母大写不触发保护。BM25 在查询和索引两侧均保留完整单词并拆分驼峰和缩写边界，不任意拆分小写子串，不改变权重或预算。

`full` 每文件保存一份完整源码快照，不作为额外向量索引文本，源码变更后须刷新；`candidate_local` 使用本次请求读取的源码，不持久化该快照。`full` 使用 `grep_terms` 时，快照缺失返回 `source_snapshot_required`，同文件快照内容冲突返回 `source_snapshot_conflict`，均需管理员逐文件刷新。未使用 `grep_terms` 时不做此检查；本测试脚本使用 `candidate_local`，无需迁移旧索引。

管理员可通过现有认证调用 `POST /api/v1/resources/code/refresh`，正文如 `{"root_slug":"ROOT","path":"src/module.py"}`，分别填写已注册工作区名称和相对文件路径。普通增量同步可能跳过未修改文件；`bootstrap` 仅用于首次导入，不清理旧块，不能代替刷新。零命中不代表仓库中不存在目标，应检查路径、快照和过滤条件，必要时减少过滤词或缩小范围重试。

流程为：

```text
固定题集 → 下载起始源码与官方镜像
→ 创建每题、每方案独立工作目录和测试容器
→ 在挂载后的源码目录执行官方项目安装，恢复编译扩展等运行文件
→ baseline 配置原生工具 / SQL 启动独立数据库、HTTP 和 MCP
→ Claude Code 在宿主机运行，调用工具修改宿主机工作目录中的源码；按需通过 docker exec 在挂载该目录的题目容器中运行测试
→ 测试脚本在宿主机提取并保存相对 base_commit 的补丁（包含新文件）
→ 另起官方容器评分 → 保存结果 → 清理本次临时服务
```

解题模型在宿主机工作目录操作代码；本地测试在官方镜像已有的 `testbed` Conda 环境执行。工作目录挂载至容器 /testbed，任务提示中提供 docker exec 命令。最终评分使用另一干净容器，不沿用模型可能改动过的测试环境。

挂载会遮住镜像原有源码目录中的编译产物，因此脚本在每次解题前执行 SWE-bench 固定版本为该项目和版本定义的预安装、安装命令；安装失败或超时则记录 error，不启动模型。命令和输出分别保存在 setup.sh、setup.log。安装完成后恢复被安装步骤改动的受版本控制文件，保留构建产物；新增的安装文件逐项加入本工作区的 Git 排除规则并记录到 setup.json，避免混入模型补丁。该步骤可能需要联网下载构建依赖，默认超时 1800 秒，可通过 evaluation.setup_timeout_seconds 调整。

这与宿主机另建 Conda 的流程不同：减少依赖安装差异，但容器启动和 ARM 模拟耗时不能直接与其他流程比较。模型自行执行系统 python 的结果不能代替官方评分。

## 2. 文件与环境要求

```text
benchmarks/swebench/
├── README.md
├── config.example.yaml
├── requirements.txt
├── prepare.py     # 题集、源码、镜像与 testbed 环境检查
├── run.py         # 预检查、解题、官方评分、续跑、独立重评
└── report.py      # 逐题明细、统计和共同成功题对比
```

完整执行要求 Linux 或 WSL2、Python 3.11/3.12、Git、可访问的本机 Docker Engine、Claude Code CLI，以及模型和 Embedding 服务。Python 运行环境与 Docker 必须看到相同宿主机路径；不支持通过远程 DOCKER_HOST 挂载本地工作目录。Windows 原生可查看文档及运行离线测试，实际解题请在 WSL2 内安装依赖并运行。

推荐 Python 3.12。在仓库根目录安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r benchmarks/swebench/requirements.txt
# SQL 方案需要以下 NLTK 数据；仅运行 baseline 时可跳过。
python -m nltk.downloader punkt punkt_tab stopwords
git rev-parse HEAD
claude --version
docker info
```

产品依赖由当前仓库 pyproject.toml 决定，测试工具固定 swebench 4.1.0。保持虚拟环境已激活，插件使用的 python3 应来自同一环境。CLI 需支持 stream-json、--setting-sources、--strict-mcp-config、--plugin-dir 和 --tools 等参数，预检查会核对关键选项。

脚本使用非交互权限模式自动执行模型工具。请使用专门的普通用户或测试机，保证其文件和 Docker 权限符合实验范围；不以 root 运行 Claude。工具白名单不等于操作系统沙箱，模型仍具有 Bash 权限。原始官方题目、评分补丁保存在工作目录的输入区，不作为模型提示输入；这是输入隔离，不是针对恶意 Agent 的安全隔离。

镜像、每题源码和会话日志会占用大量空间。先验证单题并估算全量容量，再准备 300 题；脚本不自动清理其他项目的数据或容器。

### ARM 主机

本实现统一选择官方 x86_64 题目镜像。ARM Linux 需要预先安装并注册支持 x86_64 的 QEMU 用户态解释器和 binfmt_misc。prepare.py 会启动镜像并检查 testbed 的 Python，出现 exec format error 时先修复模拟环境。官方评分同样使用 x86_64，不能只更换镜像名而沿用另一架构的结果。

有 Docker 管理权限时，可在测试主机注册 amd64 解释器，再检查注册状态：

```bash
docker run --privileged --rm tonistiigi/binfmt --install amd64
cat /proc/sys/fs/binfmt_misc/qemu-x86_64
```

状态应为 enabled，flags 应包含 F。该操作修改宿主机的 binfmt 注册；机器重启后需重新检查。注册成功后仍须运行 prepare.py 验证实际题目镜像能够启动。

ARM 上首次编译大型项目可能接近默认 1800 秒安装超时。可在配置中将 `evaluation.setup_timeout_seconds` 提高到 3600；这是项目安装时限，与模型解题时限和官方评分时限分别配置。

## 3. 配置

从仓库根目录进入本目录：

```bash
cd benchmarks/swebench
cp config.example.yaml config.yaml
```

SQL 测试要求当前代码仓库中不存在 `config/.env`。产品启动时会自动加载该文件，可能引入未记录的检索参数或模型覆盖设置；预检查及每次启动 SQL 服务前都会检查，发现文件则停止，不读取或修改其内容。已有开发配置时，请使用同一提交的干净代码副本运行测试，将测试参数放入本目录配置文件并按下文提供环境变量。仅 baseline 和独立重评不受此限制。测试期间不要创建或修改仓库的 `config/.env`。

config.yaml 和默认 work/ 已在仓库忽略规则中。路径相对于配置文件解析；其他位置的 work_dir 会自动写入忽略文件。不要将真实密钥写入 YAML。

在同一终端通过自己的凭据管理方式设置：

```bash
export ANTHROPIC_BASE_URL='https://模型服务地址'
# ANTHROPIC_AUTH_TOKEN：设置为有权访问所选模型的凭据。
export SWE_EMBEDDING_BASE_URL='https://向量服务地址/v1'
export SWE_EMBEDDING_MODEL='实际向量模型名称'
# SWE_EMBEDDING_API_KEY：设置为向量服务凭据。
```

将 config.yaml 的 `model.name: YOUR_MODEL_NAME` 改为模型服务实际支持的解题模型名称；保留占位值会在预检查时被拒绝，两组共用这一配置。模型地址必须兼容 Claude Code 使用的 Messages 协议，支持结构化工具调用。能返回普通文字不代表工具调用正常。本目录不内置特定供应商的协议转换器；若由外部兼容网关转换，两组必须使用同一网关和转换路径，并记录其版本。

| 配置 | 用途 |
| --- | --- |
| work_dir | 缓存、私有运行配置和结果位置 |
| dataset / dataset_revision | 题集与版本；首次下载保存完整快照和校验值 |
| instances | 题目 ID 列表，或 all |
| arms | baseline、sql 中的一个或两个 |
| concurrency | 同时执行的任务数量，首次使用 1 |
| model | 名称、CLI、凭据环境变量名、最大轮次和超时 |
| evaluation | 官方镜像命名空间、标签、项目安装超时 setup_timeout_seconds、评分超时 timeout_seconds |
| ogmemory | openGauss 镜像 opengauss_image、启动超时、Embedding 配置、输出过滤开关 |

示例使用 `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-lite:7.0.0lts`。数据库镜像须支持执行机架构；与统一使用 amd64 的题目镜像分别选择。替换镜像时须保留 `GS_PASSWORD`、`GS_DB`、`GS_USERNAME` 初始化接口，以及容器中的 `omm` 用户和 `gsql` 命令；初始化结束后主进程应为 `gaussdb`。脚本等待初始化完成及 TCP 连接成功后，以 `omm` 创建 `swebench` 数据库，指定 `OWNER swebench DBCOMPATIBILITY 'PG'`，再校验 `version()` 确实为 openGauss、`SHOW sql_compatibility` 为 PG。PG 是 openGauss 的 SQL 兼容模式，并不表示运行 PostgreSQL；初始化命令中的 `postgres` 只是 openGauss 自带的维护数据库名。无需手工建库。

数据库使用每次随机生成的复杂密码，凭据写入权限为 0600 的 `private/opengauss.env` 和服务配置；数据库端口只映射到宿主机 127.0.0.1 的随机端口。默认启动超时 300 秒，可用 `ogmemory.startup_timeout_seconds` 调整。旧配置的 `postgres_image` 应替换为 `opengauss_image`，并使用新的 work_dir 重新执行，不能续跑旧 PostgreSQL 实验。

更换题集版本时使用新的 work_dir。main 是下载入口，真正复现依赖保存的数据快照和校验值。prepare.py 复用已冻结数据，不静默刷新。保留已有镜像时记录实际镜像 ID。每个任务启动解题容器前、提交官方评分前，均核对批次记录的评分版本和镜像 ID；解题容器按固定镜像 ID 启动。官方评测子进程在准备好测试配置后再次核对镜像，若发现镜像标签已指向其他 ID，则记录 error 并停止该任务，不纳入有效评分。

## 4. 准备与单题执行

以下步骤在本目录、激活虚拟环境的终端执行。

### 4.1 准备题目

```bash
python prepare.py --config config.yaml
```

脚本下载 test split，选择配置中的题目，获取 base_commit 与环境安装提交，拉取缺失的官方 amd64 镜像，并验证镜像中的 testbed 环境。成功每题输出 READY。只需先下载题集时可以加 --dataset-only；该选项不准备解题所需的源码和镜像。

READY 仅表示镜像中 Conda Python 能启动；挂载后的项目安装在 run.py 中执行，不能据此认定项目测试已经通过。

默认示例题为 pylint-dev__pylint-5859。准备记录保存在 work/inputs/，不包含模型调用。

执行机不能访问 Hugging Face 时，可在能联网的机器上使用相同 dataset、dataset_revision 配置运行 `python prepare.py --config config.yaml --dataset-only`，将生成的 `work/inputs/dataset.json` 和 `dataset-source.json` 一起复制到执行机相应的 inputs/ 目录，再执行正常准备命令。脚本会检查来源和内容校验值后复用快照；不要手工裁剪或修改题集文件。执行机仍需能获取 GitHub 源码和官方容器镜像。

### 4.2 预检查

```bash
python run.py --config config.yaml --check
```

检查版本、模型环境变量、Claude 参数、源码准备记录和镜像 ID。SQL 还检查服务依赖及 Embedding 配置，必要时拉取 openGauss 镜像。此命令不启动数据库、不调用解题模型，也不验证实际模型凭据或 Embedding 返回是否正确；实际数据库启动和真实工具调用需由单题执行确认。

### 4.3 执行两组

```bash
python run.py --config config.yaml
```

脚本打印本次 Batch 路径。默认顺序执行同一道题的 baseline 和 SQL，各用独立工作目录、测试容器和会话。SQL 自动启动数据库、本仓库源码的 HTTP 服务和 Streamable HTTP MCP，创建用户及 Agent，注册本题工作区；正常结束或捕获到异常后清理其临时服务。无需手工常驻数据库或预先安装用户级插件。

若需要断开终端运行，可将上述命令放在 tmux 中，或用 nohup 重定向到自行指定的日志。不要因为暂时没有评分再次启动相同运行。

SQL 初始化必须包含 og-memory 插件和已连接的 ogmemory MCP，并至少产生一条成功的 oGMemory 工具结果。若接口只输出文字、MCP 未连接或只有失败调用，记录为异常，不当成正常接入结果。

### 4.4 查看结果

```bash
# 将下面路径换成启动时打印的 Batch 路径。
python report.py work/runs/实际运行目录
```

生成 tasks.csv、summary.csv、summary.json 和 report.md。report.py 不调用模型，可在运行中查看，但 pending/running 任务列为无判定，不当成已完成全量。

```text
work/
├── inputs/                       # 固定题集、原始题目、源码缓存、镜像 ID
└── runs/<运行ID>/
    ├── manifest.json             # 配置、版本、源码指纹、模型地址等
    ├── dataset.json              # 本次所选题目（仅供执行器与评分）
    ├── jobs/<方案>--<题目ID>/attempt-0001/
    │   ├── workspace/            # 模型实际修改的源码
    │   ├── prompt.txt、session.jsonl、agent.log
    │   ├── setup.sh、setup.log、setup.json # 挂载后项目安装与生成文件记录
    │   ├── produced.patch
    │   ├── result.json
    │   ├── database.json         # SQL 方案的 openGauss 版本、PG 模式和镜像 ID
    │   ├── private/              # 临时服务密钥、配置、独立 Claude 目录
    │   └── evaluation/           # 提交补丁、官方日志、validation.json
    └── report.md、tasks.csv、summary.csv、summary.json
```

SQL 数据库是本次执行的临时数据库，结束会删除本次数据库容器及其卷；这只针对脚本生成的唯一容器名。结果、补丁和日志保留。不要公开 private/；会话和日志也应按实际内容检查后再分享。

## 5. 连续执行完整 Lite 300 题

先完成同一配置下的两组单题验证，再将配置改为：

```yaml
instances: all
arms: [baseline, sql]
concurrency: 1
```

使用相同入口：

```bash
python prepare.py --config config.yaml
python run.py --config config.yaml --check
python run.py --config config.yaml
```

300 题 × 两组共 600 个任务，由程序连续调度，不需要人工拆分题单。增加 concurrency 会增加临时数据库、服务和题目容器数量，按本机资源与 API 限额调整。每题每方案使用独立 attempt，不共享记忆或工作目录。

运行中再次执行 report.py 查看进度。运行结束不代表全部修复成功，需核对计划数、有效评分数、异常数和用量完整数。

## 6. 续跑与独立重评

继续未启动任务：

```bash
python run.py --config config.yaml --resume work/runs/实际运行目录
```

为 error 或 empty_patch 任务建立新 attempt：

```bash
python run.py --config config.yaml --resume work/runs/实际运行目录 --retry-errors
```

已有 completed 任务（包括官方判定未修复）不自动重复。续跑检查配置、题集、模型地址、CLI 版本、产品（包括 `cli/` 中的 MCP 启动代码）和脚本源码指纹、镜像及 Embedding 标识；发生变化时创建新运行，不把不同条件混在一起。

异常退出可能遗留 running 状态。先确认原进程已停止，按该 attempt 的资源记录清理其容器与服务；确认后把其 result.json 的 status 改为 error，并按上述命令重试。脚本不会自动接管仍标记运行的任务或清理不属于它的资源。SQL 服务会随新 attempt 重新创建，不依赖上一次常驻进程。

仅重新评分已有补丁：

```bash
python run.py --evaluate work/runs/实际运行目录/jobs/方案--题目ID/attempt-0001
```

新评分保存在该 attempt 的 regrade-目录中，不调用模型，不自动替换原结果。主表默认只统计原 evaluation；采用重评结论时必须另行说明关联和原因，不能把重评解释为新的模型解题。

即使官方镜像已缓存，SWE-bench harness 在准备评分时仍可能从 GitHub 读取该题环境提交中的依赖文件，因此评分阶段也需要网络访问。网络阻塞导致评分失败时，先确认原评分进程已经结束，再对同一补丁执行上述独立重评命令。另行记录原 attempt、重评目录、补丁一致性和采用重评的原因；主表中的原失败记录保持不变。

独立重评要求 Linux/WSL2、swebench==4.1.0 及本地 Docker，不需要模型或 Embedding 凭据。脚本会核对原 manifest.json 中的评分版本、镜像名称和镜像 ID；版本变化、镜像缺失或 ID 不一致时停止，需先恢复原版本和原镜像。每次评分在 grading-environment.json 中记录实际版本和镜像 ID。

模型超时、非正常退出或接入校验失败时，脚本仍尝试将工作区修改保存为 produced.patch，原任务保持 error，不自动评分。可使用上述 --evaluate 命令诊断补丁，但不能将异常解题当作正常接入成绩。若补丁提取失败，result.json 的 patch_error 会说明原因；若尚未形成有效工作区，则没有可重评补丁。

## 7. 官方评分与统计口径

评分调用固定版本 SWE-bench 的 make_test_spec、run_instance；每次评分使用新的日志 ID，避免复用旧报告。官方 FAIL_TO_PASS 验证目标缺陷，PASS_TO_PASS 验证原有行为。validation.json 仅在官方评测完成且实例报告包含布尔值 resolved 时记为 completed。

| 结果 | 含义 |
| --- | --- |
| completed + resolved=true | 官方判定修复成功 |
| completed + resolved=false | 官方判定修复未通过 |
| empty_patch | 没有实际修改，无实例级官方判定 |
| error / running / pending | 异常、执行中或未启动，无有效判定 |

脚本从不可变 base_commit 提取补丁，并包含新文件，即使模型自行提交了修改也不会丢失差异。回答中的 diff 文本不直接作为补丁执行。

每个“题目 ID + 方案”只取最后记录的 attempt。报告中的成功率为成功数 / 计划数；完整 Lite 每组分母是 300。有效判定正确率为成功数 / 有效判定数，后者为 0 时不计算。两种分母在 summary.json 中分别保存。

Token 来自 Claude 最终 result 的会话累计 usage，合计输入、缓存创建输入、缓存读取输入和输出，缓存只计一次。字段缺失、日志损坏或终态缺失时为 PARTIAL，不补零。工具调用从 assistant 事件按 tool_use.id 去重，失败调用也计入；一次 Bash 内部执行多个测试仍是一条工具调用。

每组平均数要求该组全部记录完整，否则为空。比较效率时另取两组均修复成功、Token 和调用均完整的共同题目集合，列出样本数和（baseline − SQL）/ baseline 的下降比例；分母为 0 时不计算。各组自己答对的不同题目集合不直接互相比平均数。

Token 不等同于账单费用，不包含 Embedding、服务运行或官方评测成本。早期 attempt 和额外诊断用量也不计入最终 attempt 主表；实际总成本应另计。

## 8. 故障排查与验证状态

| 现象 | 检查 |
| --- | --- |
| 模型只输出文字，工具调用为 0 | 网关是否支持结构化工具调用；核对 session.jsonl，不能把文字 diff 当实际修改 |
| 解题前项目安装失败或超时 | setup.sh、setup.log、构建依赖网络；必要时调整 evaluation.setup_timeout_seconds |
| SQL 启动失败 | http.log、mcp.log、数据库镜像、Embedding 设置和本机端口 |
| MCP 连接失败 | 当前源码及 MCP 依赖版本、Streamable HTTP、临时鉴权；不回退成 baseline |
| MCP 编辑失败 | 复制 `edit_target.locator` 原值，按错误反馈核对定位符、精确文本和匹配范围后重试；未知内部错误查看 mcp.log。失败调用仍计入统计 |
| MCP 调用成功但搜索显示 partial_failure | 查看返回状态和候选检索诊断；连接成功、工具无异常不代表所有检索路径正常。分别核对候选文件读取、Embedding 和检索路径状态 |
| 镜像启动 exec format error | Docker 架构与 QEMU/binfmt 注册 |
| 本地缺少 Python 模块 | 是否使用题目容器中的 testbed 环境 |
| 官方评分缺失 | evaluation/harness.log 及 logs/run_evaluation；不要按 false 补齐 |
| 独立重评拒绝执行 | 核对原 manifest.json 中的 SWE-bench 版本和镜像 ID，恢复相同环境后重试 |
| 用量不完整 | 终态事件和字段是否完整，不计算虚假的零消耗 |

本目录提供测试流程，不修改 oGMemory 产品实现。模型是否修复问题，以实际运行的官方评分为准；接入 SQL 不保证降低 Token 或提高正确率。先使用同一配置完成两组单题验证，再运行完整题集，不将单题结果外推为全量结论。

离线回归覆盖补丁提取、新文件、安装产物排除、安装超时、异常解题保留补丁、重评环境校验、最终 attempt 选择、缺失用量、工具调用去重、SQL 本地配置隔离和 CLI 源码变更检测。离线检查不能代替真实模型解题和官方评分；可选 SQL 接口检查也不包含完整模型调用。

维护者验证命令（仓库根目录）：

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/unit/benchmarks/test_swebench.py -q
python -m ruff check benchmarks/swebench tests/unit/benchmarks/test_swebench.py
# 可选：启动并清理一个临时 openGauss 容器，验证当前产品接口；不调用付费模型。
SWE_SQL_INTEGRATION=1 python -m pytest tests/unit/benchmarks/test_swebench.py -q
```

官方参考：[题集](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)、[评测指导](https://www.swebench.com/SWE-bench/guides/evaluation/)、[固定版本评分实现](https://github.com/SWE-bench/SWE-bench/blob/v4.1.0/swebench/harness/run_evaluation.py)。服务配置字段以本仓库 [配置参考](../../config/ogmem.reference.yaml) 为准；默认 Compose 的 AGFS 部署不等同于这里的 SQL 测试部署。
