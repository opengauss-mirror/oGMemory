# 记忆导入导出操作说明

通过正常服务的迁移 API 或 `ogmem migration` 迁移持久化记忆。源端仅在固定内容副本期间暂停持久化，读取、模型计算和索引投影继续；超时放弃副本并恢复写入，副本固定后后台打包，不补齐后续变化；目标可为空或已有数据，导入期间暂停业务读写，先备份再合并。数据范围见[设计说明](/docs/architecture/memory-migration.md)。

仅支持相同内容存储类型迁移：内容存储类型必须一致，允许 SQL 数据库类型、向量后端、模型和维度变化。缺少来源环境描述的旧包依据清单中的内容存储类型检查，其他差异显示为未知。

## 配置

- 使用配置中的 `auth.root_api_key` 调用迁移 API；账号 Admin Key 不能执行实例迁移。迁移范围为全部账号，不受页面当前账号选择影响。
- 同一实例的 HTTP、MCP、独立索引进程（`ogmem start index`）和后台服务必须位于同一主机，使用同一个持久化 `storage.migration_workspace`（环境变量 `OGMEM_MIGRATION_WORKSPACE`，默认 `.ogmemory/migration`）。应配置绝对路径，并为迁移包、临时数据及完整备份预留空间。不能协调的跨主机进程和绕过服务的直接访问必须停止。
- 目标使用自己的模型和存储配置。默认检索模式下，模型实际维度须与向量索引一致；已有向量还须具有一致的模型身份。`memory` 向量后端不能作为持久化迁移目标。
- 使用 AGFS HTTP 存储时，AGFS 服务端须同步升级到本版本，以正确返回已完成的创建冲突；仅更新 Python 服务不能修复旧 AGFS 的错误响应。
- SQL 全实例迁移使用可绕过租户 RLS 的管理员：PostgreSQL superuser 或 openGauss SYSADMIN；openGauss 数据库使用 `DBCOMPATIBILITY 'PG'`。
- 迁移保留账号、用户及 Agent 标识，不复制凭证和实例角色；新增归属的访问权限须在目标配置。

## 命令行

CLI 使用正常服务的迁移 API、权限和任务。先启动业务服务，将 ROOT 密钥放入 `OGMEM_ROOT_API_KEY` 环境变量；也可通过 `--api-key-env` 指定已有凭据变量。`--base-url` 指向源或目标 HTTP 服务，不是数据库或模型地址。

```bash
ogmem migration --base-url http://source:8090 export --output memories.tar.gz
ogmem migration --base-url http://target:8090 preflight memories.tar.gz
ogmem migration --base-url http://target:8090 import --preflight-id <预检任务ID> --yes
ogmem migration --base-url http://target:8090 status <任务ID>
ogmem migration --base-url http://target:8090 recover <中断的导入任务ID> --yes
```

预检输出包含新增、重复及冲突统计；导入使用该预检绑定的包。`--yes` 确认目标维护与数据变更。默认等待任务结束，进度和任务 ID 写入标准错误，JSON 结果写入标准输出。`--no-wait` 仅提交；`status <任务ID> --wait` 可继续等待，`status <导出任务ID> --output <文件>` 可重新下载。下载校验大小与摘要，不覆盖已有文件。

等待超时或中断不会取消服务端任务；响应丢失时先查询已打印的任务 ID，必要时使用 `--request-id <原ID>` 重试提交，不能直接生成新请求重复导入。退出码：成功或提交成功为 0，失败为 1，查询到运行中任务为 2，用户中断等待为 130。原有启动独立服务的 `--workspace`、`--maintenance-confirmed` 参数不再使用。

## 本地附件

源、目标分别配置 `storage.migration_asset_root` 或 `OGMEM_MIGRATION_ASSET_ROOT`。仅读取该目录内由 `source_path`、`blob_path`、`media_path`、`image_path`、`video_path`、`audio_path` 声明的普通文件。附件按摘要去重，导入后更新声明路径；正文中的历史路径不改写。目标附件目录应专用于实例持久化数据，并可由业务服务访问。



## 数据限制

AGFS 备份拒绝符号链接和特殊文件。

不接续运行中会话，不迁移任务队列、锁、请求回执或工作区同步状态。场景状态、异构图、技能候选和未支持的外部 trace/tool 证据依赖明确拒绝。SQL 关系端点须与存储 URI 精确一致；启用 SQL 图索引时端点必须是节点。

仅接受工具生成的 gzip／USTAR 包。上传及解压上限为 10 GiB，单附件或 JSONL 数据集不超过 8 GiB−1 字节，单条 JSONL 不超过 16 MiB。来源描述不会覆盖目标配置；默认检索模式下，新增及合并内容使用目标模型重建向量，检索排序可能变化。

目标配置 `retrieval.mode=agfs` 时，无需 Embedding 或向量库；导入及恢复后，全文缓存由正常检索重新建立。源目标须使用相同的向量后端。

启用共享空间的实例及包含共享空间的迁移包不在当前支持范围内，预检会明确拒绝。
