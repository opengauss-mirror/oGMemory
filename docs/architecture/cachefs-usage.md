# CacheFS 使用与验证

CacheFS 默认关闭。适用于由一个 AGFS 实例独占写入的 LocalFS、S3FS、SQLite 版 SQLFS 挂载；多个 Python HTTP 客户端可共享该 AGFS 服务。所有修改必须经过该服务，不能另起 embedded AGFS 或直接修改同一存储。程序拒绝已识别的重叠存储别名。两个 LocalFS 挂载可能通过符号链接或硬链接写入同一对象，共存时该 AGFS 实例的 LocalFS 缓存会停止使用，直至重建实例。MySQL/TiDB 版 SQLFS 保留现有直连用法，启用 CacheFS 前需要验证数据库排序规则。

## 启用

在实际传给 `agfs-server -c` 的配置文件中加入以下内容，同时保留原有 `server` 和 `plugins`：

```yaml
cache:
  provider: redis
  command_timeout_ms: 50
  max_inflight: 32
  params:
    mode: standalone
    endpoints: ["redis://127.0.0.1:6379"]
    pool_size: 32
    connect_timeout_ms: 1000
    # password_env: OGMEM_CACHE_REDIS_PASSWORD

cachefs:
  enabled: true
  namespace: ogmemory-dev
  mounts: [/local]
  max_file_size_bytes: 1048576
  max_dir_entries: 4096
  max_dir_encoded_bytes: 4194304
  file_ttl_seconds: 300
  dir_ttl_seconds: 30
  traversal_mode: backend
  bypass_prefixes: []
```

配置完整示例位于 `config/agfs/config.reference.yaml`。独立 AGFS 读取自己的配置；本次尚未增加 Python `ogmem.yaml` 的 cache 配置自动渲染。未开启 CacheFS 时不会创建缓存连接池。

`namespace` 用于部署隔离，`mounts` 是 AGFS 绝对挂载路径。每次挂载及故障恢复都会生成新 epoch，旧 key 等待 TTL 回收，无需 FLUSHDB。单文件配置上限 16 MiB，目录编码上限 32 MiB / 65536 项，TTL 最大 86400 秒。默认只缓存完整文件 `Read(0,-1)` 与原始 ReadDir；Stat、范围读取和流式读取保持回源。`cached_traversal` 会关闭本地 ripgrep 快速路径，使普通遍历经过包装 FS，尚未实现批量预取。

锁、lease/claim 等协调文件和默认不缓存的暂存/工作状态直接 bypass，其余对象按通用大小和操作条件缓存。自定义 bypass 只能追加，使用 AGFS 绝对路径和分量边界。bypass 对象的修改仍使相关目录失效。可写 stream/handle 存续期间整个挂载 bypass，关闭成功后通过新 epoch 恢复缓存。若关闭报错且写入结果不确定，挂载继续回源，排空资源后可卸载重挂。

S3FS/SQLFS 被包装时自动关闭插件内部缓存。Redis 不可用时返回真实 backend 结果，进入 bypass；每秒有界探测成功后换 epoch 恢复。DynamicProvider 在成功创建实例后适用相同恢复流程，动态库加载/ABI/create 失败则拒绝启动。

显式请求 `fresh=true` 的目录读取直接回源。创建 workspace 注册锁和锁数据库期间，受影响的挂载暂时回源，会话关闭后切换 epoch。SQLite SQLFS 继续使用原有路径匹配规则，遇到匹配范围不确定的递归变更时切换整个挂载的 epoch。

Redis CacheFS 读取使用有长度上限的 `GETRANGE`。若配置 Redis ACL，缓存账号要有专用 key 范围的 `GETRANGE`、`GET`、`SET`、`DEL`、`PING` 权限。缓存值包含文件原文，跨主机连接应使用受信任的 `rediss://` 服务，并按业务信任边界隔离缓存访问权限。namespace 用于键空间区分。

## DynamicProvider

将 `cache` 替换为：

```yaml
cache:
  provider: dynamic
  command_timeout_ms: 50
  max_inflight: 32
  params:
    library: /opt/ogmem/providers/libcache_adapter.so
    endpoint: 127.0.0.1:31501
```

adapter 必须遵循 `cache_runtime/include/cache_provider_v1.h`，通过 `OV_CACHE_PROVIDER_EXPORT_ENTRY` 导出 v1 兼容入口，并实现 create/ping/close/get/set/del。library 之外的 params 原样传给 create。回调必须支持并发，返回内存使用 host allocator；基础 KV 的成功写删需对后续读取可见。首版不调用可选原子程序或列表/集合接口，不提供专门的 Yuanrong/Mooncake Provider。

只加载可信 native adapter；ABI 校验无法隔离任意 C 代码崩溃。adapter 自身不得再嵌入 Go c-shared runtime。调用超时不会强行终止 native 回调；在途槽位和库保持存活直到回调结束，避免卸载仍在执行的代码。

## 构建与 native 使用

以下命令从仓库根目录执行，使用满足依赖要求的 Go 工具链和 C 编译器（已验证 Go 1.25.4）：

```sh
go -C agfs build -o /tmp/agfs-server ./cmd/server
go -C agfs build -buildmode=c-shared -o /tmp/libagfsbinding.so ./cmd/pybinding
```

macOS 可将输出命名为 `libagfsbinding.dylib`。`agfs/go.mod` 使用 `replace ../cache_runtime`，构建时必须包含两个相邻目录；两个主 Dockerfile 已包含该目录。

HTTP pyagfs 调用方不需要加载动态库。native 调用方继续使用同一份 `libagfsbinding`，在任何 mount 前调用 `AGFS_ConfigureCache(config_json)`，JSON 顶层为上述 `cache` 与 `cachefs`。成功返回 NULL，失败返回需用 `AGFS_FreeString` 释放的错误字符串。`AGFS_CacheStats()` 返回同样需释放的 JSON。进程 owner 最后调用 `AGFS_Shutdown()`，返回值约定与 ConfigureCache 相同；不在单个 client 释放时关闭共享 Runtime。

本次没有交付独立 `libogmemory_cache_runtime`，也没有实现 Python 公共 KV 薄绑定。公共 Go Runtime 可直接导入，应用使用 `UseCache` 注入时保留其关闭所有权。

## 观察与回滚

```sh
curl http://127.0.0.1:1833/api/v1/cachefs/stats
```

按 mount 返回 Hits、Misses、Bypasses、Failures、Fills、Coalesced、Epochs。连续读取同一普通小文件应增加 Hits；锁文件应增加 Bypasses；故障时 Failures/Epochs 增加，恢复后可重新命中。这些计数不等同于延迟或性能验收结果。

将 `cachefs.enabled` 改回 false 后重启 AGFS 即回滚。持久化格式不变，不清空共享 Redis；若有其他 Go consumer，共享 Runtime 仍由其应用 owner 管理。

## 回归命令

```sh
go -C cache_runtime test -race ./...
go -C agfs test -race ./pkg/cachefs ./pkg/mountablefs ./pkg/handlers ./pkg/config ./pkg/plugins/localfs ./pkg/plugins/sqlfs ./pkg/plugins/s3fs ./cmd/server ./cmd/pybinding
go -C agfs test ./...
CGO_ENABLED=0 go -C cache_runtime test ./...
python agfs/tests/cachefs_native_smoke.py /tmp/libagfsbinding.so redis://127.0.0.1:6379
```

Go Redis 契约和 HTTP 故障测试使用 miniredis；DynamicProvider 使用本地编译的 C fixture。它们不能代替真实 Redis 拓扑、真实第三方 SDK、S3/MySQL 或生产性能验收。原生 smoke 使用一次性 LocalFS 目录及独立 epoch，可对测试 Redis 执行。

多进程快照测试可在缓存开关两种模式运行：设置 `OGMEM_TEST_AGFS_BINARY` 指向编译好的 server；额外设置 `OGMEM_TEST_CACHE_REDIS_URL` 时，`tests/integration/test_agfs_node_snapshot_reads.py` 的一次性服务会启用 CacheFS。session 恢复、回执、目录分页等现有测试使用 `OGMEM_AGFS_TEST_URL` 指向一次性 AGFS 服务。

## 本次验证记录（2026-11-19）

- macOS arm64：Runtime 及 CacheFS/挂载/HTTP 定向 race 测试通过；Runtime 的 `CGO_ENABLED=0` 测试通过。
- openGauss237（Linux arm64，Go 1.25.4）：Runtime 及相关 AGFS 包的 race 测试通过，AGFS `go test ./...` 通过；server 与 c-shared 原生库构建成功。
- 原生库 smoke：通过同一 `libagfsbinding.so` 配置 CacheFS、挂载、读写二进制/空值、锁文件 bypass 和重复 shutdown；观察到 Hits=3、Bypasses=2、Failures=0。
- Python/HTTP：关闭缓存的旧卷重启及多进程快照测试 2 项通过；开启缓存的快照、回执、session 恢复、窄状态写入、目录分页及并发 Outbox 测试 40 项通过。共享测试服务记录 Hits=396、Bypasses=1867、Failures=0；这只是功能证据，不是性能基准。Redis 服务使用 miniredis。
- 全量 `go test -race ./...` 在本机发现未改动的 `gptfs_test.go:32/82` 对测试变量的并发读写；未在此变更中修改 GPTFS。
- Linux ASan 尝试因环境缺少 `libasan` 链接失败，不能声称通过原生 sanitizer/泄漏检查。真实第三方 adapter、S3/MySQL、容器镜像及 Linux amd64 尚未实机验证。

远端测试副本、工具链与产物位于 `/tmp/ogmemory-cachefs-validation-20261119`；未覆盖服务器上的现有项目目录，测试服务已退出。
