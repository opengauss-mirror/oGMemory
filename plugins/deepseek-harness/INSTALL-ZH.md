# DeepSeek Harness 接入 oGMemory

本文使用 `web` profile 连接已经运行的 oGMemory HTTP 服务，并在 Web UI 验证跨会话记忆。
在命令行运行一次性任务，请参阅 [Headless 安装和验证步骤](../../docs/integrations/deepseek-harness.md#headless)。

开始前请确认：

- 已安装 DSH `0.1.5-rc.2` 或 `0.2.1-rc.2`，并已配置可用的模型提供方。
- 已知可访问的 oGMemory HTTP 服务地址。

DSH `0.1.5-rc.2` 对应插件 `0.1.1`，DSH `0.2.1-rc.2` 对应插件 `0.2.1`。

## 1. 确认 oGMemory 可访问

将下面地址替换为用户实际使用的 oGMemory 地址：

```powershell
curl.exe http://127.0.0.1:8090/api/v1/health
```

健康检查成功后继续安装插件。

## 2. 安装插件

DSH `0.1.5-rc.2` 可以从仓库根目录安装到 `web` profile。

```powershell
dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev
```

DSH `0.2.1-rc.2` 使用对应的安装包。在仓库中运行以下命令，打包需要 Node.js 和 `tar`。
使用目标 DSH 安装目录中的 `dsh`，将插件安装到 `web` profile。

```sh
cd plugins/deepseek-harness
node scripts/pack.mjs 0.2.1-rc.2
dsh plugin --profile web add ./dist/opengauss-ogmemory-dsh-plugin-0.2.1.tgz
```

运行 `node scripts/pack.mjs` 会在 `dist/` 下生成两个安装包。两个包分别携带各自的 DSH
适配器，并共享其余运行代码。保留安装时使用的安装包路径，供 profile 恢复依赖时使用。

## 3. 配置连接并启动 DSH

```powershell
$env:OGMEM_URL='http://127.0.0.1:8090'
$env:OGMEM_ACCOUNT_ID='acct-demo'
$env:OGMEM_USER_ID='user-dsh'
$env:OGMEM_AGENT_ID='deepseek-harness'
```

如果 oGMemory 开启了 API 认证，再设置：

```powershell
$env:OGMEM_DSH_API_KEY='<oGMemory API key>'
```

将 oGMemory Tenant Key 填入 `OGMEM_DSH_API_KEY` 后，在同一终端启动 Web UI。

```powershell
dsh --profile web
```

## 4. 确认插件已加载

```powershell
dsh --profile web --dump-config
```

输出中应包含：

```text
# == @opengauss/ogmemory-dsh-plugin
id: ogmemory-lifecycle
name: '@opengauss/ogmemory-dsh-plugin'
```

## 5. 在 DSH 页面验证跨会话记忆

1. 打开 DSH 页面并新建会话。
2. 发送：`请记住测试标记 OGMEM-DSH-TEST-001，只回复“已记住”。` 每次测试使用新的标记。
3. 回复后保持 DSH 运行，供插件在后台自动写回、提取和建立索引。
   建议先等待 1–2 分钟，实际处理耗时取决于模型服务和索引速度。
4. 保持相同的 Account ID、User ID 和 Agent ID，新建一个会话，发送：
   `上一个会话要求你记住的测试标记是什么？只回复标记。`
5. 确认返回 `OGMEM-DSH-TEST-001`（或本次使用的标记）。展开回复的过程区，
   查看 `上下文注入 ogmemory-lifecycle`（英文界面为 `Context injection ogmemory-lifecycle`）。
   也可在 `Trajectory` 标签页查看注入的 `<ogmemory-context>` 内容。

后台仍在处理时，继续等待，随后另建新会话重复第 4 步。

同时满足“出现 oGMemory 上下文注入”和“新会话准确返回上一会话标记”，即表示
DSH 已成功对接 oGMemory。

## 6. 卸载插件

```powershell
dsh plugin --profile web remove @opengauss/ogmemory-dsh-plugin
```
