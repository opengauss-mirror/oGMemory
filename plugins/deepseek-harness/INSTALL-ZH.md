# DeepSeek Harness 接入 oGMemory

本文说明如何让用户的 DSH 连接一个已经运行的 oGMemory HTTP 服务，以及如何在
DSH 页面确认接入成功。

开始前请确认：

- DSH `0.1.5-rc.2` 已安装，并已配置可用的模型提供方。
- 已知可访问的 oGMemory HTTP 服务地址。

插件 `0.1.1` 支持 DSH CLI `0.1.5-rc.2`、核心包 `0.1.5-rc.2`
与 Cordis `4.0.2`，提供已完成对话的自动写回和跨会话记忆召回。

## 1. 确认 oGMemory 可访问

将下面地址替换为用户实际使用的 oGMemory 地址：

```powershell
curl.exe http://127.0.0.1:8090/api/v1/health
```

健康检查成功后继续安装插件。

## 2. 安装插件

在 DSH 所在环境安装 oGMemory 插件包：

```powershell
dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev
```

## 3. 配置连接并启动 DSH

```powershell
$env:OGMEM_URL='http://127.0.0.1:8090'
$env:OGMEM_ACCOUNT_ID='acct-demo'
$env:OGMEM_USER_ID='user-dsh'
$env:OGMEM_AGENT_ID='deepseek-harness'
dsh --profile web
```

如果 oGMemory 开启了 API 认证，再设置：

```powershell
$env:OGMEM_DSH_API_KEY='<oGMemory API key>'
```

`OGMEM_DSH_API_KEY` 是访问 oGMemory HTTP API 的密钥，不是模型提供方密钥。

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
