# M3/GPU 验证修复迁移到 dev（2026-10-23）

## 分支约定

- 开发分支：`dev_1023`，后续修改在此分支提交，准备合回 `dev`。
- 本次基线：`origin/dev` 的 `fc13d829`。
- 本地 `dev` 保持与该远端基线一致，不直接堆放本轮功能提交。
- 原分支 `fix_m3_1019` 和备份
  `backup/fix_m3_1019-before-dev-20261023` 都保留在 `2765f9d2`。
- 本次只做本地迁移，不推送、不强制更新远端、不创建合并请求。

拉取远端后确认，`dev` 已通过 `84f8c049` / `30894d27` 合入 anttrail。
共同祖先为 `d139c5fb`。不再重复合入 anttrail，而是按原顺序移植此后的
10 个本地独有提交。

## 提交映射

| 原提交 | dev_1023 提交 | 内容 |
| --- | --- | --- |
| `a3342e92` | `7fd0b561` | M3 恢复及 embedding batch size |
| `b402c671` | `763eea97` | YAML 配置 CV 服务 |
| `2b08fba3` | `bf55d80f` | 本地 Omni 抽取和 M3 验证修复 |
| `6f85f883` | `466f7fe2` | lazy 模型失败不作为零记忆成功 |
| `154f772c` | `396bca94` | 冻结评测记录与评委局限 |
| `c3a0f86e` | `f95406b6` | M3 LLM 请求超时配置 |
| `f80037a7` | `7715067a` | 保留多模态提示、限制重复检索上下文 |
| `b3b31b08` | `deee78be` | 全视频 GPU 实验记录 |
| `29724e58` | `656b2b68` | MiniMax 唯一映射位置参数兼容 |
| `2765f9d2` | `d79d9f4d` | MiniMax 完整视频对照记录 |

原 52 个改动文件中，41 个未与 dev 改动重叠，迁移后逐文件字节一致。
11 个文件两边都有修改，按 dev 的现有接口适配，不以旧文件整体覆盖新实现。

## 冲突及兼容处理

### 抽取流程

保留 dev 新增的 `check_deadline()`、候选校验、版本化读取、事件与来源协调逻辑；
在其上加入外层消息索引范围的 schema 约束。复用 dev 已有 `deepcopy` 导入，
不重复定义。

lazy 抽取保留操作截止时间异常的传播，同时保留本地修复中
`finish_reason=llm_error` 不被误报为空结果成功的行为。

### LLM provider

保留 dev 的流式返回处理、usage 记录、瞬时错误和截止时间约束。
不重新添加另一套返回 `LLMCallError` 的截断检查，而是扩展 dev 的
`_ensure_complete_response`，继续抛出 `LLMOutputTruncatedError`：

- 增加 completion/reasoning token 元数据诊断，不记录私有输入正文；
- 多模态 JSON 与文本回退也执行相同检查；
- 文本回退保留截断、瞬时错误、截止时间和 usage 持久化异常类型；
- 保留调用方多模态 system 指令，不被 schema 格式指令覆盖。

dev 已有 `_is_required_tool_choice_unsupported`，因此只补充针对 vLLM
生成结果数组解析错误的识别，不重复定义已有方法。

首轮测试暴露出自动合并未识别的接口差异：dev 使用 `tool_parameters`，
旧补丁的异常分支却引用 `api_tool_choice`。修正为读取实际发送参数中的
`tool_choice`；没有 tools 的请求不误归类为 required-tool 输出错误。

新增回归覆盖无工具请求、多模态瞬时错误恢复、响应后截止时间传播、
以及截断 token 诊断，避免仅靠“Git 合并成功”判断迁移正确。

## 验证

- provider 专项：164 passed。
- providers、M3、extraction、session、受影响 server 和 deadline 回归：
  2925 passed，36 skipped。
- `tests/integration/m3/test_detect_faces.py`：13 passed。
- M3 pipeline/eval CLI 导入及 `--help` 正常；原 ignored MiniMax 实验配置可加载。
- 8 个私有配置文件迁移前后 SHA-256 不变；仍被 Git 忽略。
- 原 fix 分支及备份未改；本地 dev 等于迁移基线；locomo-test 未改。

完整单元测试最终重跑：**7877 passed，38 skipped，15 warnings，无失败**。
警告为未注册的 `unit` 测试标记及依赖弃用提示。
首轮完整测试的 5 个失败均为前述旧变量引用问题，修复后重新完整执行，
不是仅用专项通过替代全量结果。

日志和机器可读提交映射位于相邻部署仓库的 ignored 目录：

```text
deploy-llm-on-gpu/results/dev-migration-20261023/
```

保留首轮失败日志和修复后的最终日志，不以修复后结果覆盖原失败证据。
测试中的 skip 和 warning 不代表相关可选能力已验证通过。

## 范围与后续

本轮是代码基线迁移，不引入新的采帧、事件 schema 或检索策略优化。
没有重建/重启服务，没有更新数据库，也没有重新运行 GPU/远端 MiniMax
完整视频质量评测。旧实验文档中的结果仍对应其记录的原提交，
本轮单元测试不能替代新基线上的端到端质量验证。

后续多模态记忆优化统一基于 `dev_1023`，在合并前再次同步并验证最新 dev。
