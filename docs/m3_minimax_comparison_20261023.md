# MiniMax 与本地 Omni-7B 完整视频对照（2026-10-23）

## 结论与主比较

相同视频、帧、原始音频和 ASR 输入下，MiniMax 链路的答案明显优于此前修复后
Omni-7B 链路；两种评委均给出同方向的提升。**不能据此认为所有问题只需换模型：
MiniMax 仍有明显错答，且原预算下最后一题两次因输出耗尽失败。**

| 答案来源 | Omni-7B 评委 | MiniMax-M3 评委 | 非空且无运行错误的答案 |
| --- | ---: | ---: | ---: |
| 修复后 Omni-7B 基线 | 0.0833 | 0.1667 | 12/12 |
| MiniMax，QA 上限 2048 | 0.5000 | 0.5417 | 11/12 |

均分为 12 题 `{0, 0.5, 1}` 自动评分的平均值，失败题计 0，**不是独立验证的准确率**。
四格均使用同一标注、评分提示和 1024 评委输出上限，`judge_error_count=0`；
API 审计也没有评委 `length` 响应。旧答案重新交叉评分，不重新生成。
MiniMax 主结果保留第一次 QA，不择优挑选答案。

主比较有一个必须保留的例外：记忆片段 51 最后一次重试上限为 8192，而非 4096。
它实际只消耗 1935 completion tokens，因此不能确定提高上限是恢复成功的原因。
这不是“所有配置完全相同”的严格实验，也不是记忆模型/QA 模型的独立消融。

## 工作负载与隔离

- 用户指向的凭据位于 `deploy/ogmemory.yaml` 的注释配置块。
  实际请求和响应均报告 `MiniMax-M3`；远端 host 为 `api.minimaxi.com`。
  原 YAML、全局 `.env` 未改，凭据只复制到 gitignored、权限 600 的实验 YAML。
- 使用真正的 `scripts/run_m3_random10.sh`，`SAMPLE_SIZE=1`、`SEED=42`。
  视频 `study_05`，2044.889 秒（约 34 分 5 秒），`--seconds 0`，69 个片段、12 题。
- 1093 张 JPEG 与固定 7B 基线逐文件字节一致；69 段 WAV 也逐文件一致；
  277 条 ASR 的文本与时间戳完全一致，见 `input-comparison.json`。
- 真实执行人脸检测、ASR、说话人分离、ECAPA、记忆生成、索引、聚类、
  语义整合和问答，并非只测多模态 API。未使用 mock。
- 本机 CV/ASR/声纹与 BGE-M3 保持不变。69 个记忆请求均附加原始音频。
- 新独立数据库 `ogmem_m3_minimax_20261023`，账户
  `m3-sample1-minimax-20261023`，向量表 `m3_sample1_minimax_20261023`。
  避免旧视频片段主键不含账户导致的数据冲突，不覆盖原基线。
- SQL 审计：1231 raw、4378 extracted、70 memory 节点；69 非空向量、
  69 EVOLVES 关系；仅本实验账户，无跨账户或悬空边。

默认每 30 秒片段的视觉采样仍主要覆盖前 0–15 秒，而非均匀覆盖整段。
“完整视频”表示处理全部片段和完整音频，不表示每个视觉时刻都已观察。
本次刻意保留采样策略，避免混入额外变量。

## 预算、失败和恢复

原配置：记忆 `4096/.2`、QA `2048/0`、评委 `1024/0`
（输出上限/temperature），请求超时 300 秒。QA 最多 6 轮、180 秒，
top_k=8、candidate_k=20、transcript_top_k=20、RRF、context_k=8、
neighbors=1、current_clip 时间策略，与基线一致。
模型可在工具参数中请求不同 top_k，这不是全局配置被修改。

### 记忆

1. 首遍生成 66/69；片段 19、45、51 耗尽输出预算而被拒收，未拿残缺 JSON 继续。
   其中两次几乎所有输出 tokens 都计入 reasoning。
2. 同预算定向重试补齐 19、45；51 再次耗尽 4096，达到 68/69。
3. 仅片段 51 改用 8192 上限、600 秒超时重试成功：实际 completion=1935，
   reasoning=1398。前面 68 条正文逐项验证不变。
4. 一次 HTTP 529 overloaded 经有界退避恢复，和答案质量错误分别统计。

前处理与首遍失败存于 `first-attempt/`，同预算恢复存于 `retry-4096/`，
最终记忆摘要为 69 完成、0 失败，69 条 description 非空且音频已附加。
续跑的 pipeline summary 中上游 `skipped` 指复用本轮已验证产物，不是没测。

### 问答

全部 12 题均执行真实检索，共 67 次 search executions。首轮 11 题产生答案；
Q12（剪刀使用次数）第 6 轮返回：

```text
finish_reason=length max_tokens=2048
completion_tokens=2047 reasoning_tokens=2047
```

只对 Q12 按原设置恢复一次，11 个成功答案跳过，仍同样耗尽输出预算。
这次失败不是 API 报告的输入 context-window 超限。
主流水线退出码 **2**、QA stage **degraded** 原样保留，不能写成一次全绿通过。
主比较仍使用首轮答案，Q12 计 0。

另有仅 Q12 的 8192 上限诊断，独立文件和 usage 审计，不计入上表。
结果仍失败：第 6 轮 `completion_tokens=8191`、`reasoning_tokens=8191`、
`finish_reason=length`，输入 110531 tokens，调用约 50.8 秒。
其余 11 题未重跑。没有继续盲目增加预算；此诊断支持长检索对话下 reasoning
不收敛的可能性，而不是“调大 max_tokens 就一定解决”。

累计主实验和同预算 Q12 重试的实际 QA prompt 峰值为 **129886 tokens**。
MiniMax 产生的记忆更长、检索对话也不同，不能把此结果解释为只改变参数量、
固定 token 输入的推理能力比较；不同模型 tokenizer 的 token 数也不可直接等同。

主实验审计（包含记忆恢复和原预算 Q12 重试，不含预检、交叉评分和 QA8192
诊断）记录 155 次逻辑 completion 调用：154 个响应、1 个 HTTP 529 错误；
响应合计 4,589,210 prompt tokens、142,832 completion tokens，其中
84,953 reasoning tokens，3,016,745 prompt tokens 报告为缓存命中。
6 个 `length` 分别来自 4 次记忆生成和 2 次 Q12。
另行 QA8192 诊断为 6 次调用、1 个 `length`、370,500 prompt tokens、
10,285 completion tokens。以上不是独立输入大小，也不是账单金额；
不含 SDK 内部未单独记录的 HTTP 重试用量。

## 逐题检查与剩余问题

- 树干步骤：标注 3；MiniMax 答 3，旧 7B 未能给出正确答案。
- 树叶步骤：标注 3；MiniMax 答 3，但理由是否完整两评委意见不同。
- 装饰材料：标注“黏土苹果”；MiniMax 答案包含 clay apples，旧 7B 为树叶/纸叶。
  新记忆中可找到 clay 证据，但 MiniMax 答案还添加叶子、felt 等细节，不能都视为已证实。
- 工具箱物品：标注剪刀和胶水；MiniMax 仅确认剪刀，漏掉胶水，两评委均给半分。
- 彩纸位置：标注最右书柜顶层抽屉；MiniMax 答桌上，错误。
  新 description 中也未检出 `drawer`，但这不能证明原视频没有该证据。
- 树的总数：标注 1；MiniMax 把“再做一棵”的话当作已经完成，答 2，错误。
- 主要步骤数：标注 5；MiniMax 答 3。Omni 评委仍给半分，MiniMax 给 0，
  说明自动评委也有偏差。
- 书包位置：标注桌子左侧；回答桌上，两评委都给半分，不等于空间定位正确。

需要进一步区分视觉覆盖、记忆抽取、检索召回、人物关联、意图与已发生事件、
以及最终回答推理。当前实验同时替换了记忆与回答模型，尚不能分摊各环节贡献。
本轮只做单视频 M3 对照，没有新增 LoCoMo 实验，也没有重跑 30B 全视频。

## 代码与验证

`29724e58`（`fix_m3_1019`）修复 MiniMax 工具参数兼容：
远端预检有时返回位置数组，而不是工具 schema 要求的命名对象。
仅当简单标量 schema **存在唯一映射**时无损转换，歧义/复杂情况拒绝，
不依赖 property 顺序，不虚构工具名、值、默认值；原生对象与请求提示保持不变。
输出 `length` 仍由共享截断检查拒绝。不是替模型伪造工具调用。

真实 16 帧＋音频预检、原生工具往返均通过；相关测试日志记录
**1220 passed**，并在报告阶段重新运行 MiniMax 专项测试，**23 passed**。
最终另跑 `tests/unit/providers/llm` 与 `tests/unit/m3`，**858 passed**。
主 QA 的审计工具参数实际均为对象；位置数组兼容不是本次答案语义提升的证据。

## 产物与重现

结果根目录（下文简称 R）：

```text
/home/yuanjian/Development/deploy-llm-on-gpu/results/m3-minimax-sample1-20261023/
```

运行目录：

```text
/data/m3_runs_yuanjian/sample1_minimax_seed42_20261023/
```

- `R/run-minimax.sh`：实际 wrapper 入口；私有配置在 oGMemory 的 ignored `data/`。
- `R/comparison-report.json`：四格评分、用量、输入与 SQL 审计。
- `R/answer-comparison.json`：逐题旧/新答案、金标、两评委分数及理由。
- `R/minimax-final-qa.jsonl`：用于主比较的首轮 12 题（包括 Q12 失败），
  **文件名 final 不表示 12 题成功**。
- `R/qa-first-attempt/`、`R/qa-retry.jsonl`：首轮与原预算定向重试，不删除失败行。
- `R/cross-grade/`：固定预测的四格评分；不重新生成记忆。
- `R/model-usage.jsonl`：仅模型、tokens、模态数量、finish reason 等元数据；
  无提示内容、媒体、回答正文或凭据。SDK 内部 HTTP 重试不单独计数。
- `R/recovery-audit.json`、`input-comparison.json`、`database-audit.json`：
  恢复、相同输入和隔离验证。

重新评分可使用 `python -m m3.eval --config <ignored YAML> --annotations
/data/m3_data/annotations/robot.json --video study_05 --input-qa-results <JSONL>
--output-dir <新目录>`。不要把密钥写进 CLI 或提交配置。

保留已有 GPU 服务和部署仓库改动；`locomo-test` 未修改；提交不推送。
