# 完整视频 SAMPLE 验证：Omni-7B 与 30B 对照（2026-10-22）

## 结论

**Omni-7B 能在本机两张 T4 上完成完整视频的分片记忆流水线，但目前不能通过
稳定性和答案质量验收。** 不是只测试图片/音频 API，也不是只跑视频前一分钟。

同一个完整视频做了两轮 Omni-7B 实验。两轮都在定向恢复失败片段后完成
69 条记忆、索引、聚类、语义整合和全部 12 题问答。30B 则完成了对同一份
7B 记忆的 QA-only 对照，以及少量同输入记忆抽取诊断，**没有完成 30B
全视频 69 条记忆重新生成**，不能把混合结果称为 30B 端到端通过。

| 实验 | 记忆生成 | QA 执行覆盖 | 同一 Omni-7B 评委的参考均分 |
| --- | --- | --- | --- |
| 7B 原提示链路 | 首遍 68/69；片段 46 定向重试后 69/69 | 12/12，全部有实际检索 | 0.0417 |
| 30B QA-only，读取上述 7B 记忆 | 不重建整片记忆；正文 hash 前后相同 | 初始 11/12，执行修复后 12/12 | 0.0000 |
| 7B 修复后独立库重跑 | 首遍 66/69；片段 23、58、68 重试后 69/69 | 12/12，全部有实际检索 | 0.0833 |

这些均分是 `{0, 0.5, 1}` 自动评分的平均值，**不是经独立确认的准确率**；
不能用这个小样本或同模型评委断言某个模型总体更好。三次评分均覆盖 12 题，
没有 judge API/解析错误。实际答案仍有明显问题，例如：

- 装饰材料的标注为“黏土做的苹果”，修复后 7B 回答树叶/纸叶；
- 树干、树叶的制作步骤，多次回答“没有信息”；
- 剪刀使用次数标注为三次，7B 回答两次；
- 30B 在相同旧记忆上也经常找不到证据，并未靠更大模型自动解决问题。

因此本次分别记录：

- `pipeline_coverage_complete=true`
- `first_pass_success=false`
- `quality_accepted=false`

## 工作负载与资源

实际脚本名称为 `scripts/run_m3_random10.sh`。设置 `SAMPLE_SIZE=1`、
`SEED=42`，从 100 个候选视频中固定抽到 `study_05`，没有因结果不好换样本。

- 视频时长：2044.889 秒，约 **34 分 5 秒**；`--seconds 0`。
- 69 个片段；默认 30 秒切片，最后一个较短。
- 1,093 帧：68 个片段各 16 帧，最后一个 5 帧。
- 69 段原始音频全部附加到记忆生成请求。
- 67 个片段有 ASR 转写，共 277 条；两轮 ASR 文本和时间戳逐条一致。
- 1,598 条人脸检测结果；547 个说话人时间段；140 条 ECAPA 声纹记录。
- 人脸聚类 82 个、声纹聚类 51 个，**不是 82/51 个真实人物的准确识别证明**。
- 69 条情景索引记录、69 个片段 MemoryNode、1 个整视频语义 MemoryNode。
- 69 条 EVOLVES 关系；独立库只有本实验账户，无跨账户关系。

真实使用 faster-whisper、pyannote、SpeechBrain、InsightFace 和 BGE-M3，
没有替换为 mock。M3 CLI 在仓库 `.venv` 中执行；原应用容器没有为本次
CLI 实验重建。模型与辅助服务均为本机服务：

- Omni-7B：GPU 0、3，FP16 TP=2，32768 上下文，端口 8000。
- embedding：GPU 1，端口 8001。
- CV：GPU 2，端口 18081；本次音频处理也在物理 GPU 2 上顺序执行。
- 30B 对照期间暂停两个旧 LLM，NF4/FP16 分层放置在四张 T4 上，
  禁用语音生成，但保留图像/原始音频输入；这不是 vLLM TP=4。

当前部署 `.env` 的 `OMNI_MAX_IMAGES` 已由 4 调整为 16，其他值保留。
原 Omni-7B、text-7b、embedding、CV 服务在 30B 对照后恢复。
30B 试验服务停止，权重、镜像和配置保留。

### 采帧限制

当前默认提帧器在 1 fps 下采到 16 帧就停止。数据库核对首片段帧时间为
0、1、…、15 秒，**并非均匀覆盖整个 30 秒片段**。
“全视频”在这里表示所有片段和完整音频都被处理，不能说视觉覆盖了每个时刻。
此次保留该默认策略以符合脚本基线，没有把采帧策略改变混入模型对照。

## 实验经过与失败记录

### 原提示链路 7B

使用独立账户 `m3-sample1-omni7b-16f-20261022`。开始前该账户节点数为零。
首轮新生成 68 条有效记忆，片段 46 缺少非空 `description`，被拒收。
另有 3 次 SDK 120 秒请求重试。首轮以失败结束，没有用不完整记忆进入 QA。

保存首轮日志及 summary 后，通过 `--retry-failed` 仅重试片段 46。
它以 16 帧＋原始音频正常返回，其他 68 条内容逐项比对未变化。
随后继续下游阶段并完成 12 题。因此最后一次 pipeline summary 的上游
`skipped` 是**对本轮已验证产物的续跑**，不是没有执行这些功能。

### 30B 受控对照

QA 使用原 7B 生成的同一份 69 条记忆，账户、向量索引和问题范围不变。
整个对照前后，原记忆正文 hash 相同。另对片段 7、46、52 调用纯抽取算子，
只保存诊断结果，不写回旧图；这些片段不是随机准确率样本。

30B 首次完成 11 题，第 12 题多轮检索后失败，逐层定位为：

1. Thinker 给整个长输入生成词表 logits，最后一张卡额外申请 8.61 GiB；
   推理服务改为只计算自回归生成使用的最后一个 token 的 logits。
   数值回归通过，实际 30,407-token 输入也成功执行。
2. 随后相同 Search 的完整缓存证据被重复写入对话，超过 32K。
   在 M3 保留第一份完整证据、将重复结果引用化后补测成功。
   重复工具结果从 45,085 字符降为 2,621 字符引用，最终请求输入为
   18,342 tokens，没有丢弃独有证据，也没有虚构工具结果。

30B 原始 QA JSONL 保留两次失败重试行；`qwen3-qa-final.jsonl` 为最后
12 个唯一题目的结果，评分只读取这个文件。

30B 的原生工具接口使用官方模板，只解析和校验模型实际生成的工具调用；
真实函数往返已验证。JSON object 模式是完成后的校验，不是受约束解码。
一次显式 `enable_thinking=False` 试验产生空音频回答，被启动自检拦截；
恢复官方模板默认行为后通过音频自检，失败记录未删除。

### 修复后完整 7B 重跑

最初尝试仅更换账户时，首个视频节点触发唯一键冲突。原因是现有
`load_clips` 使用 `sha1(video_id/clip_id)`，SQL 主键没有账户维度。
新账户未插入节点，旧数据未覆盖。没有重写旧 ID，而是创建空的独立测试库
`ogmem_m3_fixed_20261022`，同时隔离图和向量存储。

使用新账户 `m3-sample1-omni7b-fixed-20261022` 从头重跑所有前处理，
没有复用前一账户的记忆或 ASR 缓存。使用已提交代码 `f80037a7`。

首遍 66/69 有效，片段 23、58、68 均出现：

```text
finish_reason=length max_tokens=4096 completion_tokens=4096
```

这些响应在 JSON 修补之前被拒绝。对失败清单做一次定向恢复：

| 片段 | 相同输入的 prompt tokens | 重试 completion tokens | 结束原因 |
| --- | ---: | ---: | --- |
| 23 | 4988 | 533 | stop |
| 58 | 4996 | 546 | stop |
| 68 | 2101 | 439 | stop |

输入加原先 4096 输出预算远低于 32768，因此这三次是**输出预算耗尽**，
不是整个视频把上下文塞满。重试未提高 token 上限，其他 66 条内容保持不变。
这说明能恢复完成，但不证明模型首遍生成稳定。

恢复后继续创建 MemoryNode、索引、聚类、语义整合和 QA，18 个阶段的执行
证据跨首轮、恢复、续跑完整覆盖。最终 69 条记忆都有非空描述和原始音频；
12 题均有非空、无协议占位符的回答以及实际 Search 记录，judge 错误为零。

## 本次修复

- `c3a0f86e`：M3 继承/覆盖并实际传递 `request_timeout_seconds`。
  此前配置 300 秒却使用默认 120 秒；同时覆盖 OpenAI-compatible/MiniMax。
- `f80037a7`：
  - 多模态客户端保留调用方系统提示，再追加 JSON schema；此前保留具体
    语音事实等抽取要求被静默丢弃。
  - 多模态输出 `length` 不再被 JSON repair 修补成成功。
  - ReAct QA 任意题失败返回非零，保留其他结果以便定向恢复。
  - 缓存检索结果只在完整、完全相同的证据仍位于历史中时引用化；
    完整 cache、`tools_used`、检索审计和第一份原文保留。

提示修复后对 30B 片段 52 额外进行了同输入抽取，描述更关注黏土任务，
但仍不能据此认定事实保留或整体 QA 质量通过。完整 7B 修复后均分依然很低。

相关验证：**1206 项 M3/provider/operator 测试通过**；
部署与协议/最后-token-logits 回归 **36 项通过**。
所有 oGMemory 修改提交在 `fix_m3_1019`，没有修改 locomo-test。

## 复现与文件位置

原始入口：

```bash
SAMPLE_SIZE=1 SEED=42 \
CONFIG=data/m3_sample1_omni7b_fixed_20261022.yaml \
RUN_ROOT=data/m3_runs_yuanjian/sample1_omni7b_fixed_db_seed42_20261022 \
CUDA_VISIBLE_DEVICES=2 UV_NO_SYNC=1 \
bash scripts/run_m3_random10.sh \
  --account-id m3-sample1-omni7b-fixed-20261022 \
  --set voice_asr.params.device=cuda \
  --set speaker_diarization.params.device=cuda \
  --set speaker_embedding.params.device=cuda
```

需要提前导出已授权的 Hugging Face token 和 CUDA 12 cuBLAS/cuDNN 动态库路径。
本机完整环境包装命令保存在相邻部署仓库的 ignored
`results/m3-sample1-20261022/run-omni7b-fixed.sh`。
不要把包含凭据的 YAML、环境文件或备份提交到 Git。

已完成的同目录重跑默认跳过完整 QA；新实验应使用新目录和独立数据库。
定向恢复不应把 `retry_failed=true` 无差别加到整个随机 wrapper 上。

数据目录：

- 原 7B：`data/m3_runs_yuanjian/sample1_omni7b_16f_seed42_20261022/`
- 最终 7B：`data/m3_runs_yuanjian/sample1_omni7b_fixed_db_seed42_20261022/`
- 本机 ignored 配置：`data/m3_sample1_omni7b_fixed_20261022.yaml`

相邻 `deploy-llm-on-gpu/results/m3-sample1-20261022/` 保存：

- `initial-attempt/`、`fixed-initial-attempt/`：首遍失败，不删除、不冒充成功。
- `recovery-audit.json`、`fixed-recovery-audit.json`：仅失败片段重生成。
- `omni7b-audit.json`、`fixed-audit.json`：完整阶段、SQL、QA 覆盖审计。
- `qwen3-qa-final.jsonl`、`qwen3-judge/`：30B QA-only 对照与相同评委评分。
- `qwen3-memory-pilot*.json`：只读片段诊断，不是整片 30B 记忆生成。
- `qwen3-frozen-memory-audit.json`：30B 对照没有覆盖原 7B 记忆。
- `answer-comparison.json`：每道题的标注与三组答案。
- `frame-sampling-audit.json`、`fixed-asr-input-comparison.json`：输入覆盖与对照。
- `run_report.json`：最终汇总；`final-*-smoke.log`：服务恢复后的真实请求检查。
