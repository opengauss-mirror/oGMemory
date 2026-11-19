# UX Contract

## Product context

中文工作界面，服务于记忆浏览、问答、资料整理和计划执行。视觉规范见 `DESIGN.md`。时间按用户浏览器区域显示，计划保留明确时区。交互基线为原生语义、可见焦点、键盘可操作和 WCAG 2.2 AA。

## Business-context sources

| Domain / scope | Authoritative source | Source type | Reviewed date |
|---|---|---|---|
| 功能与流程 | `docs/memorybot-design.md`、`docs/memorybot.md` | 产品与设计 | 2026-11-18 |
| 身份、Agent 与作用域 | `memorybot/http.py`、`memorybot/service.py` | API | 2026-11-18 |
| 任务、确认、记忆动作 | `memorybot/models.py`、`memorybot/service.py` | 生命周期 | 2026-11-18 |
| 查看器导航与团队空间 | `server/memory_viewer_static/app.js`、`spaces.js` | 现有行为 | 2026-11-18 |
| 迁移操作与反馈 | `server/memory_viewer_static/migration.js` | 现有行为 | 2026-11-18 |

## Visual contract

运行时 `server/memory_viewer_static/theme.css` 维护语义颜色、字体、焦点、动效和滚动条。查看器、团队空间、迁移和 memoryBot 共用。宿主通过同源 iframe 消息同步 `data-theme`。

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | 原生 select | DESIGN.md | native | 浏览器弹层与键盘 |
| Date | 原生 datetime-local / time | memoryBot 计划表单 | native | 计划创建与编辑 |
| Form | memoryBot formDialog；宿主既有表单 | API 字段与 UX Contract | create / edit | DOM、浏览器 |
| Scrollbar | `server/memory_viewer_static/theme.css` | DESIGN.md | 面板滚动区域与留槽 | 浏览器计算样式 |
| Toast | memoryBot toast / report；宿主状态区 | UX Contract | success / info / error | live region |
| CRUD | memoryBot action / API；宿主 switchView | API 生命周期 | return / stay | DOM、浏览器 |

## Dataset navigation

会话、资料、计划和记忆使用服务端 cursor 与“加载更多”。关键词搜索使用“相关结果”文案。会话搜索支持输入法合成、延迟请求、即时清除和请求序号过滤；记忆搜索显式查询。

宿主导航使用 URL hash。会话和密钥通过 sessionStorage 保持当前标签页上下文；查询词、草稿、选中文件和内部筛选属于临时工作状态。切换页面保留聊天草稿，切换身份清理旧身份的会话与数据。

## Flow ledger

| Operation | Trigger | Pending | Success destination | Success feedback | Failure recovery | Focus outcome | Source ref |
|---|---|---|---|---|---|---|---|
| 对话 | 发送消息 | 当前回复执行过程、停止 | 当前会话 | 逐段正文与最终结果 | 输入与任务提示保留 | 输入区 | ChatInput / task |
| 改名 | 会话设置 | 保存按钮 busy | 当前会话 | 标题与列表更新 | 表单内错误 | 原触发按钮 | SessionPatch |
| 归档/恢复 | 会话菜单 | 请求处理中 | 当前会话筛选列表 | 列表更新 | 页面错误提示 | 会话列表 | SessionPatch |
| 上传资料 | 上传资料 / 收录网页 | 后台任务步骤 | 资料列表与处理记录 | 资料可用状态 | 停止 / 核对结果 | dialog / 资料卡 | ingest task |
| 新建计划 | 新建计划 | 保存按钮 busy | 计划列表 | 成功提示 | 保留表单 | 原触发按钮 | ScheduleCreate |
| 管理记忆 | 记住 / 更正 / 遗忘 | 后台任务步骤 | 处理记录与记忆列表 | 真实记忆状态 | 停止 / 核对结果 | dialog | MemoryAction |

## Interaction and recovery

按钮与导航使用原生交互元素。表单统一应用内校验、字段关联错误、必填焦点、重复提交防护；密钥默认遮盖并支持显示。文本域自动增长，dialog 内容内部滚动。

所有后台状态来自任务与 SSE。正文结束、任务完成、后台记忆完成分别呈现。执行步骤按 step_id 和 event_seq 合并；刷新由快照恢复。用户向上阅读时保留位置，可点击“最新消息”。

嵌入入口为 `/memory-viewer/#bot`，独立入口为 `/bot`。iframe 消息校验同源和发送窗口；宿主连接设置更新后重新载入 bot。API Key 保存在当前标签页的 sessionStorage。

原生 dialog 提供焦点隔离、Escape 和关闭后焦点恢复。会话抽屉保持已有非模态导航行为。网络或权限错误呈现在可见状态区，编辑内容保留供重试。

## Verification

维护现有 DOM 场景并运行浏览器验收：Agent 选择、首次建会话、流式正文、工具执行过程、来源、复制、停止、资料、计划、记忆范围、改名归档恢复、连接和主题同步；比较查看器与 memoryBot 在桌面和窄屏的布局。
