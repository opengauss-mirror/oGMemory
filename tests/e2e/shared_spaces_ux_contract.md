# UX Contract

## Product context

团队空间面向中文开发者与团队管理者。主要工作为查阅团队记忆、管理注册成员和编写文本规则。界面 locale 为 `zh-CN`，日期沿用查看器本地格式化。可访问性目标为 WCAG 2.2 AA 的键盘、语义、焦点、可读性与反馈要求。

## Business-context sources

| Domain / scope | Authoritative source | Source type | Reviewed date |
|---|---|---|---|
| 成员、来源、读取权限 | `server/space_repository.py` | 已实现业务与授权接口 | 2026-11-14 |
| 生成与生命周期 | `service/space_lifecycle.py`、`server/space_job_repository.py` | 领域实现 | 2026-11-14 |
| 幂等、修订和错误 | `server/space_management.py` | HTTP 契约 | 2026-11-14 |
| 文本规则、后台隐私 | `server/space_rules.py` | 当前用户决策及领域实现 | 2026-11-14 |

## Visual contract

[团队空间界面设计](../../docs/shared_spaces_ui_design.md) 记录团队工作台的视觉方向。`server/memory_viewer_static/styles.css` 是运行时 token 来源。支持浅、深色主题，维持查看器共享侧栏和主题开关。团队特有布局在 `#spaces-view` 所属样式中维护。

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | `spaces.js` 的 select helper 与原生 select | 本契约和 团队空间界面设计 | native | 键盘选择、打开弹层、窄屏 |
| Date | `app.js` 的日期格式化与现有原生日期输入 | 查看器日期约定 | native | 现有查看器流程 |
| Form | `spaces.js` 字段 helper、统一验证及 submit handler | HTTP 字段约束 | create / edit / inline | 保存与字段关联检查 |
| Scrollbar | `styles.css` 全局滚动条变量及基线 | 团队空间界面设计 | 列表独立滚动 | computed style、键盘滚动 |
| Toast | `spaces.js` message/failure 与 `#space-message` | 本契约 | success / error | live-region、重复反馈替换 |
| CRUD | `spaces.js` open/showTab/guardNavigation/mutate | HTTP 修订、幂等契约 | owning-list / inline | 完整浏览器业务流程 |

## Dataset navigation

空间、记忆、规则、候选使用服务端游标分页，每页 20 项。已获取的成员和来源集合在本地分页，草稿跨页保留。搜索是显式提交变体：语义检索会调用后台检索服务，输入期间只使旧响应失效；点击检索或完成输入后按 Enter 执行。清除立即恢复相应列表并回焦。

搜索条件与游标保留在内存中：查询可能包含敏感记忆内容，游标绑定身份和授权修订。团队子流程沿用查看器 `#spaces` 入口和内部返回操作。更换租户、重新加载或权限变更后重新取列表。

加载、空列表、无匹配和错误分别给出中文状态。列表区保留稳定占位；错误提供重试或刷新路径，旧请求由 epoch/sequence 失效规则阻止回写。

## Flow ledger

| Operation | Trigger | Pending | Success destination | Success feedback | Failure recovery | Focus outcome | Source ref |
|---|---|---|---|---|---|---|---|
| 创建团队 | 创建团队空间 | 稳定忙态、阻止重复提交 | 新团队成员页 | 团队空间已创建 | 保留表单与错误 | 当前团队区域 | spaces.js create |
| 保存规则 | 保存规则 | 等待服务端确认 | 规则列表 | 规则已保存 | 保留输入；修订冲突刷新 | 规则列表 | spaces.js rule |
| 保存成员/来源 | 保存成员/保存来源贡献 | 等待服务端确认 | 当前成员页 | 配置已保存/来源贡献已保存 | 草稿保留 | 成员工作区 | spaces.js members/source |
| 保存读取权限 | 保存读取权限 | 等待服务端确认 | 当前记忆详情 | 读取权限已保存 | 旧版本提示重开当前记忆 | 详情权限区 | spaces.js acl |
| 删除 | 对象对应删除按钮 | 对话框内忙态 | 所属有效列表 | 对象已删除 | 对话框内错误，可重试 | 原触发器或所属导航 | confirmAction |
| 移出成员 | 移出 | 对话框确认 | 当前成员草稿 | 待保存后生效 | 保留原草稿 | 保存成员操作 | confirmAction |
| 检索/查找 | 显式提交、清除 | 稳定列表状态 | 当前列表 | 结果与分页 | 当前条件与错误 | 搜索输入 | searchPage/findPrincipals |
| 返回/切换 | 返回、导航、切换团队 | 草稿变更时确认 | 所属列表/选中页面 | 必要时说明草稿恢复 | 继续编辑 | 原上下文 | guardNavigation |

## Navigation and responsive behavior

团队标题、功能导航和主体按阅读顺序排列。桌面成员主列与注册目录并排；窄屏按当前成员、来源、配置、注册目录排列。记忆详情和规则编辑随宽度切换为单列，所有操作保持可达。正文自然滚动，注册列表可独立滚动，长字段在可用宽度内换行。

## Overlays and feedback

确认使用 `dialog.showModal()`，包含标题、对象名称、后果、取消与动作按钮。默认聚焦取消，Escape 关闭，忙态时保护正在执行的动作，关闭后恢复焦点。删除和成员移出复用这一入口。

未保存更改通过同一 `guardNavigation()` 处理；真实页面关闭沿用 `beforeunload`。服务端失败保留输入，对应错误与修正位置可见。`#space-message` 使用工作区左下的固定被动提示与 polite live region，`pointer-events: none` 保持底层操作可达，后一次反馈替换前一次。普通反馈 6 秒后收起，错误持续显示至下一次操作。字段错误仍紧邻字段，对话框错误保持在对话框内。

## Async and resilience

保存采用服务端确认后更新。单次逻辑 mutation 使用一个 `Idempotency-Key`，按既有机制处理限流、服务暂不可用与超时重试。所有配置携带当前修订，冲突提示刷新。旧请求通过 epoch/sequence 核验，权限失效时清理旧团队内容。

忙态保持按钮尺寸、原标签与 `aria-busy`；重复提交由 form busy 状态拦截。中文 IME 组合输入期间保持编辑，候选确认 Enter 后再由明确提交触发检索。

## Validation

产品表单声明 `novalidate`，统一读取字段 validity 并展示中文内联错误。错误字段具有 `aria-invalid` 与有效的 `aria-describedby`，提交时聚焦首个错误；输入修正后清理错误。后台业务错误复用 `failure()`。规则内容为自由文本，隐私策略由后台处理。

## Permission UI

按钮与字段使用后端 capabilities。管理操作按能力显示；普通成员按只读能力浏览规则和当前记忆。授权变化由服务端确认。注册候选来自当前租户目录，来源和读者选择使用已保存成员；身份字符串保留原值。

## Verification

静态：JavaScript syntax、团队空间界面设计 lint、Premium 原始审计与逐项复核。浏览器：既有布局回归与真实注册用户完整业务流程；桌面/窄屏、浅深色、键盘、下拉框、清除、草稿和确认对话框。对照查看器记忆和控制台页面，保持共享导航尺寸。

执行所需浏览器和私有测试凭据通过环境变量传入，凭据保留在本机临时文件。
