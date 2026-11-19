---
version: alpha
name: oGMemory
description: 以记忆浏览、来源和执行过程为核心的工作界面
colors:
  primary: "#222222"
  canvas: "#f5f5f5"
  surface: "#ffffff"
  ink: "#181818"
  muted: "#525252"
  line: "#e0e0e0"
  evidence: "#383838"
  success: "#454545"
  warning: "#505050"
  danger: "#181818"
typography:
  body:
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans SC", sans-serif'
    fontSize: "14px"
    lineHeight: "1.7"
rounded:
  control: "8px"
  panel: "12px"
omitted:
  - section: spacing
    reason: 现有布局在各自运行时样式中维护
  - section: components
    reason: 复用原生控件与各文档共享的 CSS 类
---

## Overview

面向记忆查询、资料整理和系统管理。以索引、正文、来源和执行记录建立一致的阅读顺序。侧栏定位工作区，主区呈现数据，细边框区分筛选、列表和详情。明暗模式均采用黑白灰。实色按钮标记主要动作，浅灰底和边线标记当前选择。memoryBot 沿回复逐步展示实际执行记录。

中文为界面语言，Agent、URI 等原生标识保持原样。主要使用场景为桌面工作，窄屏提供会话抽屉。产品范围以 `docs/memorybot-design.md`、`docs/memorybot.md` 和服务端契约为准。

运行时以 `server/memory_viewer_static/theme.css` 为取值源。查看器、团队空间和迁移页面继承语义变量，memoryBot 通过同一主题适配。页面样式负责布局和组件密度。

## Colors

`--canvas` 用于页面，`--surface-solid` 用于实色面板，`--surface-soft` 用于次级内容，`--surface-rail` 用于导航列。`--ink`、`--muted`、`--line` 建立文字与边界层次。现有语义变量保持兼容，`--violet` 与 `--on-accent` 映射主要动作，`--cyan` 映射来源，`--green`、`--amber`、`--red` 映射状态，各变量均取中性灰度。状态由文字和图标表达，趋势图使用与图例对应的线型区分序列。

memoryBot 通过 `--accent → --violet`、`--soft → --surface-soft`、`--warning → --amber` 适配共享主题。`data-theme` 切换明暗语义值。滚动条使用共享 thumb/track/hover/active token，并提供系统高对比模式。

## Typography

正文使用 `--font-body`，页面标题使用 `--font-display`，标识、路径和分值使用 `--font-mono`。中文回退来自操作系统。页面标题、正文和辅助信息各有明确层级；长段文字保持约 1.7 行高。

## Layout

查看器左侧为工作区导航。记录浏览围绕列表和详情展开。memoryBot 的会话列表与对话区独立滚动，输入区固定在对话区底部。资料、计划、记忆采用浏览记录的纵向阅读顺序。团队空间和迁移页面沿用标题、主操作、任务反馈的层次。

窄屏沿用查看器折叠导航，memoryBot 会话列表由“会话”按钮展开。卡片、表单和消息中的长内容允许换行或在自身区域滚动。

## Elevation & Depth

主要层级由底色、边框和留白表达；浮层使用既有 dialog 遮罩和阴影。消息执行过程通过折叠层级呈现。

## Shapes

按钮和输入框以 `--radius-control` 为基础，面板使用 `--radius-panel`。图标使用当前前台内嵌线条 SVG，按钮保留文本或可访问名称。

## Components

按钮提供悬停、可见焦点、禁用和处理中状态。焦点与常用动效时长分别由 `--focus-ring` 和 `--motion-fast` 管理。原生 select/date 控件保留平台弹层与键盘交互。表单由 `formDialog` 统一提供校验、提交状态和错误提示；dialog 负责焦点隔离与 Escape。

任务进度使用服务端阶段、耗时和结果摘要；连续文本使用 Markdown 渲染并净化。自动滚动尊重用户向上阅读。动画只用于短时交互反馈，遵循 reduced-motion。

## Do's and Don'ts

- 新界面复用共享语义 token 和现有动作名称。
- 团队空间、memoryBot、查看器和迁移界面共用黑白灰主题，演示截图从对应版本重新截取。
- 将复杂过程放在可展开的步骤中，默认摘要保留当前状态。
- 业务状态和权限动作以服务端响应为依据。
