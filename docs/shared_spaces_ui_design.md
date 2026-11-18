---
version: alpha
name: "oGMemory team workspace"
description: "面向团队记忆管理的紧凑工作台：石墨文字、冷白分区与青绿操作强调。"
colors:
  primary: "#147d73"
  primary-hover: "#0f655d"
  primary-pressed: "#0a504a"
  on-primary: "#ffffff"
  background: "#f5f7f8"
  surface: "#ffffff"
  surface-soft: "#eef3f4"
  foreground: "#202b33"
  muted: "#5b6b73"
  border: "#e2e8ea"
  border-strong: "#c3ced3"
  selected: "#e8f3f1"
  selected-text: "#0f655d"
  quiet: "#65757d"
typography:
  sans:
    fontFamily: '"Segoe UI", "PingFang SC", "Microsoft YaHei UI", system-ui, sans-serif'
    fontSize: "14px"
    lineHeight: "1.6"
  mono:
    fontFamily: 'Consolas, "SFMono-Regular", monospace'
rounded:
  control: "8px"
  panel: "12px"
spacing:
  compact: "8px"
  control-gap: "12px"
  panel-gap: "20px"
  panel-padding: "22px"
components:
  button:
    backgroundColor: "#147d73"
    textColor: "#ffffff"
    rounded: "8px"
  panel:
    backgroundColor: "#ffffff"
    rounded: "12px"
---

# oGMemory 界面设计

## Overview

团队空间采用产品工作台布局：团队身份是入口，记忆是阅读对象，成员与规则是管理对象。面向中文界面的开发者、运维人员和团队管理者，桌面承担日常配置，窄屏承担查看与轻量编辑。业务依据为 团队空间的实现 与服务端能力字段。

本轮设计覆盖团队空间的列表、详情、成员和规则流程。主导航沿用查看器的共享布局；模块内通过青绿团队标识、内容卡片和连续主列形成辨识度。文字只用于名称、操作、状态和必要反馈。

运行时 CSS 是 token 唯一来源，本文件镜像已采用的值及用途。修改颜色先更新 `server/memory_viewer_static/styles.css` 的团队语义变量，再同步本文件。共享 `app.js` 负责查看器导航、连接与主题；`spaces.js` 复用按钮、表单、分页、确认和状态反馈函数。

## Colors

冷灰背景承托白色内容区，石墨色用于阅读，青绿集中用于当前团队、选中标签和主要操作。描边区分输入和内容边界；删除使用现有语义危险色，成功与错误同时提供文字。

| 文档 token | 运行时角色 | 使用方 |
|---|---|---|
| primary / primary-hover / primary-pressed | `--space-action` 及其 hover/pressed 变量 | 主按钮、焦点、选中导航 |
| on-primary | `--space-on-action` | 主按钮文字 |
| background / surface / surface-soft | `--canvas` / `--surface-solid` / `--surface-soft` | 页面、面板、低强调区域 |
| foreground / muted | `--ink` / `--muted` | 标题、正文、辅助信息 |
| border / border-strong | `--line` / `--line-strong` | 内容分区、输入框 |
| selected | `--violet-soft` 兼容别名 | 选中区域 |
| selected-text | `--space-selected-text` | 浅色选中背景上的文字 |
| quiet | `--quiet` | 占位符和低强调信息 |

深色主题保持相同语义：背景 `#111a1d`，面板 `#182328`，低强调面板 `#213036`，正文 `#e7eff1`，辅助文字 `#a9bbc2`，描边 `#30434b` / `#50666f`；操作色 `#6dd4c4`，hover `#91e0d3`，pressed `#b4ece3`，操作文字 `#102c29`，选中背景 `#203d39`。

浅色选中文字与成功文字使用 `#0f655d`；深色选中文字使用 `#91e0d3`，使文字在各自低强调背景上保持可读对比度。

滚动条由全局 CSS 基线提供，包含 thumb、track、hover、active 变量；forced-colors 使用系统颜色。团队和其他查看器页面继承同一滚动条行为。

## Typography

使用系统中英文字体避免网络字体加载位移。团队名 24px / 600，分区标题 16px / 600，正文 14px，辅助文字 13px。正文采用约 1.6 行高，阅读内容适当增加行高；技术代码使用等宽字体。成员名称保留完整注册身份，长文本自然换行，卡片摘要在详情中可完整查看。

## Layout

顶部单行连接团队标识、团队名称、切换和创建操作。功能导航独占一行；内容区紧接搜索与列表。页面随内容自然增长。

成员页主列连续排列当前成员、来源贡献和团队配置，右列放注册目录。注册列表独立滚动和分页，主列依据自身内容排列。详情以阅读区为主、读取权限为辅；规则采用基本信息与文本编辑器并排。宽度不足时按原顺序切换单列。

共享侧栏保持桌面 232px、折叠 58px 的既有尺寸。常规间距使用 8 / 12 / 20 / 24px，标准面板内边距 22px；控件和主要操作至少 36px 高，输入约 40px。控制长度由内容和可用空间决定。

## Elevation & Depth

内容层级依靠背景、间距和描边。静态面板保持低阴影或平面；确认对话框使用顶层遮罩及清楚的容器边界。提示固定在工作区左下，随侧栏宽度调整起点，底层分页和操作保持可点击。

## Shapes

控件圆角 8px，内容面板 12px。头像和文档图标使用克制的方形容器。图标统一为 24 单位 viewBox 的线性 SVG，装饰图标隐藏于辅助技术，操作保留文字或中文可访问名称。

## Components

主按钮使用实色，常规操作使用描边，次要操作使用低强调文字。hover、focus-visible、pressed、disabled、busy 均由同一 CSS 规则族定义。忙态保留按钮文字和尺寸，使用状态指示与 `aria-busy`。

记忆与规则使用可读标题、正文摘要和明确的详情按钮。成员行统一头像、名称、身份、角色和操作列。分页复用 `pagination()`，表单复用 `input()` / `select()` / `textArea()`；确认复用 `confirmAction()`。

原生 select 负责选项弹层与键盘交互，界面负责标签、状态与触发器视觉。搜索使用明确提交和独立清除按钮。文本框固定宽度、足够初始高度，内容超长时在文本框内滚动。验证使用同一内联反馈结构，错误与对应字段关联。

动效仅用于 hover 和忙态反馈，遵循 `prefers-reduced-motion`。主题变化保持组件尺寸与信息顺序。

布局、草稿、权限反馈和恢复行为见 [浏览器验收约定](../tests/e2e/shared_spaces_ux_contract.md)。
