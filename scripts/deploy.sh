#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# ContextEngine 快速部署脚本
# 使用 uv 包管理器

set -e

echo "🚀 ContextEngine 部署脚本"
echo "=========================="

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ uv 版本: $(uv --version)"

# 创建虚拟环境并安装依赖
echo ""
echo "📦 安装依赖..."
uv sync

# 检查环境变量
echo ""
echo "🔍 检查环境变量..."

if [ -z "$OGMEM_API_KEY" ]; then
    echo "⚠️  警告: OGMEM_API_KEY 未设置"
    echo "   export OGMEM_API_KEY='sk-...'"
fi

if [ -z "$AGFS_BASE_URL" ]; then
    echo "📝 AGFS_BASE_URL 未设置，使用默认值: http://localhost:1833"
    export AGFS_BASE_URL="http://localhost:1833"
fi

echo "✅ OGMEM_API_KEY: ${OGMEM_API_KEY:+已设置}"
echo "✅ AGFS_BASE_URL: $AGFS_BASE_URL"

# 运行测试
echo ""
echo "🧪 运行测试..."
uv run pytest tests/contract/ -v --tb=short

echo ""
echo "✨ 部署完成！"
echo ""
echo "使用方式:"
echo "  uv run python examples/basic_usage.py"
echo "  uv run python -m pytest tests/ -v"
