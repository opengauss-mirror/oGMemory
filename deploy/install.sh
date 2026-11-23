#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

################################################################################
# OpenClaw + oGMemory 安装脚本
#
# 功能：下载部署脚本和配置文件到本地
#
# 使用方法：
#   远程一键安装（推荐）
#     curl -fsSL https://raw.gitcode.com/opengauss/oGMemory/raw/dev/deploy/install.sh | bash
#
# 执行后会下载以下文件到 ./ogmem-deploy/ 目录：
#   - .env.example:   环境变量模板（请复制为 .env 并填写）
#   - ogmemory.example.yaml: oGMemory 服务配置模板
#   - docker-compose.yml:  docker compose 部署定义（推荐方式）
#   - deploy.sh:      旧版 docker run 部署脚本（向后兼容）
################################################################################
set -euo pipefail

REPO_RAW="https://raw.gitcode.com/opengauss/oGMemory/raw/dev"
TARGET_DIR="./ogmem-deploy"

echo "下载 OpenClaw + oGMemory 部署文件..."
mkdir -p "${TARGET_DIR}"

curl -fsSL "${REPO_RAW}/deploy/.env.example"          -o "${TARGET_DIR}/.env.example"
curl -fsSL "${REPO_RAW}/deploy/ogmemory.example.yaml" -o "${TARGET_DIR}/ogmemory.example.yaml"
curl -fsSL "${REPO_RAW}/deploy/docker-compose.yml"    -o "${TARGET_DIR}/docker-compose.yml"
curl -fsSL "${REPO_RAW}/deploy/deploy.sh"             -o "${TARGET_DIR}/deploy.sh"
chmod +x "${TARGET_DIR}/deploy.sh"

# Bootstrap a runnable .env from the template. Without this, `docker compose`
# complains "variable not set" for the required-password vars (compose treats
# `${X:?...}` as a hard error when X is unset).
if [[ ! -f "${TARGET_DIR}/.env" ]]; then
    cp "${TARGET_DIR}/.env.example" "${TARGET_DIR}/.env"
    echo "已生成 ${TARGET_DIR}/.env（从 .env.example 复制）"
else
    echo "${TARGET_DIR}/.env 已存在，未覆盖"
fi

echo ""
echo "下载完成！文件已保存到 ${TARGET_DIR}/"
echo ""
echo "使用 docker compose（推荐）："
echo "  1. cd ${TARGET_DIR}"
echo "  2. 编辑 .env 填写 OG_PASSWORD / LLM_* / 等（替换所有 CHANGE_ME_* 占位符）"
echo "  3. cp ogmemory.example.yaml ogmemory.yaml"
echo "  4. docker compose --profile with-db up -d        # openGauss + ogmem"
echo "  5. docker compose --profile with-db --profile with-openclaw up -d   # + openclaw gateway"
echo ""
echo "使用旧版 docker run："
echo "  1. cd ${TARGET_DIR}"
echo "  2. 编辑 .env 填写配置（deploy.sh 也读取此文件）"
echo "  3. bash deploy.sh -password <openGauss密码>"
echo ""
echo "首次访问："
echo "  oGMemory API:  http://<HOST_IP>:8090"
echo "  OpenClaw Web:   http://<HOST_IP>:18789"
echo ""
echo "配置文件说明："
echo "  .env:                统一环境变量配置（compose 和 deploy.sh 共用）"
echo "  .env.example:        .env 的模板"
echo "  ogmemory.yaml:       oGMemory 服务配置（LLM、向量数据库等）"
echo "  docker-compose.yml:  compose 部署定义"
echo ""
echo "详细配置说明请参考各文件中的注释"
