#!/bin/sh
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -e

# Disable Bonjour / mDNS by default. With network_mode: host the openclaw
# container shares the host's interface list (eth0, wlan0, docker0, ...);
# the ciao probe packet that the responder fires at startup then includes
# an A/AAAA record per interface and blows past the 1440-byte UDP MTU
# (`AssertionError: Probe query packet exceeds the mtu size (2115>1440)`),
# which crashes the gateway. mDNS over a host-network container also serves
# no real purpose — the multicast stays on the host loopback and isn't
# discoverable from other devices. Operators who actually want LAN
# discovery can `unset OPENCLAW_DISABLE_BONJOUR` or run on bridge network.
export OPENCLAW_DISABLE_BONJOUR="${OPENCLAW_DISABLE_BONJOUR:-1}"

OC_HOME="/home/node/.openclaw"
BACKUP="/opt/ogmem-openclaw-defaults"
TEMPLATE="$BACKUP/openclaw.template.json"
CONFIG="$OC_HOME/openclaw.json"
# Render and validate before touching the active config. Existing user settings
# are preserved; known template placeholders and missing gateway fields migrate
# in place with a private backup. Invalid configs abort startup rather than
# being silently replaced or passed to the gateway.
node /usr/local/bin/configure-openclaw.cjs "$TEMPLATE" "$CONFIG"

mkdir -p "$OC_HOME/extensions"
# This extension is image-managed, not user state. Refresh bundled files even
# on persisted volumes, otherwise rebuilding the image leaves old plugin code
# (which may not support the current environment switches) active indefinitely.
cp -r "$BACKUP/extensions/og-memory-context-engine" "$OC_HOME/extensions/"

chown -R node:node "$OC_HOME"
chmod 755 "$OC_HOME"
chmod -R 755 "$OC_HOME/extensions"
chmod -R a+rwX "$OC_HOME/agents" 2>/dev/null || true
# The gateway runs as node; API keys must not be world-readable/writable.
chmod 600 "$CONFIG"

mkdir -p /tmp/ogmem_ws
chmod 777 /tmp/ogmem_ws

exec su -s /bin/sh node -c "umask 0000; exec openclaw $*"
