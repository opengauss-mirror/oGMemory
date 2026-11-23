# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Docker CLI wrapper using subprocess."""

from __future__ import annotations

import json
import subprocess


def docker_available() -> bool:
    """Check if docker is installed and running."""
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def docker_pull(image: str) -> bool:
    """Pull a docker image."""
    return subprocess.run(["docker", "pull", image], capture_output=True).returncode == 0


def docker_run(
    name: str,
    image: str,
    env: dict[str, str] | None = None,
    volumes: dict[str, str] | None = None,
    ports: dict[str, str] | None = None,
    network: str = "host",
    extra_args: list[str] | None = None,
) -> bool:
    """Run a docker container."""
    cmd = ["docker", "run", "-d", "--name", name, "--network", network]
    if env:
        for k, v in env.items():
            cmd.extend(["-e", f"{k}={v}"])
    if volumes:
        for h, c in volumes.items():
            cmd.extend(["-v", f"{h}:{c}"])
    if ports:
        for h, c in ports.items():
            cmd.extend(["-p", f"{h}:{c}"])
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(image)
    return subprocess.run(cmd, capture_output=True).returncode == 0


def docker_rm(name: str, force: bool = False) -> bool:
    """Remove a docker container."""
    cmd = ["docker", "rm", "-f", name] if force else ["docker", "rm", name]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def docker_ps(names: list[str] | None = None) -> list[dict]:
    """List running containers."""
    result = subprocess.run(["docker", "ps", "--format", "json"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    containers = []
    for line in result.stdout.strip().split("\n"):
        if line:
            try:
                data = json.loads(line)
                containers.append({"name": data.get("Names", ""), "status": data.get("Status", ""), "image": data.get("Image", "")})
            except json.JSONDecodeError:
                continue
    return [c for c in containers if not names or c["name"] in names]


def docker_logs(name: str, follow: bool = False) -> None:
    """Show docker container logs."""
    subprocess.run(["docker", "logs"] + (["-f"] if follow else []) + [name], check=False)


def container_running(name: str) -> bool:
    """Check if a container is running."""
    return any(c["name"] == name for c in docker_ps())
