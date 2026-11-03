import shutil
import subprocess

import pytest


def require_docker_compose():
    docker = shutil.which('docker')
    if docker is None:
        pytest.skip('Docker CLI unavailable; config checks need no daemon')
    result = subprocess.run([docker, 'compose', 'version'], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        pytest.skip('Docker Compose plugin unavailable; config checks need no daemon')
    return docker


@pytest.fixture
def docker_compose_cli():
    return require_docker_compose()
