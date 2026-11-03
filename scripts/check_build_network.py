#!/usr/bin/env python3
"""Check selected images and build dependencies without starting application services."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD_ARGS = ('GO_BASE_IMAGE', 'PYTHON_BASE_IMAGE', 'GOPROXY', 'GOSUMDB',
              'DEBIAN_MIRROR', 'DEBIAN_SECURITY_MIRROR', 'PIP_INDEX_URL',
              'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY')


def load_config():
    result = subprocess.run(['docker', 'compose', 'config', '--format', 'json'],
                            cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Invalid Compose configuration')
    return json.loads(result.stdout)


def build_spec(config, target='agfs-deps'):
    service = config['services']['ogmemory']
    build = service['build']
    args = {key: build['args'][key] for key in BUILD_ARGS
            if build.get('args', {}).get(key)}
    spec = {'name': 'ogmemory-source-check', 'services': {'source-check': {
        'platform': service['platform'],
        'build': {'context': build['context'], 'dockerfile': build['dockerfile'],
                  'target': target, 'args': args},
    }}}
    if build.get('extra_hosts'):
        spec['services']['source-check']['build']['extra_hosts'] = build['extra_hosts']
    return spec


def check_images(config):
    args = config['services']['ogmemory']['build']['args']
    images = [args['GO_BASE_IMAGE'], args['PYTHON_BASE_IMAGE'],
              config['services']['opengauss']['image']]
    for image in images:
        result = subprocess.run(['docker', 'image', 'inspect', image,
                                 '--format', '{{.Architecture}}'],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == 'arm64':
            print('ARM64 image available in local cache.', flush=True)
            continue
        print('Pulling required ARM64 image from selected registry.', flush=True)
        result = subprocess.run(['docker', 'pull', '--platform', 'linux/arm64', image])
        if result.returncode:
            return result.returncode
    return 0


def main():
    try:
        config = load_config()
        result = check_images(config)
        if result:
            print('Image check failed; select reachable image addresses in .env.')
            return result
        for target in ('agfs-deps', 'python-deps'):
            print(f'Checking {target}: downloading dependencies with selected sources.', flush=True)
            result = subprocess.run(
                ['docker', 'compose', '-f', '-', 'build', '--no-cache', 'source-check'],
                input=json.dumps(build_spec(config, target)), text=True, cwd=ROOT)
            if result.returncode:
                print(f'Build source check failed at {target}; update .env and retry.')
                return result.returncode
    except (OSError, ValueError, KeyError):
        print('Build source check failed. Check Docker and run docker compose config --quiet.')
        return 1
    print('Build source check passed: images available, Go dependencies verified, Debian/Python packages installed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
