#!/usr/bin/env python3
"""Prepare an isolated instance without evaluating shell configuration."""
import json
import fcntl
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile

import yaml


def private_write(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def prepare(env, root):
    # Deployment roots own one generated dotenv; serialize all preparation there.
    root = Path(root).resolve()
    fd = os.open(root / '.ogmemory-prepare.lock', os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_locked(env, root)


def _prepare_locked(env, root):
    root = Path(root).resolve()
    project = env.get('COMPOSE_PROJECT_NAME', 'ogmemory')
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,40}', project):
        raise ValueError('invalid project name')
    ports = [int(env.get(k, default)) for k, default in
             [('OGMEM_HTTP_HOST_PORT', '18090'), ('OGMEM_MCP_HOST_PORT', '18091')]]
    if len(set(ports)) != 2 or any(p < 1024 or p > 65535 for p in ports):
        raise ValueError('invalid or overlapping port')
    for key in ['LLM_BASE_URL', 'LLM_MODEL', 'LLM_API_KEY', 'EMBEDDING_BASE_URL',
                'EMBEDDING_MODEL', 'EMBEDDING_API_KEY', 'EMBEDDING_DIMENSION']:
        if not env.get(key) or any(x in env[key] for x in ['替换为', '你的']):
            raise ValueError('missing configuration: ' + key)
    dimension = int(env['EMBEDDING_DIMENSION'])
    threshold = int(env.get('OGMEM_AFTER_TURN_THRESHOLD', '400'))
    if dimension < 1 or threshold < 1:
        raise ValueError('dimension and threshold must be positive')
    enabled = env.get('OGMEM_MCP_ENABLED', 'true').lower()
    if enabled not in ('true', 'false'):
        raise ValueError('OGMEM_MCP_ENABLED must be true or false')
    data_root = Path(env.get('DATA_ROOT', './data')).expanduser()
    if not data_root.is_absolute():
        data_root = root / data_root
    stack = (data_root / project).resolve()
    if stack == root or root.is_relative_to(stack):
        raise ValueError('data directory cannot contain deployment files')
    state_file = stack / 'deployment-state.json'
    state = dict(deployment_root=str(root), project=project,
                 embedding={k: env.get(k, 'openai') for k in
                            ['EMBEDDING_PROVIDER', 'EMBEDDING_MODEL', 'EMBEDDING_DIMENSION']})
    if state_file.exists():
        previous = json.loads(state_file.read_text())
        if previous['deployment_root'] != str(root):
            raise ValueError('instance directory belongs to another deployment')
        if previous['embedding'] != state['embedding']:
            raise ValueError('Embedding space changed; create a new instance')
    elif stack.exists() and any(p.name != '.gitignore' for p in stack.iterdir()):
        raise ValueError('existing data directory has no ownership record')
    key_file = stack / 'secrets.env'
    if key_file.exists():
        keys = dict(line.split('=', 1) for line in key_file.read_text().splitlines() if '=' in line)
        if not all(keys.get(k) for k in ['OG_DB_PASSWORD', 'OG_ROOT_API_KEY', 'OG_TENANT_API_KEY']):
            raise ValueError('existing credentials are incomplete')
    else:
        # Only an interrupted first preparation with no published config/data may retry.
        if state_file.exists() and (
            (root / '.env.compose').exists()
            or any(p.name not in {'.gitignore', 'deployment-state.json'} for p in stack.iterdir())
        ):
            raise ValueError('existing instance credentials are missing; restore secrets.env')
        keys = dict(OG_DB_PASSWORD='OgM9_' + secrets.token_hex(12) + '!a',
                    OG_ROOT_API_KEY=secrets.token_hex(32), OG_TENANT_API_KEY=secrets.token_hex(32))
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'ogmemory.yaml').read_text())
    values = dict(env, **keys)
    values.update(LLM_PROVIDER=env.get('LLM_PROVIDER', 'openai'),
                  EMBEDDING_PROVIDER=env.get('EMBEDDING_PROVIDER', 'openai'),
                  OG_ACCOUNT_ID=env.get('OG_ACCOUNT_ID', project + '-account'),
                  OG_USER_ID=env.get('OG_USER_ID', project + '-user'),
                  OG_AGENT_ID=env.get('OG_AGENT_ID', project + '-api-agent'),
                  OGMEM_MCP_ENABLED=enabled,
                  OGMEM_MCP_ALLOWED_HOSTS=env.get('OGMEM_MCP_ALLOWED_HOSTS', 'localhost,127.0.0.1,ogmemory,host.docker.internal'),
                  OGMEM_AFTER_TURN_THRESHOLD=str(threshold))
    def expand(x):
        if isinstance(x, dict): return {k: expand(v) for k, v in x.items()}
        if isinstance(x, list): return [expand(v) for v in x]
        if isinstance(x, str): return re.sub(r'\$\{(\w+)\}', lambda m: values[m[1]], x)
        return x
    cfg = expand(cfg)
    cfg['vector_db']['dimension'] = dimension
    cfg['memory']['after_turn_threshold'] = threshold
    cfg['mcp']['enabled'] = enabled == 'true'
    cfg['mcp']['allowed_hosts'] = [s.strip() for s in values['OGMEM_MCP_ALLOWED_HOSTS'].split(',') if s.strip()]
    extra = env.get('LLM_EXTRA_BODY', '')
    if extra:
        parsed = json.loads(extra)
        if not isinstance(parsed, dict): raise ValueError('LLM_EXTRA_BODY must be a JSON object')
        # Fixed image may not implement this field; no claim that adding it changes provider behavior.
        cfg['llm']['extra_body'] = parsed
    # Credentials are hexadecimal/generated and DSN-safe. Model credentials are YAML scalars.
    cfg['vector_db']['connection_string'] += ' password=' + keys['OG_DB_PASSWORD']
    def dotenv(value):
        if '\n' in value or '\r' in value: raise ValueError('multiline deployment path is unsupported')
        return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"
    # Validate and serialize everything before creating instance files.
    config_text = yaml.safe_dump(cfg, allow_unicode=True)
    compose_text = ''.join(k + '=' + dotenv(v) + '\n' for k, v in dict(keys, STACK_DIR=str(stack)).items())
    state_text = json.dumps(state, ensure_ascii=False, indent=2) + '\n'
    caller = None
    if os.environ.get('DEPLOY_UID') is not None:
        caller = (int(os.environ['DEPLOY_UID']), int(os.environ['DEPLOY_GID']))
    def write_metadata(path, text):
        private_write(path, text)
        if caller is not None:os.chown(path, *caller)

    stack.mkdir(parents=True, exist_ok=True)
    stack.chmod(0o700)
    if caller is not None:os.chown(stack, *caller)
    # Protect runtime contents before credentials, including custom DATA_ROOT.
    write_metadata(stack / '.gitignore', '*\n')
    # Claim ownership first. An interrupted preparation can then resume safely.
    write_metadata(state_file, state_text)
    if not key_file.exists():
        write_metadata(key_file, ''.join(k + '=' + v + '\n' for k, v in keys.items()))
    key_file.chmod(0o600)
    for name in ['opengauss', 'agfs']:
        (stack / name).mkdir(exist_ok=True)
        (stack / name).chmod(0o777)
    private_write(stack / 'ogmemory.yaml', config_text)
    # Config remains private; ogmem must be able to read it when bind-mounted.
    try:
        import pwd
        os.chown(stack / 'ogmemory.yaml', pwd.getpwnam('ogmem').pw_uid, pwd.getpwnam('ogmem').pw_gid)
    except (KeyError, PermissionError):
        pass
    write_metadata(root / '.env.compose', compose_text)

    return dict(stack_dir=str(stack), keys=keys)


def parse_environment(text):
    model = json.loads(text)
    values = dict(model['x-deployment-inputs'])
    values['COMPOSE_PROJECT_NAME'] = model['name']
    if not all(isinstance(value, str) for value in values.values()):
        raise ValueError('deployment inputs must be strings')
    # Compose's rendered model escapes literal dollars for re-interpolation.
    return {key: value.replace('$$', '$') for key, value in values.items()}


def main():
    root = Path.cwd()
    try:
        env = parse_environment(sys.stdin.read())
        # The helper sees the canonical bind target, not external host symlinks.
        if os.environ.get("DEPLOY_DATA_ROOT"):
            env["DATA_ROOT"] = os.environ["DEPLOY_DATA_ROOT"]
        prepare(env, root)
        print(json.dumps({'configuration_ready': True}))
        return 0
    except Exception as exc:
        # Validation messages contain field names only, never values or DSNs.
        print(json.dumps({'configuration_ready': False, 'error_type': type(exc).__name__,
                          'hint': 'check required fields, project, ports, data ownership and Embedding configuration'}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
