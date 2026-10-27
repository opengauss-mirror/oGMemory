import importlib.util
import json
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_root_env_template_uses_the_buffer_default():
    values = dict(line.split('=', 1) for line in (ROOT / '.env.example').read_text().splitlines()
                  if line and not line.startswith('#') and '=' in line)
    from providers.unified_config import OgMemConfig
    assert int(values['OGMEM_AFTER_TURN_THRESHOLD']) == OgMemConfig().after_turn_threshold


@pytest.fixture(autouse=True)
def forbid_network_in_deployment_unit_tests(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('Deployment unit tests must not access the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


def module(name):
    path = ROOT / 'deploy/compose/scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_fake_docker(path, source):
    # Fake CLI returns the same structured contract as real Compose config.
    has_config = "'config'" in source
    source = source.replace("args[-2:] == ['config','--environment']", "args[-3:] == ['config','--format','json']")
    source = source.replace("args[-2:]==['config','--environment']", "args[-3:]==['config','--format','json']")
    shim = """
import builtins,json
_real_print = builtins.print
def print(*items, **kwargs):
    if args[-3:] == ['config','--format','json'] and items:
        text = str(items[0])
        try: json.loads(text)
        except ValueError:
            values = dict(line.split('=',1) for line in text.splitlines() if '=' in line)
            text = json.dumps({'name':values.get('COMPOSE_PROJECT_NAME','demo'), 'x-deployment-inputs':values})
        return _real_print(text, **kwargs)
    return _real_print(*items, **kwargs)
if args[:1] == ['create']:
    print('fake-project-lock');sys.exit(0)
"""
    if not has_config:
        shim += "if args[-3:] == ['config','--format','json']:\n print('COMPOSE_PROJECT_NAME=foreign');sys.exit(0)\n"
    source = source.replace('args=sys.argv[1:]', 'args=sys.argv[1:]\n'+shim, 1)
    path.write_text(source)

def config():
    return dict(COMPOSE_PROJECT_NAME='test-demo', DATA_ROOT='./data',
                LLM_BASE_URL='https://example.test/v1', LLM_MODEL='test-model',
                LLM_API_KEY='key-with-quote-"-and-$', EMBEDDING_PROVIDER='openai',
                EMBEDDING_BASE_URL='https://example.test/v1', EMBEDDING_MODEL='test-embed',
                EMBEDDING_API_KEY='embedding-key', EMBEDDING_DIMENSION='1024',
                OG_ACCOUNT_ID='test-account', OG_USER_ID='test-user', OG_AGENT_ID='test-agent',
                OGMEM_HTTP_HOST_PORT='18090', OGMEM_MCP_HOST_PORT='18091')


def test_compose_does_not_build_source():
    compose = yaml.safe_load((ROOT / 'compose.yaml').read_text())
    assert set(compose['services']) == {'opengauss', 'ogmemory'}
    assert all('build' not in x for x in compose['services'].values())
    assert 'ogmemory:1013' in compose['services']['ogmemory']['image']


def test_prepare_preserves_keys_and_special_character_values(tmp_path):
    prepare = module('prepare')
    first = prepare.prepare(config(), tmp_path)
    second = prepare.prepare(config(), tmp_path)
    assert first['keys'] == second['keys']
    assert first['stack_dir'] == str(tmp_path / 'data/test-demo')
    parsed = yaml.safe_load((Path(first['stack_dir']) / 'ogmemory.yaml').read_text())
    assert parsed['llm']['api_key'] == config()['LLM_API_KEY']
    assert parsed['mcp']['default_toolset'] == 'read-write'
    assert parsed['memory']['after_turn_threshold'] == 400
    assert (Path(first['stack_dir']) / 'secrets.env').stat().st_mode & 0o777 == 0o600


def test_prepare_rejects_changed_embedding_space(tmp_path):
    prepare = module('prepare')
    prepare.prepare(config(), tmp_path)
    changed = config(); changed['EMBEDDING_MODEL'] = 'another-model'
    with pytest.raises(ValueError, match='Embedding'):
        prepare.prepare(changed, tmp_path)


def test_prepare_rejects_overlapping_ports_and_unsafe_project(tmp_path):
    prepare = module('prepare')
    changed = config();changed['OGMEM_MCP_HOST_PORT'] = '18090'
    with pytest.raises(ValueError, match='port'):
        prepare.prepare(changed, tmp_path)
    changed = config();changed['COMPOSE_PROJECT_NAME'] = '../oops'
    with pytest.raises(ValueError, match='project'):
        prepare.prepare(changed, tmp_path)


def test_errors_never_include_credentials():
    client = module('runtime')
    def fail():raise ValueError('password=very-secret')
    assert client.run_checks({'database':fail}) == {'database':{'ok':False,'error_type':'ValueError'}}


def test_foreign_project_cannot_be_stopped_or_removed(tmp_path, monkeypatch):
    import os
    import shutil
    import subprocess
    root = tmp_path / 'deployment'
    (root / 'deploy/compose').mkdir(parents=True)
    shutil.copy(ROOT / 'deploy/compose/deploy.sh', root / 'deploy/compose/deploy.sh')
    (root / '.env').write_text('COMPOSE_PROJECT_NAME=foreign\n')
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    docker = bin_dir / 'docker'
    write_fake_docker(docker,'''#!/usr/bin/env python3
import sys
args=sys.argv[1:]
if args[:1] == ['ps']: print('foreign-container')
elif args[:1] == ['inspect']: print('/another/deployment')
elif args[:1] == ['compose'] and args[-1] in ['stop','down']: print('UNSAFE_ACTION');sys.exit(0)
''')
    docker.chmod(0o755)
    environment = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'])
    for action in ['stop', 'down', 'logs', 'status']:
        r = subprocess.run(['/bin/bash',str(root/'deploy/compose/deploy.sh'),action],capture_output=True,text=True,env=environment)
        assert r.returncode != 0
        assert 'UNSAFE_ACTION' not in r.stdout
        assert '另一个部署目录' in r.stderr


def test_existing_model_network_is_optional():
    path = ROOT / 'deploy/compose/compose.model-network.yaml'
    assert path.exists()
    value = yaml.safe_load(path.read_text())
    assert value['networks']['model']['external'] is True
    assert 'model' in value['services']['ogmemory']['networks']
    assert 'model' not in yaml.safe_load((ROOT/'compose.yaml').read_text())['networks']


def test_prepare_consumes_compose_resolved_environment(tmp_path, docker_compose_cli):
    import subprocess
    m = module('prepare')
    env_file = tmp_path / '.env'
    env_file.write_text('INSTANCE_NAME=demo\nCOMPOSE_PROJECT_NAME=${INSTANCE_NAME}\nBASE_DIR=./data\nDATA_ROOT=${BASE_DIR}\nLLM_API_KEY=\'literal-${UNCHANGED}-key\'\n')
    r = subprocess.run([docker_compose_cli,'compose','--env-file',str(env_file),'-f',str(ROOT/'compose.yaml'),'config','--format','json'],capture_output=True,text=True)
    assert r.returncode == 0
    parsed = m.parse_environment(r.stdout)
    assert parsed['COMPOSE_PROJECT_NAME'] == 'demo'
    assert parsed['DATA_ROOT'] == './data'
    assert parsed['LLM_API_KEY'] == 'literal-${UNCHANGED}-key'


def test_start_recreates_application_after_config_change(tmp_path):
    import os,shutil,subprocess
    root=tmp_path/'deployment';(root/'deploy/compose').mkdir(parents=True)
    shutil.copy(ROOT/'deploy/compose/deploy.sh',root/'deploy/compose/deploy.sh')
    (root/'.env').write_text('COMPOSE_PROJECT_NAME=demo\n')
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    events=tmp_path/'events.jsonl'
    d=bin_dir/'docker'
    write_fake_docker(d,'''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
with open(EVENT_PATH,'a') as f:f.write(json.dumps(args)+'\\n')
if args[-2:] == ['config','--environment']:print('COMPOSE_PROJECT_NAME=demo\\nDATA_ROOT=./data')
'''.replace('EVENT_PATH',repr(str(events))));d.chmod(0o755)
    listener=bin_dir/'lsof';listener.write_text('#!/bin/sh\nexit 1\n');listener.chmod(0o755)
    r=subprocess.run(['/bin/bash',str(root/'deploy/compose/deploy.sh'),'start'],capture_output=True,text=True,
                     env=dict(os.environ,EVENTS=str(events),PATH=str(bin_dir)+os.pathsep+os.environ['PATH']))
    assert r.returncode==0,r.stderr
    commands=[json.loads(x) for x in events.read_text().splitlines()]
    assert any('up' in x and '--force-recreate' in x and '--no-deps' in x and x[-1]=='ogmemory' for x in commands)


def test_log_redactor_masks_database_and_service_keys(tmp_path):
    redactor = module('redact')
    cfg = tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump({'llm':{'api_key':'model-secret'},'auth':{'root_api_key':'root-secret','admin_api_keys':['account:tenant-secret']},'vector_db':{'connection_string':'host=opengauss password=database-secret'}}))
    result = redactor.redact('password=database-secret token=tenant-secret key=model-secret root=root-secret',cfg)
    assert all(x not in result for x in ['database-secret','tenant-secret','model-secret','root-secret'])


def test_deployment_environment_uses_private_dotenv_not_shell(tmp_path, docker_compose_cli):
    import os, shutil, subprocess
    real_docker = docker_compose_cli
    root = tmp_path/'deployment';(root/'deploy/compose').mkdir(parents=True)
    shutil.copy(ROOT/'compose.yaml',root/'compose.yaml')
    shutil.copy(ROOT/'deploy/compose/deploy.sh',root/'deploy/compose/deploy.sh')
    (root/'.env').write_text('COMPOSE_PROJECT_NAME=private-demo\nLLM_MODEL=file-model\n'
        'OPENAI_API_KEY=file-key\nLLM_API_KEY=${OPENAI_API_KEY}\n'
        'EMBEDDING_MODEL=file-embedding\nOG_USER_ID=file-user\n'
        'CUSTOM_DEPLOYMENT_VALUE=file-custom\nEMBEDDING_API_KEY=${CUSTOM_DEPLOYMENT_VALUE}\n'
        'HTTP_PROXY=http://private-proxy:3128\nHTTPS_PROXY=http://private-proxy:3128\nNO_PROXY=local-model\n')
    captured=tmp_path/'resolved.json';bin_dir=tmp_path/'bin';bin_dir.mkdir()
    names=['LLM_MODEL','LLM_API_KEY','EMBEDDING_MODEL','EMBEDDING_API_KEY','OG_USER_ID','HTTP_PROXY','HTTPS_PROXY','NO_PROXY']
    docker=bin_dir/'docker'
    write_fake_docker(docker,'#!/usr/bin/env python3\nimport json,subprocess,sys\n'
        "args=sys.argv[1:]\nif args[:1]==['compose'] and args[-2:]==['config','--environment']:\n"
        f' r=subprocess.run([{real_docker!r}]+args,capture_output=True,text=True,check=True)\n'
        " values=json.loads(r.stdout)['x-deployment-inputs']\n"
        f' with open({str(captured)!r},"w") as f:json.dump({{k:values.get(k) for k in {names!r}}},f)\n'
        ' print(r.stdout,end="")\n')
    docker.chmod(0o755)
    env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],
             LLM_MODEL='shell-model',OPENAI_API_KEY='shell-key',EMBEDDING_MODEL='shell-embedding',
             OG_USER_ID='shell-user',CUSTOM_DEPLOYMENT_VALUE='shell-custom',HTTP_PROXY='http://shell-proxy:9999',
             HTTPS_PROXY='http://shell-proxy:9999',NO_PROXY='shell-bypass')
    result=subprocess.run(['/bin/bash',str(root/'deploy/compose/deploy.sh'),'status'],
                          capture_output=True,text=True,env=env)
    assert result.returncode==0,result.stderr
    assert json.loads(captured.read_text())==dict(LLM_MODEL='file-model',LLM_API_KEY='file-key',
        EMBEDDING_MODEL='file-embedding',EMBEDDING_API_KEY='file-custom',OG_USER_ID='file-user',
        HTTP_PROXY='http://private-proxy:3128',HTTPS_PROXY='http://private-proxy:3128',NO_PROXY='local-model')


def test_invalid_optional_config_leaves_no_instance_and_can_retry(tmp_path):
    m=module('prepare');env=config();env['LLM_EXTRA_BODY']='{'
    with pytest.raises(ValueError):m.prepare(env,tmp_path)
    assert not (tmp_path/'data/test-demo').exists()
    env['LLM_EXTRA_BODY']='{}'
    prepared=m.prepare(env,tmp_path)
    assert (Path(prepared['stack_dir'])/'deployment-state.json').exists()


def test_prepare_can_retry_interrupted_write_without_changing_credentials(tmp_path,monkeypatch):
    m=module('prepare');original=m.private_write
    def interrupted(path,text):
        if Path(path).name=='.env.compose':raise OSError('simulated write interruption')
        original(path,text)
    monkeypatch.setattr(m,'private_write',interrupted)
    with pytest.raises(OSError):m.prepare(config(),tmp_path)
    key_file=tmp_path/'data/test-demo/secrets.env';before=key_file.read_text()
    monkeypatch.setattr(m,'private_write',original)
    m.prepare(config(),tmp_path)
    assert key_file.read_text()==before
    assert (tmp_path/'.env.compose').exists()


@pytest.mark.parametrize('mode', ['old-container','old-volume','changed-bind','quickstart','managed','changed-data-root','changed-data-root-down'])
def test_start_blocks_legacy_storage_before_any_mutation(tmp_path, mode):
    import os, shutil, subprocess
    root=tmp_path/'deployment';(root/'deploy/compose').mkdir(parents=True)
    shutil.copy(ROOT/'deploy/compose/deploy.sh',root/'deploy/compose/deploy.sh')
    (root/'.env').write_text('COMPOSE_PROJECT_NAME=demo\n')
    (root/'data').mkdir()
    events=tmp_path/'events';bin_dir=tmp_path/'bin';bin_dir.mkdir()
    fake=bin_dir/'docker'
    write_fake_docker(fake,'''#!/usr/bin/env python3
import sys
ROOT=ROOT_VALUE
args=sys.argv[1:]
with open(EVENTS,'a') as f:f.write(' '.join(args)+'\\n')
if args[-2:]==['config','--environment']:print('COMPOSE_PROJECT_NAME=demo\\nSTACK_DIR='+ROOT+'/data/demo'+('\\nDATA_ROOT='+ROOT+'/new-data' if MODE in ('changed-data-root','changed-data-root-down') else '') if MODE in ('managed','changed-data-root','changed-data-root-down') else 'COMPOSE_PROJECT_NAME=demo')
elif args[:2]==['volume','ls'] and MODE=='old-volume':print('demo_opengauss-data')
elif args[:1]==['ps'] and '-aq' in args:
 if MODE in ('old-container','changed-bind','managed','changed-data-root') and 'label=com.docker.compose.project=demo' in args:print('existing-container')
 elif MODE=='quickstart' and any(x.startswith('name=') for x in args):print('quickstart-container')
elif args[:1]==['inspect']:
 fmt=args[-1]
 if 'working_dir' in fmt:print(ROOT)
 elif 'Mounts' in fmt:
  print(('volume|demo_opengauss-data' if MODE=='old-container' else ('bind|'+ROOT+'/data/demo/opengauss' if MODE in ('managed','changed-data-root') else 'bind|/old/storage'))+'|/var/lib/opengauss')
elif args[:1]==['compose'] and 'pull' in args:print('UNSAFE_PULL')
'''.replace('EVENTS',repr(str(events))).replace('MODE',repr(mode)).replace('ROOT_VALUE',repr(str(root))))
    fake.chmod(0o755)
    listener=bin_dir/'lsof';listener.write_text('#!/bin/sh\nexit 1\n');listener.chmod(0o755)
    r=subprocess.run(['bash',str(root/'deploy/compose/deploy.sh'),'start'],capture_output=True,text=True,
        env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH']))
    if mode=='managed':
        assert r.returncode==0,r.stderr
        assert ' up ' in events.read_text()
        return
    assert r.returncode!=0
    assert '旧存储' in r.stderr
    assert ' pull' not in events.read_text() and ' up ' not in events.read_text()
    assert not (root/'data'/'demo').exists()


def test_application_proxies_and_internal_bypass_are_configurable():
    env=yaml.safe_load((ROOT/'compose.yaml').read_text())['services']['ogmemory']['environment']
    assert 'HTTP_PROXY' in env and 'HTTPS_PROXY' in env
    assert '${NO_PROXY' in env['NO_PROXY']
    assert all(x in env['NO_PROXY'] for x in ['localhost','127.0.0.1','opengauss','ogmemory'])


@pytest.mark.parametrize('available', ['missing-docker','missing-compose'])
def test_compose_cli_gate_skips_missing_tools(monkeypatch,available):
    import importlib.util,shutil,subprocess
    spec=importlib.util.spec_from_file_location('deployment_conftest',Path(__file__).with_name('conftest.py'))
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    monkeypatch.setattr(shutil,'which',lambda name: None if available=='missing-docker' else '/fake/docker')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw: subprocess.CompletedProcess(a,1))
    with pytest.raises(pytest.skip.Exception):gate.require_docker_compose()


def test_proxy_credentials_are_redacted_from_logs(tmp_path,monkeypatch):
    m=module('redact');p=tmp_path/'config.yaml';p.write_text('{}')
    proxy='http://user:proxy-password@proxy.example:3128'
    monkeypatch.setenv('DEPLOY_HTTP_PROXY',proxy)
    redacted=m.redact(proxy+' auth=proxy-password',p)
    assert proxy not in redacted and 'proxy-password' not in redacted


@pytest.mark.parametrize('action', ['start', 'logs'])
@pytest.mark.parametrize('location', ['relative', 'absolute', 'symlink', 'separate'])
def test_helper_mount_targets_are_unique(tmp_path, action, location):
    import os, shutil, subprocess
    root=tmp_path/'deployment';(root/'deploy/compose').mkdir(parents=True)
    shutil.copy(ROOT/'deploy/compose/deploy.sh',root/'deploy/compose/deploy.sh')
    alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True)
    data={'relative':'.','absolute':str(root),'symlink':str(alias),'separate':str(tmp_path/'data')}[location]
    if location=='separate':Path(data).mkdir()
    (root/'.env').write_text('COMPOSE_PROJECT_NAME=demo\n')
    if action=='logs':(root/'.env.compose').write_text('')
    events=tmp_path/'events';bin_dir=tmp_path/'bin';bin_dir.mkdir()
    fake=bin_dir/'docker'
    write_fake_docker(fake,'''#!/usr/bin/env python3
import json,sys
args=sys.argv[1:]
with open(EVENTS,'a') as f:f.write(json.dumps(args)+'\\n')
if args[-2:]==['config','--environment']:print('COMPOSE_PROJECT_NAME=demo\\nDATA_ROOT='+DATA_VALUE)
'''.replace('EVENTS',repr(str(events))).replace('DATA_VALUE',repr(data)))
    fake.chmod(0o755)
    listener=bin_dir/'lsof';listener.write_text('#!/bin/sh\nexit 1\n');listener.chmod(0o755)
    r=subprocess.run(['/bin/bash',str(root/'deploy/compose/deploy.sh'),action],capture_output=True,text=True,
        env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH']))
    assert r.returncode==0,r.stderr
    runs=[args for args in map(json.loads,events.read_text().splitlines()) if args[0]=='run']
    assert len(runs)==1
    mounts=[runs[0][i+1] for i,v in enumerate(runs[0]) if v=='-v']
    targets=[v.split(':')[1] for v in mounts]
    assert len(targets)==len(set(targets))
    assert set(targets)==({str(root.resolve()),str(Path(data).resolve())} if location=='separate' else {str(root.resolve())})


def test_external_symlink_preparation_persists_in_mounted_real_directory(tmp_path):
    import os, shutil, subprocess, sys
    root=tmp_path/'deployment';shutil.copytree(ROOT/'deploy/compose',root/'deploy/compose')
    real=tmp_path/'real-data';real.mkdir()
    alias=tmp_path/'outside-link';alias.symlink_to(real,target_is_directory=True)
    inputs=config();inputs.update(DATA_ROOT=str(alias),COMPOSE_PROJECT_NAME='demo')
    (root/'.env').write_text('COMPOSE_PROJECT_NAME=demo\n')
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    fake=bin_dir/'docker'
    write_fake_docker(fake,'''#!PYTHON
import json,os,subprocess,sys
args=sys.argv[1:]
if args[-2:]==['config','--environment']:
 print('\\n'.join(k+'='+v for k,v in INPUTS_VALUE.items()))
elif args[0]=='run':
 env=dict(os.environ)
 for i,value in enumerate(args):
  if value=='-e':
   key,_,val=args[i+1].partition('=');env[key]=val
 # Docker only exposes the bind targets. The original external symlink is unavailable.
 mounts=[args[i+1].split(':')[1] for i,value in enumerate(args) if value=='-v']
 assert str(REAL_VALUE) in mounts
 assert env.get('DEPLOY_DATA_ROOT')==str(REAL_VALUE), 'preparation input still uses inaccessible symlink'
 os.unlink(INPUTS_VALUE['DATA_ROOT'])  # external symlink is absent inside the helper
 result=subprocess.run([sys.executable,args[-1]],input=sys.stdin.read(),text=True,cwd=ROOT_VALUE,env=env)
 sys.exit(result.returncode)
'''.replace('PYTHON',sys.executable).replace('INPUTS_VALUE',repr(inputs)).replace('REAL_VALUE',repr(str(real.resolve()))).replace('ROOT_VALUE',repr(str(root.resolve()))))
    fake.chmod(0o755)
    listener=bin_dir/'lsof';listener.write_text('#!/bin/sh\nexit 1\n');listener.chmod(0o755)
    result=subprocess.run(['/bin/bash',str(root/'deploy/compose/deploy.sh'),'start'],capture_output=True,text=True,
        env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH']))
    assert result.returncode==0,result.stderr
    stack=real/'demo'
    assert (stack/'ogmemory.yaml').is_file() and (stack/'secrets.env').is_file()
    assert (stack/'opengauss').is_dir() and (stack/'agfs').is_dir()
    generated=(root/'.env.compose').read_text()
    assert str(stack.resolve()) in generated and str(alias) not in generated


def test_concurrent_preparation_reuses_one_credential_set(tmp_path):
    import subprocess,sys
    inputs=config()
    script='''import importlib.util,json,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('prepare',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original=m.private_write
def delayed(path,text):
 original(path,text)
 if Path(path).name=='deployment-state.json':time.sleep(.3)
m.private_write=delayed
result=m.prepare(json.loads(sys.argv[3]),Path(sys.argv[2]))
print(json.dumps(result['keys']))
'''
    args=[sys.executable,'-c',script,str(ROOT/'deploy/compose/scripts/prepare.py'),str(tmp_path),json.dumps(inputs)]
    first=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    import time
    time.sleep(.1)
    second=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    a,err_a=first.communicate(timeout=15);b,err_b=second.communicate(timeout=15)
    assert first.returncode==second.returncode==0,(err_a,err_b)
    keys=json.loads(a);assert keys==json.loads(b)
    stack=tmp_path/'data/test-demo'
    saved=dict(line.split('=',1) for line in (stack/'secrets.env').read_text().splitlines())
    assert keys==saved
    cfg=yaml.safe_load((stack/'ogmemory.yaml').read_text())
    assert cfg['auth']['root_api_key']==saved['OG_ROOT_API_KEY']
    assert 'password='+saved['OG_DB_PASSWORD'] in cfg['vector_db']['connection_string']
    for value in saved.values():assert value in (tmp_path/'.env.compose').read_text()


def test_concurrent_start_holds_lock_through_service_initialization(tmp_path):
    import os,shutil,subprocess
    root=tmp_path/'deployment';(root/'deploy/compose').mkdir(parents=True)
    shutil.copy(ROOT/'deploy/compose/deploy.sh',root/'deploy/compose/deploy.sh')
    (root/'.env').write_text('COMPOSE_PROJECT_NAME=demo\n')
    events=tmp_path/'events';bin_dir=tmp_path/'bin';bin_dir.mkdir()
    fake=bin_dir/'docker'
    write_fake_docker(fake,'''#!/usr/bin/env python3
import sys,time
args=sys.argv[1:]
if args[-2:]==['config','--environment']:
 with open(EVENTS,'a') as f:f.write('config\\n')
 time.sleep(.3)
 print('COMPOSE_PROJECT_NAME=demo')
elif args[-1:] == ['connection']:
 with open(EVENTS,'a') as f:f.write('ready\\n')
'''.replace('EVENTS',repr(str(events))))
    fake.chmod(0o755)
    listener=bin_dir/'lsof';listener.write_text('#!/bin/sh\nexit 1\n');listener.chmod(0o755)
    env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
    args=['/bin/bash',str(root/'deploy/compose/deploy.sh'),'start']
    processes=[subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    for process in processes:
        out,err=process.communicate(timeout=20)
        assert process.returncode==0,err
    assert events.read_text().splitlines()==['config','ready','config','ready']


def test_experiment_can_override_default_extraction_threshold(tmp_path):
    m=module('prepare');env=config();env['OGMEM_AFTER_TURN_THRESHOLD']='1'
    result=m.prepare(env,tmp_path)
    cfg=yaml.safe_load((Path(result['stack_dir'])/'ogmemory.yaml').read_text())
    assert cfg['memory']['after_turn_threshold']==1


@pytest.mark.parametrize('data_root', ['./storage', '.'])
def test_custom_instance_runtime_is_git_ignored(tmp_path, data_root):
    import subprocess
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    env=config();env['DATA_ROOT']=data_root
    result=module('prepare').prepare(env,tmp_path)
    stack=Path(result['stack_dir'])
    for file in ['secrets.env','ogmemory.yaml','opengauss/probe','agfs/probe']:
        target=stack/file
        if not target.exists():target.write_text('test runtime')
        result=subprocess.run(['git','-C',str(tmp_path),'check-ignore',str(target)],capture_output=True,text=True)
        assert result.returncode==0,file


def test_compose_structured_inputs_preserve_multiline_json(tmp_path, docker_compose_cli):
    import subprocess
    extra={'thinking':{'type':'disabled'}}
    env=config();env['LLM_EXTRA_BODY']=json.dumps(extra,indent=2)
    path=tmp_path/'.env';path.write_text(''.join(k+"='"+v+"'\n" for k,v in env.items()))
    result=subprocess.run([docker_compose_cli,'compose','--env-file',str(path),'-f',str(ROOT/'compose.yaml'),'config','--format','json'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    parsed=module('prepare').parse_environment(result.stdout)
    assert json.loads(parsed['LLM_EXTRA_BODY'])==extra
    prepared=module('prepare').prepare(parsed,tmp_path)
    cfg=yaml.safe_load((Path(prepared['stack_dir'])/'ogmemory.yaml').read_text())
    assert cfg['llm']['extra_body']==extra


def test_same_daemon_project_cannot_be_started_from_two_directories(tmp_path):
    import os,shutil,subprocess
    daemon=tmp_path/'daemon';daemon.mkdir()
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    fake=bin_dir/'docker'
    source='''#!/usr/bin/env python3
import json,os,sys,time
from pathlib import Path
args=sys.argv[1:];daemon=Path(DAEMON);owner=daemon/'owner'
if args[-3:]==['config','--format','json']:
 print(json.dumps({'name':'shared','x-deployment-inputs':{}}))
elif args[:1]==['create']:
 claim=daemon/args[args.index('--name')+1]
 try:fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError:sys.exit(1)
 os.close(fd);print(claim.name)
elif args[:1]==['rm']:
 (daemon/args[1]).unlink()
elif args[:1]==['ps'] and '-aq' in args:
 existed=owner.exists();time.sleep(.1)
 if existed:print('app')
elif args[:1]==['inspect']:
 if owner.exists():print(owner.read_text())
elif args[:1]==['compose'] and 'up' in args:
 if owner.exists() and owner.read_text()!=os.getcwd():
  (daemon/'hijacked').touch()
 owner.write_text(os.getcwd());time.sleep(.2)
'''.replace('DAEMON',repr(str(daemon)))
    Path.write_text(fake,source);fake.chmod(0o755)
    listener=bin_dir/'lsof';listener.write_text('#!/bin/sh\nexit 1\n');listener.chmod(0o755)
    roots=[tmp_path/'first',tmp_path/'second']
    for root in roots:
        (root/'deploy/compose').mkdir(parents=True)
        shutil.copy(ROOT/'deploy/compose/deploy.sh',root/'deploy/compose/deploy.sh')
        (root/'.env').write_text('COMPOSE_PROJECT_NAME=shared\n')
    env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
    processes=[subprocess.Popen(['/bin/bash',str(root/'deploy/compose/deploy.sh'),'start'],env=env,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for root in roots]
    outputs=[p.communicate(timeout=30) for p in processes]
    assert sorted(p.returncode for p in processes)==[0,1],outputs
    assert not (daemon/'hijacked').exists()
    assert not list(daemon.glob('ogmemory-deploy-guard-*'))
    assert any('占用' in err or '另一个部署目录' in err for out,err in outputs), outputs


def test_start_only_initializes_identity_and_reports_connection():
    script=(ROOT/'deploy/compose/deploy.sh').read_text()
    assert '/opt/ogmem-deploy/runtime.py bootstrap' in script
    assert 'verify|recall)' not in script
    assert '/opt/ogmem-deploy/client.py' not in script


def test_runtime_bootstrap_is_idempotent_without_memory_writes(monkeypatch):
    m=module('runtime');calls=[]
    cfg={'identity':{'account_id':'test-account','user_id':'test-user','agent_id':'test-agent'}}
    def call(cfg,method,route,payload=None,**kwargs):
        calls.append((method,route))
        if route.endswith('/users'):return {'users':[{'user_id':'test-user'}]}
        return {'agents':[{'agent_id':'test-agent','owner_user_id':'test-user'}]}
    monkeypatch.setattr(m,'call',call)
    assert m.bootstrap(cfg)=={'identity_ready':True}
    assert all(method=='GET' for method,route in calls)


def test_missing_persisted_credentials_never_rewrites_existing_configuration(tmp_path):
    m=module('prepare');result=m.prepare(config(),tmp_path)
    stack=Path(result['stack_dir']);(stack/'opengauss/PG_VERSION').write_text('initialized')
    before={p:p.read_bytes() for p in [stack/'ogmemory.yaml',tmp_path/'.env.compose',stack/'deployment-state.json']}
    (stack/'secrets.env').unlink()
    with pytest.raises(ValueError,match='credentials'):
        m.prepare(config(),tmp_path)
    assert not (stack/'secrets.env').exists()
    assert all(p.read_bytes()==value for p,value in before.items())


@pytest.mark.parametrize('action',['stop','down'])
@pytest.mark.parametrize('existing',[True,False])
def test_stop_and_down_use_existing_image_without_pulling_target(tmp_path,action,existing):
    import os,shutil,subprocess
    root=tmp_path/'deployment';(root/'deploy/compose').mkdir(parents=True)
    shutil.copy(ROOT/'deploy/compose/deploy.sh',root/'deploy/compose/deploy.sh')
    (root/'.env').write_text('COMPOSE_PROJECT_NAME=demo\n')
    bin_dir=tmp_path/'bin';bin_dir.mkdir();events=tmp_path/'events'
    fake=bin_dir/'docker'
    source='''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
with open(EVENTS,'a') as f:f.write(json.dumps(args)+'\\n')
if args[-3:]==['config','--format','json']:
 print(json.dumps({'name':'demo','x-deployment-inputs':{'OGMEM_IMAGE':'invalid.example/not-present:broken'}}))
elif args[:2]==['image','inspect']:
 sys.exit(1 if args[-1].startswith('invalid.') else 0)
elif args[:1]==['pull']:sys.exit(1)
elif args[:1]==['ps'] and '-aq' in args:
 if EXISTING:print('existing')
elif args[:2]==['image','ls']:print('sha256:cached-original')
elif args[:1]==['inspect']:
 print('sha256:cached-original' if args[-1]=='{{.Image}}' else ROOT)
elif args[:1]==['create']:
 assert args[-1]=='sha256:cached-original'
 assert '--platform' not in args
 print('owned-lock')
'''.replace('EVENTS',repr(str(events))).replace('ROOT',repr(str(root.resolve()))).replace('EXISTING',repr(existing))
    Path.write_text(fake,source);fake.chmod(0o755)
    r=subprocess.run(['/bin/bash',str(root/'deploy/compose/deploy.sh'),action],capture_output=True,text=True,
        env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH']))
    assert r.returncode==0,r.stderr
    commands=[json.loads(line) for line in events.read_text().splitlines()]
    assert not any(command[0]=='pull' for command in commands)
    assert any(command[0]=='compose' and command[-1]==action for command in commands)


def test_prepare_can_retry_before_first_credentials_are_published(tmp_path,monkeypatch):
    m=module('prepare');original=m.private_write
    def interrupted(path,text):
        if Path(path).name=='secrets.env':raise OSError('interrupted')
        return original(path,text)
    monkeypatch.setattr(m,'private_write',interrupted)
    with pytest.raises(OSError,match='interrupted'):m.prepare(config(),tmp_path)
    monkeypatch.setattr(m,'private_write',original)
    result=m.prepare(config(),tmp_path)
    assert (Path(result['stack_dir'])/'secrets.env').is_file()
    assert (tmp_path/'.env.compose').is_file()
