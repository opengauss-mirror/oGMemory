"""Command-line client for the normal service's ROOT migration API."""
import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit


def add_arguments(parser):
    parser.add_argument('--base-url', default=None, help='Service URL (or OGMEM_HTTP_BASE_URL)')
    parser.add_argument('--api-key-env', default='OGMEM_ROOT_API_KEY', help='Environment variable holding the ROOT key')
    parser.add_argument('--request-id', type=str, help='Stable UUID for a submission retry')
    parser.add_argument('--no-wait', action='store_true', help='Return after submitting a task')
    parser.add_argument('--wait-timeout', type=float, default=3600, help='Maximum polling seconds; does not cancel the task')
    actions = parser.add_subparsers(dest='migration_action', required=True)
    export = actions.add_parser('export', help='Export a snapshot and download it')
    export.add_argument('--output', required=True)
    preview = actions.add_parser('preflight', help='Upload a package and inspect the merge')
    preview.add_argument('package')
    merge = actions.add_parser('import', help='Import a successfully inspected package')
    merge.add_argument('--preflight-id', required=True)
    merge.add_argument('--yes', action='store_true', help='Confirm pausing target business reads and writes')
    status = actions.add_parser('status', help='Inspect the instance or an existing task')
    status.add_argument('job_id', nargs='?')
    status.add_argument('--wait', action='store_true')
    status.add_argument('--output', help='Download a completed export without starting another task')
    recover = actions.add_parser('recover', help='Recover an interrupted import using its backup')
    recover.add_argument('job_id')
    recover.add_argument('--yes', action='store_true', help='Confirm restoring the import backup')


class MigrationClient:
    def __init__(self, client):
        self.client = client

    def request(self, method, path, *, missing_ok=False, **kwargs):
        response = self.client.request(method, path, **kwargs)
        if response.status_code == 404 and missing_ok:
            return None
        if not response.is_success:
            try:
                reason = response.json().get('error', 'request rejected')
            except (ValueError, AttributeError):
                reason = 'request rejected'
            raise ValueError(f'HTTP {response.status_code}: {reason}')
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError('migration API returned a non-object response')
        return value

    def job(self, identity, *, missing_ok=False):
        return self.request('GET', 'jobs/' + str(uuid.UUID(identity)), missing_ok=missing_ok)

    def wait(self, identity, timeout):
        deadline, last = time.monotonic() + timeout, None
        while True:
            job = self.job(identity)
            progress = (job.get('status'), job.get('phase'), job.get('index_records'))
            if progress != last:
                print(f"{identity}: {job.get('phase')} ({job.get('status')})", file=sys.stderr)
                last = progress
            if job['status'] != 'running':
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f'task {identity} is still running; use migration status {identity}')
            time.sleep(1)

    def submit(self, kind, identity, **fields):
        # Print before sending: a lost response must not hide the task identity.
        print(f'task_id={identity}', file=sys.stderr, flush=True)
        return self.request('POST', 'jobs', json={'kind': kind, 'request_id': identity, **fields})

    def download(self, job, output):
        if job.get('kind') != 'export' or job.get('status') != 'succeeded':
            raise ValueError('only a completed export can be downloaded')
        destination = Path(output).expanduser()
        report = job['result']
        identity = str(uuid.UUID(job['id']))
        size, digest = 0, hashlib.sha256()
        fd, temporary = tempfile.mkstemp(prefix='.ogmem-download-', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as out, self.client.stream('GET', f'jobs/{identity}/download') as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > report['archive_bytes']:
                        raise ValueError('download exceeds declared size')
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            if size != report['archive_bytes'] or digest.hexdigest() != report['archive_sha256']:
                raise ValueError('download size or checksum differs from export report')
            os.link(temporary, destination)  # Never overwrite an existing file.
        finally:
            Path(temporary).unlink(missing_ok=True)


def execute(args, api):
    action = args.migration_action
    if getattr(args, 'output', None) and Path(args.output).expanduser().exists():
        raise ValueError('output file already exists')
    if action == 'status':
        if not args.job_id:
            if args.output or args.wait:
                raise ValueError('--wait and --output require a job ID')
            return api.request('GET', 'status')
        job = api.wait(args.job_id, args.wait_timeout) if args.wait else api.job(args.job_id)
    else:
        if action in {'import', 'recover'} and not args.yes:
            raise ValueError('use --yes to confirm target maintenance and data changes')
        identity = str(uuid.UUID(args.request_id)) if args.request_id else str(uuid.uuid4())
        fields = {}
        if action == 'preflight':
            if args.request_id:
                previous = api.job(identity, missing_ok=True)
                if previous:
                    if previous['kind'] != 'preflight':
                        raise ValueError('request ID belongs to another operation')
                    if previous['status'] == 'running':
                        if args.no_wait:
                            raise ValueError(f'task {identity} exists; use migration status {identity}')
                        previous = api.wait(identity, args.wait_timeout)
                    if previous['status'] != 'succeeded':
                        raise ValueError(f'previous preflight {identity} failed; submit a new preflight')
                    with Path(args.package).expanduser().open('rb') as source:
                        digest = hashlib.file_digest(source, 'sha256').hexdigest()
                    if previous['result'].get('archive_sha256') != digest:
                        raise ValueError('package differs from the original preflight request')
                    return previous
            with Path(args.package).expanduser().open('rb') as source:
                uploaded = api.request('POST', 'uploads', headers={'Content-Type': 'application/octet-stream',
                                                                   'Content-Length': str(os.fstat(source.fileno()).st_size)},
                                       content=iter(lambda: source.read(1024 * 1024), b''))
            fields['artifact_id'] = uploaded['artifact_id']
        elif action == 'import':
            preview = api.job(args.preflight_id)
            if preview['kind'] != 'preflight' or preview['status'] != 'succeeded':
                raise ValueError('a successful preflight is required')
            fields.update(preflight_id=preview['id'], artifact_id=preview['artifact_id'], maintenance_confirmed=True)
        elif action == 'recover':
            fields['recover_task'] = str(uuid.UUID(args.job_id))
        submitted = api.submit(action, identity, **fields)
        if args.no_wait:
            return submitted
        job = api.wait(submitted['job_id'], args.wait_timeout)
    if getattr(args, 'output', None) and job.get('status') == 'succeeded':
        api.download(job, args.output)
    return job


def run(args):
    import httpx

    try:
        if not 0 < args.wait_timeout < float('inf'):
            raise ValueError('--wait-timeout must be finite and positive')
        base = args.base_url or os.environ.get('OGMEM_HTTP_BASE_URL') or os.environ.get('OGMEM_SERVER_URL') or 'http://127.0.0.1:8090'
        url = urlsplit(base)
        if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('use an HTTP(S) service URL without credentials, query or fragment')
        key = os.environ.get(args.api_key_env)
        if not key:
            raise ValueError(f'ROOT key environment variable {args.api_key_env} is not set')
        with httpx.Client(base_url=base.rstrip('/') + '/api/v1/migration/',
                          headers={'X-API-Key': key}, timeout=30, follow_redirects=False) as client:
            result = execute(args, MigrationClient(client))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get('status') == 'failed' else 2 if result.get('status') == 'running' else 0
    except (ValueError, OSError, httpx.HTTPError) as exc:
        print(f'ogmem migration: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Stopped waiting; the server task continues. Use migration status with the printed task ID.', file=sys.stderr)
        return 130
