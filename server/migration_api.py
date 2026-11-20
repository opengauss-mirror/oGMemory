"""ROOT-only migration routes on the normal Memory Viewer server."""
import hmac
from uuid import UUID

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.exceptions import (
    BadRequest,
    HTTPException,
    RequestEntityTooLarge,
    UnsupportedMediaType,
)
from werkzeug.wsgi import get_input_stream

from server.auth import AuthenticationError, AuthService
from service.migration.instance import InstanceMigration
from service.migration.jobs import MAX_UPLOAD_BYTES, InstanceJobs
from service.migration.maintenance import MaintenanceError
from service.migration.package import PackageError


def create_migration_blueprint(get_service):
    api = Blueprint('instance_migration', __name__, url_prefix='/api/v1/migration')

    def authorize(headers=None):
        cfg = get_service()._cfg
        headers = dict(request.headers) if headers is None else headers
        key = AuthService._extract_api_key({str(k).lower(): v for k, v in headers.items()})
        if not key:
            raise AuthenticationError('请使用 ROOT 管理员身份连接。')
        # ROOT is the configured instance key in AuthService. Resolving it does
        # not require bootstrapping SQL business tables during maintenance.
        if (not cfg.role_control_enabled or not cfg.root_api_key
                or not hmac.compare_digest(key.encode(), cfg.root_api_key.encode())):
            raise PermissionError('记忆迁移仅允许已认证的 ROOT 管理员操作。')

    def jobs():
        service = get_service()
        return InstanceJobs(InstanceMigration(service, asset_root=getattr(service._cfg, 'migration_asset_root', None)))

    @api.before_request
    def check_identity():
        authorize()

    @api.errorhandler(AuthenticationError)
    def authentication_error(exc):
        return jsonify(error=str(exc)), 401

    @api.errorhandler(PermissionError)
    def permission_error(exc):
        return jsonify(error=str(exc)), 403

    @api.errorhandler(MaintenanceError)
    def maintenance_error(exc):
        return jsonify(error=str(exc)), 409

    @api.errorhandler(PackageError)
    @api.errorhandler(ValueError)
    def bad_request(exc):
        return jsonify(error=str(exc)), 400

    @api.errorhandler(FileNotFoundError)
    def not_found(exc):
        return jsonify(error=str(exc)), 404

    @api.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description), exc.code

    def public_state(state):
        return {key: state[key] for key in ('id', 'mode', 'phase', 'modified')} if state else None

    @api.get('/status')
    def status():
        queue = jobs()
        cfg = queue.migration.cfg
        recent = queue.recent()
        return jsonify(mode='online', storage=cfg.storage_backend,
                       vector='none' if getattr(cfg, 'agfs_retrieval_enabled', False) else cfg.vector_db_type,
                       maintenance=public_state(queue.gate.status()), jobs=recent)

    @api.post('/uploads')
    def upload():
        if request.mimetype != 'application/octet-stream':
            raise PackageError('请上传迁移包文件。')
        stream = get_input_stream(request.environ, max_content_length=MAX_UPLOAD_BYTES)
        return jsonify(jobs().upload(stream)), 201

    @api.post('/jobs')
    def submit():
        if not request.is_json:
            raise UnsupportedMediaType()
        raw = request.stream.read(16 * 1024 + 1)
        if len(raw) > 16 * 1024:
            raise RequestEntityTooLarge()
        try:
            body = current_app.json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise BadRequest('invalid migration control JSON') from exc
        if not isinstance(body, dict):
            raise PackageError('invalid migration request')
        kind = body.get('kind')
        if not isinstance(kind, str) or kind not in {'export', 'preflight', 'import', 'recover'}:
            raise PackageError('unknown migration operation')
        required = ['request_id']
        if kind in {'preflight', 'import'}:
            required.append('artifact_id')
        if kind == 'import':
            required.append('preflight_id')
        if kind == 'recover':
            required.append('recover_task')
        for field in required:
            if not isinstance(body.get(field), str):
                raise PackageError(f'{field} must be a UUID string')
            UUID(body[field])
        queue = jobs()
        if kind == 'import':
            if body.get('maintenance_confirmed') is not True:
                raise PackageError('请确认暂停目标业务并导入。')
            preview = queue.get(body.get('preflight_id', ''))
            if (preview['kind'] != 'preflight' or preview['status'] != 'succeeded'
                    or preview['artifact_id'] != body.get('artifact_id')):
                raise PackageError('请先对选中的迁移包完成预检。')
        headers = dict(request.headers)
        selection = preview.get('result', {}).get('shared_spaces') if kind == 'import' else None
        job = queue.submit(kind, identity=body.get('request_id', ''), artifact=body.get('artifact_id'),
                           recover_task=body.get('recover_task'), authorize=lambda: authorize(headers),
                           shared_selection=selection)
        return jsonify(job_id=job['id']), 202

    @api.get('/jobs/<identity>')
    def job(identity):
        queue = jobs()
        queue.recent()
        result = queue.get(identity)
        result['maintenance'] = public_state(queue.gate.status(result.get('recover_task') or identity))
        return jsonify(result)

    @api.get('/jobs/<identity>/download')
    def download(identity):
        queue = jobs()
        task = queue.get(identity)
        if task['kind'] != 'export' or task['status'] != 'succeeded':
            raise PackageError('导出任务尚未完成。')
        path = queue.root / f'export-{UUID(identity)}.tar.gz'
        return send_file(path, as_attachment=True, download_name='ogmemory-snapshot.tar.gz')

    return api
