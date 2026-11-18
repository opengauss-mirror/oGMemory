"""Same-origin memory viewer and shared-space management gateway."""

from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from server.app import _enforce_ip_allowlist, _error_response, _get_service, _shutdown_service
from server.app import app as public_app
from server.space_management import create_space_management_blueprint


def create_app():
    management = Flask(__name__)
    management.before_request(_enforce_ip_allowlist)
    management.json.ensure_ascii = False
    management.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    management.register_blueprint(create_space_management_blueprint(_get_service, _error_response))
    return DispatcherMiddleware(public_app, {"/management": management})


app = create_app()


if __name__ == "__main__":
    from werkzeug.serving import run_simple

    try:
        run_simple("127.0.0.1", 8091, app, threaded=True)
    finally:
        _shutdown_service()
