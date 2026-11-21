"""Viewer routes serve the migration shell and its required assets."""
import pytest
from flask import Flask
from server.memory_viewer import create_memory_viewer_blueprint


@pytest.mark.parametrize('path,marker', [
    ('/memory-viewer/', '/memory-viewer/migration'),
    ('/memory-viewer/migration', 'id="progress"'),
    ('/memory-viewer/migration.js', '/api/v1/migration'),
    ('/memory-viewer/styles.css', '.migration-shell'),
])
def test_migration_page_and_assets(path, marker):
    def forbidden(*args, **kwargs):
        pytest.fail('serving the static shell must not initialize business resources')
    app = Flask(__name__)
    app.register_blueprint(create_memory_viewer_blueprint(forbidden, forbidden, forbidden))
    response = app.test_client().get(path)
    assert response.status_code == 200
    assert marker in response.get_data(as_text=True)
