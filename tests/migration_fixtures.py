"""Shared backend fixtures for migration component tests."""

class LocalClient:
    def __init__(self, root):
        self.root = root
        self.timeout = 5
        self.api_base = "http://test"
        self.session = self

    def put(self, url, *, params, data, timeout):
        from types import SimpleNamespace
        value = data if isinstance(data, bytes) else data.read()
        self._path(params["path"]).write_bytes(value)
        return SimpleNamespace(raise_for_status=lambda: None, close=lambda: None)

    def _path(self, path):
        return self.root / path.lstrip("/")

    path = _path

    def ls(self, path):
        return [{"name": p.name, "isDir": p.is_dir(), "isLink": p.is_symlink()}
                for p in self._path(path).iterdir()]

    def read(self, path, offset=0, size=-1):
        with self._path(path).open("rb") as f:
            f.seek(offset)
            return f.read(size)

    def write(self, path, data, max_retries=1):
        self._path(path).write_bytes(data)

    def mkdir(self, path):
        self._path(path).mkdir()

    def rm(self, path, recursive=False):
        import shutil
        shutil.rmtree(self._path(path))
