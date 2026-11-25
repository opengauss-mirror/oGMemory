"""Process-lifetime locks on a local migration workspace."""
import errno
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno == errno.EACCES:
                    raise BlockingIOError(errno.EAGAIN, "file is already locked", path) from exc
                raise
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
