"""Start durable source recovery inside each serving worker."""


def post_worker_init(worker):
    from server.app import _get_service
    _get_service()
