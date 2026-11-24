from uuid import uuid4

from core.document_backend import DocumentIndexStatus, DocumentRef
from core.resource_models import ResourceChunk
from server.document_resource_service import DocumentResourceService
from server.document_resource_store import DocumentBinding


def test_document_service_accepts_a_local_backend():
    class LocalBackend:
        name = "local/agfs"
        namespace = "acme-knowledge"
        creates_distinct_objects = True

        def __init__(self):
            self.objects = {}

        def ingest(self, resource, content):
            ref = DocumentRef(resource, self.name, self.namespace, uuid4().hex)
            self.objects[ref.object_id] = content
            return ref

        def status(self, ref):
            assert self.objects[ref.object_id]
            return DocumentIndexStatus("ready", 1)

        def download(self, ref):
            return self.objects[ref.object_id]

        def search(self, query, refs, *, limit):
            matches = [
                (ref, self.download(ref).decode())
                for ref in refs
                if query.encode() in self.download(ref)
            ]
            return [
                ResourceChunk(ref.resource.uri, "paragraph-1", text, rank)
                for rank, (ref, text) in enumerate(matches[:limit], start=1)
            ]

    class Bindings:
        def __init__(self):
            self.refs = {}

        def add(self, ref):
            self.refs[(ref.resource.account_id, ref.resource.resource_id)] = ref

        def get(self, account_id, resource_id):
            return DocumentBinding(self.refs[account_id, resource_id])

        def list_active(self, account_id, backend, namespace):
            return [
                ref
                for ref in self.refs.values()
                if (ref.resource.account_id, ref.backend, ref.namespace)
                == (account_id, backend, namespace)
            ]

    store = Bindings()
    backend = LocalBackend()
    service = DocumentResourceService(store, backend, "acme")
    content = b"Maintenance starts at 02:00 UTC."
    resource = service.ingest("maintenance.md", "text/markdown", content)

    uri = f"ctx://acme/resources/doc-{resource.resource_id.hex}"
    assert resource.uri == uri
    assert store.get("acme", resource.resource_id).ref == DocumentRef(
        resource, "local/agfs", "acme-knowledge", next(iter(backend.objects))
    )
    assert service.status(resource.resource_id) == (resource, DocumentIndexStatus("ready", 1))
    assert service.download(resource.resource_id) == (resource, content)
    assert service.search("Maintenance", limit=1) == [
        ResourceChunk(uri, "paragraph-1", content.decode(), 1)
    ]
