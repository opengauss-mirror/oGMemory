"""An absent graph annotation keeps historical durable write identities."""
import hashlib
import json
from dataclasses import asdict, replace

from commit.write_receipts import operation_digest
from core.models import CandidateMemory, RequestContext


def test_empty_graph_metadata_preserves_pre_graph_operation_digest():
    candidate=CandidateMemory('entity','user','project','Project','Project overview','Project facts',.9)
    ctx=RequestContext('acct','user','agent','session','trace')
    legacy=asdict(candidate)
    for key in ('metadata','write_operation_id','verified_routing_aliases','verified_object_aliases',
                'source_contexts','node_draft_indices','source_occurrence_id',
                'node_reconciliation_role','node_reconciliation_group'):
        legacy.pop(key)
    evidence=dict(candidate=legacy,account_id=ctx.account_id,user_id=ctx.user_id,
                  agent_id=ctx.agent_id,session_id=ctx.session_id)
    expected=hashlib.sha256(json.dumps(evidence,ensure_ascii=False,sort_keys=True,
                                      separators=(',',':')).encode()).hexdigest()
    assert operation_digest(candidate,ctx)==expected
    annotated=replace(candidate,metadata={'update_activation':'Project uses SQL.',
                                          'update_activation_version':'c1-integrated-activation-v1'})
    assert operation_digest(annotated,ctx)!=expected


def test_consuming_native_relations_does_not_break_accepted_write_receipt():
    from extraction.relation_proposals import attach, take
    candidate=CandidateMemory('entity','user','project','Project','Project overview','Project facts',.9,
                              metadata={'update_activation':'Project uses SQL.'})
    ctx=RequestContext('acct','user','agent','session','trace')
    digest=operation_digest(candidate,ctx)
    attach([candidate],[{'from_routing_key':'project','to_routing_key':'sql'}])
    assert operation_digest(candidate,ctx)==digest
    assert len(take([candidate]))==1
    assert operation_digest(candidate,ctx)==digest
