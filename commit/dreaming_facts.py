"""Source-bound assessment and temporal projection of committed memory facts."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from itertools import takewhile

from commit.dreaming_policy import Evidence, eligible_for_promotion, evidence_score, valid_at
from core.dreaming_evidence import claim_text, fact_blocks, timestamp, verified_assertions
from core.errors import MemoryCardFormatError
from core.models import CandidateMemory
from extraction.card_format import validate_memory_card


def _assessment(proofs):
    evidence = [Evidence(item["occurrence_id"], item["weight"], item["supports"]) for item in proofs]
    score = evidence_score(evidence)
    grade = "unrated" if score is None else "high" if score >= .8 else "medium" if score >= .5 else "low"
    return {
        "evidence": proofs,
        "confidence": score,
        "grade": grade,
        "support": "corroborated" if eligible_for_promotion(evidence) else
                   "disputed" if any(not item.supports for item in evidence) else
                   "observation" if evidence else "needs_evidence",
    }


def _facts(node, evidence_claims=frozenset()):
    lines = [line.rstrip() for line in node.content.splitlines() if line.strip()]
    parts = []
    if node.category in {"entity", "preference"}:
        try:
            validate_memory_card(CandidateMemory(
                node.category, "user", str(node.metadata.get("routing_key") or ""),
                node.abstract, node.overview, node.content, 1.0,
            ))
        except MemoryCardFormatError:
            pass
        else:
            heading, intro = "", []
            for line in lines[1:]:
                if line.startswith("## "):
                    heading = line
                elif line.startswith("- "):
                    parts.append(([lines[0], *intro, heading], [line]))
                elif parts and line.startswith("  "):
                    parts[-1][1].append(line)
                elif not heading and not parts:
                    intro.append(line)
    elif node.category == "profile" and lines and all(line.startswith("- ") for line in lines):
        parts = [([], [line]) for line in lines]
    if parts:
        return [(" ".join(line.strip()[2:] if index == 0 else line.strip() for index, line in enumerate(body)),
                 "\n".join(line for line in [*header, *body] if line))
                for header, body in parts]
    if evidence_claims and claim_text(node.content) not in evidence_claims:
        records = fact_blocks(node.content, evidence_claims)
        if any(claim in evidence_claims for claim, _ in records):
            return records
    return [(node.content.strip(), node.content)]


@dataclass(frozen=True)
class _Observation:
    proof: dict
    start: datetime | None
    end: datetime | None
    slot: str
    value: str
    effect: str
    retain: bool

    @property
    def correction(self):
        return self.effect == "correction"

    @property
    def reason(self):
        if not self.proof["supports"]:
            return ""
        if self.retain:
            return "explicit"
        if self.end:
            return "deadline"
        return "correction" if self.correction else ""


def _revocation(observation):
    return observation.start if observation.effect == "revocation" else None


def _assess_claim(node, claim, content, observations, window=(None, None)):
    identities = {(item.slot, item.value) for item in observations if item.slot}
    slot, value = next(iter(identities)) if len(identities) == 1 else ("", "")
    supporting = [item for item in observations if item.proof["supports"]]
    periods = supporting or observations
    start = min((item.start for item in periods), default=None) if all(item.start for item in periods) else None
    end = max((item.end for item in periods), default=None) if all(item.end for item in periods) else None
    renewed = min((item.start for item in supporting if item.start), default=None)
    correction_from = window[0] or start
    expiry = min((item for item in observations if supporting and _revocation(item)
                  and valid_at(item.start, valid_from=start, valid_to=end) and item.start != start),
                 key=lambda item: item.start, default=None)
    start = max(filter(None, (start, window[0])), default=None)
    end = min(filter(None, (expiry.start if expiry else end, window[1])), default=None)
    proofs, reasons = {}, []
    correction = False
    for item in observations:
        proof = item.proof
        if ((item.start and end and item.start >= end) or (item.end and start and item.end <= start)
                or (_revocation(item) and renewed and item.start < renewed)):
            continue
        key = (proof["source_ref"], proof["occurrence_id"], proof["weight"], proof["supports"])
        proofs.setdefault(key, proof)
        correction |= item.correction and item.start == correction_from
        reasons.append(item.reason)
    refs = list(dict.fromkeys(proof["source_ref"] for proof in proofs.values()))
    if expiry and expiry.proof["source_ref"] not in refs:
        refs.append(expiry.proof["source_ref"])
    return {
        "abstract": (next((line[2:] for line in content.splitlines() if line.startswith("- ")), claim)
                     if claim != content or content != node.content else node.abstract or claim),
        "content": content,
        "provenance_ids": refs or list(node.metadata.get("provenance_ids") or []),
        "slot": slot, "value": value,
        "valid_from": start.isoformat() if start else None,
        "valid_to": end.isoformat() if end else None,
        "expiry_evidence": expiry.proof if expiry else None,
        "correction": correction,
        "value_reason": "explicit" if "explicit" in reasons else next(filter(None, reasons), ""),
        "evidence_state": "verified" if proofs else "pending",
        **_assessment(list(proofs.values())),
    }


def _claim_periods(observations, departures):
    earliest = datetime.min.replace(tzinfo=UTC)
    supporting = [item for item in observations if item.proof["supports"]]
    refutations = [item for item in observations if not item.proof["supports"]]
    revocations = {bound for item in refutations if (bound := _revocation(item))}
    replacements = {departure.start for departure in departures}
    period = []
    period_end = None
    for item in sorted(supporting, key=lambda item: item.start or earliest):
        if period and item.start:
            boundaries = revocations | replacements if item.correction else revocations
            has_gap = period_end is not None and period_end < item.start
            restarted = any((period[0].start or earliest) < boundary < item.start for boundary in boundaries)
            if has_gap or restarted:
                yield period + refutations
                period = []
        if not period:
            period_end = item.end
        elif period_end is not None:
            period_end = max(period_end, item.end) if item.end is not None else None
        period.append(item)
    yield period + refutations


def _fact_applies(fact, when):
    return valid_at(when, valid_from=timestamp(fact.get("valid_from")), valid_to=timestamp(fact.get("valid_to")))


def _assessed_periods(node, claim, content, observations, departures, initial):
    periods = list(_claim_periods(observations, departures))
    for period in periods:
        fact = initial if len(periods) == 1 else _assess_claim(node, claim, content, period)
        start, end = timestamp(fact["valid_from"]), timestamp(fact["valid_to"])
        bounds = sorted({edge for item in period for edge in (item.start, item.end)
                         if edge and edge != start and valid_at(edge, valid_from=start, valid_to=end)})
        if bounds:
            for window in zip([None, *bounds], [*bounds, None]):
                yield _assess_claim(node, claim, content, period, window) | {"episode_valid_from": fact["valid_from"]}
        else:
            yield fact


def _assess_node_claims(node, records, assertions):
    by_claim = {}
    for assertion, proof in assertions:
        by_claim.setdefault(claim_text(assertion.claim), []).append(_Observation(
            proof, timestamp(assertion.valid_from), timestamp(assertion.valid_to),
            assertion.slot, assertion.value, assertion.effect, assertion.retain,
        ))
    prepared, departures = {}, {}
    for claim, content in records:
        observations = by_claim.get(claim_text(claim), [])
        initial = _assess_claim(node, claim, content, observations)
        prepared[content] = claim, initial, observations
        if initial["slot"] and (changes := [item for item in observations if item.correction]):
            for fact in _assessed_periods(node, claim, content, observations, (), initial):
                if fact["grade"] == "high" and fact["support"] != "disputed":
                    departures.setdefault(fact["slot"], []).extend(
                        (fact["value"], item) for item in changes if _fact_applies(fact, item.start)
                    )
    facts = {}
    for content, (claim, initial, observations) in prepared.items():
        other_values = [item for value, item in departures.get(initial["slot"], ()) if value != initial["value"]]
        for index, item in enumerate(_assessed_periods(node, claim, content, observations, other_values, initial)):
            identity = f'{content}\0{item["valid_from"]}\0{item["valid_to"]}' if index else content
            facts[sha256(identity.encode()).hexdigest()[:20]] = item
    return facts


def _value(items):
    if reason := next((item["value_reason"] for item in items if item.get("value_reason")), ""):
        return "high", reason
    if any(item["support"] == "corroborated" for item in items):
        return "high", "reusable_supported"
    return "medium", "observed" if any(item["evidence"] for item in items) else "pending_source_review"


def _correction_start(fact):
    if fact and fact["slot"] and fact["grade"] == "high" and fact["correction"] and fact["support"] != "disputed":
        return start if (start := timestamp(fact["valid_from"])) and _fact_applies(fact, start) else None
    return None


def _episode(fact):
    return fact["content"], fact.get("episode_valid_from", fact["valid_from"])


def _close_facts(facts, prior):
    by_slot = {}
    for key, fact in facts.items():
        if (start := _correction_start(fact)) is not None:
            by_slot.setdefault(fact["slot"], []).append((start, key, fact))
    for candidates in by_slot.values():
        candidates.sort(key=lambda candidate: candidate[0])
    result = {}
    supported = {_episode(fact) for fact in facts.values() if fact["grade"] == "high"}
    supported.update(_episode(fact) for fact in prior.values() if fact.get("superseded_by"))
    for key, old in facts.items():
        if _episode(old) not in supported:
            continue
        old_start = timestamp(_episode(old)[1])
        source_end = timestamp(old.get("valid_to"))
        decisions = (
            (start, new_key, new) for start, new_key, new in by_slot.get(old.get("slot"), ())
            if new_key != key and new["value"] != old.get("value")
            and (old_start is None or start > old_start)
            and (source_end is None or start < source_end)
        )
        if (earliest := next(decisions, None)) is None:
            continue
        start, new_key, new = earliest
        same_day = takewhile(lambda decision: decision[0] == start, decisions)
        if any(fact["value"] != new["value"] for _, _, fact in same_day):
            continue
        result[key] = {
            **old, "valid_to": new["valid_from"], "superseded_by": new_key,
            "source_valid_to": old.get("valid_to"),
            "supersession_evidence": [item for item in new["evidence"] if item["supports"]],
        }
    return result


def _retain_history(node, records, prior):
    present = {content for _, content in records}
    retained = dict.fromkeys(old["content"] for old in prior.values()
                             if (old.get("superseded_by") or old.get("retained_history")) and old["content"] not in present)
    for content in retained:
        records.extend(_facts(replace(node, content=content)))
    return retained


def _effective_bounds(fact, node_start, node_end):
    start = max(filter(None, (timestamp(fact.get("valid_from")), node_start)), default=None)
    end = min(filter(None, (timestamp(fact.get("valid_to")), node_end)), default=None)
    return {"valid_from": start.isoformat() if start else None, "valid_to": end.isoformat() if end else None}


def assess_facts(node, *, read_source=None):
    """Build auditable fact and value metadata for the current node version."""
    assertions = list(verified_assertions(node, read_source))
    records = list(dict.fromkeys(_facts(node, {claim_text(assertion.claim) for assertion, _ in assertions})))
    prior = node.metadata.get("dreaming_review", {}).get("facts", {})
    retained = _retain_history(node, records, prior)
    facts = _assess_node_claims(node, [record for record in records if record[0]], assertions)
    facts.update(_close_facts(facts, prior))
    for fact in facts.values():
        if fact["content"] in retained:
            fact["retained_history"] = True
    current = [fact for fact in facts.values() if not fact.get("superseded_by")]
    value_band, value_reason = _value(current)
    single = next(iter(facts.values())) if len(facts) == 1 else {}
    valid_from, valid_to = timestamp(node.metadata.get("valid_from")), timestamp(node.metadata.get("valid_to"))
    scores = [item["confidence"] for item in current if item["confidence"] is not None]
    confidence = min(scores) if scores and len(scores) == len(current) else None
    recommendation = "pending"
    if confidence is not None:
        recommendation = "recommended" if all(item["support"] == "corroborated" for item in current) else "observation"
    return {
        "facts": facts,
        "evidence_state": "verified" if confidence is not None else "pending",
        "confidence": confidence,
        "value_band": value_band, "value_reason": value_reason,
        "recommendation": recommendation,
        **_effective_bounds(single, valid_from, valid_to),
    }


def visible_facts(node, when):
    """Return the independently eligible facts for the requested point in time."""
    review = node.metadata.get("dreaming_review", {})
    start, end = timestamp(review.get("valid_from")), timestamp(review.get("valid_to"))
    if review.get("version") != node.metadata.get("version") or not valid_at(when, valid_from=start, valid_to=end):
        return {}
    return {key: fact | _effective_bounds(fact, start, end)
            for key, fact in review.get("facts", {}).items() if _fact_applies(fact, when)}
