"""Source-grounded fact/entity/event candidates, independently implemented rules.

The original quote is always authoritative. Conservative rules decline questions,
uncertain claims and unsupported coreference rather than inventing relationships.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import islice

from core.evidence import deterministic_payload, digest, message_source, reassemble_messages
from core.evidence_timing import operation
from core.models import CandidateMemory
from core.operation_deadline import check_deadline
from core.validation import MAX_ABSTRACT_LEN, MAX_CONTENT_LEN, MAX_OVERVIEW_LEN
from extraction.deterministic.nlp import DeterministicConfig, TextParser, parser_language

RULE_VERSION = "1"
_SENTENCE = re.compile(r"(?:[^.!?。！？\r\n]|(?<=\d)\.(?=\d))+(?:[.!?。！？]+|(?=[\r\n]|$))")
_QUOTED = re.compile(r'''"[^"]*(?:"|$)|“[^”]*(?:”|$)|「[^」]*(?:」|$)|『[^』]*(?:』|$)|(?<!\w)'(?=\S).*?(?:'(?!\w)|$)''', re.S)
_UNCERTAIN = re.compile(r"\b(if|maybe|might|perhaps|would|could|suppose|heard|said that|will|plan to)\b|如果|可能|也许|听说|据说|打算|计划|将要", re.I)
_NEGATED = re.compile(r"\b(not|never|no|neither|nor|cannot)\b|n['’]t\b|没有|从未|未曾|并非", re.I)
_ACTION = re.compile(r"\b(attended|visited|completed|finished|joined|adopted|bought|moved|graduated|won|ran|played|went|repaired|met|started|lost)\b|参加了|去了|完成了|加入了|收养了|买了|搬到|毕业|赢得|修好|开始了")
_RECOMMEND = re.compile(r"\b(recommend|suggest|try|consider)\b|推荐|建议|可以试试", re.I)
_DIRECT_RECOMMEND = re.compile(r"^(?:(?:I|We)\s+(?:recommend|suggest)\b|(?:try|consider)\b|(?:我(?:们)?(?:推荐|建议)|推荐|建议|可以试试))", re.I)
_QUANTITY_DATE = re.compile(
    r"(?<!\d)(?:\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|\d{4}年(?:\d{1,2}月(?:\d{1,2}[日号])?)?)(?!\d)"
)
_QUANTITY_DATE_OR_CLOCK = (
    rf"(?:{_QUANTITY_DATE.pattern}|\b\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:\s*[ap]m)?\b|\b\d{{1,2}}\s*[ap]m\b)"
)
_YEAR_CONTEXT = re.compile(
    r"\b(?:(?:in|on|at|during|since|before|after|until|by|around|about|circa|throughout|from|to|through|between|as of)\s+"
    r"(?:\d{4}(?:\s+(?:and|or|to)\s+|\s*[-–—]\s*))*"
    r"|(?:early|late|mid|year)[\s-]+"
    r"|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(?:\d{1,2},?\s+)?)$",
    re.I,
)
_YEAR_APPROXIMATE_COUNT = re.compile(r"\b(?:count(?:ed|s)?|total(?:l?ed|s)?|bought|sold)\s+(?:around|about)\s+$", re.I)
_CALENDAR_NOUNS = {"year", "spring", "summer", "autumn", "fall", "winter", "quarter"}
_QUANTITY_NON_OBJECTS = set("and or but with without in on at to from for of by as into onto during before after "
                            "since until over under through near across between throughout "
                            "is are was were be been being have has had will would can could should may might "
                            "i we you he she it they me us him her them my our your his its their the a an "
                            "together alone again then later earlier once also still instead nearby abroad "
                            "already recently finally".split())
_SELF = re.compile(r"^(I|We)\b|^(我|我们)", re.I)
_POSSESSIVE = re.compile(r"^(my|our)\b|^(我|我们)的", re.I)
_NUMBER_SIGNS = "+−－＋–-"
_NUMBER = rf"(?:[{_NUMBER_SIGNS}][ \t]*)?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?"
_NUMBER_NORMALIZE = str.maketrans("−－–＋", "---+", ", \t")
_QUANTITY_TEMPORAL = re.compile(
    # A date can also begin a subordinate clause. Only consume its connector
    # when the following phrase is an explicit count introduced by with/for.
    rf"\b(?:as of|since|until)\s+{_QUANTITY_DATE_OR_CLOCK}(?=\s+(?:with|for)\s+{_NUMBER}\s)"
    rf"|{_QUANTITY_DATE_OR_CLOCK}",
    re.I,
)
_QUANTITY = re.compile(
    rf"(?<![\w.,{_NUMBER_SIGNS}])({_NUMBER})\s*([^\W\d_]+)"
    rf"|(?<![\d.,{_NUMBER_SIGNS}])({_NUMBER})(个|篇|次|只|本|公里|小时)"
)
_QUANTITY_RANGE = re.compile(
    rf"(?<![\d.,])(?:\bbetween\s+{_NUMBER}\s+and\s+|{_NUMBER}\s*(?:[-−－–—~～至到]|\bto\b)\s*){_NUMBER}", re.I
)
_QUANTITY_COUNT_TAIL = rf"{_NUMBER}\s+[^\W\d_]+(?:\s+in\s+(?:total|all))?[.!]?\s*$"
_QUANTITY_BOUNDARY = re.compile(
    r"(?<!\d),|,(?!\d)|[;:，；：]"
    r"|\b(?:while|although|though|whereas|because|when|after|before|who|which|where|but|yet)\b"
    r"|\b(?:as|since|until|that|unless|whether)\b"
    rf"|\b(?:and|or)\b(?!\s+{_QUANTITY_COUNT_TAIL})"
    r"|但|而|同时|然后|随后|后来|接着|并且|因为|虽然|直到|等到",
    re.I,
)
_QUANTITY_INDIRECT = re.compile(
    # These verbs can introduce an embedded assertion with no conjunction.
    # A standalone counted object can still belong to the main action.
    r"\b(?:say|said|tell|told|report(?:ed)?|hear|heard|see|saw|watch(?:ed)?|notice(?:d)?"
    r"|find|found|learn(?:ed)?|know|knew|think|thought|believe(?:d)?)\b"
    r"(?!\s+(?:(?:the\s+)?(?:first|last)\s+|(?:another|about|around)\s+)?"
    rf"{_QUANTITY_COUNT_TAIL})"
    rf"|(?:看到|看见|听到|得知|发现|告诉|通知|要求|认为|觉得|以为)(?!了?\s*{_NUMBER})[\u4e00-\u9fff]",
    re.I,
)
_QUANTITY_KIND = re.compile(r"(?P<total>\b(?:total|in all)\b|总共|一共)|(?P<increment>\b(?:more|another)\b|又|新增)", re.I)
_NAMED = re.compile(r"^([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})\s+(.+)")
_ZH_SUBJECT = re.compile(r"^(我(?:们)?|[\u4e00-\u9fff]{2,4}?)(参加了|完成了|去了|收养了|买了|搬到|喜欢|拥有|是|在)(.*)")


def overlaps_quotation(text, start, end):
    """Keep unresolved quoted speech as raw evidence rather than an observation."""
    return any(match.start() < end and start < match.end() for match in _QUOTED.finditer(text))


def quotation_ranges(text):
    starts, ends = [], []
    for match in _QUOTED.finditer(text):
        check_deadline()
        starts.append(match.start())
        ends.append(match.end())
    return starts, ends


def is_negated(quote, doc):
    """The current assertion schema cannot safely express negative polarity."""
    negative = (any(token.dep_ == "neg" or token.text in {"不", "没", "未", "无"} for token in doc)
                if doc is not None else any(char in quote for char in "不没未无"))
    return bool(_NEGATED.search(quote) or negative)


def _event_time(quote, source):
    from datetime import datetime, timedelta

    absolute = re.search(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", quote)
    if absolute:
        try:
            return datetime.fromisoformat(absolute[1]).date().isoformat(), absolute[0]
        except ValueError:
            return None, absolute[0]
    relative = re.search(r"\b(yesterday|today|tomorrow)\b|昨天|今天|明天", quote, re.I)
    if relative and source.recorded_at:
        anchor = datetime.fromisoformat(source.recorded_at)
        if anchor.tzinfo is None and not source.timezone:
            return None, relative[0]
        if source.timezone:
            from zoneinfo import ZoneInfo
            zone = ZoneInfo(source.timezone)
            anchor = anchor.replace(tzinfo=zone) if anchor.tzinfo is None else anchor.astimezone(zone)
        offset = {"yesterday": -1, "昨天": -1, "today": 0, "今天": 0, "tomorrow": 1, "明天": 1}
        return (anchor.date() + timedelta(days=offset[relative[0].lower()])).isoformat(), relative[0]
    return None, relative[0] if relative else ""


def _parts(quote, speaker, role, doc):
    """Resolve explicit grammatical subjects; ownership is never speaker identity."""
    if doc is not None:
        roots = [token for token in doc if token.dep_ == "ROOT"]
        if roots:
            root = roots[0]
            subjects = [token for token in root.children if token.dep_ in {"nsubj", "nsubjpass"}]
            if subjects:
                subject = " ".join(t.text for t in subjects[0].subtree)
                if subject.lower() in {"i", "we"} or subject in {"我", "我们"}:
                    subject = speaker or role
                obj = " ".join(t.text for t in root.children if t.dep_ in {"dobj", "obj", "attr", "acomp"})
                return subject, root.lemma_ or root.text, obj
    if _POSSESSIVE.match(quote):
        return "", "", ""  # Without a parsed subject, ownership cannot identify the actor.
    zh = _ZH_SUBJECT.match(quote)
    if zh:
        return (speaker or role) if zh[1] in {"我", "我们"} else zh[1], zh[2], zh[3]
    words = quote.split(maxsplit=2)
    if _SELF.match(quote):
        return speaker or role, words[1] if len(words) > 1 else "asserts", words[2] if len(words) > 2 else quote
    named = _NAMED.match(quote)
    if named and named[1] not in {"Hello", "Thanks", "Hi", "The", "This", "That"}:
        predicate, _, obj = named[2].partition(" ")
        return named[1], predicate, obj
    return "", "", ""


def _year_counts_object(quote, match, doc):
    number = next((token for token in doc or ()
                   if getattr(token, "idx", None) == match.start(1) and getattr(token, "pos_", "")), None)
    if number is not None:
        # NER also labels genuine four-digit counts as DATE. Require a counted noun phrase.
        return (number.dep_ == "nummod" and number.head.pos_ == "NOUN"
                and number.head.idx >= match.start(2) and number.head.text.casefold() not in _CALENDAR_NOUNS)
    prefix = quote[:match.start(1)]
    # The rules-only fallback cannot distinguish a calendar prepositional phrase
    # from a count in the same position. Keep the source instead of guessing.
    return not _YEAR_CONTEXT.search(prefix) or bool(_YEAR_APPROXIMATE_COUNT.search(prefix))


def _quantity_clause(doc, position):
    span = doc.char_span(position, position + 1, alignment_mode="expand")
    if span is None:
        return None
    token = span[0]
    return next((head for head in (token, *token.ancestors)
                 if head.dep_ in {"ROOT", "advcl", "acl", "relcl", "ccomp", "xcomp", "csubj", "csubjpass", "parataxis"}
                 or head.dep_ == "conj" and head.pos_ in {"VERB", "AUX"}), None)


def _owns_quantity(root, clause, quote):
    if clause is None:
        return False
    if clause == root:
        return True
    if clause.head != root:
        return False
    between = quote[root.idx + len(root.text):clause.idx].strip()
    subjects = {child.text.casefold() for child in clause.children if child.dep_ in {"nsubj", "nsubjpass"}}
    if _RECOMMEND.fullmatch(root.lemma_ or root.text) and (
            clause.dep_ in {"xcomp", "ccomp"} or clause.dep_ == "conj" and not between):
        return subjects <= {"you", "你", "你们", "您"}
    if subjects:
        return False
    if clause.dep_ == "advcl":
        return clause.tag_ == "VBN"  # A subjectless passive modifier of the main action.
    # The Chinese model can tag 在 before an ISO date as ROOT and attach the
    # main action to it as a conjunct.
    return clause.dep_ == "conj" and root.text == "在" and bool(_QUANTITY_DATE.fullmatch(between))


def _quantity(quote, doc=None):
    # Keep offsets aligned with the already parsed original quote.
    def mask(match):
        return " " * len(match[0])

    text = _QUANTITY_RANGE.sub(mask, _QUANTITY_TEMPORAL.sub(mask, quote))
    root = next((token for token in doc or () if token.dep_ == "ROOT"), None)
    if root is None:
        # A connector can introduce an arbitrary subject. When the syntax is
        # unknown, keep the quote and only look for counts before that boundary.
        boundary = _QUANTITY_BOUNDARY.search(text)
        if boundary:
            text = text[:boundary.start()]
        indirect = _QUANTITY_INDIRECT.search(text)
        if indirect:
            text = text[:indirect.start()]
    for match in _QUANTITY.finditer(text):
        value, obj = match[1] or match[3], match[2] or match[4]
        if obj.casefold() in _QUANTITY_NON_OBJECTS:
            continue
        if (match[2] and obj.isascii() and len(value) == 4 and value.isdecimal()
                and not _year_counts_object(quote, match, doc)):
            continue
        if root is not None:
            position = match.start() + len(value) - len(value.lstrip(_NUMBER_SIGNS + " \t"))
            clause = _quantity_clause(doc, position)
            if not _owns_quantity(root, clause, quote):
                continue
        kinds = {marker.lastgroup for marker in _QUANTITY_KIND.finditer(text)
                 if root is None or _quantity_clause(doc, marker.start()) == clause}
        number = value.translate(_NUMBER_NORMALIZE)
        return {"value": float(number) if "." in number else int(number), "object": obj,
                "unit": match[4] or None, "kind": "total" if "total" in kinds
                else "increment" if "increment" in kinds else "observation"}
    return None


def is_speaker_recommendation(quote, speaker, role, doc):
    """Promote only the speaker's recommendation; reported actors stay evidence."""
    if not _RECOMMEND.search(quote):
        return False
    if _DIRECT_RECOMMEND.match(quote):
        return True
    if doc is not None:
        root = next((token for token in doc if token.dep_ == "ROOT"), None)
        if root is not None:
            if not _RECOMMEND.fullmatch(root.lemma_ or root.text):
                return False
            if any(token.dep_ in {"nsubj", "nsubjpass"} for token in root.children):
                return _parts(quote, speaker, role, doc)[0] == (speaker or role)
    return False


def is_assertion_quote(quote):
    return bool(quote and len(quote) <= 4000 and "?" not in quote and "？" not in quote
                and not _UNCERTAIN.search(quote))


def build_assertion(quote, speaker, role, source, doc):
    """Derive the persisted assertion fields from a source quote and its parse."""
    if is_negated(quote, doc):
        return None
    recommendation = role == "assistant"
    if recommendation:
        if not is_speaker_recommendation(quote, speaker, role, doc):
            return None
        subject, predicate, obj = speaker or "assistant", "recommended", quote
    else:
        subject, predicate, obj = _parts(quote, speaker, role, doc)
        if not subject:
            return None
    event_time, expression = _event_time(quote, source)
    return {"subject": subject, "predicate": predicate, "object": obj,
            "assertion_type": "recommendation" if recommendation else "observation",
            "event_time": event_time, "time_expression": expression, "quantity": _quantity(quote, doc),
            "scope": re.findall(r"\b\d{4}\b|this year|last year|今年|去年", quote)}


def fact_id(ref, assertion):
    return digest(ref["source_key"], ref["evidence_unit_id"], ref["span"][0], ref["quote"],
                  assertion["subject"], assertion["predicate"], assertion["object"])


def render_records(title, records):
    """Bound the rendered view; structured records and source lineage stay complete."""
    quotes = list(dict.fromkeys(record["quote"].strip() for record in records))
    overview = "\n".join("- " + quote for quote in quotes)
    header = f"# {title}\n\n## Facts\n"
    lines, length = [], len(header)
    for quote in quotes:
        if length + len(quote) + 3 > MAX_CONTENT_LEN:
            break
        lines.append("- " + quote)
        length += len(quote) + 3
    return ((quotes[0] if quotes else title)[:MAX_ABSTRACT_LEN], overview[:MAX_OVERVIEW_LEN],
            header + "\n".join(lines) + "\n")


class DeterministicExtractor:
    def __init__(self, schema_registry, config=None, *, parser=None):
        self.registry = schema_registry
        self.parser = parser if parser is not None else TextParser(config or DeterministicConfig())
        self.diagnostics = {}

    def extract(self, messages, ctx, session_time=None, session_summary="", tool_stats_text="", archive_id=None):
        with operation("EXTRACT_DETERMINISTIC", ctx.trace_id):
            return self._extract(messages, ctx)

    def _parsed_sentences(self, sources):
        def sentences():
            for source in sources:
                for unit in source.units:
                    check_deadline()
                    quotations = iter(_QUOTED.finditer(unit.text))
                    quoted = next(quotations, None)
                    for index, match in enumerate(_SENTENCE.finditer(unit.text), 1):
                        if index % 32 == 0:
                            check_deadline()
                        while quoted is not None and quoted.end() <= match.start():
                            quoted = next(quotations, None)
                        quote = match[0].strip()
                        if quoted is not None and quoted.start() < match.end():
                            continue
                        if not is_assertion_quote(quote):
                            continue
                        if unit.role == "assistant" and not _RECOMMEND.search(quote):
                            continue
                        yield source, unit, match, quote

        iterator = sentences()
        while batch := list(islice(iterator, 32)):
            check_deadline()
            texts = [item[3] for item in batch]
            if callable(getattr(type(self.parser), "parse_many", None)):
                docs = self.parser.parse_many(texts)
            else:
                docs = [self.parser.parse(text) for text in texts]
            check_deadline()
            if len(docs) != len(batch):
                raise ValueError("NLP batch returned an unexpected document count")
            for item, doc in zip(batch, docs):
                yield *item, doc
        check_deadline()

    def _extract(self, messages, ctx):
        sources = [message_source(message, ctx) for message in reassemble_messages(messages)]
        facts, events, entity_records = [], [], defaultdict(list)
        entity_units = defaultdict(set)
        entity_signals = defaultdict(set)
        previous_unit = None
        for source, unit, match, quote, doc in self._parsed_sentences(sources):
            if unit is not previous_unit:
                fact_count, event_count = 0, 0
                previous_unit = unit
            assertion = build_assertion(quote, unit.speaker, unit.role, source, doc)
            if assertion is None:
                continue
            subject = assertion["subject"]
            recommendation = assertion["assertion_type"] == "recommendation"
            raw = match[0]
            start = unit.parent_span[0] + match.start() + len(raw) - len(raw.lstrip())
            ref = {"source_key": source.source_key, "evidence_unit_id": unit.evidence_unit_id,
                   "content_hash": source.content_hash, "span": [start, start + len(quote)],
                   "quote": quote, "session_id": source.session_id, "message_id": source.message_id}
            record = {**assertion, "fact_id": fact_id(ref, assertion),
                      "role": unit.role, "speaker": unit.speaker, "quote": quote,
                      "source_refs": [ref], "recorded_at": source.recorded_at,
                      "rule_version": RULE_VERSION,
                      "nlp_model": getattr(self.parser, "signature", {}).get(parser_language(quote))}
            if fact_count < (6 if recommendation else 4):
                facts.append(record)
                fact_count += 1
            # Event/entity channels consume parsed sources independently of fact selection.
            if not recommendation and _ACTION.search(quote) and event_count < 3:
                events.append(record)
                event_count += 1
            names = set()
            if doc is not None:
                names.update(entity.text for entity in doc.ents
                             if entity.label_ in {"PERSON", "PER", "ORG", "GPE", "LOC", "PRODUCT"})
            if subject not in {"user", "assistant"}:
                names.add(subject)
            for name in names:
                identity = digest(ctx.account_id, ctx.user_space_name(), source.session_id, name.casefold())
                entity_records[(identity, name)].append(record)
                entity_units[identity].add(unit.evidence_unit_id)
                if name == subject:
                    entity_signals[identity].add("subject")
                if doc is not None and any(entity.text == name for entity in doc.ents):
                    entity_signals[identity].add("ner")
        candidates = []

        def add(category, key, title, records):
            schema = self.registry.get_compatible(category)
            if schema is None or not schema.enabled:
                return
            abstract, overview, content = render_records(title, records)
            candidates.append(CandidateMemory(
                category=category, owner_scope="user", routing_key="det_" + key,
                abstract=abstract, overview=overview, content=content, confidence=1.0,
                title=title, who=records[0]["subject"], when=records[0]["event_time"],
                object_name=title, scope_ref="det_" + key,
                metadata={"deterministic": deterministic_payload(records)},
            ))

        for record in facts:
            add("fact", record["fact_id"], record["subject"], [record])
        for record in events[:20]:
            add("event", record["fact_id"], record["subject"] + " " + record["predicate"], [record])
        confirmed = [(identity, name, records) for (identity, name), records in entity_records.items()
                     if len(entity_units[identity]) >= 2 or len(entity_signals[identity]) >= 2]
        confirmed.sort(key=lambda item: (-len(entity_units[item[0]]), item[0]))
        for identity, name, records in confirmed[:30]:
            add("entity", identity, name, records)
        self.diagnostics = {"degraded_languages": sorted(self.parser.degraded_languages),
                            "events_truncated": max(0, len(events) - 20),
                            "entities_truncated": max(0, len(confirmed) - 30)}
        return candidates
