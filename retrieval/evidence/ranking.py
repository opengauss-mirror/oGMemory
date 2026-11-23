"""Deterministic lexical ranking, query intent and whole-group evidence selection."""

import math
import re
from collections import Counter

_LEMMA = {}
for _root in "visit attend complete finish join adopt move graduate play repair start recommend suggest enjoy hide like love own mention".split():
    for _form in (_root + "s", _root + "d" if _root.endswith("e") else _root + "ed",
                  (_root[:-1] if _root.endswith("e") else _root) + "ing"):
        _LEMMA[_form] = _root
_LEMMA.update(dict(went="go", gone="go", bought="buy", won="win", ran="run", met="meet",
                   hid="hide", hidden="hide", lost="lose", essays="essay", books="book"))

_STOP = set("a an the i my me we our you your what which when where how did do does is are was were have has had and or of to in on at for about tell please".split())
_FUNCTIONAL = set("am be been being can could shall should will would may might must he she it they him her them their his its us this that these those there not cannot "
                  "with from by as into onto upon through during before after between among over under than then if because while so".split())
_CONVERSATIONAL = set("hi hello hey thanks thank okay ok welcome".split())
_QUERY_OPERATORS = set("long many much kind kinds type types different total number both together".split())


def terms(text):
    normalized = text.casefold().replace("’", "'")
    normalized = re.sub(r"\b(ca|wo|sha)n't\b",
                        lambda match: {"ca": "can", "wo": "will", "sha": "shall"}[match[1]] + " not", normalized)
    normalized = re.sub(r"n't\b", " not", normalized)
    normalized = re.sub(r"\b([a-z]+)(?:'ve|'re|'m|'ll|'d|'s)\b", r"\1", normalized)
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[\u4e00-\u9fff]+", normalized)
    result = []
    for token in tokens:
        if "\u4e00" <= token[0] <= "\u9fff":
            result.extend(token[i:i + 2] for i in range(max(1, len(token) - 1)))
        elif token not in _STOP:
            result.append(_LEMMA.get(token, token))
    return result


def topic_terms(text):
    """Retrieval topics exclude grammatical glue and generic query operators."""
    return set(terms(text)) - _FUNCTIONAL - _CONVERSATIONAL - _QUERY_OPERATORS


def query_topics(query, units):
    base = topic_terms(query)
    if not base and re.fullmatch(r"[a-z]+[.!?]?", query.strip(), re.I):
        base = set(terms(query))  # Literal single-word lookups may name an entity.
    speakers = {word for unit in units.values() for word in terms(unit.get("speaker", ""))}
    topics = base - speakers
    frequencies = Counter(word for unit in units.values() for word in set(terms(unit["text"])) & topics)
    distinctive = {word for word in topics if 0 < frequencies[word] <= len(units) / 2}
    # Name-only and single-topic queries remain searchable.
    return distinctive or topics or base


def supplement_scores(query, units, texts, limit):
    """Rank verified supplements below the bounded primary channel, without a vote twice."""
    topics = query_topics(query, units)
    documents = {key: {"text": text} for key, text in texts.items()
                 if key in units and topics & set(terms(text))}
    scores = lexical_scores(query, documents)
    return {key: 1 / (60 + limit + rank) for rank, key in enumerate(
        sorted(scores, key=lambda key: (-scores[key], key))[:limit], 1)}


def lexical_scores(query, units):
    qterms = set(terms(query))
    docs = {key: Counter(terms(unit["text"])) for key, unit in units.items()}
    frequencies = Counter(word for doc in docs.values() for word in doc)
    average = sum(sum(doc.values()) for doc in docs.values()) / max(1, len(docs))
    scores = {}
    for key, doc in docs.items():
        length = sum(doc.values())
        score = sum(math.log(1 + (len(docs) - frequencies[word] + .5) / (frequencies[word] + .5))
                    * doc[word] * 2.2 / (doc[word] + 1.2 * (.25 + .75 * length / max(1, average)))
                    for word in qterms if doc[word])
        if score:
            # Reward a normalized multi-word phrase, preserving token boundaries.
            phrase = " ".join(terms(query))
            if len(qterms) > 1 and " " + phrase + " " in " " + " ".join(terms(units[key]["text"])) + " ":
                score *= 1.2
            scores[key] = score
    return scores


def reciprocal_ranks(scores):
    return {key: 1 / (60 + rank) for rank, key in enumerate(
        sorted(scores, key=lambda key: (-scores[key], key)), 1)}


def fuse_channels(query, units, lexical, semantic, limit):
    """Keep strong independent hits; bound corroboration to 25% of a rank vote.

    Speaker names, greetings and terms in most documents cannot establish an
    independent topic. Semantic-only hits remain eligible (including synonyms).
    """
    topics = query_topics(query, units)
    matches = {key: topics & set(terms(unit["text"])) for key, unit in units.items()}
    lexical = {key: score for key, score in lexical.items()
               if key in units and (not topics or topics & matches[key])}
    semantic = {key: score for key, score in semantic.items() if key in units}
    channels = [reciprocal_ranks(dict(sorted(channel.items(), key=lambda item: (-item[1], item[0]))[:limit]))
                for channel in (lexical, semantic)]
    scores = {}
    speakers = {word for unit in units.values() for word in terms(unit.get("speaker", ""))}
    support_topics = topics - speakers
    for key in channels[0].keys() | channels[1].keys():
        left, right = channels[0].get(key, 0), channels[1].get(key, 0)
        support = len(support_topics & matches[key]) / len(support_topics) if support_topics else 0
        scores[key] = max(left, right) + .25 * min(left, right) * support
    return scores


def fact_matches(query, record):
    # A capitalized destination ("Paris") alone is not a subject constraint.
    names = re.findall(r"\b(?i:did|does|has|had|was|is)\s+([A-Z][a-z]+)\b", query)
    leading = re.match(r"([A-Z][a-z]+)(?:'s\b|\s+(?:visited|attended|completed|and)\b)", query)
    if leading:
        names.append(leading[1])
    self_query = bool(re.search(r"\b(I|my|me|we|our)\b|我", query, re.I))
    recommendation = bool(re.search(r"\b(recommend|recommended|suggest|suggested)\b|推荐|建议", query, re.I))
    if recommendation:
        self_query = bool(re.search(
            r"\b(I|we)\s+(?:(?:have|had|previously|ever|once)\s+)*(?:recommend|recommended|suggest|suggested)\b"
            r"|我(?:们)?(?:之前|曾经|曾)?(?:推荐|建议)", query, re.I))
    assistant = recommendation and not names and bool(re.search(
        r"\b(you|(?:the )?(?:AI )?assistant)\s+(?:(?:have|had|previously|ever|once)\s+)*"
        r"(?:recommend|recommended|suggest|suggested)\b|(?:你|助手)(?:之前|曾经|曾)?(?:推荐|建议)", query, re.I))
    if assistant:
        return record.get("role") == "assistant" and record.get("assertion_type") == "recommendation"
    if record.get("role") == "assistant":
        return recommendation and not names and not self_query and record.get("assertion_type") == "recommendation"
    if record.get("role") != "user":
        return False
    if recommendation and not ({"recommend", "suggest", "推荐", "建议"} & set(terms(record.get("predicate", "")))):
        return False
    # A named actor wins over first-person recipients ("Alex recommended to me").
    if names:
        return bool(set(name.casefold() for name in names) & set(terms(record.get("subject", ""))))
    if self_query and record.get("subject") not in {"user", "我", "我们"}:
        return False
    return True


def _question(text):
    parts = re.split(r"[?？]", text)
    return re.split(r"[.!。！\n]", parts[-2])[-1].strip() if len(parts) > 1 else ""


def _short_answer(question, text):
    """Typed, short replies need not repeat the question's topic words."""
    if not question or len(text.encode("utf-8")) > 512 or len(terms(text)) > 20 or _question(text):
        return False
    text = text.strip()
    number = bool(re.match(r"(?:(?:for|since|about|around|roughly|nearly|almost|over|under|at least|at most|more than|less than|just|only)\s+|大约|大概|已经|至少|不到)*(?:\d|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|hundred|thousand)\b|[一二三四五六七八九十百千万两])", text, re.I))
    if re.search(r"\bhow long\b|多久|多长时间", question, re.I):
        return number and bool(re.search(r"\b(seconds?|minutes?|hours?|days?|weeks?|months?|years?|since)\b|秒|分钟|小时|天|周|个月|年", text, re.I))
    if re.search(r"\bhow many\b|多少|几个", question, re.I):
        return number
    if re.match(r"when\b|什么时候|何时", question, re.I):
        return number or bool(re.match(r"(yesterday|today|tomorrow|last|next)\b|昨天|今天|明天|去年|上周", text, re.I))
    if re.match(r"where\b|哪里|在哪", question, re.I):
        return bool(re.match(r"(at|in|near|by|inside|outside)\b|在|就在|附近", text, re.I))
    if re.match(r"(did|do|does|have|has|had|is|are|was|were|can|could|will|would)\b|是否|有没有|.*[吗么]$", question, re.I):
        return bool(re.match(r"(yes|no|yeah|yep|nope|sure|of course)\b|(i|we) (did|have|do|can)\b|是的|不是|有的|没有|我们.*了", text, re.I))
    # A short noun/name reply to an open question, not an unrelated full clause.
    return (bool(re.match(r"(what|which|who)\b|什么|哪个|谁", question, re.I))
            and len(terms(text)) <= 4
            and not re.match(r"(i|we|he|she|they|it|you)\b|我|他|她|它", text, re.I))


def _adjacent(left, right):
    return (left["session_id"] == right["session_id"]
            and (left.get("message_id") is not None and left["message_id"] == right.get("message_id")
                 and left["sequence"] == right["sequence"]
                 or left.get("sequence_verified", True) and right.get("sequence_verified", True)
                 and right["sequence"] - left["sequence"] == 1))


def select_groups(query, scores, units, top_k, candidate_limit, *, costs=None, byte_limit=64 * 1024):
    """Select independent seeds with only the context required to interpret them."""
    speaker_terms = {word for unit in units.values() for word in terms(unit.get("speaker", ""))}
    reference_terms = set("it that this they he she there him her them these those yes no hi hello hey thanks okay ok".split())

    def topic_words(text):
        return topic_terms(text) - speaker_terms - reference_terms

    query_topics = topic_words(query)
    branches = [set(terms(part)) for part in re.split(r"\band\b|和|与", query, flags=re.I)]
    branches = [branch for branch in branches if branch]
    dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", query))
    ordered = sorted(units, key=lambda key: (units[key]["session_id"], units[key]["sequence"],
                                             units[key]["parent_span"][0], key))
    positions = {key: i for i, key in enumerate(ordered)}
    pending = set(sorted(scores, key=lambda key: (-scores[key], key))[:candidate_limit])
    selected, covered, used_bytes, groups = [], set(), 0, []
    pronoun = re.compile(r"^(yes|no|it|that|this|they|he|she|there|and then)\b|^(是的|不是|它|他|她|那里|然后)", re.I)
    while pending and len(selected) < top_k:
        def priority(key):
            text_terms = set(terms(units[key]["text"]))
            gain = sum(1 for i, branch in enumerate(branches) if i not in covered and branch & text_terms)
            exact_date = bool(dates & set(re.findall(r"\d{4}-\d{2}-\d{2}", units[key]["text"])))
            return (-int(exact_date), -scores[key] * (1 + .25 * gain), key)
        key = min(pending, key=priority)
        pending.remove(key)
        group = [key]
        pos = positions[key]
        current = units[key]
        # Resolve dependencies backwards. A complete statement does not need
        # the preceding question merely because it shares the same topic.
        while positions[group[0]] and len(group) < 4:
            first = units[group[0]]
            prior_key = ordered[positions[group[0]] - 1]
            prior = units[prior_key]
            question = _question(prior["text"])
            referring = pronoun.search(first["text"].strip()) or re.search(
                r"\b(it|he|she|they|him|her|this|that|these|those)\b|它|他|她|这|那",
                _question(first["text"]), re.I)
            shared = topic_words(prior["text"]) & (topic_words(first["text"]) | query_topics)
            media = re.search(r"\b(image|photo|picture|caption)\b|图片|照片|图注", prior["text"], re.I)
            if not (_adjacent(prior, first) and (
                    _short_answer(question, first["text"]) or referring and (shared or media))):
                break
            group.insert(0, prior_key)
        # A question needs its answer, unless the seed already states an
        # independent, query-relevant fact before asking a follow-up question.
        question = _question(current["text"])
        prefix = current["text"][:current["text"].rfind(question)] if question else ""
        independent = bool(re.search(r"[.!。！]", prefix) and topic_words(prefix) & query_topics)
        if question and not independent and pos + 1 < len(ordered) and len(group) < 4:
            following_key = ordered[pos + 1]
            following = units[following_key]
            if (_adjacent(current, following) and (
                    _short_answer(question, following["text"])
                    or topic_words(question) & topic_words(following["text"]))):
                group.append(following_key)
        # Keep shared dependency IDs in the group for final source checks and
        # caller-side truncation, but charge their cost only once.
        added = [item for item in group if item not in selected]
        size = sum(costs[item] if costs is not None else len(units[item]["text"].encode("utf-8")) for item in added)
        if len(selected) + len(added) > top_k or used_bytes + size > byte_limit:
            continue
        selected.extend(added)
        groups.append(group)
        pending.difference_update(group)
        used_bytes += size
        for item in group:
            covered.update(i for i, branch in enumerate(branches) if branch & set(terms(units[item]["text"])))
    # Optional explanations/continuations compete only for the remaining
    # capacity, after every independent seed has had its admission attempt.
    for seed_group in list(groups):
        group = list(seed_group)
        while len(group) < 4 and len(selected) < top_k:
            previous = units[group[-1]]
            pos = positions[group[-1]]
            if pos + 1 == len(ordered):
                break
            following_key = ordered[pos + 1]
            following = units[following_key]
            shared = set().union(*(topic_words(units[item]["text"]) for item in group)) & topic_words(following["text"])
            question = _question(previous["text"])
            if (following_key in selected or not _adjacent(previous, following)
                    or not (_short_answer(question, following["text"])
                            or pronoun.search(following["text"].strip()) and shared
                            or question and shared)):
                break
            size = costs[following_key] if costs is not None else len(following["text"].encode("utf-8"))
            if used_bytes + size > byte_limit:
                break
            group = group + [following_key]
            selected.append(following_key)
            groups.append(group)
            used_bytes += size
            covered.update(i for i, branch in enumerate(branches) if branch & set(terms(following["text"])))
    return selected, [i for i in range(len(branches)) if i not in covered], groups
