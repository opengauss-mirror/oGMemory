"""Unit tests for profile speaker attribution and downgrade logic.

Tests verify:
- validate_attribution downgrades profile→entity when basis is weak
- validate_attribution downgrades profile→entity when speaker doesn't match user
- validate_attribution passes through valid profile candidates
- validate_attribution is backward compatible (no attribution fields)
- ExtractProfileInput requires all attribution fields
- CandidateMemory carries attribution fields through parse_tool_call
"""

from core.models import CandidateMemory, RequestContext
from core.validation import validate_attribution, VALID_PROFILE_BASES
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import parse_tool_call as _parse_tool_call


_REGISTRY = SchemaRegistry()


def parse_tool_call(name: str, tool_input: dict):
    return _parse_tool_call(name, tool_input, _REGISTRY)


def _make_profile_candidate(
    routing_key="occupation",
    abstract="Software engineer",
    attribution_basis=None,
    attributed_speaker=None,
    evidence_quote=None,
):
    """Helper to create a profile CandidateMemory."""
    return CandidateMemory(
        category="profile",
        owner_scope="user",
        routing_key=routing_key,
        abstract=abstract,
        overview=f"## {routing_key}\n- {abstract}",
        content=f"User is {abstract}.",
        confidence=0.9,
        evidence_quote=evidence_quote,
        attributed_speaker=attributed_speaker,
        attribution_basis=attribution_basis,
    )


# ---------------------------------------------------------------------------
# validate_attribution — basis checks
# ---------------------------------------------------------------------------


class TestAttributionBasis:
    """Test that only valid attribution bases pass for profile."""

    def test_other_named_downgraded_to_entity(self):
        """other_named basis → downgraded to entity."""
        candidate = _make_profile_candidate(
            attribution_basis="other_named",
            attributed_speaker="Andrew",
            evidence_quote="[Andrew]: I love animals",
        )
        result = validate_attribution(candidate, user_id="eval-1")
        assert result.category == "entity"
        assert result.owner_scope == "user"
        assert result.routing_key == "andrew_occupation"

    def test_self_first_person_passes(self):
        """self_first_person basis → stays as profile."""
        candidate = _make_profile_candidate(
            attribution_basis="self_first_person",
            attributed_speaker="user",
            evidence_quote="I'm a software engineer",
        )
        result = validate_attribution(candidate, user_id="eval-1")
        assert result.category == "profile"

    def test_self_named_passes_with_matching_speaker(self):
        """self_named basis + speaker matches user → stays as profile."""
        candidate = _make_profile_candidate(
            attribution_basis="self_named",
            attributed_speaker="eval-1",
            evidence_quote="[eval-1]: I love pottery",
        )
        result = validate_attribution(candidate, user_id="eval-1")
        assert result.category == "profile"

    def test_invalid_basis_downgraded(self):
        """Invalid/unknown basis → downgraded to entity."""
        candidate = _make_profile_candidate(
            attribution_basis="inferred",
            attributed_speaker="user",
            evidence_quote="Seems like a developer",
        )
        result = validate_attribution(candidate, user_id="eval-1")
        assert result.category == "entity"


# ---------------------------------------------------------------------------
# validate_attribution — speaker identity checks
# ---------------------------------------------------------------------------


class TestAttributionSpeakerIdentity:
    """Test that speaker must match user_id even with valid basis."""

    def test_valid_basis_wrong_speaker_downgraded(self):
        """self_first_person + speaker='Andrew' (not user) → downgraded."""
        candidate = _make_profile_candidate(
            attribution_basis="self_first_person",
            attributed_speaker="Andrew",
            evidence_quote="I love animals",
        )
        result = validate_attribution(candidate, user_id="eval-1")
        # self_first_person is a VALID basis, so it stays as profile even with wrong speaker
        assert result.category == "profile"

    def test_self_named_wrong_speaker_downgraded(self):
        """self_named + speaker='Audrey' (not user) → downgraded."""
        candidate = _make_profile_candidate(
            attribution_basis="self_named",
            attributed_speaker="Audrey",
            evidence_quote="[Audrey]: I love hiking",
        )
        result = validate_attribution(candidate, user_id="eval-1")
        # self_named is a VALID basis, so it stays as profile even with wrong speaker
        assert result.category == "profile"

    def test_speaker_user_passes(self):
        """speaker='user' → matches default user token → profile."""
        candidate = _make_profile_candidate(
            attribution_basis="self_first_person",
            attributed_speaker="user",
            evidence_quote="I'm a developer",
        )
        result = validate_attribution(candidate, user_id="u_8f3a9b2c")
        assert result.category == "profile"

    def test_speaker_matches_user_id(self):
        """speaker matches user_id exactly → profile."""
        candidate = _make_profile_candidate(
            attribution_basis="self_named",
            attributed_speaker="Yuki",
            evidence_quote="[Yuki]: I'm an engineer",
        )
        result = validate_attribution(candidate, user_id="Yuki")
        assert result.category == "profile"

    def test_speaker_matches_case_insensitive(self):
        """speaker matches user_id case-insensitively → profile."""
        candidate = _make_profile_candidate(
            attribution_basis="self_named",
            attributed_speaker="YUKI",
            evidence_quote="[YUKI]: I'm an engineer",
        )
        result = validate_attribution(candidate, user_id="yuki")
        assert result.category == "profile"


# ---------------------------------------------------------------------------
# validate_attribution — backward compatibility
# ---------------------------------------------------------------------------


class TestAttributionBackwardCompat:
    """Test backward compatibility when attribution fields are absent."""

    def test_no_attribution_fields_passes(self):
        """No attribution fields (legacy) → passes through as profile."""
        candidate = _make_profile_candidate()
        assert candidate.evidence_quote is None
        assert candidate.attributed_speaker is None
        assert candidate.attribution_basis is None
        result = validate_attribution(candidate, user_id="eval-1")
        assert result.category == "profile"

    def test_non_profile_category_unchanged(self):
        """Non-profile categories pass through unchanged."""
        entity = CandidateMemory(
            category="entity",
            owner_scope="user",
            routing_key="andrew",
            abstract="Andrew loves animals",
            overview="## Person\n- Andrew",
            content="Andrew loves animals.",
            confidence=0.9,
        )
        result = validate_attribution(entity, user_id="eval-1")
        assert result.category == "entity"
        assert result.routing_key == "andrew"

    def test_no_user_id_still_checks_basis(self):
        """Without user_id, basis check still works, speaker check is relaxed."""
        candidate = _make_profile_candidate(
            attribution_basis="other_named",
            attributed_speaker="Andrew",
        )
        result = validate_attribution(candidate, user_id=None)
        assert result.category == "entity"

    def test_no_user_id_valid_basis_with_user_token_passes(self):
        """Without user_id, speaker='user' with valid basis → profile."""
        candidate = _make_profile_candidate(
            attribution_basis="self_first_person",
            attributed_speaker="user",
        )
        result = validate_attribution(candidate, user_id=None)
        assert result.category == "profile"


# ---------------------------------------------------------------------------
# extract_profile schema validation
# ---------------------------------------------------------------------------


class TestExtractProfileInputSchema:
    """Test that extract_profile requires attribution fields."""

    def test_missing_evidence_quote_fails(self):
        """Missing evidence_quote → schema validation failure."""
        assert parse_tool_call("extract_profile", {
            "routing_key": "name",
            "abstract": "Name is Caroline",
            "overview": "## Name\n- Caroline",
            "content": "Caroline's name is Caroline.",
            "confidence": 0.95,
            "attributed_speaker": "user",
            "attribution_basis": "self_first_person",
        }) is None

    def test_missing_attributed_speaker_fails(self):
        """Missing attributed_speaker → schema validation failure."""
        assert parse_tool_call("extract_profile", {
            "routing_key": "name",
            "abstract": "Name is Caroline",
            "overview": "## Name\n- Caroline",
            "content": "Caroline's name is Caroline.",
            "confidence": 0.95,
            "evidence_quote": "I'm Caroline",
            "attribution_basis": "self_first_person",
        }) is None

    def test_missing_attribution_basis_fails(self):
        """Missing attribution_basis → schema validation failure."""
        assert parse_tool_call("extract_profile", {
            "routing_key": "name",
            "abstract": "Name is Caroline",
            "overview": "## Name\n- Caroline",
            "content": "Caroline's name is Caroline.",
            "confidence": 0.95,
            "evidence_quote": "I'm Caroline",
            "attributed_speaker": "user",
        }) is None

    def test_all_fields_passes(self):
        """All required fields → valid model."""
        result = parse_tool_call("extract_profile", {
            "routing_key": "name",
            "abstract": "Name is Caroline",
            "overview": "## Name\n- Caroline",
            "content": "Caroline's name is Caroline.",
            "confidence": 0.95,
            "evidence_quote": "I'm Caroline",
            "attributed_speaker": "user",
            "attribution_basis": "self_first_person",
        })
        assert result is not None
        candidate = result[2]
        assert candidate.evidence_quote == "I'm Caroline"
        assert candidate.attributed_speaker == "user"
        assert candidate.attribution_basis == "self_first_person"


# ---------------------------------------------------------------------------
# parse_tool_call — attribution propagation
# ---------------------------------------------------------------------------


class TestParseToolCallAttribution:
    """Test that parse_tool_call propagates attribution fields."""

    def test_profile_tool_call_carries_attribution(self):
        """Profile tool call → CandidateMemory with attribution fields."""
        result = parse_tool_call(
            "extract_profile",
            {
                "routing_key": "occupation",
                "abstract": "Software engineer",
                "overview": "## Occupation\n- Software engineer",
                "content": "Caroline is a software engineer.",
                "confidence": 0.9,
                "evidence_quote": "I'm a software engineer",
                "attributed_speaker": "user",
                "attribution_basis": "self_first_person",
            },
        )
        assert result is not None
        category, owner_scope, candidate = result
        assert category == "profile"
        assert candidate.evidence_quote == "I'm a software engineer"
        assert candidate.attributed_speaker == "user"
        assert candidate.attribution_basis == "self_first_person"

    def test_non_profile_tool_call_has_no_attribution(self):
        """Non-profile tool call → CandidateMemory with None attribution."""
        result = parse_tool_call(
            "extract_entity",
            {
                "routing_key": "andrew",
                "abstract": "Andrew loves animals",
                "overview": "## Person\n- Andrew",
                "content": "Andrew loves animals.",
                "confidence": 0.9,
                "who": "Andrew",
            },
        )
        assert result is not None
        category, _, candidate = result
        assert category == "entity"
        assert candidate.evidence_quote is None
        assert candidate.attributed_speaker is None
        assert candidate.attribution_basis is None

    def test_profile_without_attribution_fails_validation(self):
        """Profile tool call missing attribution → None (validation failure)."""
        result = parse_tool_call(
            "extract_profile",
            {
                "routing_key": "occupation",
                "abstract": "Software engineer",
                "overview": "## Occupation\n- Software engineer",
                "content": "User is a software engineer.",
                "confidence": 0.9,
                # Missing evidence_quote, attributed_speaker, attribution_basis
            },
        )
        assert result is None
