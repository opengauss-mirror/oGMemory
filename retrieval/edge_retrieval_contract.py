"""Stable origins and outcomes shared by semantic graph retrieval and tracing."""
from enum import StrEnum


class RetrievalOrigin(StrEnum):
    VECTOR_ANCHOR = "vector_anchor"
    GRAPH_NEIGHBOR = "graph_neighbor"


class GraphExpansionOutcome(StrEnum):
    DISABLED = "disabled"
    NO_EDGES = "no_edges"
    ALL_REJECTED = "all_rejected"
    ADDED_NEIGHBORS = "added_neighbors"
    GRAPH_FAILURE = "graph_failure"
    QUERY_EMBEDDING_FAILURE = "query_embedding_failure"
    HYDRATION_REJECTION = "hydration_rejection"
