"""Analysis-specific enumerations."""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    """Misclassification taxonomy categories, ordered by classification priority."""

    OOS_AS_INSCOPE = "oos_as_inscope"
    INSCOPE_AS_OOS = "inscope_as_oos"
    NEAR_SEMANTIC_CONFUSION = "near_semantic_confusion"
    CROSS_DOMAIN_CONFUSION = "cross_domain_confusion"
    SHORT_QUERY_AMBIGUITY = "short_query_ambiguity"


class LengthBucket(StrEnum):
    """Utterance length buckets based on whitespace-split token count."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class ScopeSlice(StrEnum):
    """Supervised class scope, independent of class frequency."""

    IN_SCOPE = "in_scope"
    OOS = "oos"
