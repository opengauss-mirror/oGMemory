"""Shared cosine arithmetic with caller-specific validation policies."""
import math


def cosine_similarity(left, right, *, strict_dimensions=False,
                      require_finite=False, combined_norm=False):
    if not left or not right or len(left) != len(right):
        if strict_dimensions:
            raise ValueError("embedding dimensions do not match")
        return 0.0
    if require_finite and not all(math.isfinite(x) for x in (*left, *right)):
        raise ValueError("Non-finite graph vector")
    left_squared = sum(x * x for x in left)
    right_squared = sum(x * x for x in right)
    # Retain each existing caller's floating-point evaluation order.
    denominator = (math.sqrt(left_squared * right_squared) if combined_norm
                   else math.sqrt(left_squared) * math.sqrt(right_squared))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0
