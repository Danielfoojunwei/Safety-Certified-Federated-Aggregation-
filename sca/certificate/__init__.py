"""Certificate schema and acceptance rule for safety-certified aggregation.

The acceptance rule consumes ``estimation_stats`` only.  ``K`` (number of
regions) and ``M`` (per-region budget cap) are declared *a priori* by the
caller and must never be derived from the data -- deriving ``K`` by dropping
empty buckets gave the most degenerate run the smallest union-bound penalty
(audit finding F10).  A region with positive weight and zero samples
contributes ``UCB = 1.0`` and is never dropped; an empty statistics list yields
a vacuous certificate (``bound = 1.0``, ``accepted = False``), not the
``(True, 0.0)`` the old code returned for zero evidence.
"""

__all__: list[str] = []
