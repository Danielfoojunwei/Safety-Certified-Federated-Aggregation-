"""Cryptographic primitives for certificate commitment.

Provides canonical hashing and a Merkle tree used as the ``TraceDigest`` in
the safety certificate.

All hashing here is process-independent (SHA-256 over canonical JSON / raw
bytes).  Nothing in this module uses Python's builtin ``hash()``, which is
salted per-process unless ``PYTHONHASHSEED`` is set -- that was finding F9,
and the project-wide replacement lives in ``sca/utils/seeding.stable_hash``.

MERKLE PADDING (finding F14).  The previous implementation labelled padding
leaves in two mutually inconsistent ways: ``__init__`` used
``f"__merkle_pad_{counter}_{current_length}__"`` where *both* fields changed
on every append, while ``get_proof`` used
``f"__merkle_pad_{absolute_index}_{final_length}__"``.  The two agreed only
when no padding was needed at all, i.e. only for power-of-two leaf counts, so
inclusion proofs silently failed to verify for every other tree size.  There
is now exactly one padding rule, :func:`_pad_leaf_hash`, used by construction,
proof generation and verification alike.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "sha256_hash",
    "hash_object",
    "hash_tensor",
    "MerkleNode",
    "MerkleTree",
    "verify_merkle_proof",
]


def sha256_hash(data: bytes) -> str:
    """Collision-resistant hash using SHA-256, returned as a hex digest."""
    return hashlib.sha256(data).hexdigest()


def hash_object(obj: Any) -> str:
    """Hash an arbitrary serializable object via canonical JSON + SHA-256.

    Keys are sorted so that dictionaries with the same content hash the same
    regardless of insertion order.  Non-JSON types fall back to ``str``.
    """
    canonical = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return sha256_hash(canonical)


def hash_tensor(tensor) -> str:
    """Hash a PyTorch tensor or numpy array by hashing its raw bytes."""
    import numpy as np

    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        arr = np.array(tensor)
    return sha256_hash(np.ascontiguousarray(arr).tobytes())


def _next_power_of_two(n: int) -> int:
    """Smallest power of two ``>= n`` (with ``_next_power_of_two(1) == 1``)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pad_leaf_hash(index: int, n_leaves: int, n_padded: int) -> str:
    """THE single padding rule, used by build, proof and verify alike.

    Args:
        index: Absolute position of the pad leaf in the padded array.
        n_leaves: Number of real leaves (bound in so trees of different real
            sizes that pad to the same width still differ).
        n_padded: Padded width of the leaf array.

    Returns:
        The hex digest for that pad slot.
    """
    label = f"__merkle_pad__idx={index}__n={n_leaves}__padded={n_padded}__"
    return sha256_hash(label.encode("utf-8"))


def _leaf_hashes(leaves: list[Any]) -> list[str]:
    """Hashes of the padded leaf array for a given real leaf list."""
    n = len(leaves)
    n_padded = _next_power_of_two(n)
    hashes = [hash_object(item) for item in leaves]
    for i in range(n, n_padded):
        hashes.append(_pad_leaf_hash(i, n, n_padded))
    return hashes


def _combine(left: str, right: str) -> str:
    return sha256_hash((left + right).encode("utf-8"))


@dataclass
class MerkleNode:
    """Node in a Merkle tree for trace commitment."""

    hash_value: str
    data: Any = None
    left: "MerkleNode | None" = None
    right: "MerkleNode | None" = None


EMPTY_ROOT = sha256_hash(b"__merkle_empty__")


class MerkleTree:
    """Binary Merkle tree over a sequence of leaf data items.

    The root hash serves as the ``TraceDigest`` in the safety certificate.
    Leaf counts that are not powers of two are padded with deterministic,
    position-bound pad hashes (see :func:`_pad_leaf_hash`), so that inclusion
    proofs verify for *every* tree size.
    """

    def __init__(self, leaves: list[Any]) -> None:
        """Build a Merkle tree from leaf data items.

        Args:
            leaves: List of serializable objects (trace entries).
        """
        self.leaves: list[Any] = list(leaves)
        self.n_leaves = len(self.leaves)

        if self.n_leaves == 0:
            self.root = MerkleNode(hash_value=EMPTY_ROOT)
            self.n_padded = 0
            return

        self.n_padded = _next_power_of_two(self.n_leaves)
        hashes = _leaf_hashes(self.leaves)

        level = [
            MerkleNode(
                hash_value=h,
                data=(self.leaves[i] if i < self.n_leaves else None),
            )
            for i, h in enumerate(hashes)
        ]
        while len(level) > 1:
            level = [
                MerkleNode(
                    hash_value=_combine(
                        level[i].hash_value, level[i + 1].hash_value
                    ),
                    left=level[i],
                    right=level[i + 1],
                )
                for i in range(0, len(level), 2)
            ]
        self.root = level[0]

    @property
    def root_hash(self) -> str:
        """The Merkle root hash (``TraceDigest``)."""
        return self.root.hash_value

    def get_proof(
        self, index: int, leaves: list[Any] | None = None
    ) -> list[tuple[str, str]]:
        """Generate a Merkle inclusion proof for a leaf index.

        Args:
            index: Index of the leaf to prove, in ``[0, n_leaves)``.
            leaves: Optional leaf list. Ignored unless it differs from the
                stored one; kept for backwards compatibility with the old
                two-argument signature. Supplying a *different* list raises,
                because a proof against a different leaf set is meaningless.

        Returns:
            List of ``(sibling_hash, side)`` pairs from leaf to root, where
            ``side`` is the side the SIBLING sits on.

        Raises:
            IndexError: If the tree is empty or ``index`` is out of range.
            ValueError: If ``leaves`` is supplied and differs from the tree's.
        """
        if leaves is not None and list(leaves) != self.leaves:
            raise ValueError(
                "get_proof(leaves=...) must match the leaves the tree was "
                "built from; pass nothing to use the stored leaves."
            )
        if self.n_leaves == 0:
            raise IndexError("cannot prove inclusion in an empty Merkle tree")
        if not (0 <= index < self.n_leaves):
            raise IndexError(
                f"index {index} out of range for {self.n_leaves} leaves"
            )

        hashes = _leaf_hashes(self.leaves)
        proof: list[tuple[str, str]] = []
        idx = index
        while len(hashes) > 1:
            sibling_idx = idx ^ 1
            side = "right" if idx % 2 == 0 else "left"
            proof.append((hashes[sibling_idx], side))
            hashes = [
                _combine(hashes[i], hashes[i + 1])
                for i in range(0, len(hashes), 2)
            ]
            idx //= 2
        return proof


def verify_merkle_proof(
    leaf: Any, proof: list[tuple[str, str]], root: str
) -> bool:
    """Verify a Merkle inclusion proof produced by :meth:`MerkleTree.get_proof`.

    Args:
        leaf: The original leaf data item.
        proof: ``(sibling_hash, side)`` pairs from leaf to root.
        root: The expected root hash.

    Returns:
        ``True`` iff recomputing the path from ``leaf`` reproduces ``root``.
    """
    current = hash_object(leaf)
    for sibling, side in proof:
        if side == "right":
            current = _combine(current, sibling)
        elif side == "left":
            current = _combine(sibling, current)
        else:
            raise ValueError(f"invalid proof side {side!r}")
    return current == root
