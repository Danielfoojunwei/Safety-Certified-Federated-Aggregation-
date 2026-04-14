"""Cryptographic utilities for certificate construction.

Implements collision-resistant hashing and Merkle tree construction
for the certificate schema (Section 4.2):
    C = (h(theta), h(V), h(G), {p_hat_j}, {UCB_j}, epsilon, delta, Decision, TraceDigest)

TraceDigest is a Merkle root committing to the recursion tree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def sha256_hash(data: bytes) -> str:
    """Collision-resistant hash using SHA-256."""
    return hashlib.sha256(data).hexdigest()


def hash_object(obj: Any) -> str:
    """Hash an arbitrary serializable object via canonical JSON + SHA-256."""
    canonical = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return sha256_hash(canonical)


def hash_tensor(tensor) -> str:
    """Hash a PyTorch tensor or numpy array by hashing its raw bytes."""
    import numpy as np
    if hasattr(tensor, "detach"):
        # PyTorch tensor
        arr = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        arr = np.array(tensor)
    return sha256_hash(arr.tobytes())


@dataclass
class MerkleNode:
    """Node in a Merkle tree for trace commitment."""
    hash_value: str
    data: Any = None
    left: MerkleNode | None = None
    right: MerkleNode | None = None


class MerkleTree:
    """Merkle tree for committing to verifier traces.

    Constructs a binary Merkle tree over a sequence of leaf data items.
    The root hash serves as the TraceDigest in the certificate.
    """

    def __init__(self, leaves: list[Any]) -> None:
        """Build a Merkle tree from leaf data items.

        Args:
            leaves: List of serializable objects (trace entries).
        """
        if not leaves:
            self.root = MerkleNode(hash_value=sha256_hash(b"empty"))
            return

        # Create leaf nodes
        leaf_nodes = []
        for item in leaves:
            h = hash_object(item)
            leaf_nodes.append(MerkleNode(hash_value=h, data=item))

        # Pad to power of 2 with unique hashes to prevent collision attacks
        pad_idx = 0
        while len(leaf_nodes) & (len(leaf_nodes) - 1):
            pad_data = f"__merkle_pad_{pad_idx}_{len(leaf_nodes)}__".encode("utf-8")
            leaf_nodes.append(MerkleNode(hash_value=sha256_hash(pad_data)))
            pad_idx += 1

        # Build tree bottom-up
        level = leaf_nodes
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                combined = (level[i].hash_value + level[i + 1].hash_value).encode()
                parent = MerkleNode(
                    hash_value=sha256_hash(combined),
                    left=level[i],
                    right=level[i + 1],
                )
                next_level.append(parent)
            level = next_level

        self.root = level[0]

    @property
    def root_hash(self) -> str:
        """The Merkle root hash (TraceDigest)."""
        return self.root.hash_value

    def get_proof(self, index: int, leaves: list[Any]) -> list[tuple[str, str]]:
        """Generate a Merkle inclusion proof for a given leaf index.

        Args:
            index: Index of the leaf to prove.
            leaves: Original leaf list (for rebuilding tree structure).

        Returns:
            List of (sibling_hash, side) pairs forming the proof path.
        """
        n = len(leaves)
        # Pad
        padded = list(leaves)
        while len(padded) & (len(padded) - 1):
            padded.append(None)

        hashes = [
            hash_object(x) if x is not None
            else sha256_hash(f"__merkle_pad_{i}_{len(padded)}__".encode("utf-8"))
            for i, x in enumerate(padded)
        ]

        proof = []
        idx = index
        while len(hashes) > 1:
            next_hashes = []
            for i in range(0, len(hashes), 2):
                if i == idx - (idx % 2):
                    sibling = hashes[i + 1] if idx % 2 == 0 else hashes[i]
                    side = "right" if idx % 2 == 0 else "left"
                    proof.append((sibling, side))
                combined = (hashes[i] + hashes[i + 1]).encode()
                next_hashes.append(sha256_hash(combined))
            hashes = next_hashes
            idx //= 2

        return proof
