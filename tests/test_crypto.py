"""Tests for cryptographic utilities, including the Merkle padding fix (F14).

The old implementation labelled padding leaves one way in ``__init__`` and a
different way in ``get_proof``, so inclusion proofs verified only for
power-of-two leaf counts.  :class:`TestMerkleInclusionProofs` sweeps every
tree size from 1 to 33 and checks that every leaf verifies and that any
tampering is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from sca.utils.crypto import (
    MerkleTree,
    hash_object,
    hash_tensor,
    sha256_hash,
    verify_merkle_proof,
)


class TestHashing:
    def test_deterministic(self):
        assert sha256_hash(b"test") == sha256_hash(b"test")

    def test_different_inputs(self):
        assert sha256_hash(b"a") != sha256_hash(b"b")

    def test_hash_object_key_order_invariant(self):
        assert hash_object({"a": 1, "b": 2}) == hash_object({"b": 2, "a": 1})

    def test_hash_object_value_sensitive(self):
        assert hash_object({"a": 1}) != hash_object({"a": 2})

    def test_hash_tensor(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert hash_tensor(arr) == hash_tensor(arr.copy())

    def test_hash_tensor_different(self):
        assert hash_tensor(np.array([1.0, 2.0])) != hash_tensor(
            np.array([1.0, 3.0])
        )

    def test_hash_tensor_handles_non_contiguous(self):
        arr = np.arange(12.0).reshape(3, 4)
        view = arr[:, ::2]
        assert hash_tensor(view) == hash_tensor(np.ascontiguousarray(view))

    def test_no_dependence_on_pythonhashseed(self):
        """Hashing here must be process-independent (finding F9)."""
        import os
        import pathlib
        import subprocess
        import sys

        import sca

        repo_root = str(pathlib.Path(sca.__file__).resolve().parent.parent)
        code = (
            f"import sys; sys.path.insert(0, {repo_root!r});"
            "from sca.utils.crypto import hash_object;"
            "print(hash_object({'x': 'abc', 'y': [1, 2, 3]}))"
        )
        outs = set()
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outs.add(
                subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=True,
                ).stdout.strip()
            )
        assert len(outs) == 1, outs


class TestMerkleBasics:
    def test_single_leaf(self):
        tree = MerkleTree(["leaf1"])
        assert len(tree.root_hash) == 64
        assert tree.root_hash == hash_object("leaf1")

    def test_deterministic(self):
        assert MerkleTree(["a", "b", "c"]).root_hash == MerkleTree(
            ["a", "b", "c"]
        ).root_hash

    def test_different_content(self):
        assert MerkleTree(["a", "b"]).root_hash != MerkleTree(["a", "c"]).root_hash

    def test_order_matters(self):
        assert MerkleTree(["a", "b"]).root_hash != MerkleTree(["b", "a"]).root_hash

    def test_empty(self):
        tree = MerkleTree([])
        assert len(tree.root_hash) == 64
        with pytest.raises(IndexError):
            tree.get_proof(0)

    def test_different_lengths_differ(self):
        """Padding is bound to the real leaf count, so 3 leaves != 4 leaves."""
        assert MerkleTree(["a", "b", "c"]).root_hash != MerkleTree(
            ["a", "b", "c", "c"]
        ).root_hash


class TestMerkleInclusionProofs:
    """F14: proofs must verify for EVERY tree size, not just powers of two."""

    @pytest.mark.parametrize("n", list(range(1, 34)))
    def test_every_leaf_verifies(self, n):
        leaves = [{"i": i, "payload": f"trace-{i}"} for i in range(n)]
        tree = MerkleTree(leaves)
        for i in range(n):
            proof = tree.get_proof(i)
            assert verify_merkle_proof(leaves[i], proof, tree.root_hash), (
                f"n={n}, leaf {i} failed to verify"
            )

    @pytest.mark.parametrize("n", list(range(1, 34)))
    def test_tampered_leaf_fails(self, n):
        leaves = [{"i": i, "payload": f"trace-{i}"} for i in range(n)]
        tree = MerkleTree(leaves)
        for i in range(n):
            proof = tree.get_proof(i)
            tampered = {"i": i, "payload": f"trace-{i}-TAMPERED"}
            assert not verify_merkle_proof(tampered, proof, tree.root_hash), (
                f"n={n}, tampered leaf {i} verified against the root"
            )

    @pytest.mark.parametrize("n", [3, 5, 7, 9, 17, 33])
    def test_proof_against_wrong_root_fails(self, n):
        leaves = [f"x{i}" for i in range(n)]
        tree = MerkleTree(leaves)
        other = MerkleTree([f"y{i}" for i in range(n)])
        assert not verify_merkle_proof(leaves[0], tree.get_proof(0), other.root_hash)

    @pytest.mark.parametrize("n", [1, 2, 3, 8, 17])
    def test_proof_length_is_log2_of_padded_width(self, n):
        tree = MerkleTree([f"x{i}" for i in range(n)])
        expected = max(0, (n - 1).bit_length()) if n > 1 else 0
        assert len(tree.get_proof(0)) == expected

    def test_swapped_proof_sides_fail(self):
        leaves = [f"x{i}" for i in range(5)]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(1)
        flipped = [
            (h, "left" if s == "right" else "right") for h, s in proof
        ]
        assert not verify_merkle_proof(leaves[1], flipped, tree.root_hash)

    def test_out_of_range_index(self):
        tree = MerkleTree(["a", "b", "c"])
        with pytest.raises(IndexError):
            tree.get_proof(3)
        with pytest.raises(IndexError):
            tree.get_proof(-1)

    def test_legacy_two_argument_call_still_works(self):
        leaves = ["a", "b", "c"]
        tree = MerkleTree(leaves)
        proof = tree.get_proof(1, leaves)
        assert verify_merkle_proof("b", proof, tree.root_hash)

    def test_mismatched_leaves_argument_raises(self):
        tree = MerkleTree(["a", "b", "c"])
        with pytest.raises(ValueError):
            tree.get_proof(0, ["a", "b", "d"])
