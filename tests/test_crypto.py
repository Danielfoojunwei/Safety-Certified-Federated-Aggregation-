"""Tests for cryptographic utilities."""

import numpy as np
import pytest

from sca.utils.crypto import MerkleTree, hash_object, hash_tensor, sha256_hash


class TestHashing:
    def test_deterministic(self):
        assert sha256_hash(b"test") == sha256_hash(b"test")

    def test_different_inputs(self):
        assert sha256_hash(b"a") != sha256_hash(b"b")

    def test_hash_object_dict(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        # Canonical JSON should produce same hash regardless of key order
        assert hash_object(d1) == hash_object(d2)

    def test_hash_tensor(self):
        arr = np.array([1.0, 2.0, 3.0])
        h1 = hash_tensor(arr)
        h2 = hash_tensor(arr.copy())
        assert h1 == h2

    def test_hash_tensor_different(self):
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([1.0, 3.0])
        assert hash_tensor(arr1) != hash_tensor(arr2)


class TestMerkleTree:
    def test_single_leaf(self):
        tree = MerkleTree(["leaf1"])
        assert tree.root_hash is not None
        assert len(tree.root_hash) == 64  # SHA-256 hex digest

    def test_multiple_leaves(self):
        tree = MerkleTree(["a", "b", "c", "d"])
        assert tree.root_hash is not None

    def test_deterministic(self):
        t1 = MerkleTree(["a", "b", "c"])
        t2 = MerkleTree(["a", "b", "c"])
        assert t1.root_hash == t2.root_hash

    def test_different_content(self):
        t1 = MerkleTree(["a", "b"])
        t2 = MerkleTree(["a", "c"])
        assert t1.root_hash != t2.root_hash

    def test_order_matters(self):
        t1 = MerkleTree(["a", "b"])
        t2 = MerkleTree(["b", "a"])
        assert t1.root_hash != t2.root_hash

    def test_empty(self):
        tree = MerkleTree([])
        assert tree.root_hash is not None
