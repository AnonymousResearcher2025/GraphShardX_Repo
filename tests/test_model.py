from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from graphshardx import Settings
from graphshardx.model import (
    Header,
    IntegrityError,
    Payload,
    canonical,
    holders,
    make_block,
    merge_record,
    validate_certificate,
    validate_vectors,
)
from graphshardx.placement import assign, initial_epoch, weight
from graphshardx.storage import Store


def block(shard=0, n=8, d=4):
    p = Payload(
        7,
        np.arange(n, dtype="u8"),
        np.arange(n, dtype="i8"),
        np.arange(n * d, dtype="f4").reshape(n, d),
        tuple({"frame": i} for i in range(n)),
    )
    return make_block(shard, p)


def test_full_validation_and_serialization():
    h, data = block()
    p = Payload.decode(data, h)
    assert p.metadata[5]["frame"] == 5
    assert not p.vectors.flags.writeable
    damaged = bytearray(data)
    damaged[40] ^= 1
    with pytest.raises(IntegrityError):
        Payload.decode(bytes(damaged), h)
    coordinates = np.ones((1, 960))
    coordinates[0, 500] = np.nan
    with pytest.raises(ValueError):
        validate_vectors(coordinates, 960)
    with pytest.raises(ValueError):
        validate_vectors([[True, False]], 2)
    with pytest.raises(ValueError):
        validate_vectors([[1e100] * 4], 4)
    with pytest.raises(ValueError):
        validate_vectors([[0] * 4], 4, lambda row: row.sum() > 0)
    with pytest.raises(ValueError):
        Header.parse(h.to_dict() | {"count": 0})


def test_quorum_intersection_and_certificate_epochs():
    settings = Settings(nodes=18, leaders=4, shards=2, dimension=4)
    e = initial_epoch(settings, np.random.default_rng(5).normal(size=(20, 4)))
    total = sum(e.weights.values())
    quorums = [
        set(c)
        for n in range(1, len(e.leaders) + 1)
        for c in combinations(e.leaders, n)
        if sum(e.weights[v] for v in c) >= settings.theta * total
    ]
    assert all(a & b for a in quorums for b in quorums)
    h, _ = block()
    c = {
        "block": h.block_id,
        "epoch": 0,
        "path": "primary",
        "approvers": list(e.leaders),
        "committee": [],
    }
    validate_certificate(c, h, e, settings.theta, settings.fallback_f)
    with pytest.raises(IntegrityError):
        validate_certificate(
            c | {"approvers": [e.leaders[0]] * 4}, h, e, settings.theta, settings.fallback_f
        )
    with pytest.raises(IntegrityError):
        validate_certificate(c | {"epoch": 1}, h, e, settings.theta, settings.fallback_f)
    fallback = c | {"path": "fallback", "committee": list(range(5)), "approvers": [0, 1, 2]}
    validate_certificate(fallback, h, e, settings.theta, 2)
    with pytest.raises(IntegrityError):
        validate_certificate(fallback | {"committee": [0, 1, 2]}, h, e, settings.theta, 2)


def test_merge_set_union_observed_remove():
    h, _ = block()
    base = {
        "header": h.to_dict(),
        "certificates": [],
        "committed": False,
        "adds": {},
        "removes": [],
    }
    a = base | {"adds": {"a": {"node": 1, "shard": 0}}}
    b = base | {"adds": {"b": {"node": 2, "shard": 0}}, "removes": ["a"], "committed": True}
    assert canonical(merge_record(a, b)) == canonical(merge_record(b, a))
    assert merge_record(merge_record(a, b), a) == merge_record(a, b)
    assert holders(merge_record(a, b)) == {2: "b"}
    with pytest.raises(IntegrityError):
        merge_record(a, b | {"header": h.to_dict() | {"origin": 10}})


def test_storage_persistence_capacity_and_ledger(tmp_path):
    path = tmp_path / "node.sqlite"
    h, data = block()
    s = Store(path, len(data) * 2)
    assert s.reserve_ids(10) == 0
    token = s.put_payload(h, data, 0, 2, True)
    assert s.put_payload(h, data, 0, 2, True) == token
    assert s.meta("tokens") == 1
    assert s.verify_ledgers() == 1
    s.close()
    s = Store(path, len(data) * 2)
    assert s.reserve_ids(5) == 10
    assert s.get_payload(h.block_id)[0] == data
    s.free_payload(h.block_id, token, {"node": 3, "token": "x", "hash": "p"})
    assert "pointer" in s.get_payload(h.block_id)
    assert s.used == 0
    assert s.header(h.block_id) == h
    s.db.execute("UPDATE ledger SET hash=?", ("f" * 64,))
    with pytest.raises(IntegrityError):
        s.verify_ledgers()
    s.close()


def test_paxos_adopts_highest_and_rejects_stale(tmp_path):
    s = Store(tmp_path / "a.sqlite", 1000)
    assert s.paxos(1, "prepare", [1, 2])["ok"]
    assert s.paxos(1, "accept", [1, 2], {"epoch": 1})["ok"]
    s.close()
    s = Store(tmp_path / "a.sqlite", 1000)
    assert not s.paxos(1, "accept", [0, 9], {"epoch": 2})["ok"]
    with pytest.raises(IntegrityError):
        s.paxos(1, "accept", [1, 2], {"epoch": 3})
    assert s.paxos(1, "prepare", [2, 1])["accepted"] == {"epoch": 1}
    assert not s.paxos(1, "accept", [1, 2], {"epoch": 3})["ok"]
    s.close()


def test_placement_near_duplicates_noise_and_weights():
    s = Settings(nodes=18, leaders=4, shards=2, dimension=4, noise_clusters=2)
    x = np.array([[0] * 4, [0.01] * 4, [100] * 4, [100.01] * 4], dtype="f4")
    e = initial_epoch(s, x)
    a = assign(x, e, s, 42)
    assert a[0] == a[1] and a[2] == a[3] and a[0] != a[2]
    dense = replace(s, dbscan_eps=1, dbscan_min_samples=2)
    assert np.array_equal(assign(x, e, dense, 42), a)
    assert weight(10, 1, 100) == pytest.approx(15)
    assert canonical({2: 0, 11: 1}) == canonical({"2": 0, "11": 1})
