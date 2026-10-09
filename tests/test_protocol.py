import asyncio
from dataclasses import replace

import numpy as np
import pytest

from graphshardx import Settings
from graphshardx.model import IntegrityError, holders
from graphshardx.runtime import Cluster


@pytest.fixture
def settings():
    return Settings(
        nodes=18,
        leaders=4,
        shards=2,
        dimension=4,
        fallback_f=1,
        storage_mode="memory",
        operation_timeout=1,
        certification_timeout=0.05,
        rpc_timeout=0.1,
        noise_clusters=2,
        centroid_sample=32,
    )


@pytest.fixture
async def cluster(tmp_path, settings):
    x = np.array([[0] * 4, [0.01] * 4, [10] * 4, [10.01] * 4], dtype="f4")
    c = await Cluster.create(settings, x, tmp_path, maintenance=False)
    yield c
    await c.close()


async def committed(cluster):
    origin = 17
    r = await cluster.call(origin, "ingest", {"vectors": [[0.01] * 4] * 8})
    assert r["vectors"] == 8 and not r["pending"]
    await cluster.synchronize()
    return r["committed"][0], origin


async def test_primary_replication_exact_search_and_retry(cluster):
    b, origin = await committed(cluster)
    record = cluster.nodes[origin].store.record(b)
    assert record["committed"]
    assert record["certificates"][0]["path"] == "primary"
    assert len(holders(record)) == 3
    assert all(cluster.nodes[n].store.get_payload(b) is not None for n in holders(record))
    assert all(cluster.nodes[n].store.get_payload(b) is None for n in cluster.epoch.leaders)
    result = await cluster.call(16, "search", {"query": [0.01] * 4, "k": 10, "probes": 2})
    assert len(result["neighbors"]) == 8 and result["scanned"] == 8
    before = sum(node.store.used for node in cluster.nodes.values())
    header = cluster.nodes[origin].store.header(b)
    data = await cluster.nodes[origin].fetch(record)
    retry = await cluster.nodes[origin].commit_batch([(header, data)])
    assert retry["vectors"] == 0 and retry["committed"] == [b]
    assert sum(node.store.used for node in cluster.nodes.values()) == before


async def test_fallback_after_all_primary_leaders_crash(cluster):
    await cluster.fail(cluster.epoch.leaders)
    b, origin = await committed(cluster)
    c = cluster.nodes[origin].store.record(b)["certificates"][0]
    assert c["path"] == "fallback" and len(c["committee"]) == 3 and len(c["approvers"]) >= 2
    result = await cluster.call(16, "search", {"query": [0.01] * 4, "k": 10, "probes": 2})
    assert len(result["neighbors"]) == 8


async def test_insufficient_replicas_is_certified_but_uncommitted(cluster):
    shard = cluster.epoch.shards[0]
    await cluster.fail(shard["peers"][2:])
    values = [shard["centroid"]] * 4
    result = await cluster.call(17, "ingest", {"vectors": values})
    assert result["vectors"] == 0 and result["certified"] and result["pending"]
    assert cluster.nodes[17].store.pending()
    await cluster.synchronize()
    search = await cluster.call(16, "search", {"query": values[0], "k": 10, "probes": 2})
    assert search["neighbors"]


async def test_copy_before_delete_pointer_and_repair(cluster):
    b, origin = await committed(cluster)
    record = cluster.nodes[origin].store.record(b)
    source = next(iter(holders(record)))
    shard = cluster.epoch.shard_for(cluster.nodes[source].store.header(b))
    target = next(n for n in cluster.epoch.shards[shard]["peers"] if n not in holders(record))
    assert await cluster.nodes[source].migrate(b, target)
    assert "pointer" in cluster.nodes[source].store.get_payload(b)
    assert await cluster.nodes[origin].fetch(record)
    await cluster.synchronize()
    moved = cluster.nodes[cluster.epoch.leaders[0]].store.record(b)
    assert source not in holders(moved) and target in holders(moved)
    await cluster.fail([target])
    live_holder = min(n for n in holders(moved) if n != target)
    repaired = await cluster.nodes[live_holder].repair()
    assert repaired["repaired"] == 1
    updated = cluster.nodes[live_holder].store.record(b)
    assert len([n for n in holders(updated) if n not in cluster.failed]) == 3


async def test_failed_migration_preserves_source(cluster):
    b, origin = await committed(cluster)
    record = cluster.nodes[origin].store.record(b)
    source = next(iter(holders(record)))
    target = next(
        n for n in range(18) if n not in holders(record) and n not in cluster.epoch.leaders
    )
    await cluster.fail([target])
    from graphshardx.model import Unavailable

    with pytest.raises(Unavailable):
        await cluster.nodes[source].migrate(b, target)
    assert isinstance(cluster.nodes[source].store.get_payload(b), tuple)


async def test_sidechain_relocation_keeps_header_and_awards_tokens(cluster):
    b, origin = await committed(cluster)
    record = cluster.nodes[origin].store.record(b)
    pair = next(
        (source, target)
        for source in holders(record)
        for target in cluster.epoch.leaders
        if target not in cluster.epoch.adjacency[source]
    )
    source, target = pair
    original = record["header"]
    assert await cluster.nodes[source].migrate(b, target, relocated=True)
    assert cluster.nodes[target].store.meta("tokens") == 1
    assert cluster.nodes[target].store.header(b).to_dict() == original
    assert (
        cluster.nodes[target]
        .store.db.execute("SELECT 1 FROM ledger WHERE kind='sidechain' AND id=?", (b,))
        .fetchone()
    )
    assert "pointer" in cluster.nodes[source].store.get_payload(b)


async def test_origin_crash_after_certification_does_not_report_commit(cluster):
    shard = cluster.epoch.shards[0]
    origin = 17
    calls = 0
    for peer in shard["peers"]:
        original = cluster.nodes[peer].accept_payload

        async def crash_origin(args, original=original):
            nonlocal calls
            calls += 1
            if calls == 3:
                cluster.nodes[origin].alive = False
                from graphshardx.model import Unavailable

                raise Unavailable("origin power loss during replication")
            return await original(args)

        cluster.nodes[peer].accept_payload = crash_origin
    from graphshardx.model import Unavailable

    with pytest.raises(Unavailable):
        await cluster.call(origin, "ingest", {"vectors": [shard["centroid"]] * 8})
    assert cluster.nodes[origin].metrics["committed"] == 0
    assert cluster.nodes[origin].store.pending()
    assert all(
        not r["committed"] for n in cluster.nodes.values() for r in n.store.records().values()
    )


async def test_corrupt_replica_detection_and_healthy_retry(cluster):
    b, origin = await committed(cluster)
    record = cluster.nodes[origin].store.record(b)
    source = next(iter(holders(record)))
    cluster.nodes[source].store.db.execute("UPDATE payloads SET data=? WHERE id=?", (b"bad", b))
    assert await cluster.nodes[origin].fetch(record)
    for n in holders(record):
        cluster.nodes[n].store.db.execute("UPDATE payloads SET data=? WHERE id=?", (b"bad", b))
    with pytest.raises(IntegrityError):
        await cluster.nodes[origin].fetch(record)
    with pytest.raises(IntegrityError):
        await cluster.nodes[origin].search([0.01] * 4, 10, 2)


async def test_lost_duplicate_messages_real_resend(tmp_path, settings):
    s = replace(settings, loss=0.15, duplicate=0.3, operation_timeout=5)
    c = await Cluster.create(
        s, np.array([[0] * 4, [10] * 4], dtype="f4"), tmp_path, maintenance=False
    )
    try:
        r = await c.call(17, "ingest", {"vectors": [[0.01] * 4] * 8})
        assert r["vectors"] == 8
        assert c.transport.metrics["dropped"] > 0
        assert sum(n.store.used > 0 for n in c.nodes.values()) >= 3
    finally:
        await c.close()


async def test_late_storage_acknowledgments_survive_rpc_deadlines(tmp_path, settings):
    s = replace(settings, rpc_timeout=0.03, certification_timeout=0.02, operation_timeout=3)
    c = await Cluster.create(
        s, np.array([[0] * 4, [10] * 4], dtype="f4"), tmp_path, maintenance=False
    )
    exchange = c.transport._exchange

    async def delayed(src, dst, method, args):
        if method in {"header", "payload"}:
            await asyncio.sleep(0.12)
        return await exchange(src, dst, method, args)

    c.transport._exchange = delayed
    try:
        r = await c.call(17, "ingest", {"vectors": [[0.01] * 4] * 8})
        assert r["vectors"] == 8 and not r["pending"]
        b = r["committed"][0]
        record = c.nodes[17].store.record(b)
        assert len(holders(record)) >= s.replicas
        assert all(c.nodes[n].store.get_payload(b) is not None for n in holders(record))
        assert not any(key[0] == 17 and key[3] == b for key in c.transport.replies)
    finally:
        await c.close()
    assert not c.transport.reply_tasks


async def test_epoch_agreement_header_handoff_and_no_consensus_for_inserts(cluster):
    b, origin = await committed(cluster)
    assert not any(
        n.store.db.execute("SELECT slot FROM paxos").fetchone() for n in cluster.nodes.values()
    )
    old = cluster.epoch
    response = await cluster.nodes[old.leaders[0]].advance_epoch()
    assert response["epoch"] == 1
    assert all(n.epoch.number == 1 for n in cluster.nodes.values())
    for n in cluster.nodes[origin].epoch.leaders:
        assert cluster.nodes[n].store.record(b)
    assert await cluster.nodes[16].read_vector(origin, 0)
    # A partitioned outgoing majority cannot change the epoch; data ingestion uses the old one.
    current = cluster.nodes[origin].epoch
    await cluster.fail(current.leaders[:3])
    from graphshardx.model import Unavailable

    with pytest.raises(Unavailable):
        await cluster.nodes[current.leaders[-1]].advance_epoch()
    assert cluster.nodes[16].epoch.number == 1


async def test_partition_fallback_and_set_union_after_heal(cluster):
    leaders = set(cluster.epoch.leaders)
    # Both sides have enough shard peers and committee members, with no primary quorum on one side.
    all_nodes = list(range(18))
    left = set(all_nodes[::2])
    right = set(all_nodes[1::2])
    cluster.transport.partitions = [left, right]
    origins = [max(left - leaders), max(right - leaders)]
    replies = await asyncio.gather(
        *(
            cluster.call(n, "ingest", {"vectors": [cluster.epoch.shards[0]["centroid"]] * 4})
            for n in origins
        )
    )
    assert all(r["vectors"] == 4 for r in replies)
    cluster.transport.partitions = []
    await cluster.synchronize()
    for n in cluster.epoch.leaders:
        assert all(b in cluster.nodes[n].store.records() for r in replies for b in r["committed"])
