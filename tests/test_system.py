import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from graphshardx import Settings
from graphshardx.benchmark import run_case
from graphshardx.datasets import Dataset
from graphshardx.model import Header, holders
from graphshardx.runtime import Cluster


def config(**overrides):
    return replace(
        Settings(
            nodes=18,
            leaders=4,
            shards=2,
            dimension=4,
            fallback_f=1,
            operation_timeout=3,
            rpc_timeout=0.2,
            noise_clusters=2,
        ),
        **overrides,
    )


async def test_storage_restart_preserves_provenance_and_counter(tmp_path):
    s = config()
    data = np.array([[0] * 4, [1] * 4, [10] * 4, [11] * 4], dtype="f4")
    c = await Cluster.create(s, data, tmp_path, maintenance=False)
    r = await c.call(
        17, "ingest", {"vectors": data.tolist(), "metadata": [{"frame": i} for i in range(4)]}
    )
    assert r["vectors"] == 4
    await c.synchronize()
    await c.close()
    c = await Cluster.create(s, data, tmp_path, maintenance=False)
    try:
        record = await c.call(16, "read", {"origin": 17, "counter": 2})
        assert record["metadata"] == {"frame": 2}
        r = await c.call(17, "ingest", {"vectors": [[2] * 4]})
        assert r["identifiers"] == [[17, 4]]
    finally:
        await c.close()


async def test_batch_timeout_and_size_flush(tmp_path):
    s = config(storage_mode="memory", batch_timeout=0.02, batch_size=3)
    c = await Cluster.create(
        s, np.array([[0] * 4, [10] * 4], dtype="f4"), tmp_path, maintenance=False
    )
    try:
        r = await c.call(17, "insert", {"vector": [0] * 4, "metadata": {"image": 1}})
        assert r["identifier"] == [17, 0] and not r["pending"]
        replies = await asyncio.gather(
            *(c.call(17, "insert", {"vector": [i] * 4}) for i in range(3))
        )
        assert len({tuple(r["identifier"]) for r in replies}) == 3
        assert c.nodes[17].metrics["committed"] == 4
    finally:
        await c.close()


async def test_split_keeps_headers_and_moves_child_payloads(tmp_path):
    s = config(storage_mode="memory", storage_bytes=3000, rho=0.05)
    sample = np.array([[0] * 4, [2] * 4, [100] * 4, [102] * 4], dtype="f4")
    c = await Cluster.create(s, sample, tmp_path, maintenance=False)
    try:
        shard_id = min(c.epoch.shards, key=lambda n: c.epoch.shards[n]["centroid"][0])
        blocks = []
        for value in [0, 2]:
            r = await c.call(17, "ingest", {"vectors": [[value] * 4] * 8})
            blocks += r["committed"]
        await c.synchronize()
        records = c.nodes[17].store.records()
        headers = {b: records[b]["header"] for b in blocks}
        # Seed the full shard with real replicas, then trigger the actual monitor and epoch.
        for b in blocks:
            record = records[b]
            h = Header.parse(record["header"])
            data = await c.nodes[17].fetch(record)
            for peer in c.epoch.shards[shard_id]["peers"]:
                await c.nodes[17].call(
                    peer,
                    "payload",
                    {
                        "header": h.to_dict(),
                        "data": data,
                        "certificate": record["certificates"][-1],
                    },
                )
        await c.synchronize()
        peer = c.epoch.shards[shard_id]["peers"][0]
        # A peer with stored payloads becomes a leader at the same boundary as the split.
        c.nodes[peer].store.set_meta("tokens", 1000000)
        assert shard_id in (await c.nodes[peer].monitor())["splits"]
        outcome = await c.call(c.epoch.leaders[0], "epoch", {})
        assert outcome["shards"] == 3
        current = c.nodes[17].epoch
        assert peer in current.leaders
        assert all(not set(s["peers"]) & set(current.leaders) for s in current.shards.values())
        assert shard_id in current.retired
        assert set(current.block_shards[b] for b in blocks) == set(current.retired[shard_id])
        for _ in range(3):
            await c.synchronize()
            await asyncio.gather(*(c.nodes[n].repair() for n in range(18)))
        await c.synchronize()
        for b in blocks:
            r = c.nodes[current.leaders[0]].store.record(b)
            assert r["header"] == headers[b]
            peers = set(current.shards[current.block_shards[b]]["peers"])
            assert len(peers & set(holders(r))) >= 3
            assert not isinstance(c.nodes[peer].store.get_payload(b), tuple)
        result = await c.nodes[16].search([2] * 4, 10, len(current.shards))
        assert result["scanned"] == 16
    finally:
        await c.close()


async def test_rebalancing_ablation_preserves_shards_across_epochs(tmp_path):
    s = config(storage_mode="memory", storage_bytes=3000, rho=0.01, rebalancing=False)
    c = await Cluster.create(
        s, np.array([[0] * 4, [10] * 4], dtype="f4"), tmp_path, maintenance=False
    )
    try:
        r = await c.call(17, "ingest", {"vectors": [[0] * 4] * 8})
        assert r["vectors"] == 8
        before = set(c.epoch.shards)
        leader = c.epoch.leaders[0]
        c.nodes[leader].store.set_meta("splits", list(before))
        await c.call(leader, "epoch", {})
        assert set(c.nodes[17].epoch.shards) == before
    finally:
        await c.close()


async def test_multiple_missed_epochs_are_learned_in_order(tmp_path):
    s = config(storage_mode="memory")
    c = await Cluster.create(
        s, np.array([[0] * 4, [10] * 4], dtype="f4"), tmp_path, maintenance=False
    )
    try:
        missed = 17
        await c.fail([missed])
        for _ in range(2):
            leader = c.nodes[16].epoch.leaders[0]
            await c.call(leader, "epoch", {})
        await c.recover([missed])
        assert c.nodes[missed].epoch.number == 2
        assert set(c.nodes[missed].epochs) == {0, 1, 2}
    finally:
        await c.close()


async def test_real_tcp_workers_and_measured_experiment(tmp_path):
    rng = np.random.default_rng(18)
    data = rng.normal(size=(54, 4)).astype("f4")
    np.save(tmp_path / "input.npy", data)
    settings = config(storage_mode="disk", batch_size=8, epoch_interval=100)
    with Dataset(tmp_path / "input.npy") as dataset:
        result = await run_case(settings, dataset, 54, tmp_path / "run", "tcp", 2, reads=4)
    assert result["committed_vectors"] == 54
    assert result["throughput_vectors_per_second"] == 54 / result["ingestion_seconds"]
    assert result["payloads_checked"] >= 3 * settings.nodes
    assert result["mean_amortized_read_seconds"] > 0
    assert len(list((tmp_path / "run").glob("node-*.sqlite"))) == settings.nodes


def test_mpi_transport_in_separate_processes():
    launcher = shutil.which("mpiexec") or str(Path(sys.executable).parent / "mpiexec")
    if not Path(launcher).exists():
        pytest.skip("MPI launcher is optional; install MPI to run this integration test")
    pytest.importorskip("mpi4py")
    environment = dict(os.environ)
    environment["LOKY_MAX_CPU_COUNT"] = "2"
    result = subprocess.run(
        [launcher, "-n", "2", sys.executable, "-m", "graphshardx", "sanity", "--transport", "mpi"],
        capture_output=True,
        text=True,
        timeout=90,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout.strip())
    assert measured["committed_vectors"] == 96 and measured["replicas_verified"] == 12
