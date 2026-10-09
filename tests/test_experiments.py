import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import struct
import threading

import numpy as np
import pytest

from graphshardx import Settings
from graphshardx.baselines import Baseline, service_info
from graphshardx.benchmark import cases, ingest_dataset, provenance, run_case
from graphshardx.datasets import Dataset
from graphshardx.energy import RAPL
from graphshardx.model import Header
from graphshardx.runtime import Cluster
from graphshardx.transport import decode, encode


def test_texmex_and_hdf5_data_validation(tmp_path):
    import h5py

    data = np.arange(24, dtype="f4").reshape(6, 4)
    path = tmp_path / "vectors.fvecs"
    with path.open("wb") as stream:
        for row in data:
            stream.write(struct.pack("<i", 4) + row.tobytes())
    with Dataset(path) as ds:
        assert np.array_equal(ds.slice(1, 3), data[1:3])
        assert ds.sample(3, 42).shape == (3, 4)
        with pytest.raises(ValueError):
            ds.slice(0, 7)
    with h5py.File(tmp_path / "vectors.hdf5", "w") as f:
        f["train"] = data
    with Dataset(tmp_path / "vectors.hdf5") as ds:
        assert np.array_equal(ds.slice(0, 6), data)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError):
        Dataset(path)


def test_provenance_omits_paths_filenames_and_machine_names(tmp_path, monkeypatch):
    import graphshardx.benchmark as benchmark

    private = tmp_path / "private-account" / "unpublished-project"
    private.mkdir(parents=True)
    source = private / "private-data.npy"
    np.save(source, np.ones((2, 4), dtype="f4"))
    monkeypatch.setattr(benchmark.platform, "platform", lambda: "private-workstation")
    monkeypatch.setattr(benchmark.platform, "node", lambda: "private-workstation")
    info = provenance(source, Settings(), "local", source, source, "private-key")
    serialized = json.dumps(info)
    assert "private-" not in serialized and str(tmp_path) not in serialized
    assert info["dataset"] == "base" and info["inputs"]["base"]["key"] == "custom"
    assert info["dataset_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert set(info["inputs"]) == {"base", "queries", "groundtruth"}


def test_service_info_retains_only_public_numeric_versions():
    assert service_info({"version": "1.2.3", "hostname": "private-host", "modules": {}}) == {
        "version": "1.2.3"
    }
    assert service_info({"version": "1.2.3-custom", "hostname": "private-host"}) == {}


async def test_exact_recall_and_timing_are_measured(tmp_path):
    rng = np.random.default_rng(9)
    values = rng.normal(size=(90, 4)).astype("f4")
    queries = values[:3]
    truth = np.argsort(np.sum((queries[:, None, :] - values[None, :, :]) ** 2, axis=2), axis=1)[
        :, :10
    ]
    np.save(tmp_path / "real-input.npy", values)
    settings = Settings(
        nodes=18,
        leaders=4,
        shards=2,
        dimension=4,
        fallback_f=1,
        storage_mode="memory",
        batch_size=5,
        noise_clusters=2,
    )
    with Dataset(tmp_path / "real-input.npy") as ds:
        result = await run_case(
            settings,
            ds,
            90,
            tmp_path / "run",
            reads=0,
            queries=queries,
            groundtruth=truth,
            probes=[2],
        )
    assert result["committed_vectors"] == 90
    assert result["search"][0]["recall_at_10"] == 1.0
    assert result["search"][0]["mean_scanned_fraction"] == 1.0
    assert result["mean_commit_latency_seconds"] > 0
    assert result["protocol_counts"]["committed"] == 90
    from graphshardx.benchmark import plot_results

    (tmp_path / "results.jsonl").write_text(json.dumps(result | {"case": "run"}) + "\n")
    plot_results(tmp_path / "results.jsonl", tmp_path / "throughput.png")
    assert (tmp_path / "throughput.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="requires"):
        plot_results(tmp_path / "results.jsonl", tmp_path / "absent.png", metric="figure2")


def test_paper_experiment_grids_and_ablation():
    s = Settings(nodes=500, leaders=15, shards=64, dimension=960)
    assert [w for _, _, w, _ in cases("workload", s)] == [200000, 400000, 600000, 800000, 1000000]
    assert [c.nodes for _, c, _, _ in cases("network", s)] == [100, 200, 300, 400, 500]
    assert [c.dimension for _, c, _, _ in cases("dimension", s)] == [200, 400, 600, 800, 960]
    assert cases("network-conditions", s)[-1][1].loss == 0.01
    off = [c for _, c, _, _ in cases("ablation", s) if not c.rebalancing]
    assert all(not c.clustering and c.placement == "random" for c in off)
    assert all(c.shards == 64 for _, c, _, _ in cases("search", s))


async def test_pending_metrics_count_unique_blocks_after_retries(tmp_path):
    settings = Settings(
        nodes=18,
        leaders=4,
        shards=2,
        dimension=4,
        fallback_f=1,
        storage_mode="memory",
        batch_size=1,
        operation_timeout=0.3,
        certification_timeout=0.03,
        rpc_timeout=0.05,
        noise_clusters=2,
    )
    values = np.zeros((36, 4), dtype="f4")
    np.save(tmp_path / "vectors.npy", values)
    cluster = await Cluster.create(
        settings, np.array([[0] * 4, [10] * 4], dtype="f4"), tmp_path, maintenance=False
    )
    try:
        await cluster.fail(cluster.epoch.shards[0]["peers"][2:])
        with Dataset(tmp_path / "vectors.npy") as data:
            result = await ingest_dataset(cluster, data, 36, 4)
        pending = sum(len(n.store.pending()) for n in cluster.nodes.values())
        assert pending > 0 and result["pending_blocks"] == pending
        assert result["committed_vectors"] == 0
    finally:
        await cluster.close()


def test_counter_wrap_and_no_missing_hardware_estimate(tmp_path):
    path = tmp_path / "intel-rapl:0"
    path.mkdir()
    (path / "max_energy_range_uj").write_text("1000000")
    (path / "energy_uj").write_text("900000")
    meter = RAPL(tmp_path)
    (path / "energy_uj").write_text("100000")
    assert meter.sample() == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="unavailable"):
        RAPL(tmp_path / "absent")


def test_compact_header_and_binary_transport():
    h = Header(0, 1, 0, 0, 1, 4, 1, 1, "a" * 64)
    assert len(h.to_bytes()) == 80
    assert Header.from_bytes(h.to_bytes()) == h
    message = {
        "header": h.to_dict(),
        "payload": b"\x00\x01\xff",
        "metadata": {"__bytes__": "literal"},
    }
    assert decode(encode(message)) == message


async def test_baseline_service_failure_never_uses_a_local_fallback():
    baseline = Baseline("qdrant", "http://127.0.0.1:1", 4)
    with pytest.raises(RuntimeError, match="unavailable"):
        await baseline.initialize()


async def test_qdrant_http_write_acknowledgment_contract():
    # A local HTTP fixture checks the adapter contract, and is never used by experiments.
    class Service(BaseHTTPRequestHandler):
        def do_PUT(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert len(body["points"]) == 2 and self.path.endswith("?wait=true")
            response = json.dumps({"result": {"status": "completed"}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = Baseline("qdrant", f"http://127.0.0.1:{server.server_port}", 4)
        assert await client.insert(0, np.ones((2, 4), dtype="f4")) == 2
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join()
