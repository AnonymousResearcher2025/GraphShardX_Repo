import asyncio
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from .config import Settings
from .datasets import Dataset
from .model import Unavailable
from .runtime import Cluster


def provenance(path, settings, backend, queries=None, groundtruth=None, key="train"):
    inputs = {}
    hashes = {}
    for role, source, dataset_key in (
        ("base", path, key),
        ("queries", queries, "test"),
        ("groundtruth", groundtruth, "neighbors"),
    ):
        if source is None:
            continue
        resolved = str(Path(source).resolve())
        if resolved not in hashes:
            checksum = hashlib.sha256()
            with Path(source).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            hashes[resolved] = checksum.hexdigest()
        inputs[role] = {
            "label": role,
            "sha256": hashes[resolved],
            "key": dataset_key if dataset_key in {"train", "test", "neighbors"} else "custom",
        }
    versions = {}
    for package in ("graphshardx", "numpy", "scikit-learn", "networkx", "mpi4py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "dataset": inputs["base"]["label"],
        "dataset_sha256": inputs["base"]["sha256"],
        "inputs": inputs,
        "settings": settings.to_dict(),
        "transport": backend,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "packages": versions,
        "cpu_count": os.cpu_count(),
    }


async def ingest_dataset(
    cluster, dataset, workload, dimension, failure_fraction=0.0, failure_after=0.1
):
    settings = cluster.settings
    if workload > dataset.shape[0] or workload <= 0:
        raise ValueError("workload exceeds dataset or is nonpositive")
    ranges = [
        (workload * n // settings.nodes, workload * (n + 1) // settings.nodes)
        for n in range(settings.nodes)
    ]
    batch_latencies = []
    amortized = []
    committed = 0
    attempted = 0
    pending_blocks = set()
    fail_count = int(settings.nodes * failure_fraction)
    failure_nodes = sorted(
        cluster.epoch.node_weights, key=lambda n: (-cluster.epoch.node_weights[n], n)
    )[:fail_count]
    failed_at = None
    started = time.perf_counter()
    local_workload = sum(ranges[n][1] - ranges[n][0] for n in cluster.origin_ids)

    async def origin(node_id):
        nonlocal committed, attempted, pending_blocks, failed_at
        start, stop = ranges[node_id]
        for offset in range(start, stop, settings.batch_size):
            # Failures happen during ingestion. Failed origins stop; surviving origins keep working.
            if fail_count and failed_at is None and attempted >= local_workload * failure_after:
                failed_at = time.perf_counter() - started
                await cluster.fail(failure_nodes)
            if node_id in cluster.failed:
                break
            values = dataset.slice(offset, min(offset + settings.batch_size, stop), dimension)
            attempted += len(values)
            t = time.perf_counter()
            try:
                result = await cluster.call(node_id, "ingest", {"vectors": values.tolist()})
            except Unavailable:
                continue
            elapsed = time.perf_counter() - t
            committed += result["vectors"]
            pending_blocks.difference_update(result["committed"])
            pending_blocks.update(result["pending"])
            batch_latencies.append(elapsed)
            amortized.append(elapsed / len(values))

    await asyncio.gather(*(origin(n) for n in cluster.origin_ids))
    elapsed = time.perf_counter() - started
    contributions = await cluster.aggregate(
        {
            "committed": committed,
            "attempted": attempted,
            "elapsed": elapsed,
            "batch": batch_latencies,
            "amortized": amortized,
            "pending": len(pending_blocks),
        }
    )
    count = sum(c["committed"] for c in contributions)
    wall = max(c["elapsed"] for c in contributions)
    batches = [v for c in contributions for v in c["batch"]]
    writes = [v for c in contributions for v in c["amortized"]]
    return {
        "committed_vectors": count,
        "attempted_vectors": sum(c["attempted"] for c in contributions),
        "ingestion_seconds": wall,
        "throughput_vectors_per_second": count / wall,
        "mean_commit_latency_seconds": statistics.fmean(batches) if batches else None,
        "mean_amortized_write_seconds": statistics.fmean(writes) if writes else None,
        "pending_blocks": sum(c["pending"] for c in contributions),
        "failed_nodes": failure_nodes,
        "failure_started_seconds": failed_at,
        "origin_ranges": ranges,
    }


async def measure_reads(cluster, ingestion, count=1000):
    candidates = [n for n in range(cluster.settings.nodes) if n not in cluster.failed]
    if not candidates:
        return {"read_requests": 0, "mean_amortized_read_seconds": None}
    query_node = candidates[0]
    rng = np.random.default_rng(cluster.settings.seed)
    times = []
    unavailable = 0
    for _ in range(count):
        origin = int(rng.choice(candidates))
        start, stop = ingestion["origin_ranges"][origin]
        if stop <= start:
            continue
        counter = int(rng.integers(stop - start))
        t = time.perf_counter()
        try:
            await cluster.call(query_node, "read", {"origin": origin, "counter": counter})
            times.append(time.perf_counter() - t)
        except Unavailable:
            unavailable += 1
    return {
        "read_requests": len(times),
        "unavailable_reads": unavailable,
        "mean_amortized_read_seconds": statistics.fmean(times) if times else None,
    }


async def measure_search(cluster, queries, groundtruth, ingestion, probes, fail_random=0.0):
    workload = sum(stop - start for start, stop in ingestion["origin_ranges"])
    if groundtruth.shape[0] < queries.shape[0] or groundtruth.shape[1] < 10:
        raise ValueError("ground truth must contain at least 10 neighbors for every query")
    if fail_random:
        rng = np.random.default_rng(cluster.settings.seed + 100)
        await cluster.fail(
            [
                int(n)
                for n in rng.choice(
                    cluster.settings.nodes, int(cluster.settings.nodes * fail_random), replace=False
                )
            ]
        )
    live = [n for n in range(cluster.settings.nodes) if n not in cluster.failed]
    if not live:
        raise Unavailable("no search origin remains")
    rows = []
    for p in probes:
        recalls = []
        scanned = []
        latencies = []
        missing = []
        for i, query in enumerate(queries):
            truth = set(map(int, groundtruth[i, :10]))
            if any(n < 0 or n >= workload for n in truth):
                raise ValueError(
                    "ground truth must reference this workload, not an unfiltered larger dataset"
                )
            t = time.perf_counter()
            result = await cluster.call(
                live[0], "search", {"query": query.tolist(), "k": 10, "probes": p}
            )
            latencies.append(time.perf_counter() - t)
            found = {
                ingestion["origin_ranges"][r["origin"]][0] + r["counter"]
                for r in result["neighbors"]
            }
            recalls.append(len(found & truth) / 10)
            scanned.append(result["scanned"] / workload)
            missing.append(result["unavailable_blocks"])
        rows.append(
            {
                "probes": p,
                "queries": len(queries),
                "recall_at_10": statistics.fmean(recalls),
                "mean_scanned_fraction": statistics.fmean(scanned),
                "mean_query_latency_seconds": statistics.fmean(latencies),
                "mean_unavailable_blocks": statistics.fmean(missing),
                "random_failed_fraction": fail_random,
            }
        )
    return rows


async def run_case(
    settings,
    dataset,
    workload,
    output,
    backend="local",
    processes=2,
    failure_fraction=0.0,
    reads=1000,
    queries=None,
    groundtruth=None,
    probes=None,
    search_failure=0.0,
):
    sample = dataset.sample(settings.centroid_sample, settings.seed, workload, settings.dimension)
    cluster = await Cluster.create(settings, sample, output, backend, processes)
    try:
        if cluster.comm:
            await asyncio.to_thread(cluster.comm.Barrier)
        result = await ingest_dataset(
            cluster, dataset, workload, settings.dimension, failure_fraction
        )
        await cluster.synchronize()
        if cluster.rank == 0:
            if reads:
                result.update(await measure_reads(cluster, result, reads))
            if queries is not None:
                if groundtruth is None:
                    raise ValueError("search requires exact ground truth")
                result["search"] = await measure_search(
                    cluster,
                    queries,
                    groundtruth,
                    result,
                    probes or [1, 2, 4, 8, 16],
                    search_failure,
                )
            reports = await asyncio.gather(
                *(
                    cluster.try_call(n, "audit", {})
                    for n in range(settings.nodes)
                    if n not in cluster.failed
                )
            )
            reports = [r for r in reports if r]
            result["mean_memory_percent_per_node"] = (
                statistics.fmean(
                    r["report"]["process_rss"]
                    / r["report"]["virtual_nodes_in_process"]
                    / settings.memory_bytes
                    * 100
                    for r in reports
                )
                if reports
                else None
            )
            result["memory_attribution"] = "process RSS divided equally among its virtual nodes"
            result["ledger_entries_checked"] = sum(r["ledger_entries"] for r in reports)
            result["payloads_checked"] = sum(r["payloads"] for r in reports)
            result["protocol_counts"] = (
                {
                    key: sum(r["report"]["metrics"][key] for r in reports)
                    for key in reports[0]["report"]["metrics"]
                }
                if reports
                else {}
            )
            result["transport_counts"] = dict(cluster.transport.metrics)
            result["transport_counts_scope"] = (
                "TCP client process"
                if backend == "tcp"
                else "MPI rank zero"
                if backend == "mpi"
                else "all local virtual nodes"
            )
            result["settings"] = settings.to_dict()
        if cluster.comm:
            # Non-root event loops keep serving messages while root reads and audits.
            result = await asyncio.to_thread(
                cluster.comm.bcast, result if cluster.rank == 0 else None, root=0
            )
        return result
    finally:
        await cluster.close()


def cases(suite, settings):
    workloads = [200000, 400000, 600000, 800000, 1000000]
    if suite == "workload":
        return [(f"workload-{w}", settings, w, 0.0) for w in workloads]
    if suite == "network":
        return [
            (
                f"nodes-{n}",
                replace(
                    settings,
                    nodes=n,
                    shards=min(
                        settings.shards, (n - settings.leaders) // settings.minimum_replicas
                    ),
                    memory_bytes=settings.memory_bytes * settings.nodes // n,
                    storage_bytes=settings.storage_bytes * settings.nodes // n,
                ),
                1000000,
                0.0,
            )
            for n in [100, 200, 300, 400, 500]
        ]
    if suite == "dimension":
        return [
            (f"dimension-{d}", replace(settings, dimension=d), 1000000, 0.0)
            for d in [200, 400, 600, 800, 960]
        ]
    if suite == "network-conditions":
        return [
            (f"delay-{ms}-loss-{loss}", replace(settings, delay=ms / 1000, loss=loss), 1000000, 0.0)
            for ms, loss in [(0, 0), (10, 0), (25, 0), (50, 0), (100, 0), (50, 0.01)]
        ]
    if suite == "ablation":
        return [
            (
                f"sharding-{enabled}-{w}",
                replace(
                    settings,
                    placement="similarity" if enabled else "random",
                    clustering=enabled,
                    rebalancing=enabled,
                ),
                w,
                0.0,
            )
            for enabled in [True, False]
            for w in workloads
        ]
    if suite == "memory":
        return [
            (
                f"memory-{enabled}-{n}",
                replace(
                    settings,
                    nodes=n,
                    shards=min(
                        settings.shards, (n - settings.leaders) // settings.minimum_replicas
                    ),
                    memory_bytes=settings.memory_bytes * settings.nodes // n,
                    storage_bytes=settings.storage_bytes * settings.nodes // n,
                    placement="similarity" if enabled else "random",
                    clustering=enabled,
                    rebalancing=enabled,
                ),
                1000000,
                0.0,
            )
            for enabled in [True, False]
            for n in [100, 200, 300, 400, 500]
        ]
    if suite == "failures":
        return [(f"failed-{f}", settings, 1000000, f) for f in [0.0, 0.1, 0.2, 0.3, 0.4]]
    if suite == "search":
        return [
            (
                f"search-{placement}-{failure}",
                replace(settings, placement=placement),
                1000000,
                failure,
            )
            for placement, failure in [("similarity", 0.0), ("random", 0.0), ("similarity", 0.2)]
        ]
    if suite == "all":
        return [
            case
            for name in [
                "workload",
                "network",
                "dimension",
                "network-conditions",
                "ablation",
                "memory",
                "failures",
                "search",
            ]
            for case in cases(name, settings)
        ]
    raise ValueError(f"unknown suite {suite}")


async def experiment(args):
    settings = Settings.load(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    comm = None
    if args.transport == "mpi":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    root = comm is None or comm.rank == 0
    if root and (output / "results.jsonl").exists():
        raise ValueError("output already contains results; choose a fresh directory")
    info = (
        provenance(args.dataset, settings, args.transport, args.queries, args.groundtruth, args.key)
        if root
        else None
    )
    if root:
        (output / "provenance.json").write_text(json.dumps(info, indent=2))
    with Dataset(args.dataset, args.key) as data:
        query_values = truth_values = None
        if args.queries:
            with Dataset(args.queries, "test") as queries:
                query_values = queries.slice(0, min(args.query_count, queries.shape[0]))
            if not args.groundtruth:
                raise ValueError("--queries requires --groundtruth")
            with Dataset(args.groundtruth, "neighbors") as truth:
                truth_values = truth.slice(0, len(query_values)).astype(np.int64)
        selected = (
            cases(args.suite, settings) if args.suite else [("run", settings, args.workload, 0.0)]
        )
        for name, config, workload, failure in selected:
            is_search = name.startswith("search-")
            if is_search and query_values is None:
                raise ValueError("search suite requires --queries and --groundtruth")
            for repeat in range(args.repeats):
                run_settings = replace(config, seed=config.seed + repeat)
                result = await run_case(
                    run_settings,
                    data,
                    workload,
                    output / f"{name}-{repeat}",
                    args.transport,
                    args.processes,
                    failure_fraction=0 if is_search else failure,
                    reads=args.reads,
                    queries=query_values if is_search or not args.suite else None,
                    groundtruth=truth_values,
                    search_failure=failure if is_search else 0,
                )
                result.update({"case": name, "repeat": repeat})
                if root:
                    with (output / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in result.items()
                                if k
                                in [
                                    "case",
                                    "repeat",
                                    "throughput_vectors_per_second",
                                    "committed_vectors",
                                    "pending_blocks",
                                    "search",
                                ]
                            }
                        ),
                        flush=True,
                    )


def plot_results(input_path, output_path, baseline_paths=(), metric="throughput"):
    import matplotlib.pyplot as plt

    rows = [
        json.loads(line)
        for path in [input_path, *baseline_paths]
        for line in Path(path).read_text().splitlines()
        if line
    ]
    if not rows:
        raise ValueError("no measurements to plot")
    required = {
        "figure2": ["workload-", "nodes-", "dimension-"],
        "ablation": ["sharding-"],
        "memory": ["memory-"],
        "network-conditions": ["delay-"],
        "failures": ["failed-"],
    }
    if any(
        not any(r["case"].startswith(prefix) for r in rows) for prefix in required.get(metric, [])
    ):
        raise ValueError("the requested plot requires measurements from its experiment grid")
    if metric == "search" and not any(r.get("search") for r in rows):
        raise ValueError("search plot requires measured query results")

    def curve(ax, selected, xvalue, ykey, label):
        grouped = {}
        for row in selected:
            if row.get(ykey) is not None:
                grouped.setdefault(xvalue(row), []).append(row[ykey])
        xs = sorted(grouped)
        if xs:
            ax.errorbar(
                xs,
                [statistics.fmean(grouped[x]) for x in xs],
                yerr=[statistics.stdev(grouped[x]) if len(grouped[x]) > 1 else 0 for x in xs],
                marker="o",
                capsize=3,
                label=label,
            )
            ax.legend()

    if metric == "figure2":
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        panels = [
            ("workload-", "throughput_vectors_per_second", "Vectors", "Vectors/s"),
            ("nodes-", "throughput_vectors_per_second", "Nodes", "Vectors/s"),
            ("dimension-", "throughput_vectors_per_second", "Dimensions", "Vectors/s"),
            ("workload-", "mean_amortized_write_seconds", "Vectors", "Write seconds/vector"),
            ("nodes-", "mean_amortized_write_seconds", "Nodes", "Write seconds/vector"),
            ("workload-", "mean_amortized_read_seconds", "Vectors", "Read seconds/vector"),
        ]
        systems = sorted({r.get("system", "GraphShardX") for r in rows})
        for ax, (prefix, key, xlabel, ylabel) in zip(axes.flat, panels, strict=True):
            for system in systems:
                selected = [
                    r
                    for r in rows
                    if r["case"].startswith(prefix) and r.get("system", "GraphShardX") == system
                ]
                curve(ax, selected, lambda r: int(r["case"].rsplit("-", 1)[1]), key, system)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if "seconds" in key:
                ax.set_yscale("log")
    elif metric == "search":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for name in sorted({r["case"] for r in rows if r.get("search")}):
            selected = [point for r in rows if r["case"] == name for point in r.get("search", [])]
            for ax, key, ylabel in zip(
                axes,
                ["recall_at_10", "mean_scanned_fraction", "mean_query_latency_seconds"],
                ["Recall@10", "Scanned fraction", "Query seconds"],
                strict=True,
            ):
                curve(ax, selected, lambda r: r["probes"], key, name)
                ax.set_xlabel("Probed shards")
                ax.set_ylabel(ylabel)
    elif metric in {"ablation", "memory"}:
        keys = (
            ["mean_amortized_read_seconds", "mean_amortized_write_seconds"]
            if metric == "ablation"
            else ["mean_memory_percent_per_node"]
        )
        fig, axes = plt.subplots(
            1, len(keys), figsize=(6 * len(keys), 4), layout="constrained", squeeze=False
        )
        prefix = "sharding-" if metric == "ablation" else "memory-"
        for ax, key in zip(axes.flat, keys, strict=True):
            for enabled in [True, False]:
                selected = [r for r in rows if r["case"].startswith(prefix + str(enabled) + "-")]
                curve(
                    ax,
                    selected,
                    lambda r: int(r["case"].rsplit("-", 1)[1]),
                    key,
                    "Sharding on" if enabled else "Sharding off",
                )
            ax.set_xlabel("Workload vectors" if metric == "ablation" else "Nodes")
            ax.set_ylabel(key.replace("_", " "))
    elif metric in {"network-conditions", "failures"}:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        prefix = "delay-" if metric == "network-conditions" else "failed-"
        selected = [r for r in rows if r["case"].startswith(prefix)]
        groups = (
            sorted({r["settings"]["loss"] for r in selected})
            if metric == "network-conditions"
            else [0]
        )
        for group in groups:
            subset = [r for r in selected if metric == "failures" or r["settings"]["loss"] == group]
            xvalue = (
                (lambda r: r["settings"]["delay"] * 1000)
                if metric == "network-conditions"
                else (lambda r: float(r["case"].split("-")[1]) * 100)
            )
            for ax, key in zip(
                axes, ["throughput_vectors_per_second", "mean_commit_latency_seconds"], strict=True
            ):
                curve(
                    ax,
                    subset,
                    xvalue,
                    key,
                    f"loss={group}" if metric == "network-conditions" else "GraphShardX",
                )
                ax.set_xlabel(
                    "One-way delay (ms)" if metric == "network-conditions" else "Failed nodes (%)"
                )
                ax.set_ylabel(key.replace("_", " "))
    else:
        fig, ax = plt.subplots(
            figsize=(max(7, len({r["case"] for r in rows}) * 0.7), 4), layout="constrained"
        )
        names = list(dict.fromkeys(r["case"] for r in rows))
        systems = sorted({r.get("system", "GraphShardX") for r in rows})
        positions = np.arange(len(names))
        width = 0.8 / len(systems)
        for i, system in enumerate(systems):
            means, deviations = [], []
            for name in names:
                measurements = [
                    r["throughput_vectors_per_second"]
                    for r in rows
                    if r["case"] == name and r.get("system", "GraphShardX") == system
                ]
                means.append(statistics.fmean(measurements) if measurements else np.nan)
                deviations.append(statistics.stdev(measurements) if len(measurements) > 1 else 0)
            ax.bar(
                positions + (i - (len(systems) - 1) / 2) * width,
                means,
                width,
                yerr=deviations,
                capsize=3,
                label=system,
            )
        ax.set_xticks(positions, names)
        ax.legend()
        ax.set_ylabel("Committed vectors / second")
        ax.tick_params(axis="x", labelrotation=45)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
