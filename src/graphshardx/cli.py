import argparse
import asyncio
import json
import logging
from pathlib import Path
import tempfile

import numpy as np

from .config import Settings


async def sanity(args):
    from .runtime import Cluster

    settings = Settings.load(args.config) if args.config else Settings()
    rng = np.random.default_rng(settings.seed)
    # Fixture data checks correctness only. Scientific runs always require a supplied dataset.
    vectors = rng.normal(size=(96, settings.dimension)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="graphshardx-sanity-") as directory:
        cluster = await Cluster.create(
            settings, vectors, Path(directory), args.transport, args.processes, maintenance=False
        )
        try:
            result = (
                await cluster.call(0, "ingest", {"vectors": vectors.tolist()})
                if cluster.rank == 0
                else None
            )
            if cluster.comm:
                result = await asyncio.to_thread(cluster.comm.bcast, result, root=0)
            if result["vectors"] != len(vectors) or result["pending"]:
                raise RuntimeError("sanity ingestion did not commit every fixture vector")
            await cluster.synchronize()
            if cluster.rank != 0:
                await asyncio.to_thread(cluster.comm.Barrier)
                return
            search = await cluster.call(
                1, "search", {"query": vectors[5].tolist(), "k": 10, "probes": settings.shards}
            )
            if not search["neighbors"] or search["neighbors"][0]["counter"] != 5:
                raise RuntimeError("sanity search did not find the exact vector")
            point = await cluster.call(1, "read", {"origin": 0, "counter": 5})
            if not np.array_equal(np.asarray(point["vector"], dtype=np.float32), vectors[5]):
                raise RuntimeError("sanity retrieval returned a different vector")
            audits = await asyncio.gather(
                *(cluster.call(n, "audit", {}) for n in range(settings.nodes))
            )
            if args.transport != "mpi" or cluster.rank == 0:
                print(
                    json.dumps(
                        {
                            "transport": args.transport,
                            "committed_vectors": result["vectors"],
                            "blocks": len(result["committed"]),
                            "replicas_verified": sum(a["payloads"] for a in audits),
                            "nearest_counter": search["neighbors"][0]["counter"],
                            "ledger_entries_verified": sum(a["ledger_entries"] for a in audits),
                        }
                    )
                )
            if cluster.comm:
                await asyncio.to_thread(cluster.comm.Barrier)
        finally:
            await cluster.close()


def parser():
    p = argparse.ArgumentParser(
        prog="graphshardx", description="GraphShardX vector data management"
    )
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    commands = p.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "sanity", help="verify real ingestion, replicated storage, search and ledgers"
    )
    check.add_argument("--config", type=Path)
    check.add_argument("--transport", choices=["local", "tcp", "mpi"], default="local")
    check.add_argument("--processes", type=int, default=2)
    server = commands.add_parser("serve", help="run the nodes assigned to one TCP worker")
    server.add_argument("--manifest", type=Path, required=True)
    server.add_argument("--rank", type=int, required=True)
    run = commands.add_parser("experiment", help="measure GraphShardX on real supplied vector data")
    baseline = commands.add_parser(
        "baseline", help="measure a real Qdrant, Weaviate or Pinecone endpoint"
    )
    energy = commands.add_parser(
        "energy", help="measure idle-subtracted processor energy using Intel RAPL"
    )
    for command in (run, baseline, energy):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--key", default="train")
        command.add_argument(
            "--workload", type=int, default=1000000 if command != energy else 200000
        )
        command.add_argument("--output", type=Path, required=True)
    for command in (run, baseline):
        command.add_argument("--queries", type=Path)
        command.add_argument("--groundtruth", type=Path)
        command.add_argument("--query-count", type=int, default=1000)
        command.add_argument("--reads", type=int, default=1000)
    for command in (run, energy):
        command.add_argument("--transport", choices=["local", "tcp", "mpi"], default="local")
        command.add_argument("--processes", type=int, default=2)
        command.add_argument("--repeats", type=int, default=5 if command == energy else 1)
    run.add_argument(
        "--suite",
        choices=[
            "workload",
            "network",
            "dimension",
            "network-conditions",
            "ablation",
            "memory",
            "search",
            "failures",
            "all",
        ],
    )
    baseline.add_argument("--suite", choices=["workload", "network", "dimension", "all"])
    baseline.add_argument("--repeats", type=int, default=1)
    for command in (baseline, energy):
        command.add_argument(
            "--system",
            choices=["qdrant", "weaviate", "pinecone"]
            if command == baseline
            else ["graphshardx", "qdrant", "weaviate"],
            required=True,
        )
        command.add_argument("--endpoint", required=command == baseline)
        command.add_argument("--idle-seconds", type=float, default=300 if command == energy else 0)
    energy.add_argument("--rapl-root", default="/sys/class/powercap")
    plot = commands.add_parser("plot", help="plot measured throughput and repeated-run variability")
    plot.add_argument("--input", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--baselines", nargs="*", type=Path, default=[])
    plot.add_argument(
        "--metric",
        choices=[
            "throughput",
            "figure2",
            "search",
            "ablation",
            "memory",
            "network-conditions",
            "failures",
        ],
        default="throughput",
    )
    geo = commands.add_parser("geospatial", help="extract declared geometry and climate attributes")
    geo.add_argument("--input", type=Path, required=True)
    geo.add_argument("--output", type=Path, required=True)
    geo.add_argument("--columns", nargs="+", required=True)
    rpc = commands.add_parser("request", help="send a JSON request to a TCP worker manifest")
    rpc.add_argument("--manifest", type=Path, required=True)
    rpc.add_argument("--node", type=int, required=True)
    rpc.add_argument(
        "--method",
        required=True,
        choices=[
            "ingest",
            "insert",
            "flush",
            "read",
            "search",
            "audit",
            "epoch",
            "monitor",
            "repair",
        ],
    )
    rpc.add_argument("--json", type=Path, help="request arguments as a JSON file")
    return p


async def dispatch(args):
    if args.command == "sanity":
        await sanity(args)
    elif args.command == "serve":
        from .runtime import serve

        await serve(args.manifest, args.rank)
    elif args.command == "experiment":
        from .benchmark import experiment

        await experiment(args)
    elif args.command == "baseline":
        from .baselines import run_baseline

        await run_baseline(args)
    elif args.command == "energy":
        from .energy import run_energy

        if args.transport == "mpi":
            raise ValueError("RAPL measurement is single-host; use TCP workers or local transport")
        if args.idle_seconds <= 0 or args.repeats < 1:
            raise ValueError("energy idle interval and repeats must be positive")
        await run_energy(args)
    elif args.command == "plot":
        from .benchmark import plot_results

        plot_results(args.input, args.output, args.baselines, args.metric)
    elif args.command == "geospatial":
        from .datasets import geospatial

        geospatial(args.input, args.output, args.columns)
    elif args.command == "request":
        from .transport import TCPTransport

        manifest = json.loads(args.manifest.read_text())
        transport = TCPTransport(Settings(**manifest["settings"]), manifest["addresses"])
        value = json.loads(args.json.read_text()) if args.json else {}
        result = await transport.call(args.node, args.node, args.method, value, 120)
        print(json.dumps(result))


def main():
    p = parser()
    args = p.parse_args()
    if getattr(args, "queries", None) and not getattr(args, "groundtruth", None):
        p.error("--queries requires --groundtruth")
    for name in ("processes", "repeats", "workload", "query_count"):
        if hasattr(args, name) and getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(dispatch(args))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        if args.log_level == "DEBUG":
            raise
        raise SystemExit(1) from None
