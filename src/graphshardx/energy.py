import asyncio
import json
from pathlib import Path
import statistics
import time

from .benchmark import ingest_dataset, provenance
from .config import Settings
from .datasets import Dataset
from .runtime import Cluster


class RAPL:
    """Sample package counters often enough to account for counter wraparound."""

    def __init__(self, root=Path("/sys/class/powercap")):
        self.paths = sorted(
            p
            for p in Path(root).glob("intel-rapl:*")
            if p.name.count(":") == 1 and (p / "energy_uj").exists()
        )
        if not self.paths:
            raise RuntimeError(
                "Intel RAPL package counters are unavailable; no energy estimate is produced"
            )
        self.maximum = [int((p / "max_energy_range_uj").read_text()) for p in self.paths]
        self.previous = [int((p / "energy_uj").read_text()) for p in self.paths]
        self.joules = 0.0

    def sample(self):
        current = [int((p / "energy_uj").read_text()) for p in self.paths]
        delta = sum(
            (new - old) % maximum
            for new, old, maximum in zip(current, self.previous, self.maximum, strict=True)
        )
        self.joules += delta / 1e6
        self.previous = current
        return self.joules

    async def monitor(self):
        while True:
            await asyncio.sleep(0.05)
            self.sample()


async def run_energy(args):
    from .baselines import Baseline

    rapl = RAPL(Path(args.rapl_root))
    settings = Settings.load(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "energy.json").exists():
        raise ValueError("output already contains energy results")
    info = provenance(args.dataset, settings, args.transport, key=args.key)
    sampler = asyncio.create_task(rapl.monitor())
    results = []
    try:
        with Dataset(args.dataset, args.key) as data:
            for repeat in range(args.repeats):
                cluster = None
                client = None
                service = None
                if args.system == "graphshardx":
                    cluster = await Cluster.create(
                        settings,
                        data.sample(
                            settings.centroid_sample,
                            settings.seed,
                            args.workload,
                            settings.dimension,
                        ),
                        output / f"run-{repeat}",
                        args.transport,
                        args.processes,
                    )
                else:
                    if not args.endpoint:
                        raise ValueError("baseline energy measurement requires --endpoint")
                    client = Baseline(args.system, args.endpoint, settings.dimension)
                    service = await client.initialize()
                try:
                    idle_start_energy = rapl.sample()
                    idle_start = time.perf_counter()
                    # The CLI enforces the paper's five-minute service-idle interval by default.
                    await asyncio.sleep(args.idle_seconds)
                    idle_seconds = time.perf_counter() - idle_start
                    idle_power = (rapl.sample() - idle_start_energy) / idle_seconds
                    start_energy = rapl.sample()
                    start = time.perf_counter()
                    if cluster:
                        ingestion = await ingest_dataset(
                            cluster, data, args.workload, settings.dimension
                        )
                        count = ingestion["committed_vectors"]
                    else:
                        count = 0

                        async def origin(n, client=client):
                            start, stop = (
                                args.workload * n // settings.nodes,
                                args.workload * (n + 1) // settings.nodes,
                            )
                            inserted = 0
                            for offset in range(start, stop, settings.batch_size):
                                inserted += await client.insert(
                                    offset,
                                    data.slice(
                                        offset,
                                        min(stop, offset + settings.batch_size),
                                        settings.dimension,
                                    ),
                                )
                            return inserted

                        count = sum(
                            await asyncio.gather(*(origin(n) for n in range(settings.nodes)))
                        )
                    elapsed = time.perf_counter() - start
                    total = rapl.sample() - start_energy
                    active = total - idle_power * elapsed
                    results.append(
                        {
                            "repeat": repeat,
                            "system": args.system,
                            "service": service,
                            "collection": client.name if client else None,
                            "idle_power_watts": idle_power,
                            "idle_seconds": idle_seconds,
                            "ingestion_seconds": elapsed,
                            "package_joules": total,
                            "idle_subtracted_joules": active,
                            "inserted_vectors": count,
                            "joules_per_1000_inserts": active / count * 1000 if count else None,
                        }
                    )
                finally:
                    if cluster:
                        await cluster.close()
        measurements = [
            r["joules_per_1000_inserts"]
            for r in results
            if r["joules_per_1000_inserts"] is not None
        ]
        result = {
            "runs": results,
            "mean": statistics.fmean(measurements) if measurements else None,
            "standard_deviation": statistics.stdev(measurements) if len(measurements) > 1 else None,
            "scope": "processor packages on this host; excludes radio and DRAM",
            "settings": settings.to_dict(),
            "provenance": info,
        }
        (output / "energy.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        sampler.cancel()
        await asyncio.gather(sampler, return_exceptions=True)
