import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import uuid
from types import SimpleNamespace

import numpy as np

from .datasets import Dataset


def service_info(response):
    version = response.get("version")
    if isinstance(version, str) and re.fullmatch(r"v?\d+(?:\.\d+){1,3}", version):
        return {"version": version}
    return {}


class Baseline:
    """A real database endpoint is required. Service errors abort the measurement."""

    def __init__(self, system, endpoint, dimension, name=None):
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError("endpoint must be HTTP(S), without embedded credentials")
        self.system = system
        self.endpoint = endpoint.rstrip("/")
        self.dimension = dimension
        self.name = name or "Graphshardx" + uuid.uuid4().hex
        self.headers = {"Content-Type": "application/json"}
        if system == "qdrant" and os.environ.get("QDRANT_API_KEY"):
            self.headers["api-key"] = os.environ["QDRANT_API_KEY"]
        elif system == "weaviate" and os.environ.get("WEAVIATE_API_KEY"):
            self.headers["Authorization"] = "Bearer " + os.environ["WEAVIATE_API_KEY"]
        elif system == "pinecone":
            self.headers["Api-Key"] = os.environ["PINECONE_API_KEY"]
            self.headers["X-Pinecone-API-Version"] = "2025-10"
        if system not in {"qdrant", "weaviate", "pinecone"}:
            raise ValueError("unsupported baseline")

    async def request(self, method, path, body=None):
        def invoke():
            req = Request(
                self.endpoint + path,
                data=json.dumps(body, allow_nan=False).encode() if body is not None else None,
                headers=self.headers,
                method=method,
            )
            try:
                with urlopen(req, timeout=120) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else {}
            except HTTPError as exc:
                raise RuntimeError(
                    f"{self.system} HTTP {exc.code}: {exc.read().decode()[:500]}"
                ) from exc
            except URLError as exc:
                raise RuntimeError(f"{self.system} endpoint is unavailable: {exc.reason}") from exc

        return await asyncio.to_thread(invoke)

    async def initialize(self):
        if self.system == "qdrant":
            await self.request(
                "PUT",
                f"/collections/{self.name}",
                {"vectors": {"size": self.dimension, "distance": "Euclid"}},
            )
            return service_info(await self.request("GET", "/"))
        if self.system == "weaviate":
            await self.request(
                "POST",
                "/v1/schema",
                {
                    "class": self.name,
                    "vectorizer": "none",
                    "vectorIndexType": "hnsw",
                    "vectorIndexConfig": {"distance": "l2-squared"},
                    "properties": [{"name": "row", "dataType": ["int"]}],
                },
            )
            return service_info(await self.request("GET", "/v1/meta"))
        return service_info(await self.request("POST", "/describe_index_stats", {}))

    @staticmethod
    def object_id(row):
        return str(uuid.UUID(int=(1 << 96) + row))

    async def insert(self, start, vectors):
        if self.system == "qdrant":
            result = await self.request(
                "PUT",
                f"/collections/{self.name}/points?wait=true",
                {
                    "points": [
                        {"id": start + i, "vector": v.tolist()} for i, v in enumerate(vectors)
                    ]
                },
            )
            if result.get("result", {}).get("status") != "completed":
                raise RuntimeError("Qdrant did not acknowledge a completed write")
        elif self.system == "weaviate":
            result = await self.request(
                "POST",
                "/v1/batch/objects",
                {
                    "objects": [
                        {
                            "class": self.name,
                            "id": self.object_id(start + i),
                            "vector": v.tolist(),
                            "properties": {"row": start + i},
                        }
                        for i, v in enumerate(vectors)
                    ]
                },
            )
            if (
                not isinstance(result, list)
                or len(result) != len(vectors)
                or any(r.get("result", {}).get("errors") for r in result)
            ):
                raise RuntimeError("Weaviate returned missing or failed object writes")
        else:
            result = await self.request(
                "POST",
                "/vectors/upsert",
                {
                    "namespace": self.name,
                    "vectors": [
                        {"id": str(start + i), "values": v.tolist()} for i, v in enumerate(vectors)
                    ],
                },
            )
            if result.get("upsertedCount") != len(vectors):
                raise RuntimeError("Pinecone did not acknowledge every vector")
        return len(vectors)

    async def read(self, row):
        if self.system == "qdrant":
            r = await self.request(
                "POST", f"/collections/{self.name}/points", {"ids": [row], "with_vector": True}
            )
            if len(r["result"]) != 1:
                raise RuntimeError("Qdrant retrieval did not return the inserted vector")
            return r["result"][0]["vector"]
        if self.system == "weaviate":
            return (await self.request("GET", f"/v1/objects/{self.name}/{self.object_id(row)}"))[
                "vector"
            ]
        r = await self.request(
            "GET", "/vectors/fetch?" + urlencode({"ids": str(row), "namespace": self.name})
        )
        return r["vectors"][str(row)]["values"]

    async def search(self, query, k=10):
        if self.system == "qdrant":
            r = await self.request(
                "POST",
                f"/collections/{self.name}/points/query",
                {"query": query.tolist(), "limit": k},
            )
            return [int(p["id"]) for p in r["result"]["points"]]
        if self.system == "weaviate":
            query_text = (
                "{Get{"
                + self.name
                + "(nearVector:{vector:"
                + json.dumps(query.tolist())
                + "},limit:"
                + str(k)
                + "){_additional{id}}}}"
            )
            r = await self.request("POST", "/v1/graphql", {"query": query_text})
            if r.get("errors"):
                raise RuntimeError(f"Weaviate query failed: {r['errors']}")
            return [
                uuid.UUID(p["_additional"]["id"]).int - (1 << 96)
                for p in r["data"]["Get"][self.name]
            ]
        r = await self.request(
            "POST", "/query", {"namespace": self.name, "vector": query.tolist(), "topK": k}
        )
        return [int(p["id"]) for p in r["matches"]]


async def run_baseline_case(args, settings, provenance_info):

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("output already contains results")
    with Dataset(args.dataset, args.key) as data:
        if args.workload > data.shape[0]:
            raise ValueError("workload exceeds dataset")
        client = Baseline(args.system, args.endpoint, settings.dimension)
        service = await client.initialize()
        if args.idle_seconds:
            await asyncio.sleep(args.idle_seconds)
        count = 0
        batch_times = []
        writes = []
        start_time = time.perf_counter()

        async def origin(n):
            nonlocal count
            start, stop = (
                args.workload * n // settings.nodes,
                args.workload * (n + 1) // settings.nodes,
            )
            for offset in range(start, stop, settings.batch_size):
                values = data.slice(
                    offset, min(stop, offset + settings.batch_size), settings.dimension
                )
                t = time.perf_counter()
                count += await client.insert(offset, values)
                duration = time.perf_counter() - t
                batch_times.append(duration)
                writes.append(duration / len(values))

        await asyncio.gather(*(origin(n) for n in range(settings.nodes)))
        duration = time.perf_counter() - start_time
        read_times = []
        rng = np.random.default_rng(settings.seed)
        for _ in range(args.reads):
            row = int(rng.integers(args.workload))
            t = time.perf_counter()
            actual = await client.read(row)
            read_times.append(time.perf_counter() - t)
            if not np.allclose(
                actual, data.slice(row, row + 1, settings.dimension)[0], rtol=1e-5, atol=1e-6
            ):
                raise RuntimeError("baseline returned a different vector")
        result = {
            "system": args.system,
            "endpoint": f"{args.system}-service",
            "collection": client.name,
            "service": service,
            "acknowledged_vectors": count,
            "ingestion_seconds": duration,
            "throughput_vectors_per_second": count / duration,
            "mean_commit_latency_seconds": statistics.fmean(batch_times),
            "mean_amortized_write_seconds": statistics.fmean(writes),
            "mean_amortized_read_seconds": statistics.fmean(read_times) if read_times else None,
            "settings": settings.to_dict(),
            "provenance": provenance_info | {"settings": settings.to_dict()},
        }
        if args.queries:
            with (
                Dataset(args.queries, "test") as q,
                Dataset(args.groundtruth, "neighbors") as truth,
            ):
                recall = []
                for i in range(min(args.query_count, q.shape[0])):
                    found = await client.search(q.slice(i, i + 1, settings.dimension)[0])
                    correct = set(map(int, truth.slice(i, i + 1)[0, :10]))
                    recall.append(len(set(found) & correct) / 10)
                result["recall_at_10"] = statistics.fmean(recall)
        (output / "results.json").write_text(json.dumps(result, indent=2))
        return result


async def run_baseline(args):
    from .benchmark import cases, provenance
    from .config import Settings

    settings = Settings.load(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.jsonl").exists():
        raise ValueError("output already contains baseline results")
    info = provenance(
        args.dataset, settings, "HTTP client API", args.queries, args.groundtruth, args.key
    )
    with Dataset(args.dataset, args.key) as dataset:
        full_workload = dataset.shape[0]
    if args.suite == "all":
        selected = [
            case for name in ["workload", "network", "dimension"] for case in cases(name, settings)
        ]
    else:
        selected = (
            cases(args.suite, settings) if args.suite else [("run", settings, args.workload, 0.0)]
        )
    for name, config, workload, _ in selected:
        for repeat in range(args.repeats):
            run_args = SimpleNamespace(
                **(vars(args) | {"workload": workload, "output": output / f"{name}-{repeat}"})
            )
            if args.suite and (name.startswith("dimension-") or workload != full_workload):
                run_args.queries = None
            result = await run_baseline_case(
                run_args, replace(config, seed=config.seed + repeat), info
            )
            result.update({"case": name, "repeat": repeat})
            with (output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in [
                            "case",
                            "repeat",
                            "system",
                            "acknowledged_vectors",
                            "throughput_vectors_per_second",
                        ]
                    }
                ),
                flush=True,
            )
