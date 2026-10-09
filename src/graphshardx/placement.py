import math

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from .config import Settings
from .model import Epoch


def weight(degree: int, uptime: float, tokens: int) -> float:
    return degree * (1 + 0.3 * uptime + 0.2 * tokens / 100)


def replica_factor(peers, free: dict[int, int], settings: Settings):
    eligible = sum(free[p] > 0 for p in peers)
    return max(settings.minimum_replicas, min(settings.replicas, eligible, len(peers)))


def initial_epoch(settings: Settings, sample: np.ndarray):
    if len(sample) < settings.shards or sample.shape[1] != settings.dimension:
        raise ValueError("initial centroid sample must cover every shard and match dimension")
    graph = nx.barabasi_albert_graph(settings.nodes, settings.attachment, seed=settings.seed)
    adjacency = {int(n): tuple(sorted(graph.neighbors(n))) for n in graph}
    weights = {n: weight(len(adjacency[n]), 1.0, 0) for n in graph}
    leaders = tuple(sorted(weights, key=lambda n: (-weights[n], n))[: settings.leaders])
    centers = (
        KMeans(n_clusters=settings.shards, random_state=settings.seed, n_init=10)
        .fit(sample)
        .cluster_centers_
    )
    peers = np.array_split(
        np.asarray([n for n in range(settings.nodes) if n not in leaders]), settings.shards
    )
    free = dict.fromkeys(range(settings.nodes), settings.storage_bytes)
    shards = {
        s: {
            "centroid": centers[s].tolist(),
            "peers": [int(n) for n in group],
            "replicas": replica_factor([int(n) for n in group], free, settings),
        }
        for s, group in enumerate(peers)
    }
    return Epoch(
        0, leaders, {n: weights[n] for n in leaders}, weights, shards, adjacency, {}, {}, free
    )


def assign(vectors: np.ndarray, epoch: Epoch, settings: Settings, seed: int):
    rng = np.random.default_rng(seed)
    shard_ids = sorted(epoch.shards)
    if not settings.clustering:
        return np.full(len(vectors), int(rng.choice(shard_ids)), dtype=int)
    labels = DBSCAN(
        eps=settings.dbscan_eps, min_samples=settings.dbscan_min_samples, n_jobs=1
    ).fit_predict(vectors)
    noise = np.flatnonzero(labels == -1)
    if len(noise):
        k = min(settings.noise_clusters, len(noise), len(np.unique(vectors[noise], axis=0)))
        noise_labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
            vectors[noise]
        )
        labels[noise] = noise_labels + int(labels.max()) + 1
    centroids = np.asarray([epoch.shards[s]["centroid"] for s in shard_ids], dtype=np.float64)
    assignments = np.empty(len(vectors), dtype=int)
    for cluster in np.unique(labels):
        indices = np.flatnonzero(labels == cluster)
        center = vectors[indices].mean(axis=0, dtype=np.float64)
        target = int(np.argmin(np.sum((centroids - center) ** 2, axis=1)))
        assignments[indices] = shard_ids[target]
    if settings.placement == "random":
        # Randomize complete similarity-aware blocks, retaining their vector grouping.
        targets = {s: int(rng.choice(shard_ids)) for s in np.unique(assignments)}
        assignments = np.asarray([targets[s] for s in assignments], dtype=int)
    return assignments


def candidate_pools(epoch: Epoch, shards: list[int]):
    leaders = list(epoch.leaders)
    count = min(len(shards), len(leaders))
    pools = [leaders[i::count] for i in range(count)] if count else []
    return {s: pools[i % count] for i, s in enumerate(shards)}


def find_leader(epoch: Epoch, shard: int, pool: list[int], reports: dict, latencies: dict):
    peers = set(epoch.shards[shard]["peers"])
    return min(
        pool,
        key=lambda n: (
            -len(peers & set(epoch.adjacency[n])),
            -reports.get(n, {}).get("tokens", 0),
            latencies.get(n, math.inf),
            n,
        ),
    )
