from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Settings:
    nodes: int = 24
    leaders: int = 5
    shards: int = 4
    dimension: int = 16
    attachment: int = 2
    replicas: int = 3
    minimum_replicas: int = 3
    fallback_f: int = 2
    theta: float = 0.51
    beta: float = 0.8
    certification_timeout: float = 0.1
    rpc_timeout: float = 0.5
    operation_timeout: float = 30.0
    retry_interval: float = 0.05
    batch_size: int = 256
    batch_timeout: float = 0.1
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    noise_clusters: int = 8
    centroid_sample: int = 4096
    rho: float = 0.8
    moves_per_round: int = 2
    minimum_split_peers: int = 6
    storage_bytes: int = 268435456
    memory_bytes: int = 268435456
    heartbeat_interval: float = 0.5
    failure_timeout: float = 2.0
    monitor_interval: float = 2.0
    repair_interval: float = 3.0
    gossip_interval: float = 1.0
    epoch_interval: float = 30.0
    uptime_window: int = 10
    relocation_tokens: int = 1
    workers: int = 4
    seed: int = 42
    delay: float = 0.0
    loss: float = 0.0
    duplicate: float = 0.0
    placement: str = "similarity"
    clustering: bool = True
    rebalancing: bool = True
    epoch_mode: str = "consensus"
    storage_mode: str = "disk"

    def __post_init__(self):
        integers = (
            "nodes",
            "leaders",
            "shards",
            "dimension",
            "attachment",
            "replicas",
            "minimum_replicas",
            "batch_size",
            "dbscan_min_samples",
            "noise_clusters",
            "centroid_sample",
            "moves_per_round",
            "minimum_split_peers",
            "storage_bytes",
            "memory_bytes",
            "workers",
            "uptime_window",
            "relocation_tokens",
        )
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.fallback_f) is not int or self.fallback_f < 0:
            raise ValueError("fallback_f must be a nonnegative integer")
        if not 0.5 < self.theta <= 1 or not 0 < self.beta <= 1 or not 0 < self.rho < 1:
            raise ValueError("require 0.5 < theta <= 1, 0 < beta <= 1, 0 < rho < 1")
        if self.leaders > self.nodes or self.attachment >= self.nodes:
            raise ValueError("leader count and attachment exceed network size")
        if not self.minimum_replicas <= self.replicas:
            raise ValueError("minimum_replicas cannot exceed replicas")
        if self.nodes - self.leaders < self.shards * self.minimum_replicas:
            raise ValueError("each initial shard needs minimum_replicas distinct peers")
        if self.nodes < 2 * self.fallback_f + 1:
            raise ValueError("network cannot hold a 2f+1 committee")
        for name in (
            "certification_timeout",
            "rpc_timeout",
            "operation_timeout",
            "retry_interval",
            "batch_timeout",
            "dbscan_eps",
            "heartbeat_interval",
            "failure_timeout",
            "monitor_interval",
            "repair_interval",
            "gossip_interval",
            "epoch_interval",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.delay) or self.delay < 0:
            raise ValueError("delay must be finite and nonnegative")
        if not 0 <= self.loss < 1 or not 0 <= self.duplicate < 1:
            raise ValueError("loss and duplicate must be in [0,1)")
        if self.failure_timeout <= self.heartbeat_interval:
            raise ValueError("failure_timeout must exceed heartbeat_interval")
        if self.placement not in {"similarity", "random"}:
            raise ValueError("placement must be similarity or random")
        if self.epoch_mode not in {"consensus", "harness"}:
            raise ValueError("epoch_mode must be consensus or harness")
        if self.storage_mode not in {"disk", "memory"}:
            raise ValueError("storage_mode must be disk or memory")

    @property
    def primary_timeout(self):
        return self.certification_timeout + 2 * self.delay

    @property
    def request_timeout(self):
        return self.rpc_timeout + 2 * self.delay

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path, **overrides):
        with Path(path).open("rb") as stream:
            values = tomllib.load(stream)
        allowed = {field.name for field in fields(cls)}
        unknown = values.keys() - allowed
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        return cls(**(values | overrides))
