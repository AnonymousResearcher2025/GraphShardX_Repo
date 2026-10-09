from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from collections.abc import Callable

import numpy as np


class GraphShardXError(Exception):
    pass


class IntegrityError(GraphShardXError):
    pass


class Unavailable(GraphShardXError):
    pass


class CapacityError(GraphShardXError):
    pass


def canonical(value) -> bytes:
    def normalize(v):
        if isinstance(v, dict):
            return {str(k): normalize(item) for k, item in v.items()}
        if isinstance(v, (list, tuple)):
            return [normalize(item) for item in v]
        return v

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_vectors(vectors, dimension: int, predicate: Callable | None = None) -> np.ndarray:
    array = np.asarray(vectors)
    if array.ndim != 2 or not len(array) or array.shape[1] != dimension:
        raise ValueError(f"vectors must have nonempty shape (n,{dimension})")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("every coordinate must be finite and numeric")
    with np.errstate(over="ignore"):
        array = np.array(array, dtype="<f4", order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError("coordinates must be representable as float32")
    if predicate and not all(predicate(row) for row in array):
        raise ValueError("application predicate rejected a vector")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class Header:
    shard: int
    origin: int
    first: int
    last: int
    count: int
    dimension: int
    created_min: int
    created_max: int
    payload_digest: str

    @property
    def block_id(self) -> str:
        return self.payload_digest

    def to_dict(self):
        return asdict(self)

    def to_bytes(self):
        return struct.pack(
            "<IIQQIIqq32s",
            self.shard,
            self.origin,
            self.first,
            self.last,
            self.count,
            self.dimension,
            self.created_min,
            self.created_max,
            bytes.fromhex(self.payload_digest),
        )

    @classmethod
    def from_bytes(cls, data):
        if len(data) != 80:
            raise IntegrityError("invalid binary header size")
        fields = struct.unpack("<IIQQIIqq32s", data)
        return cls.parse(
            dict(zip(cls.__dataclass_fields__, (*fields[:-1], fields[-1].hex()), strict=True))
        )

    @classmethod
    def parse(cls, value):
        header = cls(**value)
        for name in (
            "shard",
            "origin",
            "first",
            "last",
            "count",
            "dimension",
            "created_min",
            "created_max",
        ):
            if type(getattr(header, name)) is not int or getattr(header, name) < 0:
                raise ValueError(f"invalid header {name}")
        if header.count < 1 or header.dimension < 1 or header.last < header.first:
            raise ValueError("invalid header range")
        if header.created_max < header.created_min or header.count > header.last - header.first + 1:
            raise ValueError("invalid header metadata")
        if len(header.payload_digest) != 64:
            raise ValueError("invalid SHA-256 digest")
        bytes.fromhex(header.payload_digest)
        return header


@dataclass(frozen=True)
class Payload:
    origin: int
    counters: np.ndarray
    created: np.ndarray
    vectors: np.ndarray
    metadata: tuple[dict, ...]

    def encode(self) -> bytes:
        meta = canonical(list(self.metadata))
        n, d = self.vectors.shape
        return (
            struct.pack("<4sIIIII", b"GSX1", self.origin, n, d, len(meta), 0)
            + self.counters.astype("<u8", copy=False).tobytes()
            + self.created.astype("<i8", copy=False).tobytes()
            + self.vectors.astype("<f4", copy=False).tobytes()
            + meta
        )

    @classmethod
    def decode(cls, data: bytes, header: Header, predicate=None):
        if digest(data) != header.payload_digest:
            raise IntegrityError(f"payload digest mismatch: {header.block_id}")
        if len(data) < 24:
            raise IntegrityError("truncated payload")
        magic, origin, n, d, meta_size, reserved = struct.unpack("<4sIIIII", data[:24])
        if (
            magic != b"GSX1"
            or reserved
            or (origin, n, d) != (header.origin, header.count, header.dimension)
        ):
            raise IntegrityError("payload schema does not match header")
        expected = 24 + 16 * n + 4 * n * d + meta_size
        if len(data) != expected:
            raise IntegrityError("payload length mismatch")
        counters = np.frombuffer(data, dtype="<u8", count=n, offset=24)
        created = np.frombuffer(data, dtype="<i8", count=n, offset=24 + 8 * n)
        vectors = np.frombuffer(data, dtype="<f4", count=n * d, offset=24 + 16 * n).reshape(n, d)
        if (
            len(np.unique(counters)) != n
            or int(counters.min()) != header.first
            or int(counters.max()) != header.last
        ):
            raise IntegrityError("invalid vector identifiers")
        if int(created.min()) != header.created_min or int(created.max()) != header.created_max:
            raise IntegrityError("creation metadata mismatch")
        if not np.isfinite(vectors).all() or (predicate and not all(predicate(v) for v in vectors)):
            raise IntegrityError("invalid vector in payload")
        metadata = json.loads(data[24 + 16 * n + 4 * n * d :])
        if len(metadata) != n or not all(isinstance(m, dict) for m in metadata):
            raise IntegrityError("invalid application metadata")
        canonical(metadata)
        return cls(origin, counters, created, vectors, tuple(metadata))


def make_block(shard: int, payload: Payload) -> tuple[Header, bytes]:
    data = payload.encode()
    header = Header(
        shard,
        payload.origin,
        int(payload.counters.min()),
        int(payload.counters.max()),
        len(payload.vectors),
        payload.vectors.shape[1],
        int(payload.created.min()),
        int(payload.created.max()),
        digest(data),
    )
    Payload.decode(data, header)
    return header, data


def merge_record(left: dict | None, right: dict) -> dict:
    if left is None:
        left = {
            "header": right["header"],
            "certificates": [],
            "committed": False,
            "adds": {},
            "removes": [],
        }
    if left["header"] != right["header"]:
        raise IntegrityError("immutable header conflict")
    certificates = {digest(canonical(c)): c for c in left["certificates"] + right["certificates"]}
    adds = dict(left["adds"])
    for token, location in right["adds"].items():
        if token in adds and adds[token] != location:
            raise IntegrityError("replica token conflict")
        adds[token] = location
    return {
        "header": left["header"],
        "certificates": list(certificates.values()),
        "committed": left["committed"] or right["committed"],
        "adds": adds,
        "removes": sorted(set(left["removes"]) | set(right["removes"])),
    }


def holders(record: dict) -> dict[int, str]:
    removed = set(record["removes"])
    return {
        int(location["node"]): token
        for token, location in record["adds"].items()
        if token not in removed
    }


@dataclass(frozen=True)
class Epoch:
    number: int
    leaders: tuple[int, ...]
    weights: dict[int, float]
    node_weights: dict[int, float]
    shards: dict[int, dict]
    adjacency: dict[int, tuple[int, ...]]
    block_shards: dict[str, int]
    retired: dict[int, tuple[int, ...]]
    free: dict[int, int]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def parse(cls, value):
        e = cls(
            int(value["number"]),
            tuple(value["leaders"]),
            {int(k): float(v) for k, v in value["weights"].items()},
            {int(k): float(v) for k, v in value["node_weights"].items()},
            {int(k): v for k, v in value["shards"].items()},
            {int(k): tuple(v) for k, v in value["adjacency"].items()},
            {k: int(v) for k, v in value["block_shards"].items()},
            {int(k): tuple(v) for k, v in value["retired"].items()},
            {int(k): int(v) for k, v in value["free"].items()},
        )
        if (
            not e.leaders
            or len(set(e.leaders)) != len(e.leaders)
            or set(e.weights) != set(e.leaders)
        ):
            raise ValueError("invalid epoch leader set")
        if not all(math.isfinite(w) and w > 0 for w in e.weights.values()):
            raise ValueError("epoch vote weights must be positive")
        if not e.shards or e.number < 0:
            raise ValueError("invalid epoch")
        if not set(e.leaders) <= set(e.node_weights) or set(e.adjacency) != set(e.node_weights):
            raise ValueError("epoch membership and overlay disagree")
        if not all(math.isfinite(w) and w > 0 for w in e.node_weights.values()):
            raise ValueError("node connectivity weights must be positive")
        dimensions = set()
        for shard in e.shards.values():
            p = shard["peers"]
            if (
                not p
                or len(p) != len(set(p))
                or not set(p) <= set(e.node_weights)
                or set(p) & set(e.leaders)
                or not 1 <= shard["replicas"] <= len(p)
            ):
                raise ValueError("invalid shard replication")
            center = np.asarray(shard["centroid"])
            if center.ndim != 1 or not len(center) or not np.isfinite(center).all():
                raise ValueError("nonfinite centroid")
            dimensions.add(len(center))
        if len(dimensions) != 1 or not set(e.block_shards.values()) <= set(e.shards):
            raise ValueError("invalid shard map or dimensions")
        if set(e.shards) & set(e.retired):
            raise ValueError("a shard cannot be both active and retired")
        for root in e.retired:
            visiting = set()

            def check(shard, visiting=visiting):
                if shard in visiting:
                    raise ValueError("cyclic retired shard map")
                if shard in e.retired:
                    children = e.retired[shard]
                    if len(children) != 2 or len(set(children)) != 2:
                        raise ValueError("a retired shard must have two children")
                    visiting.add(shard)
                    for child in children:
                        check(child)
                    visiting.remove(shard)
                elif shard not in e.shards:
                    raise ValueError("retired shard references an unknown child")

            check(root)
        return e

    def shard_for(self, header: Header):
        if header.block_id in self.block_shards:
            return self.block_shards[header.block_id]
        shard = header.shard
        while shard in self.retired:
            shard = self.retired[shard][0]
        return shard


def validate_certificate(certificate: dict, header: Header, epoch: Epoch, theta: float, f: int):
    if certificate["epoch"] != epoch.number or certificate["block"] != header.block_id:
        raise IntegrityError("certificate epoch or block mismatch")
    voters = certificate["approvers"]
    if len(voters) != len(set(voters)):
        raise IntegrityError("duplicate approval")
    if certificate["path"] == "primary":
        if not set(voters) <= set(epoch.leaders):
            raise IntegrityError("approval from outside leader tier")
        if math.fsum(epoch.weights[v] for v in voters) < theta * math.fsum(epoch.weights.values()):
            raise IntegrityError("insufficient approving weight")
    elif certificate["path"] == "fallback":
        committee = certificate["committee"]
        if len(committee) != 2 * f + 1 or len(set(committee)) != len(committee):
            raise IntegrityError("invalid fallback committee")
        if (
            not set(committee) <= set(epoch.node_weights)
            or not set(voters) <= set(committee)
            or len(voters) < f + 1
        ):
            raise IntegrityError("insufficient fallback approvals")
    else:
        raise IntegrityError("unknown certification path")
