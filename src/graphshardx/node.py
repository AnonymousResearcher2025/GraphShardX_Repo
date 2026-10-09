import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import math
from pathlib import Path
import time
import psutil

import numpy as np

from .config import Settings
from .model import (
    Epoch,
    GraphShardXError,
    Header,
    IntegrityError,
    Payload,
    Unavailable,
    canonical,
    digest,
    holders,
    make_block,
    merge_record,
    validate_certificate,
    validate_vectors,
)
from .placement import assign, candidate_pools, find_leader, weight
from .storage import Store

LOG = logging.getLogger(__name__)


class Node:
    def __init__(
        self,
        node_id: int,
        settings: Settings,
        epoch: Epoch,
        transport,
        directory: Path,
        predicate=None,
    ):
        self.id = node_id
        self.settings = settings
        self.transport = transport
        self.store = Store(
            directory / f"node-{node_id}.sqlite" if settings.storage_mode == "disk" else None,
            settings.storage_bytes,
            predicate,
        )
        saved = self.store.meta("epochs", {})
        self.epochs = {int(k): Epoch.parse(v) for k, v in saved.items()} or {epoch.number: epoch}
        self.epoch = self.epochs[max(self.epochs)]
        self.alive = True
        self.reports = {}
        self.last_seen = {}
        self.latencies = {}
        self.uptime_history = self.store.meta("uptime_history", [])
        self.started = time.monotonic()
        self.rng = np.random.default_rng(settings.seed + node_id)
        self.executor = ThreadPoolExecutor(
            max_workers=settings.workers, thread_name_prefix=f"node-{node_id}"
        )
        self.tasks = set()
        self.periodic = []
        self.ingestion_lock = asyncio.Lock()
        self.buffer = []
        self.buffer_lock = asyncio.Lock()
        self.buffer_timer = None
        self.decisions = self.store.meta("decisions", {})
        self.move_locks = {}
        self.metrics = {
            "primary": 0,
            "fallback": 0,
            "committed": 0,
            "migrations": 0,
            "repairs": 0,
            "splits_requested": 0,
            "validation_failures": 0,
        }
        transport.register(self)
        self.store.set_meta("epochs", {k: v.to_dict() for k, v in self.epochs.items()})

    async def work(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(self._completed)
        return task

    def _completed(self, task):
        if not task.cancelled() and task.exception():
            LOG.warning("node %s background operation: %s", self.id, task.exception())

    async def call(self, target, method, args, timeout=None):
        return await self.transport.call(self.id, target, method, args, timeout)

    async def reachable(self, targets=None):
        targets = list(targets if targets is not None else self.epoch.node_weights)

        async def probe(target):
            start = time.monotonic()
            try:
                report = await self.call(target, "heartbeat", {})
                self.last_seen[target] = time.monotonic()
                self.latencies[target] = time.monotonic() - start
                self.reports[target] = report
                return target
            except Unavailable:
                return None

        live = await asyncio.gather(*(probe(t) for t in targets))
        return [n for n in live if n is not None]

    def report(self):
        used = self.store.used
        uptime = sum(self.uptime_history) / len(self.uptime_history) if self.uptime_history else 1.0
        tokens = self.store.meta("tokens", 0) * self.settings.relocation_tokens
        return {
            "node": self.id,
            "used": used,
            "capacity": self.settings.storage_bytes,
            "free": max(0, self.settings.storage_bytes - used),
            "tokens": tokens,
            "uptime": uptime,
            "weight": weight(len(self.epoch.adjacency[self.id]), uptime, tokens),
            "process_rss": psutil.Process().memory_info().rss,
            "virtual_nodes_in_process": len(self.transport.nodes),
            "epoch": self.epoch.number,
            "metrics": dict(self.metrics),
        }

    async def handle(self, src, method, args):
        if not self.alive:
            raise Unavailable(f"node {self.id} crashed")
        if method == "heartbeat":
            return self.report()
        if method == "header":
            epoch = self.epochs.get(args["epoch"])
            if epoch is None or epoch.number != self.epoch.number:
                raise Unavailable("approval requires current epoch")
            header = Header.parse(args["header"])
            if (
                header.dimension != self.settings.dimension
                or header.origin not in epoch.node_weights
            ):
                raise ValueError("header collection or origin mismatch")
            if epoch.shard_for(header) not in epoch.shards:
                raise ValueError("header names an unknown shard")
            if args["path"] == "primary" and self.id not in epoch.leaders:
                raise ValueError("only epoch leaders approve the primary path")
            if args["path"] == "fallback" and self.id not in args["committee"]:
                raise ValueError("node is not in the fallback committee")
            await self.work(self.store.append_header, header)
            return {"node": self.id, "epoch": epoch.number, "block": header.block_id}
        if method == "certify":
            return await self.certify(
                Header.parse(args["header"]), args["epoch"], args["path"], args.get("committee", [])
            )
        if method == "payload":
            return await self.accept_payload(args)
        if method == "merge":
            for record in args["records"]:
                await self.merge(record)
            return {"ok": True}
        if method == "catalog":
            records = await self.work(self.store.records)
            requested = set(args.get("shards", self.epoch.shards))
            return [
                r
                for r in records.values()
                if self.epoch.shard_for(Header.parse(r["header"])) in requested
            ]
        if method == "fetch":
            value = await self.work(self.store.get_payload, args["block"])
            if value is None:
                raise Unavailable("copy is absent")
            if isinstance(value, dict):
                return value
            data, token, shard = value
            header = self.store.header(args["block"])
            await self.work(Payload.decode, data, header, self.store.predicate)
            return {"data": data, "token": token, "shard": shard, "header": header.to_dict()}
        if method == "scan":
            return await self.scan(args)
        if method == "ingest":
            return await self.ingest(
                np.asarray(args["vectors"]), args.get("metadata"), args.get("created")
            )
        if method == "flush":
            return await self.flush_pending()
        if method == "insert":
            return await self.insert(args["vector"], args.get("metadata", {}), args.get("created"))
        if method == "search":
            return await self.search(args["query"], args.get("k", 10), args.get("probes", 1))
        if method == "read":
            return await self.read_vector(args["origin"], args["counter"])
        if method == "monitor":
            return await self.monitor()
        if method == "repair":
            return await self.repair()
        if method == "retire":
            record = await self.merge(args["record"])
            h = Header.parse(record["header"])
            current = self.epoch.shards[self.epoch.shard_for(h)]
            if len(set(holders(record)) & set(current["peers"])) < current["replicas"]:
                raise Unavailable("new shard copies are not yet complete")
            return {"moved": await self.migrate(h.block_id, args["successor"], allow_existing=True)}
        if method == "gossip":
            return await self.gossip()
        if method == "snapshot":
            return {
                "records": await self.work(self.store.records),
                "epochs": {k: e.to_dict() for k, e in self.epochs.items()},
            }
        if method == "summary":
            return {b: digest(canonical(r)) for b, r in self.store.records().items()}
        if method == "records":
            return [r for b in args["blocks"] if (r := self.store.record(b))]
        if method == "configuration":
            return {
                "epochs": {k: e.to_dict() for k, e in self.epochs.items()},
                "decisions": self.decisions,
            }
        if method == "paxos":
            if self.id not in self.epochs[args["slot"] - 1].leaders:
                raise ValueError("only outgoing leaders accept epoch proposals")
            return await self.work(
                self.store.paxos,
                args["slot"],
                args["action"],
                args.get("ballot"),
                args.get("value"),
            )
        if method == "activate":
            return await self.activate(
                args["epoch"], args.get("decision"), args.get("harness", False)
            )
        if method == "epoch":
            return await self.advance_epoch()
        if method == "split_request":
            splits = set(self.store.meta("splits", []))
            splits.add(args["shard"])
            self.store.set_meta("splits", sorted(splits))
            return {"ok": True}
        if method == "audit":
            checked = await self.work(self.store.verify_ledgers)
            for block in self.store.local_blocks():
                value = self.store.get_payload(block["block"])
                await self.work(
                    Payload.decode,
                    value[0],
                    self.store.header(block["block"]),
                    self.store.predicate,
                )
            return {
                "ledger_entries": checked,
                "payloads": len(self.store.local_blocks()),
                "report": self.report(),
            }
        raise ValueError(f"unknown method {method}")

    async def certify(self, header, number, path, committee):
        epoch = self.epochs.get(number)
        if epoch is None or epoch.number != self.epoch.number:
            raise Unavailable("coordinator epoch has changed")
        if path == "primary":
            targets = epoch.leaders
            limit = self.settings.primary_timeout
        else:
            if len(committee) != 2 * self.settings.fallback_f + 1:
                raise Unavailable("a full 2f+1 reachable committee is required")
            targets = committee
            limit = self.settings.request_timeout
        requests = {
            asyncio.create_task(
                self.call(
                    n,
                    "header",
                    {
                        "header": header.to_dict(),
                        "epoch": number,
                        "path": path,
                        "committee": committee,
                    },
                    limit,
                )
            ): n
            for n in targets
        }
        approvers = []
        deadline = time.monotonic() + limit
        try:
            while requests and time.monotonic() < deadline:
                done, _ = await asyncio.wait(
                    requests,
                    timeout=max(0, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    break
                for task in done:
                    node = requests.pop(task)
                    try:
                        reply = task.result()
                        if reply == {"node": node, "epoch": number, "block": header.block_id}:
                            approvers.append(node)
                    except Unavailable:
                        pass
                approved = (
                    math.fsum(epoch.weights[n] for n in approvers)
                    >= self.settings.theta * math.fsum(epoch.weights.values())
                    if path == "primary"
                    else len(approvers) >= self.settings.fallback_f + 1
                )
                if approved:
                    certificate = {
                        "epoch": number,
                        "block": header.block_id,
                        "path": path,
                        "approvers": sorted(approvers),
                        "committee": committee,
                    }
                    validate_certificate(
                        certificate, header, epoch, self.settings.theta, self.settings.fallback_f
                    )
                    record = {
                        "header": header.to_dict(),
                        "certificates": [certificate],
                        "committed": False,
                        "adds": {},
                        "removes": [],
                    }
                    await self.merge(record)
                    self.spawn(self.publish(record, targets))
                    self.metrics[path] += 1
                    self.transport.forget_replies(self.id, header.block_id)
                    return certificate
            return None
        finally:
            for task in requests:
                task.cancel()
            await asyncio.gather(*requests, return_exceptions=True)

    async def merge(self, record):
        header = Header.parse(record["header"])
        if not record["certificates"]:
            raise IntegrityError("location metadata requires a certificate")
        for c in record["certificates"]:
            epoch = self.epochs.get(c["epoch"])
            if epoch is None:
                raise Unavailable("certificate configuration not yet available")
            validate_certificate(c, header, epoch, self.settings.theta, self.settings.fallback_f)
        if record["committed"]:
            # At least R distinct successful writes must be evidenced, even after moves tombstone them.
            required = min(
                self.epochs[c["epoch"]].shards[self.epochs[c["epoch"]].shard_for(header)][
                    "replicas"
                ]
                for c in record["certificates"]
            )
            if len({p["node"] for p in record["adds"].values()}) < required:
                raise IntegrityError("commit record lacks replication acknowledgments")
        merged = await self.work(self.store.merge, record)
        self.transport.forget_replies(self.id, header.block_id, "header")
        return merged

    async def publish(self, record, targets=None):
        destinations = set(targets or self.epoch.leaders)
        for c in record["certificates"]:
            destinations.update(c["approvers"])

        async def send(n):
            try:
                await self.call(n, "merge", {"records": [record]})
                return True
            except Unavailable:
                return False

        return await asyncio.gather(*(send(n) for n in destinations))

    async def accept_payload(self, args):
        header = Header.parse(args["header"])
        certificate = args["certificate"]
        epoch = self.epochs.get(certificate["epoch"])
        if not epoch:
            raise Unavailable("unknown payload certification epoch")
        validate_certificate(
            certificate, header, epoch, self.settings.theta, self.settings.fallback_f
        )
        shard = self.epoch.shard_for(header)
        relocated = args.get("relocated", False)
        if self.id not in self.epoch.shards[shard]["peers"] and not relocated:
            raise ValueError("replication target is outside the current shard")
        if relocated and self.id not in self.epoch.leaders:
            raise ValueError("sidechain relocation requires an epoch leader")
        try:
            token = await self.work(
                self.store.put_payload, header, args["data"], shard, self.id, relocated
            )
        except (ValueError, IntegrityError):
            self.metrics["validation_failures"] += 1
            raise
        sidechain = self.store.sidechain_copy(header.block_id)
        record = {
            "header": header.to_dict(),
            "certificates": [certificate],
            "committed": False,
            "adds": {
                token: {
                    "node": self.id,
                    "shard": self.store.get_payload(header.block_id)[2],
                    "sidechain": sidechain,
                }
            },
            "removes": [],
        }
        await self.merge(record)
        self.spawn(self.publish(record))
        return {
            "node": self.id,
            "token": token,
            "shard": self.store.get_payload(header.block_id)[2],
            "sidechain": sidechain,
            "pointer": digest(
                canonical({"node": self.id, "block": header.block_id, "token": token})
            ),
        }

    async def ingest(self, vectors, metadata=None, created=None):
        vectors = validate_vectors(vectors, self.settings.dimension, self.store.predicate)
        metadata = metadata if metadata is not None else [{} for _ in vectors]
        if len(metadata) != len(vectors) or not all(isinstance(m, dict) for m in metadata):
            raise ValueError("one metadata dictionary is required per vector")
        canonical(metadata)
        created = np.asarray(created if created is not None else [time.time_ns()] * len(vectors))
        if created.dtype.kind not in "iu" or (created > np.iinfo(np.int64).max).any():
            raise ValueError("creation times must be integer nanoseconds representable as int64")
        created = created.astype("<i8", copy=False)
        if created.shape != (len(vectors),) or (created < 0).any():
            raise ValueError("creation times must be nonnegative nanoseconds")
        async with self.ingestion_lock:
            start = await self.work(self.store.reserve_ids, len(vectors))
            counters = np.arange(start, start + len(vectors), dtype="<u8")
            seed = int((self.settings.seed + self.id + start) % (2**32 - 1))
            assignments = await self.work(assign, vectors, self.epoch, self.settings, seed)
            for shard in sorted(set(assignments)):
                indices = np.flatnonzero(assignments == shard)
                payload = Payload(
                    self.id,
                    counters[indices],
                    created[indices],
                    vectors[indices],
                    tuple(metadata[i] for i in indices),
                )
                header, data = make_block(int(shard), payload)
                await self.work(self.store.queue, header, data)
            result = await self.commit_batch(await self.work(self.store.pending))
            result["identifiers"] = [[self.id, int(counter)] for counter in counters]
            return result

    async def insert(self, vector, metadata, created):
        validate_vectors([vector], self.settings.dimension, self.store.predicate)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        canonical(metadata)
        future = asyncio.get_running_loop().create_future()
        async with self.buffer_lock:
            self.buffer.append(
                (vector, metadata, time.time_ns() if created is None else created, future)
            )
            if self.buffer_timer is None or self.buffer_timer.done():
                self.buffer_timer = self.spawn(self._flush_buffer_after_timeout())
            if len(self.buffer) >= self.settings.batch_size:
                self.spawn(self._flush_buffer())
        return await future

    async def _flush_buffer_after_timeout(self):
        await asyncio.sleep(self.settings.batch_timeout)
        await self._flush_buffer()

    async def _flush_buffer(self):
        async with self.buffer_lock:
            entries, self.buffer = (
                self.buffer[: self.settings.batch_size],
                self.buffer[self.settings.batch_size :],
            )
        if not entries:
            return
        try:
            result = await self.ingest(
                [v for v, _, _, _ in entries],
                [m for _, m, _, _ in entries],
                [t for _, _, t, _ in entries],
            )
            for entry, identifier in zip(entries, result["identifiers"], strict=True):
                future = entry[3]
                if not future.done():
                    future.set_result(
                        {
                            "identifier": identifier,
                            "pending": result["pending"],
                            "committed_blocks": result["committed"],
                        }
                    )
        except Exception as exc:
            for _, _, _, future in entries:
                if not future.done():
                    future.set_exception(exc)
        if self.buffer:
            self.spawn(self._flush_buffer_after_timeout())

    async def commit_batch(self, blocks):
        if not blocks:
            return {"committed": [], "certified": [], "pending": [], "vectors": 0}
        deadline = time.monotonic() + self.settings.operation_timeout
        results = {
            h.block_id: h
            for h, _ in blocks
            if (self.store.record(h.block_id) or {}).get("committed")
        }
        already_committed = set(results)
        while time.monotonic() < deadline and self.alive:
            epoch = self.epoch
            remaining = [(h, d) for h, d in blocks if h.block_id not in results]
            if not remaining:
                break

            async def primary(header, epoch=epoch, pools=None):
                record = self.store.record(header.block_id)
                if record and record["certificates"]:
                    return record["certificates"][-1]
                shard = epoch.shard_for(header)
                pool = pools[shard]
                ordered = sorted(pool, key=lambda n: (self.last_seen.get(n, 0) == 0, n))
                chosen = find_leader(epoch, shard, ordered, self.reports, self.latencies)
                try:
                    return await self.call(
                        chosen,
                        "certify",
                        {"header": header.to_dict(), "epoch": epoch.number, "path": "primary"},
                        self.settings.primary_timeout
                        + 4 * self.settings.delay
                        + self.settings.rpc_timeout,
                    )
                except Unavailable:
                    return None

            certificates = []
            # Candidate pools are disjoint within a wave. More blocks than leaders require waves.
            for offset in range(0, len(remaining), len(epoch.leaders)):
                wave = remaining[offset : offset + len(epoch.leaders)]
                pools = candidate_pools(epoch, [epoch.shard_for(h) for h, _ in wave])
                certificates.extend(
                    await asyncio.gather(*(primary(h, epoch, pools) for h, _ in wave))
                )
            certified_fraction = sum(c is not None for c in certificates) / len(remaining)
            if certified_fraction < self.settings.beta:
                live = await self.reachable()
                ranked = sorted(live, key=lambda n: (-epoch.node_weights[n], n))
                committee = ranked[: 2 * self.settings.fallback_f + 1]
                if len(committee) == 2 * self.settings.fallback_f + 1:

                    async def fallback(header, certificate, committee=committee, epoch=epoch):
                        if certificate:
                            return certificate
                        try:
                            return await self.call(
                                committee[0],
                                "certify",
                                {
                                    "header": header.to_dict(),
                                    "epoch": epoch.number,
                                    "path": "fallback",
                                    "committee": committee,
                                },
                                3 * self.settings.request_timeout,
                            )
                        except Unavailable:
                            return None

                    certificates = await asyncio.gather(
                        *(fallback(h, c) for (h, _), c in zip(remaining, certificates, strict=True))
                    )

            async def replicate(header, data, certificate):
                if not certificate:
                    return False
                record = self.store.record(header.block_id)
                record = merge_record(
                    record,
                    {
                        "header": header.to_dict(),
                        "certificates": [certificate],
                        "committed": False,
                        "adds": {},
                        "removes": [],
                    },
                )
                await self.merge(record)
                shard = self.epoch.shard_for(header)
                peers = self.epoch.shards[shard]["peers"]
                required = self.epoch.shards[shard]["replicas"]
                tie_breaks = {n: float(self.rng.random()) for n in peers}
                targets = sorted(peers, key=lambda n: (-self.epoch.free.get(n, 0), tie_breaks[n]))
                acknowledged = {}

                async def put(n):
                    try:
                        ack = await self.call(
                            n,
                            "payload",
                            {"header": header.to_dict(), "data": data, "certificate": certificate},
                        )
                        return ack
                    except (Unavailable, GraphShardXError) as exc:
                        if isinstance(exc, IntegrityError):
                            raise
                        return None

                # Select R peers initially. Substitute reachable peers after timeout, never lower R.
                for offset in range(0, len(targets), required):
                    answers = await asyncio.gather(
                        *(
                            put(n)
                            for n in targets[offset : offset + required]
                            if n not in acknowledged
                        )
                    )
                    for ack in answers:
                        if ack:
                            acknowledged[ack["node"]] = ack
                    if len(acknowledged) >= required:
                        break
                record["adds"].update(
                    {
                        ack["token"]: {
                            "node": n,
                            "shard": ack["shard"],
                            "sidechain": ack["sidechain"],
                        }
                        for n, ack in acknowledged.items()
                    }
                )
                record["committed"] = len(acknowledged) >= required
                if not self.alive:
                    raise Unavailable("origin crashed before recording the commit")
                await self.merge(record)
                self.spawn(self.publish(record))
                if record["committed"]:
                    await self.work(self.store.dequeue, header.block_id)
                    self.metrics["committed"] += header.count
                    self.transport.forget_replies(self.id, header.block_id)
                    return True
                return False

            committed = await asyncio.gather(
                *(replicate(h, d, c) for (h, d), c in zip(remaining, certificates, strict=True))
            )
            for (header, _), success in zip(remaining, committed, strict=True):
                if success:
                    results[header.block_id] = header
            if len(results) != len(blocks):
                await asyncio.sleep(self.settings.retry_interval)
        pending = [h.block_id for h, _ in blocks if h.block_id not in results]
        certified = [b for b in pending if (self.store.record(b) or {}).get("certificates")]
        return {
            "committed": list(results),
            "certified": certified,
            "pending": pending,
            "vectors": sum(h.count for b, h in results.items() if b not in already_committed),
        }

    async def flush_pending(self):
        async with self.ingestion_lock:
            return await self.commit_batch(await self.work(self.store.pending))

    async def catalog(self, shards=None):
        records = {}
        targets = set(self.epoch.leaders)
        targets.update(
            n
            for n, seen in self.last_seen.items()
            if time.monotonic() - seen < self.settings.failure_timeout
        )

        async def get(n):
            try:
                return await self.call(n, "catalog", {"shards": list(shards or self.epoch.shards)})
            except Unavailable:
                return []

        replies = await asyncio.gather(*(get(n) for n in targets))
        if not any(replies):
            live = await self.reachable()
            fallback = sorted(live, key=lambda n: (-self.epoch.node_weights[n], n))[
                : 2 * self.settings.fallback_f + 1
            ]
            replies.extend(await asyncio.gather(*(get(n) for n in fallback if n not in targets)))
        for values in replies:
            for record in values:
                header = Header.parse(record["header"])
                records[header.block_id] = merge_record(records.get(header.block_id), record)
        return records

    async def fetch(self, record):
        header = Header.parse(record["header"])
        errors = []
        visited = set()
        targets = list(holders(record))
        while targets:
            node = targets.pop(0)
            if node in visited:
                continue
            visited.add(node)
            try:
                value = await self.call(node, "fetch", {"block": header.block_id})
                if "pointer" in value:
                    pointer = value["pointer"]
                    if pointer["hash"] != digest(
                        canonical(
                            {
                                "node": pointer["node"],
                                "block": header.block_id,
                                "token": pointer["token"],
                            }
                        )
                    ):
                        raise IntegrityError("invalid relocation pointer")
                    targets.append(pointer["node"])
                    continue
                await self.work(Payload.decode, value["data"], header, self.store.predicate)
                return value["data"]
            except (Unavailable, IntegrityError) as exc:
                errors.append(exc)
        if errors and all(isinstance(e, IntegrityError) for e in errors):
            raise IntegrityError(f"all reachable copies of {header.block_id} are corrupt")
        raise Unavailable(f"no valid reachable copy of {header.block_id}")

    async def scan(self, args):
        header = Header.parse(args["header"])
        value = await self.work(self.store.get_payload, header.block_id)
        if value is None or isinstance(value, dict):
            raise Unavailable("scan copy moved or is absent")
        payload = await self.work(Payload.decode, value[0], header, self.store.predicate)
        query = np.asarray(args["query"], dtype=np.float64)

        def compute():
            distances = np.sum((payload.vectors.astype(np.float64) - query) ** 2, axis=1)
            count = min(args["k"], len(distances))
            indices = np.lexsort((payload.counters, distances))[:count]
            return [
                [payload.origin, int(payload.counters[i]), float(distances[i])] for i in indices
            ]

        return {"results": await self.work(compute), "scanned": len(payload.vectors)}

    async def search(self, query, k, probes):
        query = validate_vectors([query], self.settings.dimension)[0]
        if (
            type(k) is not int
            or k < 1
            or type(probes) is not int
            or not 1 <= probes <= len(self.epoch.shards)
        ):
            raise ValueError("k must be positive and probes must be between 1 and the shard count")
        ordered = sorted(
            self.epoch.shards,
            key=lambda s: (np.sum((np.asarray(self.epoch.shards[s]["centroid"]) - query) ** 2), s),
        )[:probes]
        records = await self.catalog(ordered)

        async def scan_one(record):
            corruption = False
            for n in holders(record):
                try:
                    return await self.call(
                        n, "scan", {"header": record["header"], "query": query.tolist(), "k": k}
                    )
                except Unavailable:
                    pass
                except IntegrityError:
                    corruption = True
            if corruption:
                raise IntegrityError("search encountered corrupt replicas with no valid copy")
            return None

        replies = await asyncio.gather(*(scan_one(r) for r in records.values()))
        values = {}
        for reply in replies:
            if reply:
                for origin, counter, distance in reply["results"]:
                    key = (origin, counter)
                    values[key] = min(values.get(key, math.inf), distance)
        results = sorted(values.items(), key=lambda v: (v[1], v[0]))[:k]
        return {
            "neighbors": [
                {"origin": key[0], "counter": key[1], "distance": math.sqrt(distance)}
                for key, distance in results
            ],
            "shards": ordered,
            "scanned": sum(r["scanned"] for r in replies if r),
            "unavailable_blocks": sum(r is None for r in replies),
            "epoch": self.epoch.number,
        }

    async def read_vector(self, origin, counter):
        for record in (await self.catalog()).values():
            h = Header.parse(record["header"])
            if h.origin == origin and h.first <= counter <= h.last:
                payload = Payload.decode(await self.fetch(record), h, self.store.predicate)
                indices = np.flatnonzero(payload.counters == counter)
                if len(indices):
                    i = indices[0]
                    return {
                        "origin": origin,
                        "counter": counter,
                        "created": int(payload.created[i]),
                        "vector": payload.vectors[i].tolist(),
                        "metadata": payload.metadata[i],
                        "committed": record["committed"],
                    }
        raise Unavailable("vector is absent from reachable location records")

    async def migrate(self, block_id, target, relocated=False, allow_existing=False):
        lock = self.move_locks.setdefault(block_id, asyncio.Lock())
        async with lock:
            value = self.store.get_payload(block_id)
            record = self.store.record(block_id)
            if value is None or isinstance(value, dict) or not record:
                return False
            data, token, shard = value
            if target == self.id or (target in holders(record) and not allow_existing):
                return False
            certificate = record["certificates"][-1]
            ack = await self.call(
                target,
                "payload",
                {
                    "header": record["header"],
                    "data": data,
                    "certificate": certificate,
                    "relocated": relocated,
                },
            )
            record["adds"][ack["token"]] = {
                "node": target,
                "shard": ack["shard"],
                "sidechain": ack["sidechain"],
            }
            record["removes"] = sorted(set(record["removes"]) | {token})
            await self.merge(record)
            # Publish the new holder before freeing. The durable local pointer survives stale records.
            await self.publish(record)
            pointer = {"node": target, "token": ack["token"], "hash": ack["pointer"]}
            await self.work(self.store.free_payload, block_id, token, pointer)
            self.metrics["migrations"] += 1
            return True

    async def monitor(self):
        if (
            not self.settings.rebalancing
            or self.store.used / self.settings.storage_bytes <= self.settings.rho
        ):
            return {"moved": 0, "splits": []}
        live = await self.reachable()
        moved = 0
        splits = []
        for shard_id, shard in self.epoch.shards.items():
            local = [
                b
                for b in self.store.local_blocks()
                if self.epoch.shard_for(self.store.header(b["block"])) == shard_id
            ]
            if not local:
                continue
            peers = shard["peers"]
            uses = [
                self.reports.get(
                    n,
                    {
                        "used": self.settings.storage_bytes - self.epoch.free.get(n, 0),
                        "capacity": self.settings.storage_bytes,
                    },
                )
                for n in peers
            ]
            mean = sum(r["used"] / r["capacity"] for r in uses) / len(peers)
            if mean > self.settings.rho and len(peers) >= max(
                self.settings.minimum_split_peers, 2 * self.settings.minimum_replicas
            ):
                splits.append(shard_id)
                self.metrics["splits_requested"] += 1
                for leader in self.epoch.leaders:
                    try:
                        await self.call(leader, "split_request", {"shard": shard_id})
                    except Unavailable:
                        pass
                continue
            for block in local[: self.settings.moves_per_round]:
                record = self.store.record(block["block"])
                existing = holders(record) if record else {}
                targets = [
                    n
                    for n in peers
                    if n in live
                    and n not in existing
                    and self.reports[n]["used"] / self.reports[n]["capacity"] < self.settings.rho
                    and self.reports[n]["free"] >= block["bytes"]
                ]
                relocated = False
                if not targets:
                    targets = [
                        n
                        for n in self.epoch.leaders
                        if n in live
                        and n not in existing
                        and n not in self.epoch.adjacency[self.id]
                        and self.reports[n]["used"] / self.reports[n]["capacity"]
                        < self.settings.rho
                        and self.reports[n]["free"] >= block["bytes"]
                    ]
                    relocated = True
                if not targets:
                    continue
                target = int(self.rng.choice(targets))
                try:
                    if await self.migrate(block["block"], target, relocated):
                        moved += 1
                        self.reports[target]["used"] += block["bytes"]
                        self.reports[target]["free"] -= block["bytes"]
                except Unavailable:
                    continue
        return {"moved": moved, "splits": splits}

    async def repair(self):
        records = await self.catalog()
        live = await self.reachable()
        repaired = 0
        for record in records.values():
            if not record["committed"]:
                continue
            h = Header.parse(record["header"])
            shard = self.epoch.shards[self.epoch.shard_for(h)]
            live_holders = [n for n in holders(record) if n in live]
            if not live_holders:
                continue
            # A live holder owns repair; failures move ownership without a coordinator.
            if self.id != min(live_holders):
                continue
            copies = holders(record)
            sidechains = {
                n
                for n, token in copies.items()
                if n in self.epoch.leaders and record["adds"][token].get("sidechain", False)
            }
            valid_holders = [n for n in live_holders if n in shard["peers"] or n in sidechains]
            if len(valid_holders) >= shard["replicas"] and all(
                n in valid_holders for n in live_holders
            ):
                continue
            targets = sorted(
                (n for n in shard["peers"] if n in live and n not in live_holders),
                key=lambda n: (-self.reports[n]["free"], n),
            )
            try:
                data = await self.fetch(record)
            except (Unavailable, IntegrityError):
                continue
            retiring = any(n not in valid_holders for n in live_holders)
            for target in targets:
                count = (
                    len(set(valid_holders) & set(shard["peers"]))
                    if retiring
                    else len(valid_holders)
                )
                if count >= shard["replicas"]:
                    break
                if self.reports[target]["free"] < len(data):
                    continue
                try:
                    ack = await self.call(
                        target,
                        "payload",
                        {
                            "header": h.to_dict(),
                            "data": data,
                            "certificate": record["certificates"][-1],
                        },
                    )
                    record["adds"][ack["token"]] = {
                        "node": target,
                        "shard": ack["shard"],
                        "sidechain": ack["sidechain"],
                    }
                    live_holders.append(target)
                    valid_holders.append(target)
                    repaired += 1
                except Unavailable:
                    pass
            await self.merge(record)
            await self.publish(record)
            if len(set(valid_holders) & set(shard["peers"])) >= shard["replicas"]:
                for former in list(live_holders):
                    if former not in shard["peers"] and former not in sidechains:
                        try:
                            await self.call(
                                former,
                                "retire",
                                {
                                    "record": record,
                                    "successor": min(set(valid_holders) & set(shard["peers"])),
                                },
                            )
                        except Unavailable:
                            pass
        self.metrics["repairs"] += repaired
        return {"repaired": repaired}

    async def gossip(self):
        local = self.store.records()
        targets = [n for n in self.epoch.leaders if n != self.id]
        for target in targets:
            try:
                summary = await self.call(target, "summary", {})
                changed = [r for b, r in local.items() if summary.get(b) != digest(canonical(r))]
                for offset in range(0, len(changed), 128):
                    await self.call(target, "merge", {"records": changed[offset : offset + 128]})
                if self.id in self.epoch.leaders:
                    missing = [
                        b
                        for b, v in summary.items()
                        if b not in local or digest(canonical(local[b])) != v
                    ]
                    for offset in range(0, len(missing), 128):
                        values = await self.call(
                            target, "records", {"blocks": missing[offset : offset + 128]}
                        )
                        for record in values:
                            await self.merge(record)
            except Unavailable:
                continue
        return {"records": len(self.store.records())}

    async def activate(self, value, decision, harness=False, record_uptime=True):
        epoch = Epoch.parse(value)
        if epoch.number <= self.epoch.number:
            if self.epochs.get(epoch.number) != epoch:
                raise IntegrityError("conflicting epoch configuration")
            return {"ok": True, "epoch": self.epoch.number}
        if epoch.number != self.epoch.number + 1:
            raise Unavailable("must learn configurations in epoch order")
        outgoing = self.epoch
        if harness:
            if self.settings.epoch_mode != "harness":
                raise ValueError("harness publication is disabled")
        else:
            if (
                not decision
                or decision["slot"] != epoch.number
                or decision["value_digest"] != digest(canonical(value))
            ):
                raise IntegrityError("epoch decision is missing")
            acceptors = decision["acceptors"]
            if (
                len(set(acceptors)) != len(acceptors)
                or not set(acceptors) <= set(outgoing.leaders)
                or len(acceptors) <= len(outgoing.leaders) // 2
            ):
                raise IntegrityError("epoch decision lacks an outgoing leader majority")
        # New leaders obtain header records before activating. Existing leaders also gossip their sets.
        if self.id in epoch.leaders:
            sources = set(outgoing.leaders)
            sources.update(n for n in self.last_seen if n not in sources)
            fetched = False
            for n in sources:
                try:
                    snapshot = await self.call(n, "snapshot", {})
                    for record in snapshot["records"].values():
                        await self.merge(record)
                    fetched = True
                except Unavailable:
                    continue
            if not fetched:
                raise Unavailable("new leader cannot obtain any outgoing header snapshot")
        if record_uptime:
            self.uptime_history = (self.uptime_history + [1.0])[-self.settings.uptime_window :]
        self.store.set_meta("uptime_history", self.uptime_history)
        self.epochs[epoch.number] = epoch
        self.store.set_meta("epochs", {k: e.to_dict() for k, e in self.epochs.items()})
        self.store.set_meta("decision", decision)
        self.decisions[str(epoch.number)] = decision
        self.store.set_meta("decisions", self.decisions)
        if not harness and self.id in outgoing.leaders:
            self.store.paxos(epoch.number, "decide", value=value)
        self.epoch = epoch
        LOG.info("node %s activated epoch %s", self.id, epoch.number)
        return {"ok": True, "epoch": epoch.number}

    async def advance_epoch(self):
        from .coordination import propose

        return await propose(self)

    async def _periodic(self, interval, function):
        await asyncio.sleep(float(self.rng.uniform(0, interval)))
        while True:
            if self.alive:
                try:
                    await function()
                except Unavailable:
                    pass
                except Exception:
                    LOG.exception("node %s maintenance failed", self.id)
            await asyncio.sleep(interval)

    async def heartbeats(self):
        targets = set(self.epoch.adjacency[self.id]) | set(self.epoch.leaders)
        await self.reachable(targets)
        # Epoch publication can be lost. Live nodes learn the decision from any reachable leader.
        for leader in sorted(self.reports):
            if self.reports.get(leader, {}).get("epoch", 0) > self.epoch.number:
                value = await self.call(leader, "configuration", {})
                for number in sorted(int(k) for k in value["epochs"]):
                    if number > self.epoch.number:
                        decision = value["decisions"].get(
                            str(number), value["decisions"].get(number)
                        )
                        await self.activate(
                            value["epochs"].get(str(number), value["epochs"].get(number)),
                            decision,
                            self.settings.epoch_mode == "harness",
                        )

    async def epoch_tick(self):
        if self.settings.epoch_mode != "consensus" or self.id not in self.epoch.leaders:
            return
        live = await self.reachable(self.epoch.leaders)
        if live and self.id == min(live):
            await self.advance_epoch()

    def start(self):
        functions = [
            (self.settings.heartbeat_interval, self.heartbeats),
            (self.settings.monitor_interval, self.monitor),
            (self.settings.repair_interval, self.repair),
            (self.settings.gossip_interval, self.gossip),
            (self.settings.epoch_interval, self.epoch_tick),
        ]
        if self.settings.epoch_mode == "harness":
            functions = [(i, f) for i, f in functions if f not in {self.repair, self.epoch_tick}]
        for interval, function in functions:
            self.periodic.append(self.spawn(self._periodic(interval, function)))

    async def stop(self):
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.store.close()
