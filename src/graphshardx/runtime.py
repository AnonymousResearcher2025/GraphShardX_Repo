import asyncio
from dataclasses import replace
import json
import logging
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .config import Settings
from .model import Epoch, Unavailable
from .node import Node
from .placement import initial_epoch
from .transport import MPITransport, TCPTransport, Transport

LOG = logging.getLogger(__name__)


class Cluster:
    def __init__(
        self, settings, epoch, transport, directory, nodes=None, processes=None, comm=None
    ):
        self.settings = settings
        self.epoch = epoch
        self.transport = transport
        self.directory = directory
        self.nodes = nodes or {}
        self.processes = processes or []
        self.comm = comm
        self.failed = set()
        self.harness_tasks = []

    @classmethod
    async def create(
        cls,
        settings: Settings,
        sample: np.ndarray,
        directory: Path,
        backend="local",
        processes=2,
        maintenance=True,
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        threadpool_limits(limits=1)
        comm = None
        if backend == "mpi":
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            epoch_data = initial_epoch(settings, sample).to_dict() if comm.rank == 0 else None
            epoch_data = await asyncio.to_thread(comm.bcast, epoch_data, root=0)
            epoch = Epoch.parse(epoch_data)
            transport = MPITransport(settings, comm)
            owned = [n for n in range(settings.nodes) if n % comm.size == comm.rank]
        else:
            epoch = initial_epoch(settings, sample)
            transport = Transport(settings)
            owned = list(range(settings.nodes))
        if backend == "tcp":
            ports = []
            for _ in range(processes):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    ports.append(sock.getsockname()[1])
            addresses = {n: ["127.0.0.1", ports[n % processes]] for n in range(settings.nodes)}
            manifest = {
                "settings": settings.to_dict(),
                "epoch": epoch.to_dict(),
                "addresses": addresses,
                "workers": [
                    {
                        "host": "127.0.0.1",
                        "port": p,
                        "nodes": [n for n in range(settings.nodes) if n % processes == rank],
                    }
                    for rank, p in enumerate(ports)
                ],
                "maintenance": maintenance,
            }
            manifest_path = directory / "cluster.json"
            manifest_path.write_text(json.dumps(manifest))
            children = []
            try:
                for rank in range(processes):
                    log = (directory / f"worker-{rank}.log").open("wb")
                    child = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "graphshardx",
                        "serve",
                        "--manifest",
                        str(manifest_path),
                        "--rank",
                        str(rank),
                        stdout=log,
                        stderr=log,
                    )
                    log.close()
                    children.append(child)
                # The external client does not inject virtual-node link delay. Worker transports do.
                client = TCPTransport(replace(settings, delay=0, loss=0, duplicate=0), addresses)
                cluster = cls(settings, epoch, client, directory, processes=children)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if any(c.returncode is not None for c in children):
                        raise RuntimeError("TCP worker exited; inspect its worker log")
                    answers = await asyncio.gather(
                        *(cluster.try_call(n, "heartbeat", {}) for n in range(settings.nodes))
                    )
                    if all(answers):
                        if maintenance:
                            cluster.start_harness()
                        return cluster
                    await asyncio.sleep(0.1)
                raise Unavailable("TCP workers did not start")
            except BaseException:
                for child in children:
                    if child.returncode is None:
                        child.terminate()
                await asyncio.gather(*(c.wait() for c in children))
                raise
        if backend not in {"local", "tcp", "mpi"}:
            raise ValueError("backend must be local, tcp, or mpi")
        nodes = {n: Node(n, settings, epoch, transport, directory) for n in owned}
        cluster = cls(settings, epoch, transport, directory, nodes=nodes, comm=comm)
        if maintenance:
            for node in nodes.values():
                node.start()
            cluster.start_harness()
        return cluster

    @property
    def rank(self):
        return self.comm.rank if self.comm else 0

    @property
    def origin_ids(self):
        return sorted(self.nodes) if self.comm else list(range(self.settings.nodes))

    async def call(self, node, method, args, timeout=None):
        result = await self.transport.call(
            node, node, method, args, timeout or self.settings.operation_timeout + 5
        )
        if method == "epoch":
            values = await self.transport.call(node, node, "configuration", {})
            self.epoch = Epoch.parse(
                values["epochs"].get(str(result["epoch"]), values["epochs"].get(result["epoch"]))
            )
        return result

    def start_harness(self):
        if self.settings.epoch_mode == "harness" and self.rank == 0:
            self.harness_tasks = [
                asyncio.create_task(self._harness_epochs()),
                asyncio.create_task(self._harness_repairs()),
            ]

    async def _harness_epochs(self):
        while True:
            await asyncio.sleep(self.settings.epoch_interval)
            reports = await asyncio.gather(
                *(self.try_call(n, "heartbeat", {}) for n in range(self.settings.nodes))
            )
            live = [r for r in reports if r]
            if live:
                leader = min(live, key=lambda r: (-r["weight"], r["node"]))["node"]
                try:
                    await self.call(leader, "epoch", {})
                except Unavailable:
                    LOG.warning("harness could not publish the next epoch")

    async def _harness_repairs(self):
        while True:
            await asyncio.sleep(self.settings.repair_interval)
            await asyncio.gather(
                *(
                    self.try_call(n, "repair", {})
                    for n in range(self.settings.nodes)
                    if n not in self.failed
                )
            )

    async def try_call(self, node, method, args):
        try:
            return await self.call(node, method, args, self.settings.request_timeout)
        except Unavailable:
            return None

    async def fail(self, nodes):
        self.failed.update(nodes)
        for n in nodes:
            if n in self.nodes:
                self.nodes[n].alive = False
            elif not self.comm:
                await self.call(n, "crash", {})

    async def recover(self, nodes):
        configuration = None
        for candidate in range(self.settings.nodes):
            if candidate not in self.failed:
                try:
                    configuration = await self.call(candidate, "configuration", {})
                    break
                except Unavailable:
                    continue
        if configuration is None:
            raise Unavailable("recovery requires a reachable later configuration")
        for n in nodes:
            if n in self.nodes or not self.comm:
                await self.call(n, "recover", {"configuration": configuration})
            self.failed.discard(n)

    async def synchronize(self):
        live = [n for n in range(self.settings.nodes) if n not in self.failed]
        targets = sorted(self.nodes) if self.comm else live
        await asyncio.gather(
            *(self.try_call(n, "gossip", {}) for n in targets if n not in self.failed)
        )
        if self.comm:
            await asyncio.to_thread(self.comm.Barrier)
        await asyncio.gather(
            *(self.try_call(n, "gossip", {}) for n in self.epoch.leaders if n not in self.failed)
        )

    async def aggregate(self, value):
        return await asyncio.to_thread(self.comm.allgather, value) if self.comm else [value]

    async def close(self):
        for task in self.harness_tasks:
            task.cancel()
        await asyncio.gather(*self.harness_tasks, return_exceptions=True)
        for child in self.processes:
            if child.returncode is None:
                child.terminate()
        await asyncio.gather(*(child.wait() for child in self.processes))
        for node in self.nodes.values():
            node.alive = False
        await self.transport.close()
        await asyncio.gather(*(node.stop() for node in self.nodes.values()))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


async def serve(manifest_path: Path, rank: int):
    manifest = json.loads(Path(manifest_path).read_text())
    settings = Settings(**manifest["settings"])
    epoch = Epoch.parse(manifest["epoch"])
    worker = manifest["workers"][rank]
    transport = TCPTransport(settings, manifest["addresses"])
    threadpool_limits(limits=1)
    nodes = [
        Node(n, settings, epoch, transport, Path(manifest_path).parent) for n in worker["nodes"]
    ]
    await transport.listen(worker["host"], worker["port"])
    if manifest.get("maintenance", True):
        for node in nodes:
            node.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        for node in nodes:
            node.alive = False
        await transport.close()
        await asyncio.gather(*(n.stop() for n in nodes))
