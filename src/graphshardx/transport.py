import asyncio
import random
import struct
import threading
import time
import uuid
import msgpack

from . import model
from .config import Settings
from .model import GraphShardXError, Unavailable

MAX_FRAME = 256 * 1024 * 1024


def encode(value):
    def compact(v):
        if isinstance(v, dict):
            if set(v) == set(model.Header.__dataclass_fields__):
                return msgpack.ExtType(42, model.Header.parse(v).to_bytes())
            return {k: compact(item) for k, item in v.items()}
        if isinstance(v, (tuple, list)):
            return [compact(item) for item in v]
        return v

    return msgpack.packb(compact(value), use_bin_type=True)


def decode(data):
    def extension(code, value):
        if code != 42:
            raise ValueError("unknown message extension")
        return model.Header.from_bytes(value).to_dict()

    return msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=extension)


def error_value(exc):
    return {"error": type(exc).__name__, "message": str(exc)}


def unwrap(response):
    if "error" in response:
        cls = getattr(model, response["error"], GraphShardXError)
        raise cls(response["message"])
    return response["result"]


class Transport:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.nodes = {}
        self.rng = random.Random(settings.seed)
        self.partitions = []
        self.metrics = {"requests": 0, "request_bytes": 0, "response_bytes": 0, "dropped": 0}
        self.reply_tasks = {}
        self.replies = {}

    def register(self, node):
        self.nodes[node.id] = node

    def blocked(self, src, dst):
        return bool(
            self.partitions and not any(src in group and dst in group for group in self.partitions)
        )

    async def dispatch(self, src, dst, method, args):
        node = self.nodes.get(dst)
        if node and method == "crash":
            node.alive = False
            return {"crashed": dst}
        if node and method == "recover":
            latest = max(int(k) for k in args["configuration"]["epochs"])
            if latest <= node.epoch.number:
                raise ValueError("recovery must rejoin a later epoch")
            node.alive = True
            missed = latest - node.epoch.number
            node.uptime_history = (node.uptime_history + [0.0] * missed)[
                -node.settings.uptime_window :
            ]
            node.store.set_meta("uptime_history", node.uptime_history)
            try:
                for number in sorted(int(k) for k in args["configuration"]["epochs"]):
                    if number > node.epoch.number:
                        epochs = args["configuration"]["epochs"]
                        decisions = args["configuration"]["decisions"]
                        await node.activate(
                            epochs.get(str(number), epochs.get(number)),
                            decisions.get(str(number), decisions.get(number)),
                            node.settings.epoch_mode == "harness",
                            record_uptime=False,
                        )
            except BaseException:
                node.alive = False
                raise
            return {"recovered": dst}
        if node is None or not node.alive:
            raise Unavailable(f"node {dst} is unavailable")
        result = await node.handle(src, method, args)
        if not node.alive:
            raise Unavailable(f"node {dst} crashed before acknowledgment")
        return result

    async def _exchange(self, src, dst, method, args):
        return await self.dispatch(src, dst, method, args)

    async def call(self, src, dst, method, args, timeout=None):
        if src in self.nodes and not self.nodes[src].alive and method != "recover":
            raise Unavailable(f"origin {src} has crashed")
        budget = timeout or self.settings.request_timeout
        key = None
        if method in {"header", "payload"}:
            header = model.Header.parse(args["header"])
            epoch = args["epoch"] if method == "header" else args["certificate"]["epoch"]
            key = (
                src,
                dst,
                method,
                header.block_id,
                header.to_bytes(),
                epoch,
                args.get("path"),
                tuple(args.get("committee", [])),
                args.get("relocated", False),
            )
            if key in self.replies:
                return self.replies.pop(key)

        async def request():
            self.metrics["requests"] += 1
            self.metrics["request_bytes"] += len(encode(args))
            if src != dst:
                await asyncio.sleep(self.settings.delay)
                if self.blocked(src, dst) or self.rng.random() < self.settings.loss:
                    self.metrics["dropped"] += 1
                    await asyncio.sleep(budget)
                    raise Unavailable("request was dropped")
            result = await self._exchange(src, dst, method, args)
            if src != dst and self.rng.random() < self.settings.duplicate:
                await self._exchange(src, dst, method, args)
            if src != dst:
                await asyncio.sleep(self.settings.delay)
                if self.rng.random() < self.settings.loss:
                    self.metrics["dropped"] += 1
                    await asyncio.sleep(budget)
                    raise Unavailable("acknowledgment was dropped")
            self.metrics["response_bytes"] += len(encode(result))
            return result

        task = asyncio.create_task(request())
        if key:
            self.reply_tasks[task] = key
            task.add_done_callback(self._retain_reply)
        try:
            # An RPC deadline does not revoke a durable write or discard its eventual acknowledgment.
            result = await asyncio.wait_for(asyncio.shield(task) if key else task, budget)
            if key:
                self.replies.pop(key, None)
            return result
        except (TimeoutError, OSError, ConnectionError) as exc:
            raise Unavailable(f"{method}: node {dst} did not respond") from exc

    def _retain_reply(self, task):
        key = self.reply_tasks.pop(task, None)
        if not task.cancelled() and task.exception() is None and key:
            self.replies[key] = task.result()

    def forget_replies(self, src, block, method=None):
        def matches(key):
            return (
                key and key[0] == src and key[3] == block and (method is None or key[2] == method)
            )

        for key in list(self.replies):
            if matches(key):
                del self.replies[key]
        for task, key in self.reply_tasks.items():
            if matches(key):
                self.reply_tasks[task] = None

    async def close(self):
        tasks = list(self.reply_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.replies.clear()


async def read_frame(reader):
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    if size > MAX_FRAME:
        raise ValueError("message exceeds maximum frame size")
    return decode(await reader.readexactly(size))


async def write_frame(writer, value):
    raw = encode(value)
    if len(raw) > MAX_FRAME:
        raise ValueError("message exceeds maximum frame size")
    writer.write(struct.pack("!I", len(raw)) + raw)
    await writer.drain()


class TCPTransport(Transport):
    def __init__(self, settings, addresses):
        super().__init__(settings)
        self.addresses = {int(k): tuple(v) for k, v in addresses.items()}
        self.server = None

    async def listen(self, host, port):
        self.server = await asyncio.start_server(self._serve, host, port, limit=MAX_FRAME)

    async def _serve(self, reader, writer):
        try:
            message = await read_frame(reader)
            try:
                result = await self.dispatch(
                    message["src"], message["dst"], message["method"], message["args"]
                )
                response = {"result": result}
            except Exception as exc:
                response = error_value(exc)
            await write_frame(writer, response)
        except (ConnectionError, OSError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _exchange(self, src, dst, method, args):
        if dst in self.nodes:
            return await self.dispatch(src, dst, method, args)
        reader, writer = await asyncio.open_connection(*self.addresses[dst], limit=MAX_FRAME)
        try:
            await write_frame(writer, {"src": src, "dst": dst, "method": method, "args": args})
            return unwrap(await read_frame(reader))
        finally:
            writer.close()
            await writer.wait_closed()

    async def close(self):
        await super().close()
        if self.server:
            self.server.close()
            await self.server.wait_closed()


class MPITransport(Transport):
    """MPI transports real requests; a progress thread dispatches into the node event loop."""

    def __init__(self, settings, comm):
        from mpi4py import MPI

        if MPI.Query_thread() < MPI.THREAD_MULTIPLE:
            raise RuntimeError("MPI_THREAD_MULTIPLE is required")
        super().__init__(settings)
        self.comm = comm
        self.loop = asyncio.get_running_loop()
        self.pending = {}
        self.running = True
        self.thread = threading.Thread(target=self._progress, daemon=True)
        self.thread.start()

    def _progress(self):
        from mpi4py import MPI

        while self.running:
            if self.comm.iprobe(source=MPI.ANY_SOURCE, tag=71):
                message = self.comm.recv(source=MPI.ANY_SOURCE, tag=71)
                self.loop.call_soon_threadsafe(self._receive, decode(message))
            else:
                time.sleep(0.001)

    def _receive(self, message):
        if "response" in message:
            future = self.pending.get(message["id"])
            if future and not future.done():
                future.set_result(message["response"])
        else:
            asyncio.create_task(self._answer(message))

    async def _answer(self, message):
        try:
            response = {
                "result": await self.dispatch(
                    message["src"], message["dst"], message["method"], message["args"]
                )
            }
        except Exception as exc:
            response = error_value(exc)
        await asyncio.to_thread(
            self.comm.send,
            encode({"id": message["id"], "response": response}),
            dest=message["rank"],
            tag=71,
        )

    async def _exchange(self, src, dst, method, args):
        if dst in self.nodes:
            return await self.dispatch(src, dst, method, args)
        identity = uuid.uuid4().hex
        future = self.loop.create_future()
        self.pending[identity] = future
        try:
            await asyncio.to_thread(
                self.comm.send,
                encode(
                    {
                        "id": identity,
                        "src": src,
                        "dst": dst,
                        "rank": self.comm.rank,
                        "method": method,
                        "args": args,
                    }
                ),
                dest=dst % self.comm.size,
                tag=71,
            )
            return unwrap(await future)
        finally:
            self.pending.pop(identity, None)

    async def close(self):
        await super().close()
        self.running = False
        await asyncio.to_thread(self.thread.join, 2)
