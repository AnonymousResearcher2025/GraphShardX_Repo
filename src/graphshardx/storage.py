import json
from pathlib import Path
import sqlite3
import threading
import uuid

from .model import CapacityError, Header, IntegrityError, Payload, canonical, digest, merge_record


class Store:
    """Each node owns its database. An acknowledgment follows a completed transaction."""

    def __init__(self, path: Path | None, capacity: int, predicate=None):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS headers(id TEXT PRIMARY KEY,value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS ledger(seq INTEGER PRIMARY KEY,kind TEXT NOT NULL,
                id TEXT NOT NULL,previous TEXT NOT NULL,hash TEXT NOT NULL,
                UNIQUE(kind,id));
            CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS payloads(id TEXT PRIMARY KEY,data BLOB NOT NULL,
                token TEXT NOT NULL,shard INTEGER NOT NULL,created INTEGER NOT NULL,
                sidechain INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS pointers(id TEXT PRIMARY KEY,value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS pending(id TEXT PRIMARY KEY,header BLOB NOT NULL,
                data BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS paxos(slot INTEGER PRIMARY KEY,promised BLOB,
                accepted_ballot BLOB,accepted BLOB,decided BLOB);
        """)
        self.capacity = capacity
        self.predicate = predicate
        self.lock = threading.RLock()

    def close(self):
        with self.lock:
            self.db.close()

    def meta(self, key, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, canonical(value)))

    def reserve_ids(self, count):
        with self.lock, self.db:
            start = self.meta("counter", 0)
            if start + count > 2**64:
                raise OverflowError("origin counter exhausted")
            self.set_meta("counter", start + count)
            return start

    def append_header(self, header: Header, kind="header"):
        raw = canonical(header.to_dict())
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT value FROM headers WHERE id=?", (header.block_id,)
            ).fetchone()
            if row and bytes(row[0]) != raw:
                raise IntegrityError("immutable header conflict")
            self.db.execute("INSERT OR IGNORE INTO headers VALUES(?,?)", (header.block_id, raw))
            if not self.db.execute(
                "SELECT 1 FROM ledger WHERE kind=? AND id=?", (kind, header.block_id)
            ).fetchone():
                previous = self.db.execute(
                    "SELECT hash FROM ledger WHERE kind=? ORDER BY seq DESC LIMIT 1", (kind,)
                ).fetchone()
                previous = previous[0] if previous else "0" * 64
                hashed = digest(bytes.fromhex(previous) + raw)
                self.db.execute(
                    "INSERT INTO ledger(kind,id,previous,hash) VALUES(?,?,?,?)",
                    (kind, header.block_id, previous, hashed),
                )

    def header(self, block_id):
        with self.lock:
            row = self.db.execute("SELECT value FROM headers WHERE id=?", (block_id,)).fetchone()
            if not row:
                raise KeyError(block_id)
            return Header.parse(json.loads(row[0]))

    def verify_ledgers(self):
        with self.lock:
            previous = {}
            for kind, block_id, prev, hashed in self.db.execute(
                "SELECT kind,id,previous,hash FROM ledger ORDER BY seq"
            ):
                raw = canonical(self.header(block_id).to_dict())
                if prev != previous.get(kind, "0" * 64) or hashed != digest(
                    bytes.fromhex(prev) + raw
                ):
                    raise IntegrityError(f"broken {kind} ledger at {block_id}")
                previous[kind] = hashed
            return len(list(self.db.execute("SELECT seq FROM ledger")))

    @property
    def used(self):
        with self.lock:
            return self.db.execute("SELECT coalesce(sum(length(data)),0) FROM payloads").fetchone()[
                0
            ]

    def put_payload(self, header: Header, data: bytes, shard: int, node: int, relocated=False):
        Payload.decode(data, header, self.predicate)
        with self.lock, self.db:
            existing = self.db.execute(
                "SELECT data,token FROM payloads WHERE id=?", (header.block_id,)
            ).fetchone()
            if existing:
                Payload.decode(bytes(existing[0]), header, self.predicate)
                return existing[1]
            if self.used + len(data) > self.capacity:
                raise CapacityError("payload storage capacity exceeded")
            self.append_header(header, "sidechain" if relocated else "header")
            token = f"{node}:{uuid.uuid4().hex}"
            self.db.execute(
                "INSERT INTO payloads VALUES(?,?,?,?,?,?)",
                (header.block_id, data, token, shard, header.created_min, int(relocated)),
            )
            self.db.execute("DELETE FROM pointers WHERE id=?", (header.block_id,))
            if relocated:
                self.set_meta("tokens", self.meta("tokens", 0) + 1)
            return token

    def sidechain_copy(self, block_id):
        with self.lock:
            row = self.db.execute(
                "SELECT sidechain FROM payloads WHERE id=?", (block_id,)
            ).fetchone()
            return bool(row and row[0])

    def get_payload(self, block_id):
        with self.lock:
            row = self.db.execute(
                "SELECT data,token,shard FROM payloads WHERE id=?", (block_id,)
            ).fetchone()
            if row:
                return bytes(row[0]), row[1], row[2]
            pointer = self.db.execute(
                "SELECT value FROM pointers WHERE id=?", (block_id,)
            ).fetchone()
            return {"pointer": json.loads(pointer[0])} if pointer else None

    def free_payload(self, block_id, token, pointer):
        with self.lock, self.db:
            row = self.db.execute("SELECT token FROM payloads WHERE id=?", (block_id,)).fetchone()
            if row and row[0] != token:
                raise IntegrityError("copy changed during migration")
            self.db.execute(
                "INSERT OR REPLACE INTO pointers VALUES(?,?)", (block_id, canonical(pointer))
            )
            self.db.execute("DELETE FROM payloads WHERE id=? AND token=?", (block_id, token))

    def local_blocks(self):
        with self.lock:
            return [
                {"block": b, "token": t, "shard": s, "bytes": n}
                for b, t, s, n in self.db.execute(
                    "SELECT id,token,shard,length(data) FROM payloads ORDER BY created,id"
                )
            ]

    def merge(self, record):
        header = Header.parse(record["header"])
        with self.lock, self.db:
            self.append_header(header)
            row = self.db.execute(
                "SELECT value FROM records WHERE id=?", (header.block_id,)
            ).fetchone()
            result = merge_record(json.loads(row[0]) if row else None, record)
            self.db.execute(
                "INSERT OR REPLACE INTO records VALUES(?,?)", (header.block_id, canonical(result))
            )
            return result

    def records(self):
        with self.lock:
            return {
                b: json.loads(value) for b, value in self.db.execute("SELECT id,value FROM records")
            }

    def record(self, block_id):
        with self.lock:
            row = self.db.execute("SELECT value FROM records WHERE id=?", (block_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def queue(self, header, data):
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO pending VALUES(?,?,?)",
                (header.block_id, canonical(header.to_dict()), data),
            )

    def pending(self):
        with self.lock:
            return [
                (Header.parse(json.loads(h)), bytes(d))
                for h, d in self.db.execute("SELECT header,data FROM pending")
            ]

    def dequeue(self, block_id):
        with self.lock, self.db:
            self.db.execute("DELETE FROM pending WHERE id=?", (block_id,))

    def paxos(self, slot, action, ballot=None, value=None):
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO paxos(slot) VALUES(?)", (slot,))
            row = self.db.execute(
                "SELECT promised,accepted_ballot,accepted,decided FROM paxos WHERE slot=?", (slot,)
            ).fetchone()
            promised, accepted_ballot, accepted, decided = [
                json.loads(x) if x else None for x in row
            ]
            response = {
                "promised": promised,
                "accepted_ballot": accepted_ballot,
                "accepted": accepted,
                "decided": decided,
                "ok": False,
            }
            if action == "read":
                return response
            if action == "decide":
                if decided and decided != value:
                    raise IntegrityError("conflicting epoch decisions")
                self.db.execute("UPDATE paxos SET decided=? WHERE slot=?", (canonical(value), slot))
                return {"ok": True}
            if promised is None or tuple(ballot) >= tuple(promised):
                if action == "prepare":
                    self.db.execute(
                        "UPDATE paxos SET promised=? WHERE slot=?", (canonical(ballot), slot)
                    )
                elif action == "accept":
                    if accepted_ballot == ballot and accepted != value:
                        raise IntegrityError("ballot reused for a different value")
                    self.db.execute(
                        "UPDATE paxos SET promised=?,accepted_ballot=?,accepted=? WHERE slot=?",
                        (canonical(ballot), canonical(ballot), canonical(value), slot),
                    )
                else:
                    raise ValueError("unknown Paxos action")
                response["ok"] = True
            return response
