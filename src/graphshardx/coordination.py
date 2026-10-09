import asyncio

import numpy as np
from sklearn.cluster import KMeans

from .model import Epoch, Header, Payload, Unavailable, canonical, digest
from .placement import replica_factor


async def next_configuration(node):
    settings = node.settings
    old = node.epoch
    live = await node.reachable()
    if len(live) < settings.leaders:
        raise Unavailable("insufficient live nodes for the next leader tier")
    weights = {n: node.reports[n]["weight"] for n in live}
    # Absence contributes a zero uptime sample; degree and trust tokens remain recorded.
    node_weights = dict(old.node_weights) | weights
    leaders = tuple(sorted(live, key=lambda n: (-weights[n], n))[: settings.leaders])
    free = dict(old.free) | {n: node.reports[n]["free"] for n in live}
    records = await node.catalog()
    shards = {s: dict(v) for s, v in old.shards.items()}
    block_shards = dict(old.block_shards)
    retired = dict(old.retired)
    requested = set(node.store.meta("splits", [])) if settings.rebalancing else set()
    if settings.rebalancing:
        for leader in old.leaders:
            if leader in node.transport.nodes:
                requested.update(node.transport.nodes[leader].store.meta("splits", []))
    for shard_id, shard in list(shards.items()):
        entries = [
            r for r in records.values() if old.shard_for(Header.parse(r["header"])) == shard_id
        ]
        samples = []
        means = {}
        sample_budget = settings.centroid_sample
        for r in sorted(entries, key=lambda r: r["header"]["payload_digest"]):
            if sample_budget <= 0 and shard_id not in requested:
                break
            try:
                header = Header.parse(r["header"])
                payload = Payload.decode(await node.fetch(r), header)
                means[header.block_id] = payload.vectors.mean(axis=0, dtype=np.float64)
                count = min(len(payload.vectors), max(1, sample_budget))
                chosen = node.rng.choice(len(payload.vectors), size=count, replace=False)
                samples.append(payload.vectors[chosen])
                sample_budget -= count
            except Unavailable:
                continue
        if samples:
            points = np.concatenate(samples)
            shard["centroid"] = points.mean(axis=0, dtype=np.float64).tolist()
        mean_use = (
            np.mean(
                [
                    node.reports[n]["used"] / node.reports[n]["capacity"]
                    for n in shard["peers"]
                    if n in live
                ]
            )
            if any(n in live for n in shard["peers"])
            else 1
        )
        if settings.rebalancing and mean_use > settings.rho:
            requested.add(shard_id)
        enough = len(shard["peers"]) >= max(
            settings.minimum_split_peers, 2 * settings.minimum_replicas
        )
        candidates = sorted(
            (n for n in live if n not in leaders and n not in shard["peers"] and free[n] > 0),
            key=lambda n: (-free[n], n),
        )
        old_peers = [n for n in shard["peers"] if n in live and n not in leaders]
        replacements = candidates[: max(0, len(shard["peers"]) - len(old_peers))]
        old_peers += replacements
        candidates = [n for n in candidates if n not in replacements]
        if (
            shard_id in requested
            and enough
            and len(candidates) >= settings.minimum_replicas
            and len(old_peers) >= settings.minimum_replicas
            and len(samples)
            and len(np.unique(points, axis=0)) >= 2
        ):
            centers = await node.work(
                lambda points=points: (
                    KMeans(n_clusters=2, n_init=10, random_state=settings.seed + old.number + 1)
                    .fit(points)
                    .cluster_centers_
                )
            )
            child0 = max(set(shards) | set(retired), default=-1) + 1
            child1 = child0 + 1
            # Existing immutable blocks remain indivisible; their mean selects a child.
            for block_id, center in means.items():
                block_shards[block_id] = (
                    child0 if np.argmin(np.sum((centers - center) ** 2, axis=1)) == 0 else child1
                )
            for r in entries:
                if r["header"]["payload_digest"] not in means:
                    block_shards[r["header"]["payload_digest"]] = child0
            new_peers = candidates[: len(old_peers)]
            shards[child0] = {
                "centroid": centers[0].tolist(),
                "peers": old_peers,
                "replicas": replica_factor(old_peers, free, settings),
            }
            shards[child1] = {
                "centroid": centers[1].tolist(),
                "peers": new_peers,
                "replicas": replica_factor(new_peers, free, settings),
            }
            retired[shard_id] = (child0, child1)
            del shards[shard_id]
        else:
            # Crashed peers can be replaced at a boundary; never reduce the configured durability floor.
            peers = [n for n in shard["peers"] if n in live and n not in leaders]
            for n in sorted((n for n in live if n not in leaders), key=lambda n: (-free[n], n)):
                if len(peers) >= len(shard["peers"]):
                    break
                if n not in peers:
                    peers.append(n)
            if len(peers) < settings.minimum_replicas:
                raise Unavailable("shard cannot meet the replication floor")
            shard["peers"] = peers
            shard["replicas"] = replica_factor(peers, free, settings)
            shards[shard_id] = shard
    for block_id, shard_id in list(block_shards.items()):
        while shard_id in retired:
            shard_id = retired[shard_id][0]
        block_shards[block_id] = shard_id
    epoch = Epoch(
        old.number + 1,
        leaders,
        {n: weights[n] for n in leaders},
        node_weights,
        shards,
        old.adjacency,
        block_shards,
        retired,
        free,
    )
    return Epoch.parse(epoch.to_dict())


async def propose(node):
    old = node.epoch
    proposal = (await next_configuration(node)).to_dict()
    if node.settings.epoch_mode == "harness":
        decision = None
    else:
        majority = len(old.leaders) // 2 + 1
        # Ballots are durable, unique (counter, node), and totally ordered.
        counter = node.store.meta("ballot", 0) + 1
        node.store.set_meta("ballot", counter)
        ballot = [counter, node.id]

        async def send(leader, action, value=None):
            try:
                response = await node.call(
                    leader,
                    "paxos",
                    {"slot": old.number + 1, "action": action, "ballot": ballot, "value": value},
                )
                return leader, response
            except Unavailable:
                return leader, None

        prepared = await asyncio.gather(*(send(n, "prepare") for n in old.leaders))
        promises = [(n, r) for n, r in prepared if r and r["ok"]]
        highest = max(
            (r["promised"][0] for _, r in prepared if r and r["promised"]), default=counter
        )
        node.store.set_meta("ballot", max(counter, highest))
        if len(promises) < majority:
            raise Unavailable("outgoing leaders cannot form an epoch consensus majority")
        accepted = [r for _, r in promises if r["accepted_ballot"]]
        if accepted:
            proposal = max(accepted, key=lambda r: tuple(r["accepted_ballot"]))["accepted"]
        replies = await asyncio.gather(*(send(n, "accept", proposal) for n in old.leaders))
        acceptors = sorted(n for n, r in replies if r and r["ok"])
        if len(acceptors) < majority:
            raise Unavailable("epoch proposal did not receive an acceptor majority")
        decision = {
            "slot": old.number + 1,
            "ballot": ballot,
            "acceptors": acceptors,
            "value_digest": digest(canonical(proposal)),
        }
    destinations = list(proposal["leaders"]) + [
        n for n in old.node_weights if n not in proposal["leaders"]
    ]

    async def activate(n):
        try:
            return await node.call(
                n,
                "activate",
                {
                    "epoch": proposal,
                    "decision": decision,
                    "harness": node.settings.epoch_mode == "harness",
                },
                node.settings.operation_timeout,
            )
        except Unavailable:
            return None

    # New leaders transfer state before publication to the remaining nodes.
    count = len(proposal["leaders"])
    leaders = await asyncio.gather(*(activate(n) for n in destinations[:count]))
    if not any(leaders):
        raise Unavailable("no incoming leader completed header handoff")
    await asyncio.gather(*(activate(n) for n in destinations[count:]))
    node.store.set_meta("splits", [])
    return {
        "epoch": proposal["number"],
        "leaders": proposal["leaders"],
        "shards": len(proposal["shards"]),
        "mode": node.settings.epoch_mode,
    }
