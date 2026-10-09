# GraphShardX

GraphShardX coordinates immutable vector inserts across a provisioned deployment of edge nodes. It places similar vectors together, certifies block headers without ordering inserts, replicates validated payloads, and makes stored-data changes detectable. It targets crash faults, message loss, and partitions that eventually heal. Updates, deletes, Byzantine participants, and ANN indexes are outside the paper's scope.

## Architecture

Origins cluster each batch with DBSCAN and apply k-means to the noise, then assign whole clusters to the nearest shard centroid. Each shard receives one immutable block per batch. Vector identifiers are `(origin, persistent counter)`; the counter is reserved durably before use. Payloads contain float32 vectors, identifiers, capture times, and JSON application metadata, all covered by SHA-256. Binary headers contain 80 bytes of content.

A Barabási–Albert overlay defines connectivity. The epoch's highest-weight nodes become leaders, using exactly `degree * (1 + 0.3 * uptime + 0.2 * tokens / 100)`. Origins choose coordinators by links into the shard, tokens, and observed latency. Concurrent blocks use disjoint leader candidate pools within a wave; batches with more blocks than leaders use successive waves.

Protocol 2 has separate **pending**, **certified**, and **committed** states. Leaders persist headers before approving them. A certificate requires at least `theta` of the fixed epoch weight, or `f+1` approvals from a complete `2f+1` reachable committee when fewer than `beta` of the batch's blocks certify within `T`. An origin commits only after `R` distinct shard peers acknowledge validated payload storage. Replica shortages leave blocks pending; they never lower `R` during a write. Stored, certified payloads can be searchable before commit. Pending blocks are retried with the next batch or an explicit flush. Duplicate delivery does not insert another copy or count another commit.

Protocol 1 moves the oldest blocks from overloaded nodes. Receivers validate and store before sources free payloads. Out-of-shard relocation uses a nonadjacent leader's hash-linked sidechain and awards participation tokens. Sources retain verified forwarding pointers. Location records use an observed-remove set of immutable copy tokens: delayed messages cannot resurrect a removed copy. Headers remain unchanged by migration. Repair restores missing replicas and places split-shard blocks on their child peers before releasing obsolete copies. Epoch shard peers exclude leaders; repair transfers ordinary copies held by newly promoted leaders, while explicitly relocated sidechain copies remain eligible.

Each node maintains its own hash-linked header ledger and, when needed, a sidechain. Leaders exchange missing or changed records. A search routes to the nearest `p` shard centroids, scans one verified replica per block with exact Euclidean distance, and merges and deduplicates the top results. Missing replicas are reported; corrupted data never become search results. A read checks the entire payload against the intact certified header.

Two epoch modes are explicit:

- `consensus`: outgoing leaders run durable, two-phase Paxos once per epoch. Incoming leaders fetch outgoing header records before activating. Decisions and old configurations remain available for delayed nodes and recovered nodes. Without a consensus majority, ingestion stays in the current epoch.
- `harness`: the experiment harness publishes configurations without Paxos and schedules replica repair, matching the distinction in Section IV. The data path still performs real placement, message exchange, certification, validation, replication, migration, and search. This mode's epoch publication is **emulated**, not a distributed agreement result.

`local` uses independent virtual-node stores and worker pools in one process. `tcp` runs actual subprocess workers with framed binary messages. `mpi` exchanges messages between MPI ranks and uses in-memory transfers within each rank. Delay, loss, and duplication affect requests and responses asynchronously; timeouts cause real resends. Late header and payload acknowledgments remain available to retries of the same immutable block and epoch. Failure injection stops the targeted virtual nodes, as in the paper; it does not terminate MPI ranks. The overlay selects roles and coordinators; the paper specifies point-to-point transport, not a multihop routing protocol.

## Structure

```text
configs/                  local and paper experiment settings
src/graphshardx/
  model.py, storage.py    immutable data, certificates, ledgers, durable state
  placement.py            overlay, weights, clustering, coordinator selection
  node.py                 ingestion, query, migration, repair, background tasks
  coordination.py         configuration computation, Paxos, epoch handoff
  transport.py, runtime.py shared-memory, TCP, MPI and process lifecycle
  datasets.py             TEXMEX, NumPy, HDF5 and geospatial input
  benchmark.py            measured experiment grids, recall, plots
  baselines.py, energy.py real database adapters and RAPL measurements
  cli.py                  command-line interface
tests/                    unit, failure, persistence and process integration tests
```

## Requirements and installation

Use Python 3.11 or newer. The verified environment uses Python 3.12; dependency versions are pinned in `requirements.lock`. MPI requires an MPI implementation with `MPI_THREAD_MULTIPLE`; MPICH 4.3.2 and mpi4py 4.1.2 were tested. Qdrant, Weaviate, and Pinecone measurements require real running services. Energy measurement requires readable Linux Intel RAPL package counters. NumPy exact scans are the search index.

```bash
cd GraphShardX
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m build
python -m pytest -q
python -m ruff check .
```

The dependency lock includes the tested MPICH runtime. If using a site's existing MPI instead, install `python -m pip install -e '.[experiments,mpi,test]'` into an environment linked to that MPI. Do not mix incompatible MPI launchers and libraries. The MPI integration test skips only when its optional launcher/package is absent; other tests require the base and test dependencies.

## Run

```bash
graphshardx sanity
graphshardx sanity --transport tcp --processes 2
mpiexec -n 2 python -m graphshardx sanity --transport mpi

graphshardx experiment --config configs/local.toml \
  --dataset /absolute/path/vectors.npy --workload 10000 \
  --transport tcp --processes 2 --output runs/local
```

The sanity command uses a small deterministic fixture solely to verify ingestion, replicated storage, exact retrieval, and ledger integrity. Experiment commands always require supplied input data. Accepted inputs are `.npy`, `.fvecs`, `.ivecs`, `.h5`, and `.hdf5`. HDF5 uses `train` for base vectors, `test` for queries, and `neighbors` for ground truth; `--key` changes the base key. Ground truth uses zero-based positions in the supplied base dataset. Input dimensions must match the configuration, except the explicitly declared dimension experiment, which uses leading coordinate subsets.

Each experiment writes dataset checksums, effective settings, dependency versions, operating-system family, processor architecture, raw measured metrics, and node databases or worker logs under its chosen run directory. Provenance uses neutral input-role labels rather than filenames or local paths; nonstandard HDF5 keys are labeled `custom`. Result files omit service addresses and raw service metadata, retaining numeric server versions when available. Use a fresh directory for each invocation. Throughput counts unique newly committed vectors only; pending blocks count distinct outstanding blocks returned by ingestion calls. Write latency is batch duration divided by submitted batch size; commit latency is the full batch duration. Memory is actual process RSS divided equally among that process's virtual nodes and compared with their configured memory shares. This includes runtime overhead and is not an invented estimate of vector bytes. Transport counters identify their scope: the TCP client, MPI rank zero, or all local virtual nodes. Energy reports measured package-counter deltas, including potentially negative idle-subtracted values due to noise.

For an anonymous submission, include source, configurations, tests, dependency files, and this README. Builds use neutral archive ownership and fixed timestamps. Installed environments, build caches, Git history, deployment manifests, node databases, and runtime logs are local operational artifacts and are excluded from the submission. Logs can contain application data or diagnostic paths; review any separately published experiment artifacts before attaching them.

For a persistent TCP deployment, keep the run's `cluster.json` and start each worker with `graphshardx serve --manifest /path/cluster.json --rank 0` (and each other rank). Adjust the manifest's worker and node addresses for multiple hosts before starting them. Workers recover node databases in that directory. External applications can use `graphshardx request --manifest /path/cluster.json --node 17 --method search --json /path/request.json`, where the request contains `{"query":[...],"k":10,"probes":4}`. Other methods support batch ingestion, size/timeout-buffered single inserts, flush, point reads, audits, maintenance, and epoch advancement. The Python `Node` constructor accepts an application validation predicate; origins and every storing peer run it over all coordinates.

## Paper experiments

Obtain the GIST-1M base, query, and ground-truth files from [TEXMEX](http://corpus-texmex.irisa.fr/), or provide the equivalent ANN-benchmarks HDF5. GIST has 1M base vectors, 960 dimensions, and 1,000 queries. On the paper's 18-server testbed, launch MPI ranks with the site's MPICH host allocation; ranks own nodes by `node_id % rank_count`. The following commands execute the paper's experiment grids and produce new measurements:

```bash
export GIST_BASE=/absolute/path/gist_base.fvecs
export GIST_QUERY=/absolute/path/gist_query.fvecs
export GIST_TRUTH=/absolute/path/gist_groundtruth.ivecs

# Figures 2 and 3, Tables II through IV, and the failure experiment.
mpiexec -n 18 python -m graphshardx experiment --config configs/paper.toml \
  --dataset "$GIST_BASE" --queries "$GIST_QUERY" --groundtruth "$GIST_TRUTH" \
  --transport mpi --suite all --repeats 5 --output runs/paper

# Individual grids: workload, network, dimension, network-conditions,
# ablation, memory, search, failures. For example, Table IV:
mpiexec -n 18 python -m graphshardx experiment --config configs/paper.toml \
  --dataset "$GIST_BASE" --queries "$GIST_QUERY" --groundtruth "$GIST_TRUTH" \
  --transport mpi --suite search --output runs/search

graphshardx plot --input runs/paper/results.jsonl --output runs/paper/throughput.png
```

Workloads are 200K, 400K, 600K, 800K, and 1M; node counts are 100, 200, 300, 400, and 500; dimension subsets are 200, 400, 600, 800, and 960. Link experiments use 0, 10, 25, 50, and 100 ms one-way delay, plus 50 ms with 1% loss; `T` becomes `100 ms + 2*delay`. Failure experiments target highest-weight nodes first at 0%, 10%, 20%, 30%, and 40% during ingestion. Search probes 1, 2, 4, 8, and 16 of 64 shards, compares similarity and uniformly randomized whole-block placement, and repeats similarity placement after 20% random failures. The combined sharding ablation disables both clustering-based placement and rebalancing; origins send whole batches to random shards. Read counts and repeated-run seeds are recorded.

Run baseline servers with the same CPU and memory budgets and default HNSW settings. Adapters change only collection identity, externally supplied vectors, and the distance metric to Euclidean for consistent GIST comparisons. They retain numeric server versions, require successful acknowledgments, check returned vectors, and abort on service errors. Fresh isolated collections/namespaces are created; existing collections are not deleted.

```bash
graphshardx baseline --system qdrant --endpoint http://localhost:6333 \
  --config configs/paper.toml --dataset "$GIST_BASE" --suite all --repeats 5 \
  --queries "$GIST_QUERY" --groundtruth "$GIST_TRUTH" --output runs/qdrant
graphshardx baseline --system weaviate --endpoint http://localhost:8080 \
  --config configs/paper.toml --dataset "$GIST_BASE" --suite all --repeats 5 \
  --queries "$GIST_QUERY" --groundtruth "$GIST_TRUTH" --output runs/weaviate

graphshardx plot --input runs/paper/results.jsonl --metric figure2 \
  --baselines runs/qdrant/results.jsonl runs/weaviate/results.jsonl --output runs/paper/figure2.png
graphshardx plot --input runs/paper/results.jsonl --metric ablation --output runs/paper/figure3.png
graphshardx plot --input runs/paper/results.jsonl --metric failures --output runs/paper/figure4.png
graphshardx plot --input runs/paper/results.jsonl --metric search --output runs/paper/search.png

# Table V, on one host with matching cgroup budgets for each system.
graphshardx energy --system graphshardx --config configs/paper.toml \
  --dataset "$GIST_BASE" --workload 200000 --idle-seconds 300 --repeats 5 \
  --transport tcp --processes 18 --output runs/energy-graphshardx
graphshardx energy --system qdrant --endpoint http://localhost:6333 \
  --config configs/paper.toml --dataset "$GIST_BASE" --workload 200000 \
  --idle-seconds 300 --repeats 5 --output runs/energy-qdrant
graphshardx energy --system weaviate --endpoint http://localhost:8080 \
  --config configs/paper.toml --dataset "$GIST_BASE" --workload 200000 \
  --idle-seconds 300 --repeats 5 --output runs/energy-weaviate
```

Baseline `--suite all` runs the workload, node, and dimension grids. Individual grids use `--suite workload`, `--suite network`, or `--suite dimension`; omit `--suite` for a single declared workload. Pinecone is optional: set `PINECONE_API_KEY` and use `baseline --system pinecone --endpoint https://YOUR_EXISTING_INDEX_HOST ...`. The existing index must have the requested dimension and Euclidean metric; its WAN timings are reference measurements. Qdrant and Weaviate keys use `QDRANT_API_KEY` and `WEAVIATE_API_KEY`. No credentials enter result files.

For NOAA data, use `graphshardx geospatial --input /path/climate.geojson --columns ATTRIBUTE1 ATTRIBUTE2 --output /path/climate.npy`, then a configuration matching the resulting dimension and a workload no larger than the file. Extraction uses WGS84 representative-point coordinates and the explicitly named numeric attributes. The paper does not identify its NOAA records or feature construction, so this command does not claim to recover that exact 258-vector dataset.

## Parameters and interpretation

The paper explicitly supplies `theta=0.51`, `beta=0.8`, `T=100 ms`, `rho=0.8`, the two weight coefficients, six peers as the split threshold, and 64 shards for the search experiment. `configs/paper.toml` preserves these. Its unspecified settings are explicit, configurable choices:

| Setting | Default choice and reason |
| --- | --- |
| `leaders`, `attachment`, `fallback_f` | 15, 2, 2; a small tier and a five-member fallback committee. These sizes are absent from the final paper. |
| `replicas`, `minimum_replicas` | 3 and 3. Per-epoch `R` is bounded by eligible peers/free storage and the configured floor; a write never silently degrades durability. Higher maxima permit capacity-dependent factors. |
| `batch_size`, `batch_timeout` | 256 vectors and 100 ms; bound clustering work and allow partially filled batches to proceed. |
| `dbscan_eps`, `dbscan_min_samples`, `noise_clusters` | 0.5, 5, 8; Euclidean float32 input with no undeclared normalization. Tune explicitly for an application. |
| `centroid_sample`, `seed` | 4096 and 42; initial centroids use seeded k-means, then shard sample means update centroids at boundaries. |
| `moves_per_round`, `relocation_tokens` | 2 and 1; bounded oldest-first work and one participation credit per distinct relocated copy. |
| `storage_bytes`, `memory_bytes` | Payload capacity and declared per-node memory share. The paper preset divides 18 × 96 GiB by 500; actual server storage allocation was not specified. Network-size cases preserve the total declared memory budget. |
| Timeouts and intervals | Explicit settings govern RPC retries, failure observations, gossip, monitor, repair, and epoch boundaries. A finite operation timeout returns pending work rather than claiming commit. |

Epoch membership is drawn from provisioned node identities. Preferential attachment constructs their overlay; recovery retains identifiers/counters and rejoins only after learning a later configuration. The final paper gives no runtime admission or bootstrap-discovery protocol for previously unprovisioned identities, so the implementation does not invent one. A failed node's missed epochs contribute zero uptime samples on recovery. Historical configurations and location tombstones are retained; safe garbage collection is not specified in the insert-only paper.

A split reclusters shard samples into two centroids and assigns each existing immutable block to the nearest child by its payload mean. One child keeps the old peer group and the other uses peers with the most free storage; groups can overlap with other shards. Blocks remain indivisible to preserve certified digests. Retired shard aliases route late old-epoch headers, while an explicit block map identifies current child shards. The paper does not specify how a split treats already-certified mixed-cluster blocks or in-flight old headers; these choices preserve immutability and allow repair to transfer capacity safely. Approval requests and certificates carry the epoch; the immutable header's shard records insertion placement.

Exact numerical reproduction of the paper is not determined by the supplied material: clustering/timing settings, initial shard allocation, `R` policy, database versions/deployment details, NOAA extraction, failure injection timing, and resource allocation are incomplete. The failure grid uses a declared 10% ingestion-submission point. The implementation uses modern Python and persistent SQLite for the disk mode; the paper reports Python 3.7 and does not specify persistent-storage serialization. The commands reproduce the described experimental workloads with recorded choices, not guaranteed published rates or recall. Full GIST/18-server, live-database, and hardware-energy measurements must be performed on the intended testbed. No published result is embedded in the code or used as a measurement.

Protocol references: the supplied GraphShardX paper, [Lamport's Paxos description](https://lamport.azurewebsites.net/pubs/paxos-simple.pdf), [MPI for Python](https://mpi4py.readthedocs.io/en/stable/), [Qdrant's API](https://api.qdrant.tech/), [Weaviate's REST API](https://docs.weaviate.io/weaviate/api/rest), and [Pinecone's data API](https://docs.pinecone.io/reference/api/2025-10/data-plane/upsert).
