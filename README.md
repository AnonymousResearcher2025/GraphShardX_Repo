# GraphShardX

GraphShardX coordinates immutable vector inserts across a provisioned deployment of edge nodes. It places similar vectors together, certifies block headers without ordering inserts, replicates validated payloads, and makes stored-data changes detectable. It targets crash faults, message loss, and partitions that eventually heal. Updates, deletes, Byzantine participants, and ANN indexes are outside the paper's scope.

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
