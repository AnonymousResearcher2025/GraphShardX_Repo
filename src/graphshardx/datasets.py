from pathlib import Path
import struct

import numpy as np


class Dataset:
    """Memory-map TEXMEX or NumPy input; HDF5 reads only the requested slices."""

    def __init__(self, path: Path, key="train"):
        self.path = Path(path)
        self.file = None
        suffix = self.path.suffix.lower()
        if suffix == ".npy":
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        elif suffix in {".h5", ".hdf5"}:
            import h5py

            self.file = h5py.File(self.path, "r")
            if key not in self.file:
                raise ValueError(f"dataset key {key} is absent")
            self.array = self.file[key]
        elif suffix in {".fvecs", ".ivecs"}:
            with self.path.open("rb") as stream:
                raw = stream.read(4)
            if len(raw) != 4:
                raise ValueError("empty TEXMEX dataset")
            dimension = struct.unpack("<i", raw)[0]
            if dimension <= 0 or self.path.stat().st_size % (4 * (dimension + 1)):
                raise ValueError("invalid TEXMEX record size")
            dtype = "<f4" if suffix == ".fvecs" else "<i4"
            mapped = np.memmap(self.path, dtype=dtype, mode="r").reshape(-1, dimension + 1)
            if not np.all(mapped[:, 0].view("<i4") == dimension):
                raise ValueError("inconsistent TEXMEX dimensions")
            self.array = mapped[:, 1:]
        else:
            raise ValueError("dataset must be .npy, .fvecs, .ivecs, .h5 or .hdf5")
        if len(self.array.shape) != 2 or min(self.array.shape) < 1:
            raise ValueError("dataset must be a nonempty numeric matrix")
        self.shape = self.array.shape

    def slice(self, start, stop, dimension=None):
        if not 0 <= start <= stop <= self.shape[0]:
            raise ValueError("dataset slice exceeds available vectors")
        d = dimension or self.shape[1]
        if not 1 <= d <= self.shape[1]:
            raise ValueError("requested dimension exceeds dataset")
        values = np.asarray(self.array[start:stop, :d])
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("dataset contains invalid coordinates")
        return values

    def sample(self, count, seed, limit=None, dimension=None):
        n = min(limit or self.shape[0], self.shape[0])
        indices = np.sort(np.random.default_rng(seed).choice(n, size=min(count, n), replace=False))
        return np.asarray(self.array[indices, : dimension or self.shape[1]], dtype=np.float32)

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def geospatial(input_path, output_path, columns):
    import geopandas as gpd

    frame = gpd.read_file(input_path)
    if frame.empty or not columns:
        raise ValueError("nonempty geospatial data and explicit numeric columns are required")
    if frame.crs is None:
        raise ValueError("geospatial input needs a coordinate reference system")
    # Geometry coordinates in a declared CRS and explicit climate attributes form the vector.
    geographic = frame.to_crs(4326)
    points = geographic.geometry.representative_point()
    values = np.column_stack(
        [points.x, points.y, *[frame[c].to_numpy(dtype=float) for c in columns]]
    )
    if not np.isfinite(values).all():
        raise ValueError("missing/nonfinite geospatial features; clean the dataset explicitly")
    np.save(output_path, values.astype(np.float32), allow_pickle=False)
