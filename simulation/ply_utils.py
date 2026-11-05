"""PLY point cloud loading and voxel downsampling."""

import struct
from typing import List, Tuple

import numpy as np

PLY_SCALAR_FMT = {
    "char": "b", "uchar": "B", "int8": "b", "uint8": "B",
    "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
    "int": "i", "uint": "I", "int32": "i", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def load_ply_xyz(ply_path: str) -> np.ndarray:
    """Load XYZ vertices from ASCII or binary_little_endian PLY.

    Returns:
        pts: ndarray [N, 3] float32
    """
    with open(ply_path, "rb") as f:
        format_name = None
        vertex_count = None
        properties: List[Tuple[str, str]] = []
        in_vertex = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (missing end_header): {ply_path}")
            s = line.decode("ascii", errors="ignore").strip()
            if s.startswith("format "):
                format_name = s.split()[1]
            elif s.startswith("element "):
                toks = s.split()
                in_vertex = (len(toks) >= 3 and toks[1] == "vertex")
                if in_vertex:
                    vertex_count = int(toks[2])
                    properties = []
            elif in_vertex and s.startswith("property "):
                toks = s.split()
                if len(toks) == 3:
                    properties.append((toks[1], toks[2]))
            elif s == "end_header":
                break

        if format_name not in ("ascii", "binary_little_endian"):
            raise ValueError(f"Unsupported PLY format: {format_name}")
        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"No vertex element in PLY: {ply_path}")

        prop_names = [n for _, n in properties]
        if not all(k in prop_names for k in ("x", "y", "z")):
            raise ValueError(f"PLY vertex has no x/y/z fields: {ply_path}")
        ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")

        if format_name == "ascii":
            pts = np.zeros((vertex_count, 3), dtype=np.float32)
            for i in range(vertex_count):
                parts = f.readline().decode("ascii", errors="ignore").strip().split()
                if len(parts) < len(properties):
                    raise ValueError(f"Bad ASCII PLY vertex line #{i} in {ply_path}")
                pts[i, 0] = float(parts[ix])
                pts[i, 1] = float(parts[iy])
                pts[i, 2] = float(parts[iz])
            return pts

        # binary_little_endian
        if any(ptype not in PLY_SCALAR_FMT for ptype, _ in properties):
            unknown = [ptype for ptype, _ in properties if ptype not in PLY_SCALAR_FMT]
            raise ValueError(f"Unsupported binary PLY property types: {unknown}")
        row_fmt = "<" + "".join(PLY_SCALAR_FMT[ptype] for ptype, _ in properties)
        row_size = struct.calcsize(row_fmt)
        unpack_row = struct.Struct(row_fmt).unpack_from

        pts = np.zeros((vertex_count, 3), dtype=np.float32)
        buf = f.read(vertex_count * row_size)
        if len(buf) < vertex_count * row_size:
            raise ValueError(f"Truncated binary PLY data: {ply_path}")
        off = 0
        for i in range(vertex_count):
            row = unpack_row(buf, off)
            off += row_size
            pts[i, 0] = float(row[ix])
            pts[i, 1] = float(row[iy])
            pts[i, 2] = float(row[iz])
        return pts


def voxel_downsample(pts: np.ndarray, voxel_size: float = 0.3) -> np.ndarray:
    """Keep one point per voxel (hash-based). Returns [M, 3] float32."""
    if pts.shape[0] == 0:
        return pts
    keys = (pts / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]
