"""Solids of a session's members, for viewing the model in 3D: one prism per column, beam, wall and slab

Each member is its plan outline extruded between two levels: columns and walls up their story, beams
down from the floor by their depth, and slabs down from the floor by their thickness. write_stl
writes them to an STL file, which GitHub shows as a 3D model.
"""

import struct
from pathlib import Path

import numpy as np
import polars as pl

from steng.geometry.section_line import parse_rectangle


def member_solids(session) -> pl.DataFrame:
    """Return a session's columns, beams, walls and slabs as solids

    Returns:
        kind ("column", "beam", "wall" or "floor"), name, story, pier (walls only), and faces: each face
        a list of [x, y, z] corners [m], counterclockwise seen from outside.
    """

    rows = []
    for kind in ["column", "beam"]:
        for row in getattr(session, kind).iter_rows(named=True):
            depth, width = parse_rectangle(row["section_line"])  # "R(m) <depth t3> <width t2>"
            if kind == "column":
                outline = _rectangle(row["xs"][0], row["ys"][0], depth, width, row["angle"] or 0.0)
                bottom, top = min(row["zs"]), max(row["zs"])
            else:
                outline = _strip(row["xs"], row["ys"], width)
                top = max(row["zs"])
                bottom = top - depth
            rows.append((kind, row[kind], row["story"], None, _prism(outline, bottom, top)))
    for row in session.wall.iter_rows(named=True):
        ends = list(dict.fromkeys(zip(row["xs"], row["ys"])))
        outline = _strip([ends[0][0], ends[-1][0]], [ends[0][1], ends[-1][1]], row["thickness"])
        rows.append(("wall", row["wall"], row["story"], row["pier"], _prism(outline, min(row["zs"]), max(row["zs"]))))
    for row in session.floor.iter_rows(named=True):
        outline = np.array(list(dict.fromkeys(zip(row["xs"], row["ys"]))))
        top = max(row["zs"])
        rows.append(("floor", row["floor"], row["story"], None, _prism(outline, top - row["thickness"], top)))

    return pl.DataFrame(
        rows, orient="row",
        schema={"kind": pl.String, "name": pl.String, "story": pl.String, "pier": pl.String,
                "faces": pl.List(pl.List(pl.List(pl.Float64)))},
    )


def write_stl(solids: pl.DataFrame, path: str | Path) -> None:
    """Write the solids' faces to a binary STL file, in m, each face split into triangles"""

    triangles = [
        (face[0], face[k], face[k + 1])
        for faces in solids["faces"] for face in faces for k in range(1, len(face) - 1)
    ]
    with open(path, "wb") as f:
        f.write(b"steng member solids".ljust(80, b" "))
        f.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            a, b, c = np.array(a), np.array(b), np.array(c)
            normal = np.cross(b - a, c - a)
            normal = normal / (np.linalg.norm(normal) or 1.0)
            f.write(struct.pack("<12fH", *normal, *a, *b, *c, 0))


def volume(faces: list) -> float:
    """Return the volume enclosed by a solid's faces [m3], by the divergence theorem"""

    total = 0.0
    for face in faces:
        face = np.array(face)
        for k in range(1, len(face) - 1):
            total += np.dot(face[0], np.cross(face[k], face[k + 1])) / 6

    return total


def _rectangle(x, y, depth, width, angle):
    """Return the outline of a column centred on (x, y), its depth along its local axis 2 at angle [deg]"""

    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    corners = [(-depth / 2, -width / 2), (depth / 2, -width / 2), (depth / 2, width / 2), (-depth / 2, width / 2)]

    return np.array([(x + c * u - s * v, y + s * u + c * v) for u, v in corners])


def _strip(xs, ys, width):
    """Return the outline of a strip of the given width along the line from the first to the last point"""

    start, end = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    along = (end - start) / np.linalg.norm(end - start)
    across = np.array([-along[1], along[0]]) * width / 2

    return np.array([start - across, end - across, end + across, start + across])


def _prism(outline, bottom, top):
    """Return the faces of an outline extruded from bottom to top, counterclockwise seen from outside"""

    outline = np.asarray(outline, dtype=float)
    # Shoelace: make the outline counterclockwise in plan
    x, y = outline[:, 0], outline[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        outline = outline[::-1]
    lower = [[x, y, bottom] for x, y in outline]
    upper = [[x, y, top] for x, y in outline]
    sides = [
        [lower[k], lower[(k + 1) % len(outline)], upper[(k + 1) % len(outline)], upper[k]]
        for k in range(len(outline))
    ]

    return [lower[::-1], upper, *sides]
