"""Write made-up ETABS text files (.$et) of a small concrete tower, for the examples and tests

The tower is 24 m square on an 8 m grid, with a central 8 m core, columns on the other grid points,
beams on the grid lines, and a slab in each bay outside the core. Doors split two of the core walls,
so the core has six wall piers of four lengths. The revisions change it the way a design develops:
rev02 thickens the core, rev03 adds a storey.

    python examples/make_example_tower.py
"""

import datetime
from pathlib import Path

GRID = [0.0, 8.0, 16.0, 24.0]
CORE = {(1, 1), (2, 1), (2, 2), (1, 2)}  # grid indices of the core corners
# Columns C1-C12 on the other grid points, row by row from y = 0
COLUMNS = [(i, j) for j in range(len(GRID)) for i in range(len(GRID)) if (i, j) not in CORE]
# Grid points are numbered row by row from y = 0; the door jambs follow
DOOR_POINTS = {"17": (10.5, 8.0), "18": (11.5, 8.0), "19": (16.0, 11.0), "20": (16.0, 12.0)}
# Core wall piers, one wall panel each, between two points. A door splits the wall at y = 8 into P1
# and P2, and the wall at x = 16 into P3 and P4; P5 and P6 are solid
PIERS = {
    "P1": ("6", "17"),   # 2.5 m in X
    "P2": ("18", "7"),   # 4.5 m in X
    "P3": ("7", "19"),   # 3.0 m in Y
    "P4": ("20", "11"),  # 4.0 m in Y
    "P5": ("10", "11"),  # 8.0 m in X
    "P6": ("6", "10"),   # 8.0 m in Y
}
REVISIONS = {
    # model: when it was saved, stories, wall thickness [m]
    "tower_rev01": {"saved": "20250506-101500", "stories": 18, "wall": 0.30},
    "tower_rev02": {"saved": "20250603-143000", "stories": 18, "wall": 0.35},
    "tower_rev03": {"saved": "20250701-091000", "stories": 19, "wall": 0.35},
}


def story_names(stories):
    """Story names from the top: the first storey is 4.5 m high, the others 3.6 m"""
    return [f"L{k}" for k in range(stories, 0, -1)]


def story_height(name):
    return 4.5 if name == "L1" else 3.6


def point_id(i, j):
    return str(j * len(GRID) + i + 1)


def points():
    """Every point's coordinates by label: the grid points, then the door jambs"""
    grid = {point_id(i, j): (x, y) for j, y in enumerate(GRID) for i, x in enumerate(GRID)}
    return grid | DOOR_POINTS


def pier_geometry(pier):
    """A pier's length [m] and direction, X or Y"""
    (x1, y1), (x2, y2) = (points()[label] for label in PIERS[pier])
    return abs(x2 - x1) + abs(y2 - y1), "X" if y1 == y2 else "Y"


def write_tower(path, saved, stories, wall):
    # As ETABS dates the file: month first, 12-hour clock
    saved = datetime.datetime.strptime(saved, "%Y%m%d-%H%M%S")
    lines = [f"$ File {path.name} saved {saved.month}/{saved.day}/{saved.year} {saved:%I:%M:%S %p}", ""]

    lines += ["$ CONTROLS", '  UNITS  "KN"  "M"  "C"', ""]

    lines += ["$ STORIES - IN SEQUENCE FROM TOP"]
    names = story_names(stories)
    for k, name in enumerate(names):
        lines.append(f'  STORY "{name}"  HEIGHT {story_height(name)}' + ("  MASTERSTORY \"Yes\"" if k == 0 else ""))
    lines += ['  STORY "Base"  ELEV 0', ""]

    lines += [
        "$ MATERIAL PROPERTIES",
        '  MATERIAL  "C40"  TYPE "Concrete"  WEIGHTPERVOLUME 25',
        '  MATERIAL  "C40"  FC 40000',
        '  MATERIAL  "B500"  TYPE "Rebar"',
        '  MATERIAL  "B500"  FY 500000',
        "",
        "$ FRAME SECTIONS",
        '  FRAMESECTION  "C800"  MATERIAL "C40"  SHAPE "Concrete Rectangular"  D 0.8  B 0.8',
        '  FRAMESECTION  "B500X800"  MATERIAL "C40"  SHAPE "Concrete Rectangular"  D 0.8  B 0.5',
        "",
        "$ SLAB PROPERTIES",
        '  SHELLPROP  "S250"  PROPTYPE "Slab"  MATERIAL "C40"  SLABTHICKNESS 0.25',
        "",
        "$ WALL PROPERTIES",
        f'  SHELLPROP  "W{int(wall * 1000)}"  PROPTYPE "Wall"  MATERIAL "C40"  WALLTHICKNESS {wall}',
        "",
    ]

    lines += ["$ POINT COORDINATES"]
    for label, (x, y) in points().items():
        lines.append(f'  POINT "{label}"  {x:g} {y:g}')
    lines.append("")

    core_edges = [((1, 1), (2, 1)), ((2, 1), (2, 2)), ((1, 2), (2, 2)), ((1, 1), (1, 2))]
    beams = []
    for j in range(4):
        for i in range(3):
            beams.append(((i, j), (i + 1, j)))
            beams.append(((j, i), (j, i + 1)))
    beams = [b for b in beams if b not in core_edges]
    floors = [(i, j) for j in range(3) for i in range(3) if (i, j) != (1, 1)]

    lines += ["$ LINE CONNECTIVITIES"]
    for k, (i, j) in enumerate(COLUMNS, 1):
        lines.append(f'  LINE  "C{k}"  COLUMN  "{point_id(i, j)}"  "{point_id(i, j)}"  1')
    for k, (a, b) in enumerate(beams, 1):
        lines.append(f'  LINE  "B{k}"  BEAM  "{point_id(*a)}"  "{point_id(*b)}"  0')
    lines.append("")

    lines += ["$ AREA CONNECTIVITIES"]
    for k, (pa, pb) in enumerate(PIERS.values(), 1):
        lines.append(f'  AREA "W{k}"  PANEL  4 "{pa}" "{pb}" "{pb}" "{pa}"  1 1 0 0')
    for k, (i, j) in enumerate(floors, 1):
        corners = [point_id(i, j), point_id(i + 1, j), point_id(i + 1, j + 1), point_id(i, j + 1)]
        lines.append(f'  AREA "F{k}"  FLOOR  4 ' + " ".join(f'"{c}"' for c in corners) + "  0 0 0 0")
    lines.append("")

    lines += ["$ POINT ASSIGNS"]
    for label in [point_id(i, j) for i, j in COLUMNS + sorted(CORE)] + list(DOOR_POINTS):
        lines.append(f'  POINTASSIGN  "{label}"  "Base"  RESTRAINT "UX UY UZ RX RY RZ"')
    lines.append("")

    lines += ["$ LINE ASSIGNS"]
    for name in names:
        for k in range(1, len(COLUMNS) + 1):
            lines.append(f'  LINEASSIGN  "C{k}"  "{name}"  SECTION "C800"')
        for k in range(1, len(beams) + 1):
            lines.append(f'  LINEASSIGN  "B{k}"  "{name}"  SECTION "B500X800"')
    lines.append("")

    lines += ["$ AREA ASSIGNS"]
    for name in names:
        for k, pier in enumerate(PIERS, 1):
            lines.append(f'  AREAASSIGN  "W{k}"  "{name}"  SECTION "W{int(wall * 1000)}"  PIER "{pier}"')
        for k in range(1, len(floors) + 1):
            lines.append(f'  AREAASSIGN  "F{k}"  "{name}"  SECTION "S250"')
    lines += ["", "  END", "$ END OF MODEL FILE", ""]

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    folder = Path(__file__).parent / "etabs"
    folder.mkdir(exist_ok=True)
    for name, revision in REVISIONS.items():
        write_tower(folder / f"{name}.$et", **revision)
        print("wrote", folder / f"{name}.$et")
