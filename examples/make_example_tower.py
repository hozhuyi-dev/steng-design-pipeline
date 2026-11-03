"""Write made-up ETABS text files (.$et) of a small concrete tower, for the examples and tests

The tower is 24 m square on an 8 m grid, with a central core of four walls, columns on the other
grid points, beams on the grid lines, and a slab in each bay outside the core. The revisions change
it the way a design develops: rev02 thickens the core, rev03 adds a storey.

    python examples/make_example_tower.py
"""

from pathlib import Path

GRID = [0.0, 8.0, 16.0, 24.0]
CORE = {(1, 1), (2, 1), (2, 2), (1, 2)}  # grid indices of the core corners
REVISIONS = {
    "tower_rev01": {"stories": 5, "wall": 0.40},
    "tower_rev02": {"stories": 5, "wall": 0.50},
    "tower_rev03": {"stories": 6, "wall": 0.50},
}


def point_id(i, j):
    return str(j * len(GRID) + i + 1)


def write_tower(path, stories, wall):
    lines = [f"$ File {path.name} saved 1/6/2025 09:00:00", ""]

    lines += ["$ CONTROLS", '  UNITS  "KN"  "M"  "C"', ""]

    lines += ["$ STORIES - IN SEQUENCE FROM TOP"]
    names = [f"L{k}" for k in range(stories, 0, -1)]
    for k, name in enumerate(names):
        height = 4.5 if name == "L1" else 3.6
        lines.append(f'  STORY "{name}"  HEIGHT {height}' + ("  MASTERSTORY \"Yes\"" if k == 0 else ""))
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
    for j, y in enumerate(GRID):
        for i, x in enumerate(GRID):
            lines.append(f'  POINT "{point_id(i, j)}"  {x:g} {y:g}')
    lines.append("")

    columns = [(i, j) for j in range(4) for i in range(4) if (i, j) not in CORE]
    beams = []
    for j in range(4):
        for i in range(3):
            beams.append(((i, j), (i + 1, j)))
            beams.append(((j, i), (j, i + 1)))
    core_edges = [((1, 1), (2, 1)), ((2, 1), (2, 2)), ((1, 2), (2, 2)), ((1, 1), (1, 2))]
    beams = [b for b in beams if b not in core_edges]
    floors = [(i, j) for j in range(3) for i in range(3) if (i, j) != (1, 1)]

    lines += ["$ LINE CONNECTIVITIES"]
    for k, (i, j) in enumerate(columns, 1):
        lines.append(f'  LINE  "C{k}"  COLUMN  "{point_id(i, j)}"  "{point_id(i, j)}"  1')
    for k, (a, b) in enumerate(beams, 1):
        lines.append(f'  LINE  "B{k}"  BEAM  "{point_id(*a)}"  "{point_id(*b)}"  0')
    lines.append("")

    lines += ["$ AREA CONNECTIVITIES"]
    for k, (a, b) in enumerate(core_edges, 1):
        pa, pb = point_id(*a), point_id(*b)
        lines.append(f'  AREA "W{k}"  PANEL  4 "{pa}" "{pb}" "{pb}" "{pa}"  1 1 0 0')
    for k, (i, j) in enumerate(floors, 1):
        corners = [point_id(i, j), point_id(i + 1, j), point_id(i + 1, j + 1), point_id(i, j + 1)]
        lines.append(f'  AREA "F{k}"  FLOOR  4 ' + " ".join(f'"{c}"' for c in corners) + "  0 0 0 0")
    lines.append("")

    lines += ["$ POINT ASSIGNS"]
    for i, j in columns + sorted(CORE):
        lines.append(f'  POINTASSIGN  "{point_id(i, j)}"  "Base"  RESTRAINT "UX UY UZ RX RY RZ"')
    lines.append("")

    lines += ["$ LINE ASSIGNS"]
    for name in names:
        for k in range(1, len(columns) + 1):
            lines.append(f'  LINEASSIGN  "C{k}"  "{name}"  SECTION "C800"')
        for k in range(1, len(beams) + 1):
            lines.append(f'  LINEASSIGN  "B{k}"  "{name}"  SECTION "B500X800"')
    lines.append("")

    lines += ["$ AREA ASSIGNS"]
    for name in names:
        for k in range(1, len(core_edges) + 1):
            lines.append(f'  AREAASSIGN  "W{k}"  "{name}"  SECTION "W{int(wall * 1000)}"  PIER "P{k}"')
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
