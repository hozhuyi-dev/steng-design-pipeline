"""Write made-up ETABS result tables for the example tower revisions, as ETABS exports them

Each revision becomes folders of parquet tables, as Etabs.write_tables names them:
<datetime>_<model>_<purpose>_etabs/<table>.parquet, with every column a string, as ETABS returns it.

- "log": story and global results, for tracking the tower across revisions
- "rc-wall": core wall geometry and pier forces, for designing the core walls
- "rc-column": column geometry and forces, for designing the columns

The numbers come from a simple load model, not an analysis:

- Wind of 1.2 kPa on the 24 m face, applied at each floor over half the storey above and below.
- 10 kPa of dead and superimposed dead load on each floor. Live load is left out for brevity.
- Each column and the core carry the floor within half a bay of them. The core's share is split
  between its wall piers by length.
- The core resists all the wind, in the plane of the piers parallel to it. The lintels over the
  doors are left out, so each pier acts as a cantilever and takes a share of the wind in proportion
  to its bending stiffness, length cubed. The columns carry gravity only.
- A cantilever sway shape scaled per revision, and made-up modal periods.

    python examples/make_example_results.py
"""

from pathlib import Path

import polars as pl
from make_example_tower import (
    COLUMNS,
    GRID,
    PIERS,
    REVISIONS,
    pier_geometry,
    point_id,
    points,
    story_height,
    story_names,
)

WIDTH = GRID[-1] - GRID[0]  # m, both directions
WIND = 1.2                  # kPa
FLOOR_LOAD = 10.0           # kPa
G = 9.81
RESULTS = {
    # model: (top sway in WX and WY [m], periods of the first three modes [s])
    "tower_rev01": ((0.085, 0.092), (1.92, 1.81, 1.25)),
    "tower_rev02": ((0.066, 0.072), (1.64, 1.55, 1.08)),
    "tower_rev03": ((0.074, 0.080), (1.75, 1.66, 1.15)),
}
# Core tributary area: the core plus half the bay on each side, less the core itself, which has no floor
CORE_SIDE = GRID[2] - GRID[1]
CORE_TRIBUTARY = (CORE_SIDE + (GRID[1] - GRID[0]) / 2 + (GRID[3] - GRID[2]) / 2) ** 2 - CORE_SIDE**2  # m2
WALL_LENGTH = sum(pier_geometry(pier)[0] for pier in PIERS)  # m, total
WALL_FLOOR_LOAD = FLOOR_LOAD * CORE_TRIBUTARY / WALL_LENGTH    # kN per m of wall per floor
COLUMN_SECTION = ("C800", 0.8, 0.8)  # name, t3, t2 [m], as in the .$et files
# Ultimate combinations to EN 1990 6.10, without live load: (name, factor on dead, factor on wind, wind case)
COMBINATIONS = [
    ("ULS1", 1.35, 0.0, None),
    ("ULS2X", 1.35, 1.5, "WX"),
    ("ULS2Y", 1.35, 1.5, "WY"),
    ("ULS3X", 1.0, 1.5, "WX"),
    ("ULS3Y", 1.0, 1.5, "WY"),
]


def stories(n):
    """Stories from the top, with heights and elevations"""
    names = story_names(n)
    heights = [story_height(name) for name in names]
    elevs = [sum(heights[k:]) for k in range(n)]
    return names, heights, elevs


def wind_per_floor(heights):
    """Wind force at each floor from the top [kN]: half the storey above and half the storey below"""
    tributary = [((heights[k - 1] if k > 0 else 0) + heights[k]) / 2 for k in range(len(heights))]
    return [WIND * WIDTH * t for t in tributary]


def half_bays(k):
    """Width of floor a grid line carries: half the bay on each side [m]"""
    return ((GRID[k] - GRID[k - 1]) / 2 if k > 0 else 0) + ((GRID[k + 1] - GRID[k]) / 2 if k < len(GRID) - 1 else 0)


def as_strings(rows):
    return pl.DataFrame(rows).with_columns(pl.all().cast(pl.String))


def log_tables(n, sway, periods):
    names, heights, elevs = stories(n)
    total_height = elevs[0]
    floor_weight = FLOOR_LOAD * WIDTH * WIDTH  # kN
    wind = wind_per_floor(heights)

    out = {}
    out["tower_and_base_story_definitions"] = as_strings({"tower": ["T1"], "bsname": ["Base"], "bselev": [0.0]})
    out["story_definitions"] = as_strings({"tower": ["T1"] * n, "story": names, "height": heights})
    out["mass_summary_by_story"] = as_strings({
        "story": names,
        "ux": [floor_weight / G] * n,
        "uy": [floor_weight / G] * n,
        "uz": [0.0] * n,
    })

    rows = {k: [] for k in ["story", "outputcase", "casetype", "location", "p", "vx", "vy", "t", "mx", "my"]}
    disp = {k: [] for k in ["story", "outputcase", "casetype", "direction", "maximum", "average", "ratio"]}
    drift = {k: [] for k in ["story", "outputcase", "casetype", "direction", "max_drift", "avg_drift", "ratio"]}
    for case in ["DL", "WX", "WY"]:
        shear = moment = weight = 0.0
        for k, name in enumerate(names):
            weight += floor_weight
            if case != "DL":
                shear += wind[k]
            for location in ["Top", "Bottom"]:
                if location == "Bottom":
                    moment += shear * heights[k]
                rows["story"].append(name)
                rows["outputcase"].append(case)
                rows["casetype"].append("LinStatic")
                rows["location"].append(location)
                rows["p"].append(-weight if case == "DL" else 0.0)
                rows["vx"].append(-shear if case == "WX" else 0.0)
                rows["vy"].append(-shear if case == "WY" else 0.0)
                rows["t"].append(0.0)
                rows["mx"].append(moment if case == "WY" else 0.0)
                rows["my"].append(-moment if case == "WX" else 0.0)
            if case == "DL":
                continue
            # Cantilever sway shape, the larger direction from torsion is 10% above the average
            top = sway[0] if case == "WX" else sway[1]
            u = top * (elevs[k] / total_height) ** 1.6
            u_below = top * ((elevs[k] - heights[k]) / total_height) ** 1.6
            direction = "X" if case == "WX" else "Y"
            for table, values in [(disp, (u * 1.1, u)), (drift, ((u - u_below) * 1.1, u - u_below))]:
                table["story"].append(name)
                table["outputcase"].append(case)
                table["casetype"].append("LinStatic")
                table["direction"].append(direction)
                table[list(table)[4]].append(values[0])
                table[list(table)[5]].append(values[1])
                table["ratio"].append(1.1)
    out["story_forces"] = as_strings(rows)
    out["story_max_over_avg_displacements"] = as_strings(disp)
    out["story_max_over_avg_drifts"] = as_strings(drift)

    base_shear = sum(wind)
    base_moment = sum(w * e for w, e in zip(wind, elevs))
    out["base_reactions"] = as_strings({
        "outputcase": ["DL", "WX", "WY"],
        "casetype": ["LinStatic"] * 3,
        "fx": [0.0, -base_shear, 0.0],
        "fy": [0.0, 0.0, -base_shear],
        "fz": [floor_weight * n, 0.0, 0.0],
        "mx": [0.0, 0.0, base_moment],
        "my": [0.0, -base_moment, 0.0],
        "mz": [0.0, 0.0, 0.0],
    })
    out["modal_participating_mass_ratios"] = as_strings({
        "case": ["Modal"] * 3,
        "mode": [1, 2, 3],
        "period": list(periods),
        "ux": [0.72, 0.0, 0.0],
        "uy": [0.0, 0.70, 0.0],
        "uz": [0.0] * 3,
        "rx": [0.0, 0.0, 0.0],
        "ry": [0.0, 0.0, 0.0],
        "rz": [0.0, 0.0, 0.74],
    })
    return out


def story_and_point_tables(n):
    """The story and point tables every purpose exports, and the point unique names by label and level"""
    names, heights, elevs = stories(n)
    out = {}
    out["tower_and_base_story_definitions"] = as_strings({"tower": ["T1"], "bsname": ["Base"], "bselev": [0.0]})
    out["story_definitions"] = as_strings({"tower": ["T1"] * n, "story": names, "height": heights})

    # One point object per point and level, numbered in order
    rows = {k: [] for k in ["uniquename", "story", "pointbay", "x", "y", "z"]}
    unique_point = {}
    for level, z in zip(names + ["Base"], elevs + [0.0]):
        for label, (x, y) in points().items():
            unique_point[(label, level)] = len(unique_point) + 1
            for key, value in zip(rows, [unique_point[(label, level)], level, label, x, y, z]):
                rows[key].append(value)
    out["point_object_connectivity"] = as_strings(rows)
    return out, unique_point


def rc_wall_tables(n, wall):
    """Core wall geometry and pier forces, as the rc-wall purpose exports them"""
    names, heights, _ = stories(n)
    levels = names + ["Base"]
    section = f"W{round(wall * 1000)}"

    out, unique_point = story_and_point_tables(n)
    out["wall_property_definitions_-_specified"] = as_strings({"name": [section], "material": ["C40"], "thickness": [wall]})

    walls = {k: [] for k in ["uniquename", "story", "wallbay", "uniquept1", "uniquept2", "uniquept3", "uniquept4",
                             "perimeter", "area"]}
    sections = {k: [] for k in ["story", "label", "uniquename", "sectprop"]}
    piers = {k: [] for k in ["story", "label", "uniquename", "piername"]}
    pier_properties = {k: [] for k in ["story", "pier", "axisangle"]}
    for level, below, height in zip(names, levels[1:], heights):
        for k, (pier, (a, b)) in enumerate(PIERS.items(), 1):
            unique_area = len(walls["uniquename"]) + 1
            length, direction = pier_geometry(pier)
            corners = [unique_point[(a, level)], unique_point[(b, level)], unique_point[(b, below)], unique_point[(a, below)]]
            for key, value in zip(walls, [unique_area, level, f"W{k}", *corners, 2 * (length + height), length * height]):
                walls[key].append(value)
            for table, value in [(sections, section), (piers, pier)]:
                for key, item in zip(table, [level, f"W{k}", unique_area, value]):
                    table[key].append(item)
            for key, value in zip(pier_properties, [level, pier, 0.0 if direction == "X" else 90.0]):
                pier_properties[key].append(value)
    out["wall_object_connectivity"] = as_strings(walls)
    out["area_assignments_-_section_properties"] = as_strings(sections)
    out["area_assignments_-_pier_labels"] = as_strings(piers)
    out["pier_section_properties"] = as_strings(pier_properties)
    out["pier_forces"] = as_strings(pier_forces(names, heights))
    return out


def pier_forces(names, heights):
    """Pier forces at the top and bottom of each storey, for each ultimate combination

    P is negative in compression. V2 and M3 act in the plane of the wall, V3 and M2 out of it.
    """
    wind = wind_per_floor(heights)
    geometry = {pier: pier_geometry(pier) for pier in PIERS}
    stiffness = {direction: sum(length**3 for length, d in geometry.values() if d == direction) for direction in "XY"}
    rows = {k: [] for k in ["pier", "story", "outputcase", "casetype", "location", "p", "v2", "v3", "t", "m2", "m3"]}
    for pier, (length, direction) in geometry.items():
        share = length**3 / stiffness[direction]
        for combination, gamma_g, gamma_w, case in COMBINATIONS:
            gamma_w = gamma_w if case == f"W{direction}" else 0.0
            shear = moment = weight = 0.0
            for story, height, w in zip(names, heights, wind):
                weight += WALL_FLOOR_LOAD * length
                shear += w * share
                for location in ["Top", "Bottom"]:
                    if location == "Bottom":
                        moment += shear * height
                    for key, value in zip(rows, [
                        pier, story, combination, "Combination", location,
                        -gamma_g * weight, gamma_w * shear, 0.0, 0.0, 0.0, gamma_w * moment,
                    ]):
                        rows[key].append(value)
    return rows


def rc_column_tables(n):
    """Column geometry and forces, as the rc-column purpose exports them

    Each column carries the floor within half a bay of it, so its axial force grows down the building
    by the same amount at each floor. With the wind on the core, the columns have no shear or moment.
    """
    names, heights, _ = stories(n)
    levels = names + ["Base"]
    section, t3, t2 = COLUMN_SECTION

    out, unique_point = story_and_point_tables(n)
    out["frame_section_property_definitions_-_concrete_rectangular"] = as_strings(
        {"name": [section], "material": ["C40"], "t3": [t3], "t2": [t2]}
    )

    columns = {k: [] for k in ["uniquename", "story", "columnbay", "uniquepti", "uniqueptj", "length"]}
    sections = {k: [] for k in ["story", "label", "uniquename", "shape", "sectprop"]}
    axes = {k: [] for k in ["story", "label", "uniquename", "angle"]}
    forces = {k: [] for k in ["uniquename", "story", "outputcase", "casetype", "station", "p", "v2", "v3", "t", "m2", "m3"]}
    for k, (i, j) in enumerate(COLUMNS, 1):
        floor_load = FLOOR_LOAD * half_bays(i) * half_bays(j)  # kN per floor
        weight = 0.0
        for level, below, height in zip(names, levels[1:], heights):
            weight += floor_load
            unique_line = len(columns["uniquename"]) + 1
            label = f"C{k}"
            for key, value in zip(columns, [
                unique_line, level, label,
                unique_point[(point_id(i, j), below)], unique_point[(point_id(i, j), level)], height,
            ]):
                columns[key].append(value)
            for key, value in zip(sections, [level, label, unique_line, "Concrete Rectangular", section]):
                sections[key].append(value)
            for key, value in zip(axes, [level, label, unique_line, 0.0]):
                axes[key].append(value)
            # Stations from the bottom of the column, as ETABS reports element forces
            for combination, gamma_g, _, _ in COMBINATIONS:
                for station in [0.0, height]:
                    for key, value in zip(forces, [
                        unique_line, level, combination, "Combination", station,
                        -gamma_g * weight, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ]):
                        forces[key].append(value)
    out["column_object_connectivity"] = as_strings(columns)
    out["frame_assignments_-_section_properties"] = as_strings(sections)
    out["frame_assignments_-_local_axes"] = as_strings(axes)
    out["element_forces_-_columns"] = as_strings(forces)
    return out


if __name__ == "__main__":
    for model, (sway, periods) in RESULTS.items():
        saved, n, wall = REVISIONS[model]["saved"], REVISIONS[model]["stories"], REVISIONS[model]["wall"]
        for purpose, tables in [
            ("log", log_tables(n, sway, periods)),
            ("rc-wall", rc_wall_tables(n, wall)),
            ("rc-column", rc_column_tables(n)),
        ]:
            folder = Path(__file__).parent / "etabs" / f"{saved}_{model}_{purpose}_etabs"
            folder.mkdir(parents=True, exist_ok=True)
            for table, df in tables.items():
                df.write_parquet(folder / f"{table}.parquet")
            print("wrote", folder)
