"""Write made-up ETABS result tables for the example tower revisions, as ETABS exports them

Each revision becomes a folder of parquet tables, as Etabs.write_tables names them:
<datetime>_<model>_<purpose>_etabs/<table>.parquet, with every column a string, as ETABS returns it.
The numbers come from a simple model, not an analysis: wind of 1.2 kPa on the 24 m face, about
10 kPa of gravity load per floor, and a cantilever sway shape scaled per revision.

    python examples/make_example_results.py
"""

from pathlib import Path

import polars as pl

WIDTH = 24.0          # m
WIND = 1.2            # kPa
FLOOR_LOAD = 10.0     # kPa, dead and superimposed dead
G = 9.81
REVISIONS = {
    # model: (saved, stories, top sway in WX and WY [m], periods of the first three modes [s])
    "tower_rev01": ("20250506-101500", 5, (0.0120, 0.0135), (0.62, 0.58, 0.41)),
    "tower_rev02": ("20250603-143000", 5, (0.0092, 0.0104), (0.54, 0.51, 0.36)),
    "tower_rev03": ("20250701-091000", 6, (0.0131, 0.0148), (0.66, 0.63, 0.44)),
}


def stories(n):
    """Stories from the top, with heights and elevations"""
    names = [f"L{k}" for k in range(n, 0, -1)]
    heights = [4.5 if name == "L1" else 3.6 for name in names]
    elevs = [sum(heights[k:]) for k in range(n)]
    return names, heights, elevs


def as_strings(rows):
    return pl.DataFrame(rows).with_columns(pl.all().cast(pl.String))


def tables(n, sway, periods):
    names, heights, elevs = stories(n)
    total_height = elevs[0]
    floor_weight = FLOOR_LOAD * WIDTH * WIDTH  # kN
    # Wind on the face tributary to each level: half the story above and half below
    tributary = [((heights[k - 1] if k > 0 else 0) + heights[k]) / 2 for k in range(n)]
    wind = [WIND * WIDTH * t for t in tributary]

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


if __name__ == "__main__":
    for model, (saved, n, sway, periods) in REVISIONS.items():
        folder = Path(__file__).parent / "etabs" / f"{saved}_{model}_log_etabs"
        folder.mkdir(parents=True, exist_ok=True)
        for table, df in tables(n, sway, periods).items():
            df.write_parquet(folder / f"{table}.parquet")
        print("wrote", folder)
