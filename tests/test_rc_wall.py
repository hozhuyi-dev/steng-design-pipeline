"""Designing the example tower's core walls without AdSec, against hand calculations

The forces come from the load model in examples/make_example_results.py. In rev03 the tower has 19
storeys, L1 4.5 m high and the rest 3.6 m. The core walls are 350 mm thick in C40, and a door splits
two of them, leaving six piers. P5 is the solid 8 m wall running in X: it shares the wind in X with
the 2.5 m P1 and the 4.5 m P2 in proportion to length cubed, and carries the core's floor load by length.
"""

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from steng.design_session import DesignSession

EXAMPLES = Path(__file__).parents[1] / "examples"
MODEL = "20250701-091000_tower_rev03_rc-wall_etabs"
UR_LIMIT = 0.9
PERC_VERT = np.array([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])

# Load model, by hand
N_FLOORS = 19
# 10 kPa over the core plus half a bay around it, less the core: 16 x 16 - 8 x 8 = 192 m2,
# shared by 30 m of wall piers, so 64 kN per m
P5_FLOOR_LOAD = 10.0 * (16.0**2 - 8.0**2) / 30.0 * 8.0  # 512 kN per floor
P5_WIND_SHARE = 8.0**3 / (8.0**3 + 4.5**3 + 2.5**3)     # 0.827 of the wind in X
WIND = 1.2 * 24.0                                       # kN/m of height on the 24 m face
# Wind at each floor over half the storey above and below: L1 at 4.5 m, L2-L18 at 3.6 m spacing, L19 at 69.3 m
BASE_SHEAR = WIND * (4.05 + 17 * 3.6 + 1.8)                                       # 1931.0 kN
BASE_MOMENT = WIND * (4.05 * 4.5 + 3.6 * (17 * 4.5 + 3.6 * 153) + 1.8 * 69.3)  # 69,156 kNm

# Material, EN 1992-1-1 3.1.6 and 3.2.7
F_CK, F_CD, F_YD = 40.0, 0.85 * 40.0 / 1.5, 500.0 / 1.15
LENGTH, THICKNESS, L1_HEIGHT = 8000.0, 350.0, 4500.0  # mm


@pytest.fixture(scope="module")
def design():
    return (
        DesignSession()
        .read_models([MODEL], abs_path=str(EXAMPLES))
        .create_rc_wall()
        .design_rc_wall(contain_outputcase="^ULS", ur_limit_pmm=UR_LIMIT, perc_vert=PERC_VERT, use_adsec=False)
    )


def at_base(df, outputcase, reo_vert="H32+0+0+0+0+0-200"):
    """The row for P5 at the bottom of L1, for one combination and trial layout"""
    return df.filter(
        pl.col("pier") == "P5", pl.col("story") == "L1", pl.col("station") == 0.0,
        pl.col("outputcase") == outputcase, pl.col("reo_vert") == reo_vert,
    ).row(0, named=True)


def test_runs_without_adsec(design):
    assert "steng.connectors.adsec" not in sys.modules
    assert design.design_rc_wall_pier["ur_vert"].null_count() == 0


def test_pier_geometry(design):
    piers = design.design_rc_wall_pier.filter(pl.col("story") == "L1").unique("pier").sort("pier")
    assert piers["length"].to_list() == [2500.0, 4500.0, 3000.0, 4000.0, 8000.0, 8000.0]
    p5 = piers.filter(pl.col("pier") == "P5").row(0, named=True)
    assert (p5["thickness_min"], p5["grade_min"], p5["angle"]) == (THICKNESS, 40, 0.0)


def test_design_forces_at_base(design):
    # 1.0G + 1.5W and 1.35G + 1.5W
    for outputcase, gamma_g in [("ULS3X", 1.0), ("ULS2X", 1.35)]:
        row = at_base(design.design_rc_wall_force_pmm, outputcase)
        assert row["p"] == pytest.approx(gamma_g * N_FLOORS * P5_FLOOR_LOAD)  # 9728 kN for 1.0G
        assert row["m3"] == pytest.approx(1.5 * P5_WIND_SHARE * BASE_MOMENT)   # 85,837 kNm


def test_imperfection_and_slenderness(design):
    row = at_base(design.design_rc_wall_force_pmm, "ULS3X")
    # 5.2: theta_i = theta_0 alpha_h with alpha_h = 2 / sqrt(l), e_i = theta_i l_0 / 2, l_0 = l for k = 1
    e_i = (1 / 200) * (2 / math.sqrt(4.5)) * L1_HEIGHT / 2  # 10.6 mm
    assert row["M_Edtotal3"] == pytest.approx(row["m3"] + row["p"] * e_i / 1000)
    # 6.1(4): e_0 = h / 30 >= 20 mm in each direction: 267 mm in plane, 20 mm out of plane
    assert (row["M_min3"], row["M_min2"]) == pytest.approx((row["p"] * LENGTH / 30 / 1000, row["p"] * 20 / 1000))
    # 5.8.3.2: lambda = l_0 / i, i = h / sqrt(12). Stocky in plane, slender out of plane
    assert row["lambda2"] == pytest.approx(L1_HEIGHT / (LENGTH / math.sqrt(12)))     # 1.9
    assert row["lambda3"] == pytest.approx(L1_HEIGHT / (THICKNESS / math.sqrt(12)))  # 44.5
    assert (row["is_slender2"], row["is_slender3"]) == (False, True)


def test_squash_load(design):
    # H32 at 200 mm on both faces: 41 bars a face over 8000 - 2 x (30 cover + 16 link) - 32 = 7876 mm
    bars = 2 * (math.ceil(7876 / 200) + 1)
    n_rd = (F_CD * LENGTH * THICKNESS + bars * math.pi * 32**2 / 4 * F_YD) / 1000  # 92,135 kN
    row = at_base(design.design_rc_wall_force_pmm, "ULS3X")
    assert row["n_rd"] == pytest.approx(n_rd, rel=1e-4)
    # 5.39: alpha from N_Ed / N_Rd, linear between 1.0 at 0.1 and 1.5 at 0.7
    assert row["alpha"] == pytest.approx(1 + 0.5 * (row["p"] / n_rd - 0.1) / 0.6)


def test_in_plane_shear(design):
    row = at_base(design.design_rc_wall_force_v2, "ULS3X")
    v_ed = 1.5 * P5_WIND_SHARE * BASE_SHEAR  # 2397 kN
    assert row["V_Ed"] == pytest.approx(v_ed)
    # 6.2.2(1), with d = 0.9 (8000 - 30 - 16 - 32 / 2) and the library's fixed rho_l of 0.2827%
    d = 0.9 * (LENGTH - 30 - 16 - 16)
    k = 1 + math.sqrt(200 / d)
    sigma_cp = min(row["N_Ed"] * 1000 / (LENGTH * THICKNESS), 0.2 * F_CD)  # 3.47 MPa, under the 0.2 f_cd cap
    v_rdc = (0.18 / 1.5 * k * (100 * 0.002827 * F_CK) ** (1 / 3) + 0.15 * sigma_cp) * THICKNESS * d / 1000
    assert row["V_Rdc"] == pytest.approx(v_rdc)  # 2059 kN
    # 6.8 with z = 0.9 d (6.2.3(1)) at cot(theta) = 2.5, as V_Ed is below V_Rd,max for that angle;
    # 9.6.3(1) then governs: horizontal bars at least a quarter of the vertical, H32 at 200 mm on both faces
    assert row["A_sw_s_req"] == pytest.approx(v_ed * 1000 / (0.9 * d) / F_YD / 2.5, rel=1e-3)  # 0.34 mm2/mm
    assert row["A_sw_s2"] == pytest.approx(0.25 * 2 * math.pi * 32**2 / 4 / 200)        # 2.01 mm2/mm


def test_picks_lightest_trial_that_passes(design):
    trials = design.design_rc_wall_pier
    chosen = design.get_rc_wall(UR_LIMIT)
    lightest = (
        trials.filter(pl.col("ur_vert") <= UR_LIMIT)
        .group_by("pier", "story")
        .agg(pl.col("A_s").min())
    )
    assert chosen.height == 6 * N_FLOORS
    assert chosen.select("pier", "story", "A_s").sort("pier", "story").equals(lightest.sort("pier", "story"))


def test_longer_piers_need_more_steel_at_the_base(design):
    # The wind goes to the stiffest piers: in X, the 8 m P5 needs more than the 4.5 m P2, then the 2.5 m P1
    base = design.get_rc_wall(UR_LIMIT).filter(pl.col("story") == "L1")
    perc = dict(zip(base["pier"], base["perc_vert"]))
    assert perc["P5"] > perc["P2"] > perc["P1"]
    assert perc["P5"] > 3.0
    top = design.get_rc_wall(UR_LIMIT).filter(pl.col("story") == "L19")
    assert top["A_s"].n_unique() == 1  # every pier at the lightest trial
