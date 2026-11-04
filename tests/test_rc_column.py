"""Designing the example tower's columns without AdSec, against hand calculations

The forces come from the load model in examples/make_example_results.py. In rev03 the tower has 19
storeys, L1 4.5 m high. The columns are 800 mm square in C40 and carry gravity only: the floor within
half a bay of them, at 10 kPa. C1 is the corner column at (0, 0); C2 is the edge column at (8, 0).
"""

import math
import sys
from pathlib import Path

import polars as pl
import pytest

from steng.design_session import DesignSession

EXAMPLES = Path(__file__).parents[1] / "examples"
MODEL = "20250701-091000_tower_rev03_rc-column_etabs"
UR_LIMIT = 0.9
N_FLOORS = 19
F_CD, F_YD = 0.85 * 40.0 / 1.5, 500.0 / 1.15
SIDE, L1_HEIGHT = 800.0, 4500.0  # mm
# Tributary floor: half of each 8 m bay along x, half the 8 m bay along y
C2_FLOOR_LOAD = 10.0 * (4.0 + 4.0) * 4.0  # 320 kN per floor


@pytest.fixture(scope="module")
def design():
    return (
        DesignSession()
        .read_models([MODEL], abs_path=str(EXAMPLES))
        .create_rc_column()
        .design_rc_column(contain_outputcase="^ULS", ur_limit_pmm=UR_LIMIT, use_adsec=False)
    )


def c2_at_base(design, reo_vert="20H20+0+0"):
    """The row for C2 at the bottom of L1, under 1.35G, for one trial layout"""
    return design.design_rc_column_force_pmm.filter(
        pl.col("column") == "C2", pl.col("story") == "L1", pl.col("station") == 0.0,
        pl.col("outputcase") == "ULS1", pl.col("reo_vert") == reo_vert,
    ).row(0, named=True)


def test_runs_without_adsec(design):
    assert "steng.connectors.adsec" not in sys.modules
    assert design.design_rc_column_column["ur_vert"].null_count() == 0


def test_section_and_axial_force(design):
    row = c2_at_base(design)
    assert (row["b"], row["h"], row["l"]) == (SIDE, SIDE, L1_HEIGHT)
    assert row["p"] == pytest.approx(1.35 * N_FLOORS * C2_FLOOR_LOAD)  # 8208 kN


def test_squash_load(design):
    # 20 H20 bars: N_Rd = f_cd A_c + f_yd A_s
    n_rd = (F_CD * SIDE**2 + 20 * math.pi * 20**2 / 4 * F_YD) / 1000  # 17,238 kN
    assert c2_at_base(design)["n_rd"] == pytest.approx(n_rd, rel=1e-4)


def test_imperfection_and_slenderness(design):
    row = c2_at_base(design)
    # 5.2: theta_i = (1/200)(2 / sqrt(l)), e_i = theta_i l_0 / 2; no first-order moment, so M_i = N e_i
    e_i = (1 / 200) * (2 / math.sqrt(4.5)) * L1_HEIGHT / 2  # 10.6 mm
    assert (row["M_i2"], row["M_i3"]) == pytest.approx((row["p"] * e_i / 1000,) * 2)  # 91.8 kNm
    # 5.8.3.2: lambda = l_0 / i, i = h / sqrt(12)
    assert row["lambda2"] == pytest.approx(L1_HEIGHT / (SIDE / math.sqrt(12)))  # 19.5


def test_minimum_eccentricity(design):
    # 6.1(4): e_0 = h / 30 >= 20 mm, here 26.7 mm, which governs over imperfection and second order
    row = c2_at_base(design)
    assert (row["e_02"], row["e_03"]) == pytest.approx((SIDE / 30,) * 2)
    assert row["M_Edtotal2"] == pytest.approx(row["p"] * SIDE / 30 / 1000)  # 219 kNm


def test_slenderness_counts_the_column_bars(design):
    # 5.8.3.1: omega = A_s f_yd / (A_c f_cd) with all 20 bars
    row = c2_at_base(design)
    A_s = 20 * math.pi * 20**2 / 4  # 6283 mm2
    assert row["A_s"] == pytest.approx(A_s)
    assert row["omega"] == pytest.approx(A_s * F_YD / (SIDE**2 * F_CD))


def test_utilisation_uses_design_moments(design):
    # 5.39 on the design moments, which include imperfection, second order and the minimum
    row = c2_at_base(design)
    ur_m = row["M_Edtotal2"] / row["m2_rd"]  # 219 / 1907 = 0.115
    alpha = 1 + 0.5 * (row["p"] / row["n_rd"] - 0.1) / 0.6  # 1.313 at N / N_Rd = 0.48
    assert (row["ur_m2"], row["ur_m3"]) == pytest.approx((ur_m, ur_m))
    assert row["ur_biaxial"] == pytest.approx(2 * ur_m**alpha)  # 0.12, square column, equal moments


def test_picks_lightest_trial_that_passes(design):
    trials = design.design_rc_column_column
    chosen = design.get_rc_column(UR_LIMIT)
    lightest = (
        trials.filter(pl.col("ur_vert") <= UR_LIMIT)
        .group_by("column", "story")
        .agg(pl.col("A_s").min())
    )
    assert chosen.height == 12 * N_FLOORS
    assert chosen.select("column", "story", "A_s").sort("column", "story").equals(lightest.sort("column", "story"))


def test_rectangular_curve_turns_with_the_section():
    # The M3 curve of a 400 x 800 column (t2 x t3) is the M2 curve of the same column turned, 800 x 400
    from steng.design.pm import calculate_pm_column

    def curve(t2, t3):
        return calculate_pm_column(pl.DataFrame({
            "grade": [40], "t3": [t3], "t2": [t2], "cover": [30], "dia_horz0": [10], "n_vert0": [14],
            "s_vert0": [150], "dia_vert0": [20], "dia_vert1": [0], "dia_vert2": [0],
        }, schema_overrides={"grade": pl.UInt32, **{c: pl.Int32 for c in [
            "cover", "dia_horz0", "n_vert0", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2"]}})).row(0, named=True)

    upright, turned = curve(400.0, 800.0), curve(800.0, 400.0)
    for k in range(1, 6):
        assert upright[f"p_m3_{k}"] == pytest.approx(turned[f"p_m2_{k}"])
        assert upright[f"m3_{k}"] == pytest.approx(turned[f"m2_{k}"])
    # The strong axis: ETABS t3 is the depth, so M3 bends across 800 mm
    assert upright["m3_4"] > 1.8 * upright["m2_4"]


@pytest.fixture(scope="module")
def rectangular():
    """One 400 x 800 column, t2 x t3, ETABS section "R(m) 0.8 0.4" (depth first), 6000 kN under 1.35G"""
    from steng.session import SCHEMA, Session, _cast_df

    model = Session("model")
    model.datetime = "20250101-000000"
    model.story = pl.DataFrame({"story": ["L1", "Base"], "elev": [4.5, 0.0], "height": [4.5, None]})
    model.column = _cast_df(pl.DataFrame({
        "unique_line": [1], "column": ["C1"], "story": ["L1"], "xs": [[0.0, 0.0]], "ys": [[0.0, 0.0]],
        "zs": [[0.0, 4.5]], "grade": [40], "section_line": ["R(m) 0.8 0.4"], "angle": [0.0],
    }), SCHEMA["column"])
    model.force_column = _cast_df(pl.DataFrame({
        "unique_line": [1, 1], "station": [0.0, 4.5], "outputcase": ["ULS1"] * 2, "outputcasetype": ["combination"] * 2,
        "fx": [-6000.0] * 2, "fy": [50.0] * 2, "fz": [20.0] * 2, "mxx": [0.0] * 2, "myy": [0.0] * 2, "mzz": [0.0] * 2,
    }), SCHEMA["force_column"])
    session = DesignSession()
    session.sessions = {"model": model}
    return session.create_rc_column().design_rc_column(perc_vert=[1.0, 2.0], use_adsec=False)


def test_rectangular_column_bends_and_shears_about_the_right_axes(rectangular):
    pmm = rectangular.design_rc_column_force_pmm.row(0, named=True)
    # M3 bends across the depth t3 = 800 mm, M2 across the width t2 = 400 mm
    assert (pmm["h"], pmm["b"]) == (800.0, 400.0)
    # 6.1(4): e_0 = h / 30 >= 20 mm
    assert (pmm["e_02"], pmm["e_03"]) == pytest.approx((800 / 30, 20.0))
    # 5.8.3.2: the column is more slender about its weak axis
    assert pmm["lambda2"] == pytest.approx(4500 / (800 / math.sqrt(12)))  # 19.5
    assert pmm["lambda3"] == pytest.approx(4500 / (400 / math.sqrt(12)))  # 39.0
    assert pmm["m3_rd"] > pmm["m2_rd"]
    # V2 acts along the depth and V3 across it
    v2 = rectangular.design_rc_column_force_v2.row(0, named=True)
    v3 = rectangular.design_rc_column_force_v3.row(0, named=True)
    assert (v2["h"], v2["b_w"], v3["h"], v3["b_w"]) == (800.0, 400.0, 400.0, 800.0)


def test_links_have_at_least_the_perimeter_legs(rectangular):
    # The perimeter link gives two legs in each direction, however few bars a face has
    trials = rectangular.design_rc_column_column
    assert trials["n2_horz0"].min() >= 2 and trials["n3_horz0"].min() >= 2
    assert trials.select(pl.col("ur_horz2", "ur_horz3").is_finite().all()).row(0) == (True, True)
