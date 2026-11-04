"""Reinforced concrete calculations on DataFrames, against hand calculations"""

import math

import polars as pl
import pytest

from steng.design import rc


def test_wall_slenderness_counts_every_bar_layer():
    # A 6 m long, 300 mm wall with four layers of bars at 200 mm on each face: 5.8.3.1 omega counts them all
    df = pl.DataFrame({
        "f_yd": [500 / 1.15], "f_cd": [0.85 * 40 / 1.5], "b": [300.0], "h": [6000.0], "p": [5000.0],
        "l_02": [3600.0], "l_03": [3600.0], "phi_ef": [1.0], "s_vert0": [200],
        "dia_vert0": [32], "dia_vert1": [32], "dia_vert2": [25], "dia_vert3": [20], "dia_vert4": [0], "dia_vert5": [0],
    })
    A_s = 2 * math.pi / 4 * (32**2 + 32**2 + 25**2 + 20**2) * 6000 / 200  # 2 faces, 30 bars a face per layer
    row = rc.calculate_slenderness(df).row(0, named=True)
    assert row["A_s"] == pytest.approx(A_s)
    assert row["omega"] == pytest.approx(A_s * (500 / 1.15) / (300 * 6000 * 0.85 * 40 / 1.5))


def test_biaxial_check_not_needed_when_bending_is_mostly_about_one_axis():
    # 5.8.9(3): no biaxial check if the slenderness ratios are within 2 of each other, and the relative
    # eccentricities differ by 5 times either way (5.38b). A square column, bending mostly about axis 3
    df = pl.DataFrame({
        "b": [800.0, 350.0], "h": [800.0, 6000.0], "p": [5000.0, 5000.0], "n_rd": [20000.0, 60000.0],
        "m2_rd": [1500.0, 1200.0], "m3_rd": [1500.0, 30000.0],
        "M_Edtotal2": [20.0, 20.0], "M_Edtotal3": [500.0, 5000.0],
        "lambda2": [20.0, 2.5], "lambda3": [20.0, 50.0],
    })
    rows = rc.calculate_ur_biaxial_eccentricity(df).to_dicts()
    # e_y / h = 0.1 / 0.8 against e_z / b = 0.004 / 0.8: a ratio of 0.04, so separate checks suffice
    assert not rows[0]["is_biaxial"]
    assert rows[0]["ur_biaxial"] == 0
    # A wall: lambda3 / lambda2 = 20, so the biaxial check is needed whatever the eccentricities
    assert rows[1]["is_biaxial"]
    assert rows[1]["ur_biaxial"] > 0


def test_side_restraint_shortens_the_out_of_plane_length():
    # Table 12.1: a 6 m long wall, 3.6 m high, held on 3 sides: beta = 1 / (1 + (l_w / 3b)^2) out of plane
    df = pl.DataFrame({"l": [3600.0], "h": [6000.0], "k2": [1.0], "k3": [1.0], "n_side_restrained": [3]})
    row = rc.calculate_effective_length_wall(df).row(0, named=True)
    beta = 1 / (1 + (3.6 / (3 * 6.0)) ** 2)  # 0.962
    assert row["l_02"] == pytest.approx(3600.0)
    assert row["l_03"] == pytest.approx(3600.0 * beta)


def test_shear_lever_arm_is_0_9_d():
    # 6.2.3(1) and 6.9: V_Rd,max = alpha_cw b_w z nu_1 f_cd / (cot + tan), at 45 degrees here, z = 0.9 d
    f_ck, f_cd, b_w, d = 40.0, 0.85 * 40.0 / 1.5, 350.0, 7000.0
    df = pl.DataFrame({"V_Ed": [2000.0], "f_ck": [f_ck], "f_cd": [f_cd], "b_w": [b_w], "d": [d]})
    row = rc.calculate_capacity_shear_max(df).row(0, named=True)
    nu_1 = 0.6 * (1 - f_ck / 250)
    assert row["z"] == pytest.approx(0.9 * d)
    assert row["V_Rd_max"] == pytest.approx(b_w * 0.9 * d * nu_1 * f_cd / 2 / 1000)  # 12,595 kN
