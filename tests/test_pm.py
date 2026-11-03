"""Force-moment interaction of a rectangular section, against hand calculations"""

import numpy as np
import pytest

from steng.design.pm import _calculate_pm_curve

F_CK, B, H = 30.0, 300.0, 600.0
F_CD, F_YD = 0.85 * F_CK / 1.5, 500 / 1.15
# 4 bars of 25 mm, 50 mm from the bottom
D_REO, DIA_REO = np.full(4, 550.0), np.full(4, 25.0)
A_S = 4 * np.pi * 25**2 / 4


@pytest.fixture(scope="module")
def curve():
    m2_direction = _calculate_pm_curve(F_CK, B, H, D_REO, DIA_REO, D_REO, DIA_REO)[:10]
    return dict(zip(["p1", "p2", "p3", "p4", "p5", "m1", "m2", "m3", "m4", "m5"], m2_direction))


def test_pure_compression(curve):
    assert curve["p1"] == pytest.approx((F_CD * B * H + A_S * F_YD) / 1000, rel=0.001)


def test_pure_tension(curve):
    assert curve["p5"] == pytest.approx(-A_S * F_YD / 1000, rel=0.001)


def test_pure_bending_matches_rectangular_stress_block(curve):
    # Rectangular block: 0.8x deep at f_cd, steel yielding
    x = A_S * F_YD / (0.8 * F_CD * B)
    moment = A_S * F_YD * (550 - 0.4 * x) / 1e6
    assert curve["p4"] == 0
    assert abs(curve["m4"]) == pytest.approx(moment, rel=0.02)
