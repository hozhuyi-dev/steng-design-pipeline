"""Eurocode 2 (EN 1992-1-1:2004) expressions against hand calculations"""

import numpy as np
import pytest

from steng.design import ec2


def test_design_strengths():
    assert ec2.exp3_15_f_cd(40) == pytest.approx(0.85 * 40 / 1.5)
    assert ec2.cl3_2_f_yd(500) == pytest.approx(434.78, abs=0.01)


def test_table_3_1_and_stress_block_above_c50():
    f_ck = np.array([40.0, 70.0])
    np.testing.assert_allclose(ec2.tb3_1_epsilon_cu3(f_ck), [0.0035, (2.6 + 35 * 0.2**4) / 1000])
    np.testing.assert_allclose(ec2.exp3_20_lambda(f_ck), [0.8, 0.75])
    np.testing.assert_allclose(ec2.exp3_22_eta(f_ck), [1.0, 0.9])


def test_biaxial_exponent_5_39():
    n_ratio = np.array([0.05, 0.4, 0.85, 1.2])
    np.testing.assert_allclose(ec2.exp5_39_alpha(n_ratio), [1.0, 1.25, 1.75, 2.0])


def test_second_order_moment_5_33_without_first_order_moment():
    # 5.8.8.2(1): M_2 = N_Ed e_2, whatever the first-order moment. A slender wall with no first-order
    # moment out of plane still bows: 8977.5 kN at e_2 = 65.8 mm is 591 kNm
    M_Ed, N_Ed, e_2 = np.array([0.0, 120.0, -120.0]), np.array([8977.5] * 3), np.array([65.8] * 3)
    np.testing.assert_allclose(ec2.exp5_33_M_2(M_Ed, N_Ed, e_2), [590.72, 590.72, -590.72], rtol=1e-4)
    # No second-order moment in tension
    assert ec2.exp5_33_M_2(np.array([120.0]), np.array([-500.0]), np.array([65.8]))[0] == 0


def test_creep_factor_for_relative_humidity_annex_b():
    # C30/37, f_cm = 38 MPa > 35 MPa, RH = 80%, h_0 = 300 mm: expressions B.3b and B.8c
    f_cm, h_0 = 38.0, 300.0
    alpha_1, alpha_2 = (35 / f_cm) ** 0.7, (35 / f_cm) ** 0.2
    expected = (1 + (1 - 0.8) / (0.1 * h_0 ** (1 / 3)) * alpha_1) * alpha_2
    assert ec2.exp5_13_N_phi_RH(np.array([30.0]), np.array([h_0]))[0] == pytest.approx(expected)


def test_doubly_reinforced_beam():
    # 300 wide, d = 500, d' = 50, C30/37, M_Ed = 500 kNm: K = 0.222 > K' = 0.167
    b, d, d_prime, f_ck, f_yd, M_Ed = 300.0, 500.0, 50.0, 30.0, 500 / 1.15, 500.0
    k_1 = ec2.cl6_1_k_1(np.array([0.8]), np.array([1.0]))
    k_prime = ec2.cl6_1_k_prime(k_1, np.array([0.4]), np.array([0.45]))
    assert k_prime[0] == pytest.approx(0.167, abs=0.001)

    x = ec2.cl6_1_x(0.45, d)
    M_1 = ec2.cl6_1_M_1(k_prime, b, d, f_ck)
    M_2 = ec2.cl6_1_M_2(M_Ed, M_1)
    f_sc = ec2.cl6_1_f_sc(f_yd, x, d_prime, 0.0035)
    A_sc = ec2.cl6_1_A_sc(M_2, f_sc, d, d_prime)
    A_s = ec2.cl6_1_A_s(k_1, b, x, f_ck, A_sc, f_sc, f_yd)
    k_1, f_sc, A_sc, A_s = (float(np.ravel(v)[0]) for v in (k_1, f_sc, A_sc, A_s))

    # 700 (x - d') / x = 544 MPa, so the compression steel yields
    assert f_sc == pytest.approx(f_yd)
    # Moments about the tension steel: the stress block and the compression steel carry M_Ed
    z = d - 0.4 * x
    assert (k_1 * f_ck * b * x * z + A_sc * f_sc * (d - d_prime)) / 1e6 == pytest.approx(M_Ed, rel=0.01)
    # Forces: the tension steel balances the stress block and the compression steel
    assert A_s * f_yd == pytest.approx(k_1 * f_ck * b * x + A_sc * f_sc)
    assert A_s == pytest.approx(2743, rel=0.01)
