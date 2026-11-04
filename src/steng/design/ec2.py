"""EN 1992-1-1:2004 on NumPy arrays, one function per expression, table or clause, named for it:
exp5_33_e_2 is expression 5.33 for e_2, cl6_1_4_e_0 is clause 6.1(4) for e_0
"""

import numpy as np

ALPHA_CC = 0.85
ALPHA_CT = 1
E_Y = 200000
GAMMA_C = 1.5
GAMMA_S = 1.15
C_RDC = 0.18 / GAMMA_C
K_1 = 0.15          # exp6_2
MOMENT_REDISTRIBUTION = 1
THETA_0 = 1 / 200   # exp5_1

# ----- Section 3

# Table 3.1
def tb3_1_epsilon_cu3(f_ck):
    epsilon_cu3 = np.full(len(f_ck), 0.0035)
    epsilon_cu3[f_ck >= 50] = (2.6 + 35 * (((90 - f_ck[f_ck >= 50]) / 100) ** 4)) / 1000
    return epsilon_cu3

def tb3_1_k1(f_ck):
    k1 = np.full(len(f_ck), 0.54)
    k1[f_ck <= 50] = 0.44
    return k1

def tb3_1_k2(epsilon_cu3):
    return 1.25 * (0.6 + 0.0014 / epsilon_cu3)

def tb3_1_x_max(k1, k2):
    return 1 / k2 * (MOMENT_REDISTRIBUTION - k1)

# Clause 3.2
def cl3_2_f_yd(f_yk):
    return f_yk / GAMMA_S

# Expression 3.15
def exp3_15_f_cd(f_ck):
    return ALPHA_CC * f_ck / GAMMA_C

# Expression 3.20
def exp3_20_lambda(f_ck):
    lambda1 = np.full(len(f_ck), 0.8)
    lambda1[f_ck > 50] = 0.8 - (f_ck[f_ck > 50] - 50) / 400
    return lambda1

# Expression 3.22
def exp3_22_eta(f_ck):
    eta = np.full(len(f_ck), 1.0)
    eta[f_ck > 50] = 1.0 - (f_ck[f_ck > 50] - 50) / 200
    return eta

# ----- Section 5

# Expression 5.1
def exp5_1_theta_i(alpha_h, alpha_m):
    return THETA_0 * alpha_h * alpha_m

def exp5_1_alpha_h(l):
    MIN_ALPHA_H = 2 / 3
    MAX_ALPHA_H = 1
    return np.maximum(np.minimum(2 / np.sqrt(l), MAX_ALPHA_H), MIN_ALPHA_H)

def exp5_1_alpha_m(m):
    return np.sqrt(0.5 * (1 + 1 / m))

def exp5_2_e_i(theta_i, l_0):
    return theta_i * l_0 / 2

def exp5_2_M_i(M_Ed, N_Ed, e_i):
    M_i = np.sign(M_Ed) * np.abs(N_Ed) * e_i / 1000
    M_i[M_Ed == 0] = np.abs(N_Ed[M_Ed == 0]) * e_i[M_Ed == 0] / 1000
    return M_i

# Expression 5.13N
def exp5_13_N_beta_fcm(f_ck):
    return 16.8 / np.sqrt(f_ck + 8)

def exp5_13_N_beta_t0():
    T_0 = 28
    return 1 / (0.1 + T_0 ** 0.2)

def exp5_13_N_h_0(b, h):
    return (b * h) / (b + h)

def exp5_13_N_phi_RH(f_ck, h_0):
    RH = 80
    phi_RH = np.full(len(f_ck), 0.0)
    phi_RH[(f_ck + 8) < 35] = 1 + (1 - (RH / 100)) / (0.1 * (h_0[(f_ck + 8) < 35] ** (1 / 3)))
    phi_RH[(f_ck + 8) >= 35] = (
        (1 + (1 - (RH / 100)) / (0.1 * (h_0[(f_ck + 8) >= 35] ** (1 / 3))) * ((35 / (f_ck[(f_ck + 8) >= 35] + 8)) ** 0.7)) 
        * ((35 / (f_ck[(f_ck + 8) >= 35] + 8)) ** 0.2)
    )
    return phi_RH

def exp5_13_N_phi_infinity_t0(beta_fcm, beta_t0, phi_RH):
    return beta_fcm * beta_t0 * phi_RH

def exp5_13_N_phi_ef(phi_infinity_t0):
    M_0EQP_M_0ED = 0.75
    return phi_infinity_t0 * M_0EQP_M_0ED

def exp5_13_N_A(phi_ef):
    return 1 / (1 + 0.2 * phi_ef)

def exp5_13_N_omega(A_s, f_yd, A_c, f_cd):
    return (A_s * f_yd) / (A_c * f_cd)

def exp5_13_N_B(omega):
    return np.sqrt(1 + 2 * omega)

def exp5_13_N_n(N_Ed, b, h, f_cd):
    return N_Ed * 1000 / (b * h * f_cd)

def exp5_13_N_lambda_lim(A, B, n):
    C = 0.7
    lambda_lim = np.full(len(A), 9999.9)
    lambda_lim[n > 0] = 20 * A[n > 0] * B[n > 0] * C / np.sqrt(n[n > 0])
    return lambda_lim

def exp5_13_N_lambda(l_0, h):
    return l_0 / (1 / np.sqrt(12) * h)

# Expression 5.33
def exp5_33_e_2(one_r, l_0):
    C = 10
    return one_r * (l_0 ** 2) / C

def exp5_33_M_2(M_Ed, N_Ed, e_2):
    # Takes the sign of the first-order moment, and is positive where there is none (5.8.8.2(1))
    sign = np.where(M_Ed == 0, 1.0, np.sign(M_Ed))
    M_2 = np.full(len(M_Ed), 0.0)
    M_2[N_Ed >= 0] = sign[N_Ed >= 0] * np.abs(N_Ed[N_Ed >= 0]) * e_2[N_Ed >= 0] / 1000
    return M_2

# Expression 5.34
def exp5_34_1_r0(f_yd, d):
    return f_yd / E_Y / (0.45 * d)

def exp5_34_1_r(K_r, K_phi, one_r0):
    return K_r * K_phi * one_r0

# Expression 5.36
def exp5_36_K_r(n_u, n):
    MAX_K_R = 1
    N_BAL = 0.4
    return np.minimum((n_u - n) / (n_u - N_BAL), MAX_K_R)

def exp5_36_n_u(omega):
    return 1 + omega

# Expression 5.37
def exp5_37_M_N(h, M_0Ed, N_Ed):
    M_N = np.copy(h)
    M_N[N_Ed != 0] = np.abs(M_0Ed[N_Ed != 0] / N_Ed[N_Ed != 0]) * 1000
    return M_N

def exp5_37_beta(f_ck, lambda1):
    return 0.35 + f_ck / 200 - lambda1 / 150

def exp5_37_K_phi(is_creep, phi_ef, beta):
    MIN_K_PHI = 1
    K_phi = np.full(len(is_creep), 1.0)
    K_phi[is_creep] = np.maximum(1 + phi_ef[is_creep] * beta[is_creep], MIN_K_PHI)
    return K_phi

# Expression 5.38b
def exp5_38_b_e_2(M_Ed3, N_Ed):
    e_2 = np.full(len(M_Ed3), 0.0)
    e_2[N_Ed != 0] = np.abs(M_Ed3[N_Ed != 0] / N_Ed[N_Ed != 0]) * 1e3
    return e_2

def exp5_38_b_x(e_2, e_3, t_2, t_3):
    x = np.full(len(e_2), 0.0)
    x[e_3 != 0] = (e_2[e_3 != 0] / t_2[e_3 != 0]) / (e_3[e_3 != 0] / t_3[e_3 != 0])
    return x

# Expression 5.39

def exp5_39_N_Ed_N_Rd(N_Ed, N_Rd):
    return np.abs(N_Ed) / N_Rd

def exp5_39_alpha(N_Ed_N_Rd):
    alpha = np.full(len(N_Ed_N_Rd), 2.0)
    alpha[N_Ed_N_Rd <= 1] = (2 - 1.5) / (1 - 0.7) * (N_Ed_N_Rd[N_Ed_N_Rd <= 1] - 0.7) + 1.5
    alpha[N_Ed_N_Rd <= 0.7] = (1.5 - 1) / (0.7 - 0.1) * (N_Ed_N_Rd[N_Ed_N_Rd <= 0.7] - 0.1) + 1
    alpha[N_Ed_N_Rd <= 0.1] = 1
    return alpha

def exp5_39_M_Ed_M_Rd(M_Ed, M_Rd):
    ur_m = np.full(len(M_Ed), 99.99)
    ur_m[M_Rd != 0] = np.abs(M_Ed[M_Rd != 0]) / M_Rd[M_Rd != 0]
    return ur_m

def exp5_39_ur_biaxial(M_Ed_M_Rd2, M_Ed_M_Rd3, alpha):
    return np.sign(M_Ed_M_Rd2) * np.abs(M_Ed_M_Rd2) ** alpha + np.sign(M_Ed_M_Rd3) * np.abs(M_Ed_M_Rd3) ** alpha

# ----- Section 6

# Clause 6.1
def cl6_1_k(M_Ed, b, d, f_ck):
    return M_Ed * 1e6 / (b * (d ** 2) * f_ck)

def cl6_1_k_1(lambda1, eta):
    return lambda1 * eta * ALPHA_CC / GAMMA_C

def cl6_1_k_2(lambda1):
    return lambda1 / 2

def cl6_1_k_prime(k_1, k_2, x_1_d):
    return k_1 * x_1_d * (1 - k_2 * x_1_d)

def cl6_1_c(eta):
    return (2 * GAMMA_C / eta / ALPHA_CC)

def cl6_1_z(d, c, k):
    z = np.full(len(d), 0.95)
    x = 0.5  * (1 + np.sqrt(1 - c * k))
    z[x <= 0.95] = x[x <= 0.95]
    z *= d
    return z

def cl6_1_A_s_singly(M_Ed, f_yd, z):
    return M_Ed * 1e6 / f_yd / z

def cl6_1_x(x_d, d):
    return x_d * d

def cl6_1_M_1(k_prime, b, d, f_ck):
    return k_prime * b * (d ** 2) * f_ck

def cl6_1_M_2(M_Ed, M_1):
    return M_Ed * 1e6 - M_1

def cl6_1_f_sc(f_yd, x, d_prime, epsilon_cu3):
    return np.minimum(f_yd, 700 * (x - d_prime) / x * (epsilon_cu3 / 0.0035))

def cl6_1_A_sc(M_2, f_sc, d, d_prime):
    return M_2 / f_sc / (d - d_prime)

def cl6_1_A_s(k_1, b, x, f_ck, A_sc, f_sc, f_yd):
    return (k_1 * b * x * f_ck + A_sc * f_sc) / f_yd

# Clause 6.1(4)
def cl6_1_4_e_0(h):
    MIN_H = 20
    return np.maximum(h / 30, MIN_H)

def cl6_1_4_M_min(M_Ed, N_Ed, e_0):
    M_min = np.sign(M_Ed) * np.abs(N_Ed) * e_0 / 1000
    M_min[M_Ed == 0] = np.abs(N_Ed[M_Ed == 0]) * e_0[M_Ed == 0] / 1000
    return M_min

# Expression 6.2.a
def exp6_2_a_V_Rdc(k, rho_l, f_ck, sigma_cp, b_w, d, v_min):
    C_RDC = 0.18 / GAMMA_C
    return (
        np.maximum(
            (C_RDC * k * (100 * rho_l * f_ck) ** (1 / 3) + K_1 * sigma_cp) * b_w * d, 
            np.maximum((v_min + K_1 * sigma_cp) * b_w * d, 0)
        )
    )

def exp6_2_a_V_Rdc_1(k, rho_l, f_ck, sigma_cp, b_w, d):
    return (C_RDC * k * (100 * rho_l * f_ck) ** (1 / 3) + K_1 * sigma_cp) * b_w * d

def exp6_2_a_k(d):
    MAX_K = 2
    return np.minimum(1 + np.sqrt(200 / d), MAX_K)

def exp6_2_a_rho_l(A_sl, b_w, d):
    MAX_RHO_L = 0.02
    return np.minimum(A_sl / (b_w * d), MAX_RHO_L)

def exp6_2_a_sigma_cp(N_Ed, A_C, f_cd):
    return np.minimum(N_Ed / A_C, 0.2 * f_cd)

# Expression 6.2.b
def exp6_2_b_V_Rdc_min(v_min, sigma_cp, b_w, d):
    return (v_min + K_1 * sigma_cp) * b_w * d

def exp6_2_b_v_min(k, f_ck):
    return 0.035 * k ** (3/2) * f_ck ** (1 / 2)

# Expression 6.6N
def exp6_6_N_v(f_ck):
    return 0.6 * (1 - f_ck / 250)

# Expression 6.8
def exp6_8_A_sw_s(V_Ed, z, f_ywd, theta):
    theta = theta / 180 * np.pi
    return V_Ed * 1000 / z / f_ywd * np.tan(theta)

# Expression 6.9
def exp6_9_V_Rd_max(alpha_cw, b_w, z, v1, f_cd, theta):
    return alpha_cw * b_w * z * v1 * f_cd / (1 / np.tan(theta) + np.tan(theta)) / 1000

def exp_6_9_theta(V_Ed, b_w, d, f_ck, N_Ed):
    theta = np.full(len(V_Ed), 89.9)
    is_selected = (V_Ed * 1000) < (0.13 * b_w * d * f_ck * (1 - (f_ck / 250)))
    theta[is_selected] = 21.8
    is_selected = np.logical_and(is_selected, N_Ed < 0)
    theta[is_selected] = 45.0
    is_selected = np.logical_and(
        (V_Ed * 1000) >= (0.13 * b_w * d * f_ck * (1 - (f_ck / 250))),
        (V_Ed * 1000) <= (0.179856 * b_w * d * f_ck * (1 - (f_ck / 250))),
    )
    theta[is_selected] = (
        0.5 * np.arcsin(5.56 * V_Ed[is_selected] * 1000 / (b_w[is_selected] * d[is_selected] * f_ck[is_selected] * (1 - (f_ck[is_selected] / 250))))
    ) / np.pi * 180
    is_selected = np.logical_and(is_selected, N_Ed < 0)
    theta[is_selected] = 45.0
    return theta

# Expression 6.47 
# k refer to exp 6.2a
# vmin refer to exp 6.2b
def exp6_47_rho_l(rho_ly, rho_lz):
    RHO_L_MAX = 0.02
    return np.minimum(np.sqrt(rho_ly * rho_lz), RHO_L_MAX)

def exp6_47_sigma_c(N_Ed, A_c):
    return N_Ed / A_c

def exp6_47_sigma_cp(sigma_cy, sigma_cz):
    return (sigma_cy + sigma_cz) / 2

def exp6_47_v_Rd_c(k, rho_l, f_ck, sigma_cp, v_min):
    return np.maximum(C_RDC * k * ((100 * rho_l * f_ck) ** (1/3)) + K_1 * sigma_cp, v_min + K_1 * sigma_cp)

def exp6_52_f_ywd_ef(d, f_yd):
    return np.minimum(250 + 0.25 * d, f_yd)

def exp6_52_v_Rd_cs(v_Rd_c, s_r, A_sw, f_ywd_ef, u_1, alpha):
    return 0.75 * v_Rd_c + 1.5 / s_r * A_sw * f_ywd_ef / u_1 * np.sin(alpha)

# ----- Section 9

# Clause 9.6.3(1)
def cl9_6_3_1_A_sw_s_min1(s_vert, *dia_vert):
    """Horizontal bars of at least 25% of the vertical: every layer on both faces at s_vert"""
    return 0.25 * 2 * sum((dia ** 2) / 4 * np.pi for dia in dia_vert) / s_vert

def cl9_6_3_1_A_sw_s_min2(b):
    return 0.001 * b

# ----- Section 12

# Clause 12.6.5.1 Table 12.1
def cl12_6_5_1_beta(n_side_restrained, b, l_w):
    beta = np.full(len(n_side_restrained), 1.0)
    is_three = n_side_restrained == 3
    beta[is_three] = 1 / (1 + (l_w[is_three] / (3 * b[is_three])) ** 2)
    is_four_b_wider = np.logical_and(
        n_side_restrained == 4,
        b >= l_w
    )
    beta[is_four_b_wider] = 1 / (1 + (l_w[is_four_b_wider] / (b[is_four_b_wider])) ** 2)
    is_four_l_w_wider = np.logical_and(
        n_side_restrained == 4,
        b < l_w
    )
    beta[is_four_l_w_wider] = b[is_four_l_w_wider] / (2 * l_w[is_four_l_w_wider])
    return beta