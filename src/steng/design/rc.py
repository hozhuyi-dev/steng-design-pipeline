"""Reinforced concrete design checks on DataFrames: each function reads its inputs from columns
and adds its results as columns, in mm, kN and MPa
"""

import numpy as np
import pandera.polars as pa
import polars as pl

from steng.design import ec2
from steng.design import expressions as rc
from steng.design._checks import _check_inputs, _drop_outputs, bar_layers

F_YK = 500.0 #MPa

# ----- column -----

def calculate_pmm_add(df):
    """Calculate the additional pmm force
    """
    outputs = ["p_add", "m2_add", "m3_add"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["p_add_m", "m2_add_m", "m3_add_m", "h"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    p_add_m = np.array(df.select("p_add_m").to_series())    #kN
    m2_add_m = np.array(df.select("m2_add_m").to_series())  #kNm
    m3_add_m = np.array(df.select("m3_add_m").to_series())  #kNm
    h = np.array(df.select("h").to_series()) / 1000         #m
    
    # Constant

    # Calculations
    p_add = p_add_m * h
    m2_add = m2_add_m * h
    m3_add = m3_add_m * h

    # Output
    df = df.hstack(
        pl.DataFrame({
            "p_add": p_add,
            "m2_add": m2_add,
            "m3_add": m3_add,
        })
    )

    return df

def calculate_pmm_design(df):
    """Calculate the design PMM by adding to the forces
    """
    outputs = ["p", "m2", "m3"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["p_base", "m2_base", "m3_base", "p_add", "m2_add", "m3_add"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    p_base = np.array(df.select("p_base").to_series())      #kN
    m2_base = np.array(df.select("m2_base").to_series())    #kNm
    m3_base = np.array(df.select("m3_base").to_series())    #kNm
    p_add = np.array(df.select("p_add").to_series())        #kN
    m2_add = np.array(df.select("m2_add").to_series())      #kNm
    m3_add = np.array(df.select("m3_add").to_series())      #kNm
    
    # Constant

    # Calculations
    p = p_base + p_add
    m2 = m2_base + m2_add
    m3 = m3_base + m3_add

    # Output
    df = df.hstack(
        pl.DataFrame({
            "p": p,
            "m2": m2,
            "m3": m3
        })
    )

    return df

def calculate_material(df):
    """Calculate f_cd and f_yd
    """

    # Input
    missing_column = _check_inputs(df, ["f_ck"])
    if missing_column: return df  
    f_ck = np.array(df.select("f_ck").to_series())  #MPa
    
    # Constant
    n_rows = df.shape[0]
    f_yk = np.full(n_rows, F_YK) #MPa

    # Calculations
    f_yd = ec2.cl3_2_f_yd(f_yk) # MPa

    f_cd = ec2.exp3_15_f_cd(f_ck) #MPa

    # 
    df = _drop_outputs(df, ["f_cd", "f_yd"])
    df = df.hstack(
        pl.DataFrame({
            "f_yd": f_yd,
            "f_cd": f_cd,
        })
    )

    return df

def calculate_effective_depth_column(df):
    """Calculate the effective depths d2 and d3 of a column to its innermost layer of bars
    """

    # Validation
    pa.DataFrameSchema({
        "b": pa.Column(pl.Float64),
        "h": pa.Column(pl.Float64),
        "cover": pa.Column(pl.Int32),
        "dia_horz0": pa.Column(pl.Int32),
        "dia_vert0": pa.Column(pl.Int32),
    }).validate(df)

    # Input
    b = df["b"].to_numpy()                #mm
    h = df["h"].to_numpy()                #mm
    cover = df["cover"].to_numpy()        #mm
    dia_horz = df["dia_horz0"].to_numpy()  #mm
    dia_vert = bar_layers(df)            #mm

    # Calculations
    d2 = rc.d3(h, cover, dia_horz, *dia_vert)

    d3 = rc.d3(b, cover, dia_horz, *dia_vert)

    # Output
    return (
        df
        .with_columns(
            pl.Series("d2", d2, dtype=pl.Float64),
            pl.Series("d3", d3, dtype=pl.Float64),
        )
    )

def calculate_effective_depth_wall(df):
    """Calculate the effective depths of a wall: d2 in plane, to the end bars, and d3 out of plane,
    to the innermost layer of bars
    """
    df = _drop_outputs(df, ["d2", "d3"])

    # Input
    missing_column = _check_inputs(df, ["b", "h", "cover", "dia_horz0", "dia_vert0"])
    if missing_column: return df
    b = np.array(df.select("b").to_series())                #mm
    h = np.array(df.select("h").to_series())                #mm
    cover = np.array(df.select("cover").to_series())        #mm
    dia_horz = np.array(df.select("dia_horz0").to_series())  #mm
    dia_vert = bar_layers(df)                              #mm

    # Calculations
    d2 = rc.d2(h, cover, dia_horz, dia_vert[0])

    d3 = rc.d3(b, cover, dia_horz, *dia_vert)

    # Output
    df = df.hstack(
        pl.DataFrame({
            "d2": d2,
            "d3": d3,
        })
    )

    return df

def calculate_effective_length_wall(df):
    """Calculate the effective length
    """

    # Input
    missing_column = _check_inputs(df, ["l", "h", "k2", "k3", "n_side_restrained"])
    if missing_column: return df
    l = np.array(df.select("l").to_series()) / 1e3  #m
    h = np.array(df.select("h").to_series()) / 1e3  #m
    k2 = np.array(df.select("k2").to_series())      #-
    k3 = np.array(df.select("k3").to_series())      #-
    n_side_restrained = np.array(df.select("n_side_restrained").to_series())
    
    # Constant
    df.shape[0]

    # Calculations
    beta_restraint = ec2.cl12_6_5_1_beta(n_side_restrained, h, l)

    l_02 = rc.l_0(l, k2) #m, in plane
    l_03 = rc.l_0_restrained(l, k3, beta_restraint) #m, out of plane
    l_02 *= 1e3 #mm
    l_03 *= 1e3 #mm

    # Output
    df = _drop_outputs(df, ["beta_restraint", "l_02", "l_03"])
    df = df.hstack(
        pl.DataFrame({
            "beta_restraint": beta_restraint,
            "l_02": l_02,
            "l_03": l_03,
        })
    )

    return df

def calculate_effective_length_column(df):
    """Calculate the effective length
    """

    # Input
    missing_column = _check_inputs(df, ["l", "k2", "k3"])
    if missing_column: return df
    l = np.array(df.select("l").to_series()) / 1e3  #m
    k2 = np.array(df.select("k2").to_series())      #-
    k3 = np.array(df.select("k3").to_series())      #-
    
    # Constant
    df.shape[0]

    # Calculations
    l_02 = rc.l_0(l, k2) #m
    l_03 = rc.l_0(l, k3) #m
    l_02 *= 1e3 #mm
    l_03 *= 1e3 #mm

    # Output
    df = _drop_outputs(df, ["l_02", "l_03"])
    df = df.hstack(
        pl.DataFrame({
            "l_02": l_02,
            "l_03": l_03,
        })
    )

    return df

def calculate_eccentricity_0_i(df):
    """Calculate e_0 and e_1
    """
    outputs = ["alpha_h", "alpha_m", "theta_i", "e_i2", "e_i3", "e_02", "e_03"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    n_rows = df.shape[0]
    columns = ["l", "b", "h", "l_02", "l_03"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    l = np.array(df.select("l").to_series()) / 1e3  #m
    b = np.array(df.select("b").to_series()) / 1e3  #m
    h = np.array(df.select("h").to_series()) / 1e3  #m
    l_02 = np.array(df.select("l_02").to_series())  #mm
    l_03 = np.array(df.select("l_03").to_series())  #mm
    
    # Constant
    m = np.full(n_rows, 1)  #-

    # Calculations
    alpha_h = ec2.exp5_1_alpha_h(l) #-

    alpha_m = ec2.exp5_1_alpha_m(m) #-

    theta_i = ec2.exp5_1_theta_i(alpha_h, alpha_m) #-

    e_i2 = ec2.exp5_2_e_i(theta_i, l_02) #mm
    e_i3 = ec2.exp5_2_e_i(theta_i, l_03) #mm

    e_02 = ec2.cl6_1_4_e_0(h * 1e3) #mm
    e_03 = ec2.cl6_1_4_e_0(b * 1e3) #mm

    # Output
    df = df.hstack(
        pl.DataFrame({
            "alpha_h": alpha_h,
            "alpha_m": alpha_m,
            "theta_i": theta_i,
            "e_i2": e_i2,
            "e_i3": e_i3,
            "e_02": e_02,
            "e_03": e_03,
        })
    )

    return df

def calculate_creep(df):
    """Calculate the slenderness limit in both directions
    """
    outputs = [
        "beta_fcm", "beta_t0", "h_0", "phi_RH", "phi_infinity_t0", "phi_ef",
    ]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["f_ck", "b", "h"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    f_ck = np.array(df.select("f_ck").to_series())          #MPa
    b = np.array(df.select("b").to_series())                #mm
    h = np.array(df.select("h").to_series())                #mm
    
    # Constant

    # Calculations
    beta_fcm = ec2.exp5_13_N_beta_fcm(f_ck) #-

    beta_t0 = ec2.exp5_13_N_beta_t0() #-

    h_0 = ec2.exp5_13_N_h_0(b, h) #mm

    phi_RH = ec2.exp5_13_N_phi_RH(f_ck, h_0) #-

    phi_infinity_t0 = ec2.exp5_13_N_phi_infinity_t0(beta_fcm, beta_t0, phi_RH) #-

    phi_ef = ec2.exp5_13_N_phi_ef(phi_infinity_t0) #-

    # Output
    df = df.hstack(
        pl.DataFrame({
            "beta_fcm": beta_fcm,
            "beta_t0": beta_t0,
            "h_0": h_0,
            "phi_RH": phi_RH,
            "phi_infinity_t0": phi_infinity_t0,
            "phi_ef": phi_ef,
        })
    )

    return df

def calculate_slenderness(df, section="wall"):
    """Calculate the slenderness limit in both directions

    section is "wall", with each layer of bars on both faces at s_vert0, or "column", with n_vert0
    bars in each layer. Every dia_vert0, dia_vert1, ... layer present counts towards A_s.
    """
    outputs = [
        "A", "A_s", "A_c", "omega", "B", "n", "lambda_lim", "lambda2",
        "lambda3", "is_slender2", "is_slender3",
    ]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["f_yd", "f_cd", "b", "h", "s_vert0", "dia_vert0", "p", "l_02", "l_03", "phi_ef"]
    if section == "column":
        columns.append("n_vert0")
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    f_cd = np.array(df.select("f_cd").to_series())          #MPa
    f_yd = np.array(df.select("f_yd").to_series())          #MPa
    b = np.array(df.select("b").to_series())                #mm
    h = np.array(df.select("h").to_series())                #mm
    s_vert = np.array(df.select("s_vert0").to_series())     #mm
    dia_vert = bar_layers(df)                              #mm
    N_Ed = np.array(df.select("p").to_series())             #kN
    l_02 = np.array(df.select("l_02").to_series())          #mm
    l_03 = np.array(df.select("l_03").to_series())          #mm
    phi_ef = np.array(df.select("phi_ef").to_series())      #-

    # Constant

    # Calculations
    A = ec2.exp5_13_N_A(phi_ef) #-

    if section == "column":
        A_s = rc.A_s_column(df["n_vert0"].to_numpy(), *dia_vert) #mm^2
    else:
        A_s = rc.A_s_wall(s_vert, h, *dia_vert) #mm^2

    A_c = rc.A_c(h, b) #mm^2
    
    omega = ec2.exp5_13_N_omega(A_s, f_yd, A_c, f_cd) #-

    B = ec2.exp5_13_N_B(omega) #-

    n = ec2.exp5_13_N_n(N_Ed, b, h, f_cd) #-

    lambda_lim = ec2.exp5_13_N_lambda_lim(A, B, n) #-

    lambda2 = ec2.exp5_13_N_lambda(l_02, h)
    lambda3 = ec2.exp5_13_N_lambda(l_03, b)

    is_slender2 = np.logical_and(n > 0, lambda2 > lambda_lim)
    is_slender3 = np.logical_and(n > 0, lambda3 > lambda_lim)

    # Output
    df = df.hstack(
        pl.DataFrame({
            "A": A,
            "A_s": A_s,
            "A_c": A_c,
            "omega": omega,
            "B": B,
            "n": n,
            "lambda_lim": lambda_lim,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "is_slender2": is_slender2,
            "is_slender3": is_slender3
        })
    )

    return df

def calculate_eccentricity_2(df):
    """Calculate the design moment considering the ecc moment etc
    """
    outputs = [
        "n_u", "K_r" "M_N2", "M_N3", "is_creep2", "is_creep3", "beta2", "beta3",
        "K_phi2", "K_phi3", "one_r02", "one_r03", "one_r2", "one_r3", "e_22", "e_23"
    ]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    n_rows = df.shape[0]
    columns = [
        "f_ck", "f_yd", "omega", "n", "b", "h", "d2", "d3", "l_02", "l_03",
        "p", "m2", "m3", "phi_infinity_t0", "phi_ef", "lambda2", "lambda3"
    ]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    f_ck = np.array(df.select("f_ck").to_series())                          #MPa
    f_yd = np.array(df.select("f_yd").to_series())                          #MPa
    omega = np.array(df.select("omega").to_series())                        #-
    n = np.array(df.select("n").to_series())                                #-
    b = np.array(df.select("b").to_series())                                #mm
    h = np.array(df.select("h").to_series())                                #mm
    d2 = np.array(df.select("d2").to_series())                              #mm
    d3 = np.array(df.select("d3").to_series())                              #mm
    l_02 = np.array(df.select("l_02").to_series())                          #mm
    l_03 = np.array(df.select("l_03").to_series())                          #mm
    p = np.array(df.select("p").to_series())                                #kN
    m2 = np.array(df.select("m2").to_series())                              #kNm
    m3 = np.array(df.select("m3").to_series())                              #kNm
    phi_infinity_t0 = np.array(df.select("phi_infinity_t0").to_series())    #-
    phi_ef = np.array(df.select("phi_ef").to_series())                      #-
    lambda2 = np.array(df.select("lambda2").to_series())                    #-
    lambda3 = np.array(df.select("lambda3").to_series())                    #-
    
    # Constant

    # Calculations
    n_u = ec2.exp5_36_n_u(omega)

    K_r = ec2.exp5_36_K_r(n_u, n)

    M_N2 = ec2.exp5_37_M_N(h, m3, p)
    M_N3 = ec2.exp5_37_M_N(b, m2, p)

    is_creep2 = np.full(n_rows, True)
    is_creep2[p == 0] = False
    is_creep2[np.logical_and(phi_infinity_t0 <= 2, np.logical_and(lambda2 <= 75, np.greater_equal(M_N2, h)))] = False
    is_creep3 = np.full(n_rows, True)
    is_creep3[p == 0] = False
    is_creep3[np.logical_and(phi_infinity_t0 <= 2, np.logical_and(lambda3 <= 75, np.greater_equal(M_N3, b)))] = False

    beta2 = ec2.exp5_37_beta(f_ck, lambda2)
    beta3 = ec2.exp5_37_beta(f_ck, lambda3)

    K_phi2 = ec2.exp5_37_K_phi(is_creep2, phi_ef, beta2)
    K_phi3 = ec2.exp5_37_K_phi(is_creep3, phi_ef, beta3)

    one_r02 = ec2.exp5_34_1_r0(f_yd, d2)
    one_r03 = ec2.exp5_34_1_r0(f_yd, d3)

    one_r2 = ec2.exp5_34_1_r(K_r, K_phi2, one_r02)
    one_r3 = ec2.exp5_34_1_r(K_r, K_phi3, one_r03)

    e_22 = ec2.exp5_33_e_2(one_r2, l_02)
    e_23 = ec2.exp5_33_e_2(one_r3, l_03)

    # Output
    df = df.hstack(
        pl.DataFrame({
            "n_u": n_u,
            "K_r": K_r,
            "M_N2": M_N2,
            "M_N3": M_N3,
            "is_creep2": is_creep2,
            "is_creep3": is_creep3,
            "beta2": beta2,
            "beta3": beta3,
            "K_phi2": K_phi2,
            "K_phi3": K_phi3,
            "one_r02": one_r02,
            "one_r03": one_r03,
            "one_r2": one_r2,
            "one_r3": one_r3,
            "e_22": e_22,
            "e_23": e_23,
        })
    )

    return df

def calculate_design_moment(df):
    """Calculate the design moment
    """
    outputs = [
        "M_22", "M_23", "M_i2", "M_i3", "M_min2", "M_min3", 
        "M_Edtotal2", "M_Edtotal3", "M_ratio2", "M_ratio3",
    ]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    n_rows = df.shape[0]
    columns = ["p", "m2", "m3", "e_i2", "e_i3", "e_22", "e_23"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    p = np.array(df.select("p").to_series())                        #kN
    m2 = np.array(df.select("m2").to_series())                      #kNm
    m3 = np.array(df.select("m3").to_series())                      #kNm
    e_02 = np.array(df.select("e_02").to_series())                  #-
    e_03 = np.array(df.select("e_03").to_series())                  #-
    e_i2 = np.array(df.select("e_i2").to_series())                  #-
    e_i3 = np.array(df.select("e_i3").to_series())                  #-
    e_22 = np.array(df.select("e_22").to_series())                  #-
    e_23 = np.array(df.select("e_23").to_series())                  #-
    is_slender2 = np.array(df.select("is_slender2").to_series())    #bool
    is_slender3 = np.array(df.select("is_slender3").to_series())    #bool
    
    # Constant

    # Calculations
    M_22 = ec2.exp5_33_M_2(m2, p, e_23)
    M_23 = ec2.exp5_33_M_2(m3, p, e_22)

    M_i2 = ec2.exp5_2_M_i(m2, p, e_i3)
    M_i3 = ec2.exp5_2_M_i(m3, p, e_i2)

    M_min2 = ec2.cl6_1_4_M_min(m2, p, e_03)
    M_min3 = ec2.cl6_1_4_M_min(m3, p, e_02)

    M_Edtotal2 = m2 + M_i2
    M_Edtotal2[is_slender3] = m2[is_slender3] + M_i2[is_slender3] + M_22[is_slender3]
    M_Edtotal2 = np.maximum(M_Edtotal2, M_min2)
    M_Edtotal3 = m3 + M_i3
    M_Edtotal3[is_slender2] = m3[is_slender2] + M_i3[is_slender2] + M_23[is_slender2]
    M_Edtotal3 = np.maximum(M_Edtotal3, M_min3)

    M_ratio2 = np.full(n_rows, 1.0)
    M_ratio2[m2 != 0] = np.abs(M_Edtotal2[m2 != 0]) / np.abs(m2[m2 != 0])
    M_ratio3 = np.full(n_rows, 1.0)
    M_ratio3[m3 != 0] = np.abs(M_Edtotal3[m3 != 0]) / np.abs(m3[m3 != 0])

    # Output
    df = df.hstack(
        pl.DataFrame({
            "M_22": M_22,
            "M_23": M_23,
            "M_i2": M_i2,
            "M_i3": M_i3,
            "M_min2": M_min2,
            "M_min3": M_min3,
            "M_Edtotal2": M_Edtotal2,
            "M_Edtotal3": M_Edtotal3,
            "M_ratio2": M_ratio2,
            "M_ratio3": M_ratio3,
        })
    )

    return df

def calculate_moment_capacity_simplified_biaxial(df):
    """Calculate moment capacity given the simplified biaxial curve
    """
    outputs = ["n_rd", "m2_rd", "m3_rd"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    n_rows = df.shape[0]
    columns = [
        "p",
        "p_m2_1", "p_m2_2", "p_m2_3", "p_m2_4", "p_m2_5",
        "m2_1", "m2_2", "m2_3", "m2_4", "m2_5",
        "p_m3_1", "p_m3_2", "p_m3_3", "p_m3_4", "p_m3_5",
        "m3_1", "m3_2", "m3_3", "m3_4", "m3_5",
    ]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    p = np.array(df.select("p").to_series())            #kN
    p_m2_1 = np.array(df.select("p_m2_1").to_series())  #kN
    p_m2_2 = np.array(df.select("p_m2_2").to_series())  #kN
    p_m2_3 = np.array(df.select("p_m2_3").to_series())  #kN
    p_m2_4 = np.array(df.select("p_m2_4").to_series())  #kN
    p_m2_5 = np.array(df.select("p_m2_5").to_series())  #kN
    m2_1 = np.array(df.select("m2_1").to_series())      #kNm
    m2_2 = np.array(df.select("m2_2").to_series())      #kNm
    m2_3 = np.array(df.select("m2_3").to_series())      #kNm
    m2_4 = np.array(df.select("m2_4").to_series())      #kNm
    m2_5 = np.array(df.select("m2_5").to_series())      #kNm
    p_m3_1 = np.array(df.select("p_m3_1").to_series())  #kN
    p_m3_2 = np.array(df.select("p_m3_2").to_series())  #kN
    p_m3_3 = np.array(df.select("p_m3_3").to_series())  #kN
    p_m3_4 = np.array(df.select("p_m3_4").to_series())  #kN
    p_m3_5 = np.array(df.select("p_m3_5").to_series())  #kN
    m3_1 = np.array(df.select("m3_1").to_series())      #kNm
    m3_2 = np.array(df.select("m3_2").to_series())      #kNm
    m3_3 = np.array(df.select("m3_3").to_series())      #kNm
    m3_4 = np.array(df.select("m3_4").to_series())      #kNm
    m3_5 = np.array(df.select("m3_5").to_series())      #kNm
    
    # Constant

    # Calculations
    n_rd = np.minimum(p_m2_1, p_m3_1)

    m2_rd = np.full(n_rows, 0.0)
    is_in_range = np.logical_and(p >= p_m2_5, p < p_m2_4)
    m2_rd[is_in_range] = m2_5[is_in_range] + (m2_4[is_in_range]  - m2_5[is_in_range]) * (p[is_in_range] - p_m2_5[is_in_range]) / (p_m2_4[is_in_range] - p_m2_5[is_in_range])
    is_in_range = np.logical_and(p >= p_m2_4, p < p_m2_3)
    m2_rd[is_in_range] = m2_4[is_in_range] + (m2_3[is_in_range]  - m2_4[is_in_range]) * (p[is_in_range] - p_m2_4[is_in_range]) / (p_m2_3[is_in_range] - p_m2_4[is_in_range])
    is_in_range = np.logical_and(p >= p_m2_3, p < p_m2_2)
    m2_rd[is_in_range] = m2_3[is_in_range] + (m2_2[is_in_range]  - m2_3[is_in_range]) * (p[is_in_range] - p_m2_3[is_in_range]) / (p_m2_2[is_in_range] - p_m2_3[is_in_range])
    is_in_range = np.logical_and(p >= p_m2_2, p <= p_m2_1)
    m2_rd[is_in_range] = m2_2[is_in_range] + (m2_1[is_in_range]  - m2_2[is_in_range]) * (p[is_in_range] - p_m2_2[is_in_range]) / (p_m2_1[is_in_range] - p_m2_2[is_in_range])

    m3_rd = np.full(n_rows, 0.0)
    is_in_range = np.logical_and(p >= p_m3_5, p < p_m3_4)
    m3_rd[is_in_range] = m3_5[is_in_range] + (m3_4[is_in_range]  - m3_5[is_in_range]) * (p[is_in_range] - p_m3_5[is_in_range]) / (p_m3_4[is_in_range] - p_m3_5[is_in_range])
    is_in_range = np.logical_and(p >= p_m3_4, p < p_m3_3)
    m3_rd[is_in_range] = m3_4[is_in_range] + (m3_3[is_in_range]  - m3_4[is_in_range]) * (p[is_in_range] - p_m3_4[is_in_range]) / (p_m3_3[is_in_range] - p_m3_4[is_in_range])
    is_in_range = np.logical_and(p >= p_m3_3, p < p_m3_2)
    m3_rd[is_in_range] = m3_3[is_in_range] + (m3_2[is_in_range]  - m3_3[is_in_range]) * (p[is_in_range] - p_m3_3[is_in_range]) / (p_m3_2[is_in_range] - p_m3_3[is_in_range])
    is_in_range = np.logical_and(p >= p_m3_2, p <= p_m3_1)
    m3_rd[is_in_range] = m3_2[is_in_range] + (m3_1[is_in_range]  - m3_2[is_in_range]) * (p[is_in_range] - p_m3_2[is_in_range]) / (p_m3_1[is_in_range] - p_m3_2[is_in_range])

    # Output
    df = df.hstack(
        pl.DataFrame({
            "n_rd": n_rd,
            "m2_rd": m2_rd,
            "m3_rd": m3_rd
        })
    )

    return df

def calculate_ur_biaxial_eccentricity(df):
    """Calculate the utilisation ratio biaxial
    """
    outputs = [
        "e_2", "e_3", "x_2", "x_3", "is_biaxial", 
        "ur_p", "alpha", "ur_m2", "ur_m3", "ur_biaxial",
    ]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["b", "h", "p", "n_rd", "m2_rd", "m3_rd", "M_Edtotal2", "M_Edtotal3"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    b = np.array(df.select("b").to_series())                        #mm
    h = np.array(df.select("h").to_series())                        #mm
    p = np.array(df.select("p").to_series())                        #kN
    M_Edtotal2 = np.array(df.select("M_Edtotal2").to_series())      #kNm
    M_Edtotal3 = np.array(df.select("M_Edtotal3").to_series())      #kNm
    n_rd = np.array(df.select("n_rd").to_series())                  #kNm
    m2_rd = np.array(df.select("m2_rd").to_series())                #kNm
    m3_rd = np.array(df.select("m3_rd").to_series())                #kNm
    lambda2 = np.array(df.select("lambda2").to_series())            #-
    lambda3 = np.array(df.select("lambda3").to_series())            #-
    
    # Constant

    # Calculations
    e_2 = ec2.exp5_38_b_e_2(M_Edtotal3, p)
    e_3 = ec2.exp5_38_b_e_2(M_Edtotal2, p)

    x_2 = ec2.exp5_38_b_x(e_2, e_3, h, b)
    x_3 = ec2.exp5_38_b_x(e_3, e_2, b, h)

    # 5.8.9(3): no biaxial check if the slenderness ratios are within 2 (5.38a) and either relative
    # eccentricity ratio is at most 0.2 (5.38b)
    is_biaxial = np.logical_not(np.logical_and(
        np.logical_and((lambda2 / lambda3) <= 2, (lambda3 / lambda2) <= 2),
        np.logical_or(x_2 <= 0.2, x_3 <= 0.2)
    ))

    ur_p = ec2.exp5_39_N_Ed_N_Rd(p, n_rd)

    alpha = ec2.exp5_39_alpha(ur_p)

    ur_m2 = ec2.exp5_39_M_Ed_M_Rd(M_Edtotal2, m2_rd)
    ur_m3 = ec2.exp5_39_M_Ed_M_Rd(M_Edtotal3, m3_rd)

    ur_biaxial = ec2.exp5_39_ur_biaxial(ur_m2, ur_m3, alpha)
    ur_biaxial[np.logical_not(is_biaxial)] = 0

    # Output
    df = df.hstack(
        pl.DataFrame({
            "e_2": e_2,
            "e_3": e_3,
            "x_2": x_2,
            "x_3": x_3,
            "is_biaxial": is_biaxial,
            "ur_p": ur_p,
            "alpha": alpha,
            "ur_m2": ur_m2,
            "ur_m3": ur_m3,
            "ur_biaxial": ur_biaxial,
        })
    )

    return df

def calculate_pv2_add(df):
    """Calculate p and v2 additional
    """

    # Input
    missing_column = _check_inputs(df, ["p_add_m", "v2_add_m", "h"])
    if missing_column: return df
    p_add_m = np.array(df.select("p_add_m").to_series())        #kN
    v2_add_m = np.array(df.select("v2_add_m").to_series())      #kN
    h = np.array(df.select("h").to_series()) / 1000         #m
    
    # Constant
    df.shape[0]

    # Calculations
    p_add = p_add_m * h
    v2_add = v2_add_m * h

    # Output
    df = _drop_outputs(df, ["p_add", "v2_add"])
    df = df.hstack(
        pl.DataFrame({
            "p_add": p_add,
            "v2_add": v2_add,
        })
    )

    return df

def calculate_pv2_design(df):
    """Calculate the design pv2 by adding to the forces
    """
    outputs = ["N_Ed", "V_Ed"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["p_base", "v2_base", "p_add", "v2_add"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    p_base = np.array(df.select("p_base").to_series())      #kN
    v2_base = np.array(df.select("v2_base").to_series())    #kN
    p_add = np.array(df.select("p_add").to_series())        #kN
    v2_add = np.array(df.select("v2_add").to_series())      #kN
    
    # Constant

    # Calculations
    N_Ed = p_base + p_add
    V_Ed = v2_base + v2_add

    # Output
    df = df.hstack(
        pl.DataFrame({
            "N_Ed": N_Ed,
            "V_Ed": V_Ed,
        })
    )

    return df

def calculate_pv3_add(df):
    """Calculate p and v2 additional
    """

    # Input
    missing_column = _check_inputs(df, ["p_add_m", "v3_add_m", "b_w"])
    if missing_column: return df
    p_add_m = np.array(df.select("p_add_m").to_series())        #kN
    v3_add_m = np.array(df.select("v3_add_m").to_series())      #kN
    b_w = np.array(df.select("b_w").to_series()) / 1000         #m
    
    # Constant
    df.shape[0]

    # Calculations
    p_add = p_add_m * b_w
    v3_add = v3_add_m * b_w

    # Output
    df = _drop_outputs(df, ["p_add", "v3_add"])
    df = df.hstack(
        pl.DataFrame({
            "p_add": p_add,
            "v3_add": v3_add,
        })
    )

    return df

def calculate_pv3_design(df):
    """Calculate the design PMM by adding to the forces
    """
    outputs = ["N_Ed", "V_Ed"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["p_base", "v3_base", "p_add", "v3_add", "b_w"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    p_base = np.array(df.select("p_base").to_series())      #kN
    v3_base = np.array(df.select("v3_base").to_series())    #kN
    p_add = np.array(df.select("p_add").to_series())    #kN
    v3_add = np.array(df.select("v3_add").to_series())  #kN
    
    # Constant

    # Calculations
    N_Ed = p_base + p_add
    V_Ed = v3_base + v3_add

    # Output
    df = df.hstack(
        pl.DataFrame({
            "N_Ed": N_Ed,
            "V_Ed": V_Ed,
        })
    )

    return df

def calculate_capacity_shear_concrete(df):
    """Calculate capacity of shear with concrete only, with d to the innermost layer of bars
    """
    outputs = ["d", "k", "rho_l", "A_c", "sigma_cp", "v_min", "V_Rdc", "ur_vc"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    n_rows = df.shape[0]
    columns = ["N_Ed", "V_Ed", "f_ck", "f_cd", "h", "b_w", "dia_vert0", "cover", "dia_horz0"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    N_Ed = np.array(df.select("N_Ed").to_series())                                      #kN
    V_Ed = np.array(df.select("V_Ed").to_series())                                      #kN
    f_ck = np.array(df.select("f_ck").to_series())                                      #MPa
    f_cd = np.array(df.select("f_cd").to_series())                                      #MPa
    h = np.array(df.select("h").to_series())                                            #mm
    b_w = np.array(df.select("b_w").to_series())                                        #mm
    dia_vert = bar_layers(df)                                                          #mm
    cover = np.array(df.select("cover").to_series())                                    #mm
    dia_horz = np.array(df.select("dia_horz0").to_series())                              #mm
    effective_depth_factor = np.array(df.select("effective_depth_factor").to_series())  #-

    # Constant

    # Calculations
    d = rc.d3(h, cover, dia_horz, *dia_vert) #mm
    d *= effective_depth_factor #mm

    A_sl = 0.002827 * b_w * d #mm^2, assumed: rho_l = 0.28%, H12 at 200 mm on each face of a 400 mm wall
    
    k = ec2.exp6_2_a_k(d) #-

    rho_l = ec2.exp6_2_a_rho_l(A_sl, b_w, d) #-

    A_c = rc.A_c(h, b_w) #mm^2

    sigma_cp = ec2.exp6_2_a_sigma_cp(N_Ed / 1e3, A_c / 1e6, f_cd) #MPa

    v_min = ec2.exp6_2_b_v_min(k, f_ck) #MPa

    V_Rdc = ec2.exp6_2_a_V_Rdc(k, rho_l, f_ck, sigma_cp, b_w, d, v_min) #N
    V_Rdc /= 1e3 #kN

    ur_vc = np.full(n_rows, 99.99)
    ur_vc[V_Rdc != 0] = V_Ed[V_Rdc != 0] / V_Rdc[V_Rdc != 0]

    # Output
    df = df.hstack(
        pl.DataFrame({
            "d": d,
            "k": k,
            "rho_l": rho_l,
            "A_c": A_c,
            "sigma_cp": sigma_cp,
            "v_min": v_min,
            "V_Rdc": V_Rdc,
            "ur_vc": ur_vc,
        })
    )

    return df

def calculate_capacity_shear_max(df):
    """Calculate the maximum shear capacity
    """
    outputs = ["v", "z", "V_Rd_max", "ur_vmax"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    n_rows = df.shape[0]
    columns = ["V_Ed", "f_ck", "f_cd", "b_w", "d"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    V_Ed = np.array(df.select("V_Ed").to_series())  #kN
    f_ck = np.array(df.select("f_ck").to_series())  #MPa
    f_cd = np.array(df.select("f_cd").to_series())  #MPa
    b_w = np.array(df.select("b_w").to_series())    #mm
    d = np.array(df.select("d").to_series())        #mm
    
    # Constant
    alpha_cw = np.full(n_rows, 1.0) # for non-prestressed member
    theta = np.full(n_rows, np.pi / 4) # most conservative for required steel

    # Calculations
    v = ec2.exp6_6_N_v(f_ck) #-

    z = rc.z9(d) #mm

    V_Rd_max = ec2.exp6_9_V_Rd_max(alpha_cw, b_w, z, v, f_cd, theta) #kN

    ur_vmax = V_Ed / V_Rd_max

    # Output
    df = df.hstack(
        pl.DataFrame({
            "v": v,
            "z": z,
            "V_Rd_max": V_Rd_max,
            "ur_vmax": ur_vmax,
        })
    )

    return df

def calculate_required_shear_reo_wall(df):
    """Calculate the required shear reo, and the minimum horizontal bars from the vertical (9.6.3)
    """
    outputs = ["theta", "A_sw_s_req", "A_sw_s_min1", "A_sw_s_min2"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["f_ck", "f_yd", "b_w", "d", "s_vert0", "dia_vert0", "V_Ed"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    f_ck = np.array(df.select("f_ck").to_series())          #MPa
    f_yd = np.array(df.select("f_yd").to_series())          #MPa
    b_w = np.array(df.select("b_w").to_series())            #mm
    d = np.array(df.select("d").to_series())                #mm
    s_vert = np.array(df.select("s_vert0").to_series())      #mm
    dia_vert = bar_layers(df)                              #mm
    N_Ed = np.array(df.select("N_Ed").to_series())          #kN
    V_Ed = np.array(df.select("V_Ed").to_series())          #kN
    
    # Constant

    # Calculations
    theta = ec2.exp_6_9_theta(V_Ed, b_w, d, f_ck, N_Ed)

    A_sw_s_req = ec2.exp6_8_A_sw_s(V_Ed, rc.z9(d), f_yd, theta)

    A_sw_s_min1 = ec2.cl9_6_3_1_A_sw_s_min1(s_vert, *dia_vert)

    A_sw_s_min2 = ec2.cl9_6_3_1_A_sw_s_min2(b_w)

    # Output
    df = df.hstack(
        pl.DataFrame({
            "theta": theta,
            "A_sw_s_req": A_sw_s_req,
            "A_sw_s_min1": A_sw_s_min1,
            "A_sw_s_min2": A_sw_s_min2,
        })
    )

    return df

def calculate_required_shear_reo_single_layer_column(df, direction):
    """Calculate the required shear reo
    """
    outputs = ["theta", "A_sw_s_req" + direction, "A_sw_s_min1", "A_sw_s_min2"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["f_ck", "f_yd", "b_w", "d", "V_Ed"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    f_ck = np.array(df.select("f_ck").to_series())          #MPa
    f_yd = np.array(df.select("f_yd").to_series())          #MPa
    b_w = np.array(df.select("b_w").to_series())            #mm
    d = np.array(df.select("d").to_series())                #mm
    N_Ed = np.array(df.select("N_Ed").to_series())          #kN
    V_Ed = np.array(df.select("V_Ed").to_series())          #kN
    
    # Constant

    # Calculations
    theta = ec2.exp_6_9_theta(V_Ed, b_w, d, f_ck, N_Ed)

    A_sw_s_req = ec2.exp6_8_A_sw_s(V_Ed, rc.z9(d), f_yd, theta)

    # Output
    df = df.hstack(
        pl.DataFrame({
            "theta": theta,
            "A_sw_s_req" + direction: A_sw_s_req,
        })
    )

    return df

def calculate_required_shear_reo2(df):
    """Logic for calculating required shear reo in direction 2
    """
    outputs = ["A_sw_s2"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    columns = ["A_sw_s_req", "A_sw_s_min1", "A_sw_s_min2"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    A_sw_s_req = np.array(df.select("A_sw_s_req").to_series())      #mm
    A_sw_s_min1 = np.array(df.select("A_sw_s_min1").to_series())    #mm
    A_sw_s_min2 = np.array(df.select("A_sw_s_min2").to_series())    #mm
    
    # Constant

    # Calculations
    A_sw_s2 = np.maximum(
        A_sw_s_req,
        np.maximum(A_sw_s_min1, A_sw_s_min2)
    )

    # Output
    df = df.hstack(
        pl.DataFrame({
            "A_sw_s2": A_sw_s2,
        })
    )

    return df

def calculate_required_shear_reo3(df, ur_limit):
    """Logic for calculating required shear reo in direction 2
    """
    outputs = ["A_sw_s3"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    n_rows = df.shape[0]
    columns = ["V_Ed", "V_Rdc", "A_sw_s_req", "A_sw_s_min1", "A_sw_s_min2"]
    for column in columns:
        if column not in df.columns:
            print("Missing column", column)
            return df

    # Input
    b_w = np.array(df.select("b_w").to_series())                    #mm
    V_Ed = np.array(df.select("V_Ed").to_series())                  #kN
    V_Rdc = np.array(df.select("V_Rdc").to_series())                #kN
    A_sw_s_req = np.array(df.select("A_sw_s_req").to_series())      #mm
    A_sw_s_min1 = np.array(df.select("A_sw_s_min1").to_series())    #mm
    A_sw_s_min2 = np.array(df.select("A_sw_s_min2").to_series())    #mm
    
    # Constant

    # Calculations
    A_sw_s3 = np.full(n_rows, 0.0)
    A_sw_s3[V_Ed > ur_limit * V_Rdc] = np.maximum(
        A_sw_s_req[V_Ed > ur_limit * V_Rdc],
        np.maximum(A_sw_s_min1[V_Ed > ur_limit * V_Rdc], A_sw_s_min2[V_Ed > ur_limit * V_Rdc])
    )
    A_sw_s3 /= (b_w / 1000)

    # Output
    df = df.hstack(
        pl.DataFrame({
            "A_sw_s3": A_sw_s3,
        })
    )

    return df

# ----- beam -----

def calculate_beam_design(df):
    """Calculate the design beam forces
    """

    # Input
    missing_column = _check_inputs(df, ["p_base", "m3_base", "v2_base", "p_add", "m3_add", "v2_add"])
    if missing_column: return df   
    
    # Constant
    p_base = np.array(df.select("p_base").to_series())      #kN
    m3_base = np.array(df.select("m3_base").to_series())    #kNm
    v2_base = np.array(df.select("v2_base").to_series())    #kN
    p_add = np.array(df.select("p_add").to_series())      #kN
    m3_add = np.array(df.select("m3_add").to_series())    #kNm
    v2_add = np.array(df.select("v2_add").to_series())    #kN

    # Calculations
    p = p_base + p_add
    m3 = m3_base + m3_add
    v2 = v2_base + v2_add

    # Output
    df = _drop_outputs(df, ["p", "m3", "v2"])
    df = df.hstack(
        pl.DataFrame({
            "p": p,
            "m3": m3,
            "v2": v2,
        })
    )

    return df

def calculate_x_max(df):
    """Calculate the maximum neutral axis depth
    """

    # Input
    missing_column = _check_inputs(df, ["f_ck"])
    if missing_column: return df
    f_ck = np.array(df.select("f_ck").to_series())          #MPa
    
    # Constant
    # n_rows = df.shape[0]

    # Calculations
    epsilon_cu3 = ec2.tb3_1_epsilon_cu3(f_ck)

    k1 = ec2.tb3_1_k1(f_ck)

    k2 = ec2.tb3_1_k2(epsilon_cu3)

    x_max = ec2.tb3_1_x_max(k1, k2)

    lambda1 = ec2.exp3_20_lambda(f_ck)
    eta = ec2.exp3_22_eta(f_ck)

    # Output
    df = _drop_outputs(df, ["epsilon_cu3", "k1", "k2", "x_max", "lambda", "eta"])
    df = df.hstack(
        pl.DataFrame({
            "epsilon_cu3": epsilon_cu3,
            "k1": k1,
            "k2": k2,
            "x_max": x_max,
            "lambda": lambda1,
            "eta": eta,
        })
    )

    return df

def calculate_beam_reo_centroid(df):
    """Calculate the centroid and depth based on current flexure reo
    """

    # Input
    missing_column = _check_inputs(
        df, 
        [
            "h", "cover", "dia_link",
            "dia_top00", "dia_bot00", "n_top00", "n_bot00",
            "dia_top01", "dia_bot01", "n_top01", "n_bot01",
            "dia_top02", "dia_bot02", "n_top02", "n_bot02",
            "dia_top03", "dia_bot03", "n_top03", "n_bot03",
            "dia_top04", "dia_bot04", "n_top04", "n_bot04",
            "dia_top05", "dia_bot05", "n_top05", "n_bot05",
            "dia_top06", "dia_bot06", "n_top06", "n_bot06",
            "dia_top07", "dia_bot07", "n_top07", "n_bot07",
            "dia_top08", "dia_bot08", "n_top08", "n_bot08",
            "dia_top09", "dia_bot09", "n_top09", "n_bot09",
        ]
    )
    if missing_column: return df, None
    h = np.array(df.select("h").to_series())                        #mm
    cover = np.array(df.select("cover").to_series())                #mm
    dia_link = np.array(df.select("dia_link").to_series())          #mm
    dia_top00 = np.array(df.select("dia_top00").to_series())        #mm
    dia_top01 = np.array(df.select("dia_top01").to_series())        #mm
    dia_top02 = np.array(df.select("dia_top02").to_series())        #mm
    dia_top03 = np.array(df.select("dia_top03").to_series())        #mm
    dia_top04 = np.array(df.select("dia_top04").to_series())        #mm
    dia_top05 = np.array(df.select("dia_top05").to_series())        #mm
    dia_top06 = np.array(df.select("dia_top06").to_series())        #mm
    dia_top07 = np.array(df.select("dia_top07").to_series())        #mm
    dia_top08 = np.array(df.select("dia_top08").to_series())        #mm
    dia_top09 = np.array(df.select("dia_top09").to_series())        #mm
    dia_bot00 = np.array(df.select("dia_bot00").to_series())        #mm
    dia_bot01 = np.array(df.select("dia_bot01").to_series())        #mm
    dia_bot02 = np.array(df.select("dia_bot02").to_series())        #mm
    dia_bot03 = np.array(df.select("dia_bot03").to_series())        #mm
    dia_bot04 = np.array(df.select("dia_bot04").to_series())        #mm
    dia_bot05 = np.array(df.select("dia_bot05").to_series())        #mm
    dia_bot06 = np.array(df.select("dia_bot06").to_series())        #mm
    dia_bot07 = np.array(df.select("dia_bot07").to_series())        #mm
    dia_bot08 = np.array(df.select("dia_bot08").to_series())        #mm
    dia_bot09 = np.array(df.select("dia_bot09").to_series())        #mm
    n_top00 = np.array(df.select("n_top00").to_series())            #-
    n_top01 = np.array(df.select("n_top01").to_series())            #-
    n_top02 = np.array(df.select("n_top02").to_series())            #-
    n_top03 = np.array(df.select("n_top03").to_series())            #-
    n_top04 = np.array(df.select("n_top04").to_series())            #-
    n_top05 = np.array(df.select("n_top05").to_series())            #-
    n_top06 = np.array(df.select("n_top06").to_series())            #-
    n_top07 = np.array(df.select("n_top07").to_series())            #-
    n_top08 = np.array(df.select("n_top08").to_series())            #-
    n_top09 = np.array(df.select("n_top09").to_series())            #-
    n_bot00 = np.array(df.select("n_bot00").to_series())            #-
    n_bot01 = np.array(df.select("n_bot01").to_series())            #-
    n_bot02 = np.array(df.select("n_bot02").to_series())            #-
    n_bot03 = np.array(df.select("n_bot03").to_series())            #-
    n_bot04 = np.array(df.select("n_bot04").to_series())            #-
    n_bot05 = np.array(df.select("n_bot05").to_series())            #-
    n_bot06 = np.array(df.select("n_bot06").to_series())            #-
    n_bot07 = np.array(df.select("n_bot07").to_series())            #-
    n_bot08 = np.array(df.select("n_bot08").to_series())            #-
    n_bot09 = np.array(df.select("n_bot09").to_series())            #-          
    
    # Constant
    df.shape[0]

    # Calculations
    c_top = rc.centroid(
        cover, dia_link, 
        dia_top00, n_top00,
        dia_top01, n_top01,
        dia_top02, n_top02,
        dia_top03, n_top03,
        dia_top04, n_top04,
        dia_top05, n_top05,
        dia_top06, n_top06,
        dia_top07, n_top07,
        dia_top08, n_top08,
        dia_top09, n_top09,
    )
    c_bot = rc.centroid(
        cover, dia_link, 
        dia_bot00, n_bot00,
        dia_bot01, n_bot01,
        dia_bot02, n_bot02,
        dia_bot03, n_bot03,
        dia_bot04, n_bot04,
        dia_bot05, n_bot05,
        dia_bot06, n_bot06,
        dia_bot07, n_bot07,
        dia_bot08, n_bot08,
        dia_bot09, n_bot09,
    )

    d_top = rc.d(h, c_top)
    d_bot = rc.d(h, c_bot)

    # Output
    df = _drop_outputs(
        df, 
        [
            "c_top", "c_bot", "d_top", "d_bot", 
        ]
    )
    df = df.hstack(
        pl.DataFrame({
            "c_top": c_top,
            "c_bot": c_bot,
            "d_top": d_top,
            "d_bot": d_bot,
        })
    )

    return df

def calculate_beam_flexure_reo(df):
    """Calculate the required beam flexure reo
    """

    # Input
    missing_column = _check_inputs(
        df, 
        [
            "c_top", "c_bot", "d_top", "d_bot", 
            "f_ck", "f_yd", "b", "h", "cover", "dia_link", "A_top_req", "A_bot_req", "m3",
            "x_max", "lambda", "eta",
        ]
    )
    if missing_column: return df  
    c_top = np.array(df.select("c_top").to_series())                #mm
    c_bot = np.array(df.select("c_bot").to_series())                #mm
    d_top = np.array(df.select("d_top").to_series())                #mm
    d_bot = np.array(df.select("d_bot").to_series())                #mm
    f_ck = np.array(df.select("f_ck").to_series())                  #MPa
    f_yd = np.array(df.select("f_yd").to_series())                  #MPa
    b = np.array(df.select("b").to_series())                        #mm
    A_top_req = np.array(df.select("A_top_req").to_series())        #mm2
    A_bot_req = np.array(df.select("A_bot_req").to_series())        #mm2
    m3 = np.array(df.select("m3").to_series())                      #kNm
    x_max = np.array(df.select("x_max").to_series())                #-
    lambda1 = np.array(df.select("lambda").to_series())             #-
    eta = np.array(df.select("eta").to_series())                    #-
    epsilon_cu3 = np.array(df.select("epsilon_cu3").to_series())    #-
    
    # Constant
    n_rows = df.shape[0]

    # Calculations
    is_bot = (m3 >= 0)
    is_top = np.logical_not(is_bot)
    k = np.full(n_rows, 0.0)
    k[is_bot] = ec2.cl6_1_k(np.abs(m3[is_bot]), b[is_bot], d_bot[is_bot], f_ck[is_bot])
    k[is_top] = ec2.cl6_1_k(np.abs(m3[is_top]), b[is_top], d_top[is_top], f_ck[is_top])
    x_1_d = x_max.copy()
    k_1 = ec2.cl6_1_k_1(lambda1, eta)
    k_2 = ec2.cl6_1_k_2(lambda1)
    k_prime = ec2.cl6_1_k_prime(k_1, k_2, x_1_d)

    A_top_req_new = np.full(n_rows, 0.0)
    A_bot_req_new = np.full(n_rows, 0.0)

    is_singly = k < k_prime
    c = ec2.cl6_1_c(eta)
    z = np.full(n_rows, 0.0)
    is_bot_singly = np.logical_and(is_singly, is_bot)
    is_top_singly = np.logical_and(is_singly, is_top)
    z[is_bot_singly] = ec2.cl6_1_z(
        d_bot[is_bot_singly], 
        c[is_bot_singly], 
        k[is_bot_singly],
    )
    z[is_top_singly] = ec2.cl6_1_z(
        d_top[is_top_singly], 
        c[is_top_singly],
        k[is_top_singly],
    )
    A_bot_req_new[is_bot_singly] = ec2.cl6_1_A_s_singly(
        np.abs(m3[is_bot_singly]),
        f_yd[is_bot_singly],
        z[is_bot_singly],
    )
    A_top_req_new[is_top_singly] = ec2.cl6_1_A_s_singly(
        np.abs(m3[is_top_singly]),
        f_yd[is_top_singly],
        z[is_top_singly],
    )

    is_doubly = np.logical_not(is_singly)
    d = d_bot.copy()
    d[is_top] = d_top[is_top]
    d_prime = c_top.copy()
    d_prime[is_top] = c_bot[is_top]
    x = ec2.cl6_1_x(x_max, d)
    M_1 = ec2.cl6_1_M_1(k_prime, b, d, f_ck) #Nmm
    M_2 = ec2.cl6_1_M_2(np.abs(m3), M_1) #Nmm
    f_sc = ec2.cl6_1_f_sc(f_yd, x, d_prime, epsilon_cu3)
    A_sc = ec2.cl6_1_A_sc(M_2, f_sc, d, d_prime)
    A_s = ec2.cl6_1_A_s(k_1, b, x, f_ck, A_sc, f_sc, f_yd)
    is_bot_doubly = np.logical_and(is_doubly, is_bot)
    is_top_doubly = np.logical_and(is_doubly, is_top)
    A_top_req_new[is_bot_doubly] = A_sc[is_bot_doubly]
    A_bot_req_new[is_bot_doubly] = A_s[is_bot_doubly]
    A_top_req_new[is_top_doubly] = A_s[is_top_doubly]
    A_bot_req_new[is_top_doubly] = A_sc[is_top_doubly]

    is_converged = np.logical_and(A_top_req == A_top_req_new, A_bot_req == A_bot_req_new)

    # Output
    df = _drop_outputs(
        df,
        [
            "k", "x_1_d", "k_1", "k_2", "k_prime", 
            "c", "z", 
            "x", "M_1", "M_2", "f_sc",
            "A_top_req", "A_bot_req", "is_converged"
        ]
    )
    df = df.hstack(
        pl.DataFrame({
            "k": k,
            "x_1_d": x_1_d,
            "k_1": k_1,
            "k_2": k_2,
            "k_prime": k_prime,
            "c": c,
            "z": z,
            "x": x,
            "M_1": M_1,
            "M_2": M_2,
            "f_sc": f_sc,
            "A_top_req": A_top_req_new,
            "A_bot_req": A_bot_req_new,
            "is_converged": is_converged,
        })
    )

    return df

def calculate_beam_shear_concrete(df):
    """Calculate the beam concrete shear capacity and ur
    """

    # Input
    missing_column = _check_inputs(
        df, 
        [
            "p", "m3", "v2", "f_ck", "f_cd",
            "h", "b", "d_top", "d_bot", "A_top_prov", "A_bot_prov",
        ]
    )
    if missing_column: return df
    N_Ed = np.array(df.select("p").to_series())                 #kN
    M_Ed = np.array(df.select("m3").to_series())                #kNm
    V_Ed = np.array(df.select("v2").to_series())                #kN
    f_ck = np.array(df.select("f_ck").to_series())              #MPa
    f_cd = np.array(df.select("f_cd").to_series())              #MPa
    h = np.array(df.select("h").to_series())                    #mm
    b = np.array(df.select("b").to_series())                    #mm
    d_top = np.array(df.select("d_top").to_series())            #mm
    d_bot = np.array(df.select("d_bot").to_series())            #mm
    A_top_prov = np.array(df.select("A_top_prov").to_series())            #mm
    A_bot_prov = np.array(df.select("A_bot_prov").to_series())            #mm
    
    # Constant
    n_rows = df.shape[0]

    # Calculations
    d = d_bot.copy()
    d[M_Ed < 0] = d_top[M_Ed < 0]

    A_sl = A_bot_prov.copy()
    d[M_Ed < 0] = A_top_prov[M_Ed < 0]
    
    k = ec2.exp6_2_a_k(d) #-

    rho_l = ec2.exp6_2_a_rho_l(A_sl, b, d) #-

    A_c = rc.A_c(h, b) #mm^2

    sigma_cp = ec2.exp6_2_a_sigma_cp(N_Ed / 1e3, A_c / 1e6, f_cd) #MPa

    v_min = ec2.exp6_2_b_v_min(k, f_ck) #MPa

    V_Rdc = ec2.exp6_2_a_V_Rdc(k, rho_l, f_ck, sigma_cp, b, d, v_min) #N
    V_Rdc /= 1e3 #kN

    ur_vc = np.full(n_rows, 99.99)
    ur_vc[V_Rdc != 0] = np.abs(V_Ed[V_Rdc != 0]) / V_Rdc[V_Rdc != 0]

    # Output
    df = _drop_outputs(df, ["k_c", "rho_l", "A_c", "sigma_cp", "v_min", "V_Rdc", "ur_vc"])
    df = df.hstack(
        pl.DataFrame({
            "k_c": k,
            "rho_l": rho_l,
            "A_c": A_c,
            "sigma_cp": sigma_cp,
            "v_min": v_min,
            "V_Rdc": V_Rdc,
            "ur_vc": ur_vc,
        })
    )

    return df

def calculate_beam_shear_max(df):
    """Calculate the maximum shear capacity and ur
    """

    # Input
    missing_column = _check_inputs(df, ["m3", "v2", "f_ck", "f_cd", "b", "d_top", "d_bot"])
    if missing_column: return 
    M_Ed = np.array(df.select("m3").to_series())                #kNm
    V_Ed = np.array(df.select("v2").to_series())  #kN
    f_ck = np.array(df.select("f_ck").to_series())  #MPa
    f_cd = np.array(df.select("f_cd").to_series())  #MPa
    b = np.array(df.select("b").to_series())    #mm
    d_top = np.array(df.select("d_top").to_series())        #mm
    d_bot = np.array(df.select("d_bot").to_series())        #mm            
    
    # Constant
    n_rows = df.shape[0]
    alpha_cw = np.full(n_rows, 1.0) # for non-prestressed member
    theta = np.full(n_rows, np.pi / 4) # most conservative for required steel

    # Calculations
    d = d_bot.copy()
    d[M_Ed < 0] = d_top[M_Ed < 0]

    v = ec2.exp6_6_N_v(f_ck) #-

    z = rc.z9(d) #mm

    V_Rd_max = ec2.exp6_9_V_Rd_max(alpha_cw, b, d, v, f_cd, theta) #kN

    ur_vmax = np.abs(V_Ed) / V_Rd_max

    # Output
    df = _drop_outputs(df, ["v", "z_max", "V_Rd_max", "ur_vmax"])
    df = df.hstack(
        pl.DataFrame({
            "v": v,
            "z_max": z,
            "V_Rd_max": V_Rd_max,
            "ur_vmax": ur_vmax,
        })
    )

    return df

def calculate_beam_shear_reo(df):
    """Calculate the shear reo required
    """

    # Input
    missing_column = _check_inputs(df, ["f_ck", "f_yd", "b", "d_top", "d_bot", "p", "v2", "m3"])
    if missing_column: return df
    f_ck = np.array(df.select("f_ck").to_series())          #MPa
    f_yd = np.array(df.select("f_yd").to_series())          #MPa
    b = np.array(df.select("b").to_series())            #mm
    d_top = np.array(df.select("d_top").to_series())                #mm
    d_bot = np.array(df.select("d_bot").to_series())                #mm
    N_Ed = np.array(df.select("p").to_series())          #kN
    V_Ed = np.array(df.select("v2").to_series())          #kN
    M_Ed = np.array(df.select("m3").to_series())                #kNm
    
    # Constant
    df.shape[0]

    # Calculations
    d = d_bot.copy()
    d[M_Ed < 0] = d_top[M_Ed < 0]

    theta = ec2.exp_6_9_theta(np.abs(V_Ed), b, d, f_ck, N_Ed)

    A_sw_s_req = ec2.exp6_8_A_sw_s(np.abs(V_Ed), d, f_yd, theta)

    # Output
    df = _drop_outputs(df, ["theta", "A_sw_s_req"])
    df = df.hstack(
        pl.DataFrame({
            "theta": theta,
            "A_sw_s_req": A_sw_s_req,
        })
    )

    return df

# ----- slab -----

def calculate_punching_shear_v_c(df):
    """Calculate v_c in accordance with exp 6.47
    """

    # Validation
    pa.DataFrameSchema({
        "d": pa.Column(pl.Float64),
        "rho_ly": pa.Column(pl.Float64),
        "rho_lz": pa.Column(pl.Float64),
        "f_ck": pa.Column(pl.Float64),
        "sigma_cy": pa.Column(pl.Float64),
        "sigma_cz": pa.Column(pl.Float64),
    }).validate(df)
    
    # Input
    d = df["d"].to_numpy()
    rho_ly = df["rho_ly"].to_numpy()
    rho_lz = df["rho_lz"].to_numpy()
    f_ck = df["f_ck"].to_numpy()
    sigma_cy = df["sigma_cy"].to_numpy()
    sigma_cz = df["sigma_cz"].to_numpy()

    # Constant
    # n_rows = df.shape[0]

    # Calculation
    k = ec2.exp6_2_a_k(d)
    rho_l = ec2.exp6_47_rho_l(rho_ly, rho_lz)
    sigma_cp = ec2.exp6_47_sigma_cp(sigma_cy, sigma_cz)
    v_min = ec2.exp6_2_b_v_min(k, f_ck)
    v_Rd_c = ec2.exp6_47_v_Rd_c(k, rho_l, f_ck, sigma_cp, v_min)

    # Output
    return (
        df
        .with_columns(
            pl.Series("k", k, dtype=pl.Float64),
            pl.Series("rho_l", rho_l, dtype=pl.Float64),
            pl.Series("sigma_cp", sigma_cp, dtype=pl.Float64),
            pl.Series("v_min", v_min, dtype=pl.Float64),
            pl.Series("v_Rd_c", v_Rd_c, dtype=pl.Float64),
        )
    )

def calculate_punching_shear_v_max(df):
    """Calculate the v max
    """

    # Validation
    pa.DataFrameSchema({
        "d": pa.Column(pl.Float64),
        "f_ck": pa.Column(pl.Float64),
        "h": pa.Column(pl.Float64),
    }).validate(df)
    
    # Input
    d = df["d"].to_numpy()
    f_ck = df["f_ck"].to_numpy()
    h = df["h"].to_numpy()

    # Constant
    n_rows = df.shape[0]
    alpha_cw = np.full(n_rows, 1.0) # for non-prestressed member
    b_w = np.full(n_rows, 1000.0) # check for per m width
    theta = np.full(n_rows, np.pi / 4) # most conservative for required steel

    # Calculation
    f_cd = ec2.exp3_15_f_cd(f_ck) #MPa
    v = ec2.exp6_6_N_v(f_ck) #-
    z = rc.z9(d) #mm
    V_Rd_max = ec2.exp6_9_V_Rd_max(alpha_cw, b_w, z, v, f_cd, theta) #kN
    v_Rd_max = V_Rd_max / b_w / h * 1000 #MPa

    # Output
    return (
        df
        .with_columns(
            pl.Series("f_cd", f_cd, dtype=pl.Float64),
            pl.Series("v", v, dtype=pl.Float64),
            pl.Series("z", z, dtype=pl.Float64),
            pl.Series("V_Rd_max", V_Rd_max, dtype=pl.Float64),
            pl.Series("v_Rd_max", v_Rd_max, dtype=pl.Float64),
        )
    )

def calculate_punching_shear_v_cs(df):
    """Calculate the shear capacity with steel reinforcement for punching shear
    """

    # Validation
    pa.DataFrameSchema({
        "v_Rd_c": pa.Column(pl.Float64),
        "d": pa.Column(pl.Float64),
        "s_r": pa.Column(pl.Float64),
        "A_sw": pa.Column(pl.Float64),
        "u_1": pa.Column(pl.Float64),
    }).validate(df)
    
    # Input
    v_Rd_c = df["v_Rd_c"].to_numpy()
    d = df["d"].to_numpy()
    s_r = df["s_r"].to_numpy()
    A_sw = df["A_sw"].to_numpy()
    u_1 = df["u_1"].to_numpy()

    # Constant
    n_rows = df.shape[0]
    f_yk = np.full(n_rows, F_YK) #MPa
    alpha = np.full(n_rows, np.pi/2) #rad

    # Calculation
    f_yd = ec2.cl3_2_f_yd(f_yk) #MPa
    f_ywd_ef = ec2.exp6_52_f_ywd_ef(d, f_yd) #MPa
    v_Rd_cs = ec2.exp6_52_v_Rd_cs(v_Rd_c, s_r, A_sw, f_ywd_ef, u_1, alpha)

    # Output
    return (
        df
        .with_columns(
            pl.Series("f_ywd", f_yd, dtype=pl.Float64),
            pl.Series("f_ywd_ef", f_ywd_ef, dtype=pl.Float64),
            pl.Series("v_Rd_cs", v_Rd_cs, dtype=pl.Float64),
        )
    )