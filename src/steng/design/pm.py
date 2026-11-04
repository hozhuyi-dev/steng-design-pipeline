"""Simplified force-moment interaction curves of walls and columns, by strain compatibility over the section
"""

import numpy as np
import pandera.polars as pa
import polars as pl
from tqdm import tqdm

SPACING_MIN = 32
CONCRETE_GAMMA = 1.5
ALPHA_CC = 0.85
REO_GAMMA = 1.15
REO_GRADE = 500     #MPa
REO_E = 200000      #Mpa
STRIP_MIN_N = 100
STRIP_MAX_D = 20    #mm
ITER_TOL = 1e-3     #kN
ITER_MAX = 1000

CURVE = [
    *[f"p_m2_{k}" for k in range(1, 6)], *[f"m2_{k}" for k in range(1, 6)],
    *[f"p_m3_{k}" for k in range(1, 6)], *[f"m3_{k}" for k in range(1, 6)],
]

def calculate_pm_wall(df, n_layers=6):
    """Generate the simplified force-moment curves of wall trials with up to n_layers layers of bars

    The curves come for each direction of each unique section, dia_vert0 the outermost layer:
    1 pure compression, 2 zero tension, 3 balance, 4 pure bending, 5 pure tension.
    """

    dia_vert = [f"dia_vert{k}" for k in range(n_layers)]

    # Validation
    pa.DataFrameSchema({
        "grade_min": pa.Column(pl.UInt32),
        "thickness_min": pa.Column(pl.Float64),
        "length": pa.Column(pl.Float64),
        "cover": pa.Column(pl.Int32),
        "dia_horz0": pa.Column(pl.Int32),
        "s_vert0": pa.Column(pl.Int32),
        **{column: pa.Column(pl.Int32) for column in dia_vert},
    }).validate(df)

    # Input
    columns = ["grade_min", "thickness_min", "length", "cover", "dia_horz0", "s_vert0", *dia_vert]
    df_unique = df.select(columns).unique()

    # Calculation
    curves = []
    for f_ck, thickness, length, cover, dia_horz, s_vert, *dias in tqdm(
        df_unique.iter_rows(), total=df_unique.height, desc="Calculating pm curve simplified"
    ):
        length_vert = length - 2 * cover - 2 * dia_horz - dias[0]
        n_vert = np.ceil(length_vert / s_vert) + 1
        dia_reo2, d_reo2 = _generate_reo_top_and_bottom(thickness, cover, dia_horz, n_vert, dias)
        dia_reo3, d_reo3 = _generate_reo_side(length, cover, dia_horz, n_vert, dias)
        curves.append(_calculate_pm_curve(f_ck, length, thickness, d_reo2, dia_reo2, d_reo3, dia_reo3))

    # Output
    return _update_curves(df, df_unique, curves, columns)

def calculate_pm_column(df, n_layers=3):
    """Generate the simplified force-moment curves of column trials with up to n_layers layers of bars

    Each layer has n_vert0 bars around the perimeter, dia_vert0 the outermost. The points on each
    curve are as calculate_pm_wall's.
    """

    dia_vert = [f"dia_vert{k}" for k in range(n_layers)]

    # Validation
    pa.DataFrameSchema({
        "grade": pa.Column(pl.UInt32),
        "t3": pa.Column(pl.Float64),
        "t2": pa.Column(pl.Float64),
        "cover": pa.Column(pl.Int32),
        "dia_horz0": pa.Column(pl.Int32),
        "n_vert0": pa.Column(pl.Int32),
        "s_vert0": pa.Column(pl.Int32),
        **{column: pa.Column(pl.Int32) for column in dia_vert},
    }).validate(df)

    # Input
    columns = ["grade", "t3", "t2", "cover", "dia_horz0", "n_vert0", "s_vert0", *dia_vert]
    df_unique = df.select(columns).unique()

    # Calculation
    curves = []
    for f_ck, t3, t2, cover, dia_horz, n_vert, s_vert, *dias in tqdm(
        df_unique.iter_rows(), total=df_unique.height, desc="Calculating pm curve simplified"
    ):
        # M2 bends across the width t2, and M3 across the depth t3
        dia_reo2, d_reo2 = _generate_reo_column(t3, t2, cover, dia_horz, n_vert, dias, s_vert)
        dia_reo3, d_reo3 = _generate_reo_column(t2, t3, cover, dia_horz, n_vert, dias, s_vert)
        curves.append(_calculate_pm_curve(f_ck, t3, t2, d_reo2, dia_reo2, d_reo3, dia_reo3))

    # Output
    return _update_curves(df, df_unique, curves, columns)

def _update_curves(df, df_unique, curves, columns):
    """Return df with the curve points of each unique section"""

    df_unique = df_unique.hstack(pl.DataFrame(curves, schema={c: pl.Float64 for c in CURVE}, orient="row"))

    return (
        df
        .with_columns(pl.lit(None).cast(pl.Float64).alias(column) for column in CURVE)
        .update(df_unique, on=columns, how="left")
    )

def _generate_reo_top_and_bottom(h, cover, dia_horz, n, dia_vert):
    """Return the diameters and depths of the bars on both faces, n bars in each layer"""

    s_top = np.maximum(dia_vert[0], SPACING_MIN)
    d = cover + dia_horz + dia_vert[0] / 2
    dia_reo = list(np.full(int(n), dia_vert[0]))
    d_reo = list(np.full(int(n), d))
    for outer, inner in zip(dia_vert, dia_vert[1:]):
        if inner > 0:
            d += outer / 2 + s_top + inner / 2
            dia_reo += list(np.full(int(n), inner))
            d_reo += list(np.full(int(n), d))

    s_bot = np.maximum(dia_vert[0], SPACING_MIN)
    d = h - (cover + dia_horz + dia_vert[0] / 2)
    dia_reo += list(np.full(int(n), dia_vert[0]))
    d_reo += list(np.full(int(n), d))
    for outer, inner in zip(dia_vert, dia_vert[1:]):
        if inner > 0:
            d -= (outer / 2 + s_bot + inner / 2)
            dia_reo += list(np.full(int(n), inner))
            d_reo += list(np.full(int(n), d))

    return np.array(dia_reo), np.array(d_reo)

def _generate_reo_side(h, cover, dia_horz, n, dia_vert):
    """Return the diameters and depths of the bars along a wall of length h, n bars in each layer on each face"""

    d_first = cover + dia_horz + dia_vert[0] / 2
    length_reo = h - 2 * d_first
    s_reo = length_reo / (n - 1)

    dia_reo = []
    d_reo = []
    for k, dia in enumerate(dia_vert):
        if k == 0 or dia > 0:
            dia_reo += [dia] * int(n) * 2
            d_reo += list(np.arange(d_first, h, s_reo)) * 2

    return np.array(dia_reo), np.array(d_reo)

def _generate_reo_column(b, h, cover, dia_horz, n, dia_vert, s):
    """Return the diameters and depths of the bars of a b x h column, n bars in each layer around the perimeter"""

    n_b = int(np.ceil((b - 2 * (cover + dia_horz) - dia_vert[0]) / s)) + 1
    n_h = int(np.ceil((h - 2 * (cover + dia_horz) - dia_vert[0]) / s)) + 1

    d_reo = []
    dia_reo = np.array([], dtype=int)
    inset = 0  # the bars and gaps of the layers outside this one, on each side
    for k, dia in enumerate(dia_vert):
        if k > 0:
            inset += dia_vert[k - 1] + np.maximum(dia_vert[k - 1], SPACING_MIN)
            if dia == 0:
                continue
        s_reo = float((h - 2 * (cover + dia_horz) - 2 * inset - dia) / (n_h - 1))
        d_first = cover + dia_horz + inset + dia / 2
        d_reo += [d_first] * (n_b - 2)
        d_reo += [h - d_first] * (n_b - 2)
        d_reo += list(np.arange(d_first, h - d_first + 1, s_reo)) * 2
        dia_reo = np.concatenate((dia_reo, np.full(n, dia)))

    return dia_reo, np.array(d_reo)

def _calculate_pm_curve(f_ck, b, h, d_reo2, dia_reo2, d_reo3, dia_reo3):
    """A function to generate the following points on the force moment
    interaction curve:

    1 Pure compression
    2 Zero tension point
    3 Balance (max bending)
    4 Pure bending (iterative process)
    5 Pure tension
    """

    # Calculate material parameters
    f_cd = f_ck / CONCRETE_GAMMA * ALPHA_CC
    if f_ck < 50:
        eps_c2 = 2
        eps_cu2 = 3.5
        n = 2
    else:
        eps_c2 = 2.0 + 0.085 * (f_ck - 50) ** 0.53
        eps_cu2 = 2.6 + 35 * ((90 - f_ck) / 100) ** 4
        n = 1.4 + 23.4 * ((90 - f_ck) / 100) ** 4

    f_yd = REO_GRADE / REO_GAMMA
    eps_s = f_yd / REO_E * 1000

    # ---- M2 direction

    strip_n = max(h / STRIP_MAX_D, STRIP_MIN_N)
    delta_h = h / strip_n
    d_concrete = np.arange(h / strip_n / 2, h, h / strip_n)

    # 1 Pure compression
    eps_top = eps_cu2
    eps_bot = eps_cu2

    force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo2, dia_reo2)
    p_m2_1 = force
    m2_1 = 0.0

    # 2 Zero tension point
    eps_top = eps_cu2
    eps_bot = 0

    force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo2, dia_reo2)
    p_m2_2 = force
    m2_2 = moment

    # 3 Balance
    eps_top = eps_cu2
    eps_bot = -eps_s / np.max(d_reo2) * h

    force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo2, dia_reo2)
    p_m2_3 = force
    m2_3 = moment

    # 4 Pure bending
    eps_top = eps_cu2
    centroid_upper = h
    centroid_lower = h / 100

    force = 1
    j = 0
    while (abs(force) > ITER_TOL) and (j < ITER_MAX):
        j += 1
        centroid = (centroid_upper + centroid_lower) / 2
        eps_bot = eps_top - eps_top * h / centroid

        force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo2, dia_reo2)

        if force > 0:
            centroid_upper = centroid
        else:
            centroid_lower = centroid
    p_m2_4 = 0.0
    m2_4 = moment

    # 5 Pure tension
    eps_top = -eps_s
    eps_bot = -eps_s

    force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo2, dia_reo2)
    p_m2_5 = force
    m2_5 = 0.0

    # ---- M3 direction

    # Initialise concrete
    b_ori = b
    b = h
    h = b_ori
    strip_n = max(h / STRIP_MAX_D, STRIP_MIN_N)
    delta_h = h / strip_n
    d_concrete = np.arange(h / strip_n / 2, h, h / strip_n)

    # 1 Pure compression
    eps_top = eps_cu2
    eps_bot = eps_cu2

    force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo3, dia_reo3)
    p_m3_1 = force
    m3_1 = 0.0

    # 2 Zero tension point
    eps_top = eps_cu2
    eps_bot = 0

    force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo3, dia_reo3)
    p_m3_2 = force
    m3_2 = moment

    # 3 Balance
    eps_top = eps_cu2
    eps_bot = -eps_s / np.max(d_reo3) * h

    force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo3, dia_reo3)
    p_m3_3 = force
    m3_3 = moment

    # 4 Pure bending
    eps_top = eps_cu2
    centroid_upper = h
    centroid_lower = h / 100

    force = 1
    j = 0
    while (abs(force) > ITER_TOL) and (j < ITER_MAX):
        j += 1
        centroid = (centroid_upper + centroid_lower) / 2
        eps_bot = eps_top - eps_top * h / centroid

        force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo3, dia_reo3)

        if force > 0:
            centroid_upper = centroid
        else:
            centroid_lower = centroid
    p_m3_4 = 0.0
    m3_4 = moment

    # 5 Pure tension
    eps_top = -eps_s
    eps_bot = -eps_s

    force, moment = _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo3, dia_reo3)
    p_m3_5 = force
    m3_5 = 0.0

    return (
        p_m2_1, p_m2_2, p_m2_3, p_m2_4, p_m2_5,
        m2_1, m2_2, m2_3, m2_4, m2_5,
        p_m3_1, p_m3_2, p_m3_3, p_m3_4, p_m3_5,
        m3_1, m3_2, m3_3, m3_4, m3_5,
    )

def _calculate_force_moment(eps_top, eps_bot, b, h, delta_h, eps_c2, eps_cu2, f_cd, n, eps_s, f_yd, d_concrete, d_reo, dia_reo):

    eps_concrete = _linear_strain_diagram(d_concrete, h, eps_top, eps_bot)
    sigma_concrete = _concrete_stress_strain_parabola_rectangle(eps_concrete, eps_c2, eps_cu2, f_cd, n)
    force_concrete = sigma_concrete * b * delta_h / 1000 #kN

    eps_reo = _linear_strain_diagram(d_reo, h, eps_top, eps_bot)
    sigma_reo = _reo_stress_strain_bilinear_flat(eps_reo, eps_s, f_yd)
    force_reo = sigma_reo * (dia_reo ** 2) / 4 * np.pi / 1000 #kN
    
    centroid = _calculate_centroid(force_concrete, d_concrete, force_reo, d_reo)
    force = _calculate_force(force_concrete, force_reo)   #kN
    moment = _calculate_moment(force_concrete, force_reo, h, centroid)  #kNm
    
    return [force, moment]

def _linear_strain_diagram(d, h, eps_top, eps_bot):
    """Return the strain at each depth given the top and bottom strains"""
    return eps_top + (eps_bot - eps_top) * d / h

def _concrete_stress_strain_parabola_rectangle(eps, eps_c2, eps_cu2, f_cd, n):
    """Return the stress of the concrete given the strain"""
    is_tension = eps < 0
    is_parabola = np.logical_and(eps >= 0, eps <= eps_c2)
    is_rectangle = np.logical_and(eps >= eps_c2, eps <= eps_cu2)

    sigma = np.zeros(len(eps))
    sigma[is_tension] = 0
    sigma[is_parabola] = f_cd * (1 - (1 - eps[is_parabola] / eps_c2) ** n)
    sigma[is_rectangle] = f_cd

    return sigma

def _reo_stress_strain_bilinear_flat(eps, eps_s, f_yd):
    """Return the stress of the reo given the strain"""
    is_yield_tension = eps < (-eps_s)
    is_yield_compression = eps > eps_s
    is_not_yield = np.abs(eps) <= eps_s

    sigma = np.zeros(len(eps))
    sigma[is_yield_tension] = -f_yd
    sigma[is_yield_compression] = f_yd
    sigma[is_not_yield] = f_yd * eps[is_not_yield] / eps_s

    return sigma

def _calculate_centroid(force_concrete, d_concrete, force_reo, d_reo):
    """Return the centroid"""
    return (np.sum(force_concrete * d_concrete) + np.sum(force_reo * d_reo)) / (np.sum(force_concrete) + np.sum(force_reo))

def _calculate_force(force_concrete, force_reo):
    """Return force"""
    return np.sum(force_concrete)+ np.sum(force_reo)

def _calculate_moment(force_concrete, force_reo, h, centroid):
    """Return the moment in kNm"""
    return (np.sum(force_concrete) + np.sum(force_reo)) * (h / 2 - centroid) / 1000