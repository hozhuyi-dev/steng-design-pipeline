"""Functions for pmm curve analysis and calculation

Functions:
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

def calculate_pm_wall_6(df):
    """Generate pm curve for wall with 6 layers of rebars
    """

    # Validation
    pa.DataFrameSchema({
        "grade_min": pa.Column(pl.UInt32),
        "thickness_min": pa.Column(pl.Float64),
        "length": pa.Column(pl.Float64),
        "cover": pa.Column(pl.Int32),
        "dia_horz0": pa.Column(pl.Int32),
        "dia_vert0": pa.Column(pl.Int32),
        "dia_vert1": pa.Column(pl.Int32),
        "dia_vert2": pa.Column(pl.Int32),
        "dia_vert3": pa.Column(pl.Int32),
        "dia_vert4": pa.Column(pl.Int32),
        "dia_vert5": pa.Column(pl.Int32),
        "s_vert0": pa.Column(pl.Int32),
    }).validate(df)

    # Input
    columns = ["grade_min", "thickness_min", "length", "cover", "dia_horz0", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5"]
    df_unique = (
        df.select(columns)
        .unique()
    )
    cols = df_unique.select(columns).to_numpy()

    # Constant
    n_rows = df_unique.shape[0]

    p_m2_1_all, p_m2_2_all, p_m2_3_all, p_m2_4_all, p_m2_5_all = [], [], [], [], []
    m2_1_all, m2_2_all, m2_3_all, m2_4_all, m2_5_all = [], [], [], [], []
    p_m3_1_all, p_m3_2_all, p_m3_3_all, p_m3_4_all, p_m3_5_all = [], [], [], [], []
    m3_1_all, m3_2_all, m3_3_all, m3_4_all, m3_5_all = [], [], [], [], []
    
    pbar = tqdm(total=n_rows, desc="Calculating pm curve simplified")
    for f_ck, thickness, length, cover, dia_horz, s_vert, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4, dia_vert5 in cols:
        f_ck, thickness, length, cover, dia_horz, s_vert, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4, dia_vert5 = (
            int(f_ck), float(thickness), float(length), int(cover), int(dia_horz), int(s_vert), int(dia_vert0), int(dia_vert1), int(dia_vert2), int(dia_vert3), int(dia_vert4), int(dia_vert5)
        )

        length_vert = length - 2 * cover - 2 * dia_horz - dia_vert0
        n_vert = np.ceil(length_vert / s_vert) + 1
        dia_reo2, d_reo2 = _generate_reo_top_and_bottom_6(thickness, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4, dia_vert5)
        dia_reo3, d_reo3 = _generate_reo_side_6(length, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4, dia_vert5)
        (
            p_m2_1, p_m2_2, p_m2_3, p_m2_4, p_m2_5,
            m2_1, m2_2, m2_3, m2_4, m2_5,
            p_m3_1, p_m3_2, p_m3_3, p_m3_4, p_m3_5,
            m3_1, m3_2, m3_3, m3_4, m3_5,
        ) = _calculate_pm_curve(f_ck, length, thickness, d_reo2, dia_reo2, d_reo3, dia_reo3)

        p_m2_1_all.append(p_m2_1)
        p_m2_2_all.append(p_m2_2)
        p_m2_3_all.append(p_m2_3)
        p_m2_4_all.append(p_m2_4)
        p_m2_5_all.append(p_m2_5)
        m2_1_all.append(m2_1)
        m2_2_all.append(m2_2)
        m2_3_all.append(m2_3)
        m2_4_all.append(m2_4)
        m2_5_all.append(m2_5)
        p_m3_1_all.append(p_m3_1)
        p_m3_2_all.append(p_m3_2)
        p_m3_3_all.append(p_m3_3)
        p_m3_4_all.append(p_m3_4)
        p_m3_5_all.append(p_m3_5)
        m3_1_all.append(m3_1)
        m3_2_all.append(m3_2)
        m3_3_all.append(m3_3)
        m3_4_all.append(m3_4)
        m3_5_all.append(m3_5)

        pbar.update(1)
    pbar.close()
    
    df_unique = df_unique.hstack(pl.DataFrame({
        "p_m2_1": p_m2_1_all, 
        "p_m2_2": p_m2_2_all, 
        "p_m2_3": p_m2_3_all, 
        "p_m2_4": p_m2_4_all, 
        "p_m2_5": p_m2_5_all,
        "m2_1": m2_1_all, 
        "m2_2": m2_2_all, 
        "m2_3": m2_3_all, 
        "m2_4": m2_4_all, 
        "m2_5": m2_5_all,
        "p_m3_1": p_m3_1_all, 
        "p_m3_2": p_m3_2_all, 
        "p_m3_3": p_m3_3_all, 
        "p_m3_4": p_m3_4_all, 
        "p_m3_5": p_m3_5_all,
        "m3_1": m3_1_all, 
        "m3_2": m3_2_all, 
        "m3_3": m3_3_all, 
        "m3_4": m3_4_all, 
        "m3_5": m3_5_all,
    }))

    return (
        df
        .with_columns(
            p_m2_1 = pl.lit(None).cast(pl.Float64), 
            p_m2_2 = pl.lit(None).cast(pl.Float64), 
            p_m2_3 = pl.lit(None).cast(pl.Float64), 
            p_m2_4 = pl.lit(None).cast(pl.Float64), 
            p_m2_5 = pl.lit(None).cast(pl.Float64), 
            m2_1 = pl.lit(None).cast(pl.Float64), 
            m2_2 = pl.lit(None).cast(pl.Float64), 
            m2_3 = pl.lit(None).cast(pl.Float64), 
            m2_4 = pl.lit(None).cast(pl.Float64), 
            m2_5 = pl.lit(None).cast(pl.Float64), 
            p_m3_1 = pl.lit(None).cast(pl.Float64), 
            p_m3_2 = pl.lit(None).cast(pl.Float64),  
            p_m3_3 = pl.lit(None).cast(pl.Float64), 
            p_m3_4 = pl.lit(None).cast(pl.Float64), 
            p_m3_5 = pl.lit(None).cast(pl.Float64), 
            m3_1 = pl.lit(None).cast(pl.Float64), 
            m3_2 = pl.lit(None).cast(pl.Float64), 
            m3_3 = pl.lit(None).cast(pl.Float64), 
            m3_4 = pl.lit(None).cast(pl.Float64), 
            m3_5 = pl.lit(None).cast(pl.Float64), 
        )
        .update(df_unique, on=columns, how="left")
    )

def calculate_pm_wall_5(df):
    """Generate pm curve for wall with 5 layers of rebars
    """

    # Validation
    pa.DataFrameSchema({
        "grade_min": pa.Column(pl.UInt32),
        "thickness_min": pa.Column(pl.Float64),
        "length": pa.Column(pl.Float64),
        "cover": pa.Column(pl.Int32),
        "dia_horz0": pa.Column(pl.Int32),
        "dia_vert0": pa.Column(pl.Int32),
        "dia_vert1": pa.Column(pl.Int32),
        "dia_vert2": pa.Column(pl.Int32),
        "dia_vert3": pa.Column(pl.Int32),
        "dia_vert4": pa.Column(pl.Int32),
        "s_vert0": pa.Column(pl.Int32),
    }).validate(df)

    # Input
    columns = ["grade_min", "thickness_min", "length", "cover", "dia_horz0", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4"]
    df_unique = (
        df.select(columns)
        .unique()
    )
    cols = df_unique.select(columns).to_numpy()

    # Constant
    n_rows = df_unique.shape[0]

    p_m2_1_all, p_m2_2_all, p_m2_3_all, p_m2_4_all, p_m2_5_all = [], [], [], [], []
    m2_1_all, m2_2_all, m2_3_all, m2_4_all, m2_5_all = [], [], [], [], []
    p_m3_1_all, p_m3_2_all, p_m3_3_all, p_m3_4_all, p_m3_5_all = [], [], [], [], []
    m3_1_all, m3_2_all, m3_3_all, m3_4_all, m3_5_all = [], [], [], [], []
    
    pbar = tqdm(total=n_rows, desc="Calculating pm curve simplified")
    for f_ck, thickness, length, cover, dia_horz, s_vert, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4 in cols:
        f_ck, thickness, length, cover, dia_horz, s_vert, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4 = (
            int(f_ck), float(thickness), float(length), int(cover), int(dia_horz), int(s_vert), int(dia_vert0), int(dia_vert1), int(dia_vert2), int(dia_vert3), int(dia_vert4)
        )

        length_vert = length - 2 * cover - 2 * dia_horz - dia_vert0
        n_vert = np.ceil(length_vert / s_vert) + 1
        dia_reo2, d_reo2 = _generate_reo_top_and_bottom_5(thickness, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4)
        dia_reo3, d_reo3 = _generate_reo_side_5(length, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4)
        (
            p_m2_1, p_m2_2, p_m2_3, p_m2_4, p_m2_5,
            m2_1, m2_2, m2_3, m2_4, m2_5,
            p_m3_1, p_m3_2, p_m3_3, p_m3_4, p_m3_5,
            m3_1, m3_2, m3_3, m3_4, m3_5,
        ) = _calculate_pm_curve(f_ck, length, thickness, d_reo2, dia_reo2, d_reo3, dia_reo3)

        p_m2_1_all.append(p_m2_1)
        p_m2_2_all.append(p_m2_2)
        p_m2_3_all.append(p_m2_3)
        p_m2_4_all.append(p_m2_4)
        p_m2_5_all.append(p_m2_5)
        m2_1_all.append(m2_1)
        m2_2_all.append(m2_2)
        m2_3_all.append(m2_3)
        m2_4_all.append(m2_4)
        m2_5_all.append(m2_5)
        p_m3_1_all.append(p_m3_1)
        p_m3_2_all.append(p_m3_2)
        p_m3_3_all.append(p_m3_3)
        p_m3_4_all.append(p_m3_4)
        p_m3_5_all.append(p_m3_5)
        m3_1_all.append(m3_1)
        m3_2_all.append(m3_2)
        m3_3_all.append(m3_3)
        m3_4_all.append(m3_4)
        m3_5_all.append(m3_5)

        pbar.update(1)
    pbar.close()
    
    df_unique = df_unique.hstack(pl.DataFrame({
        "p_m2_1": p_m2_1_all, 
        "p_m2_2": p_m2_2_all, 
        "p_m2_3": p_m2_3_all, 
        "p_m2_4": p_m2_4_all, 
        "p_m2_5": p_m2_5_all,
        "m2_1": m2_1_all, 
        "m2_2": m2_2_all, 
        "m2_3": m2_3_all, 
        "m2_4": m2_4_all, 
        "m2_5": m2_5_all,
        "p_m3_1": p_m3_1_all, 
        "p_m3_2": p_m3_2_all, 
        "p_m3_3": p_m3_3_all, 
        "p_m3_4": p_m3_4_all, 
        "p_m3_5": p_m3_5_all,
        "m3_1": m3_1_all, 
        "m3_2": m3_2_all, 
        "m3_3": m3_3_all, 
        "m3_4": m3_4_all, 
        "m3_5": m3_5_all,
    }))

    return (
        df
        .with_columns(
            p_m2_1 = pl.lit(None).cast(pl.Float64), 
            p_m2_2 = pl.lit(None).cast(pl.Float64), 
            p_m2_3 = pl.lit(None).cast(pl.Float64), 
            p_m2_4 = pl.lit(None).cast(pl.Float64), 
            p_m2_5 = pl.lit(None).cast(pl.Float64), 
            m2_1 = pl.lit(None).cast(pl.Float64), 
            m2_2 = pl.lit(None).cast(pl.Float64), 
            m2_3 = pl.lit(None).cast(pl.Float64), 
            m2_4 = pl.lit(None).cast(pl.Float64), 
            m2_5 = pl.lit(None).cast(pl.Float64), 
            p_m3_1 = pl.lit(None).cast(pl.Float64), 
            p_m3_2 = pl.lit(None).cast(pl.Float64),  
            p_m3_3 = pl.lit(None).cast(pl.Float64), 
            p_m3_4 = pl.lit(None).cast(pl.Float64), 
            p_m3_5 = pl.lit(None).cast(pl.Float64), 
            m3_1 = pl.lit(None).cast(pl.Float64), 
            m3_2 = pl.lit(None).cast(pl.Float64), 
            m3_3 = pl.lit(None).cast(pl.Float64), 
            m3_4 = pl.lit(None).cast(pl.Float64), 
            m3_5 = pl.lit(None).cast(pl.Float64), 
        )
        .update(df_unique, on=columns, how="left")
    )

def calculate_pm_wall_3(df):
    """Generate pm curve for wall with 3 layers of rebars
    """

    # Validation
    pa.DataFrameSchema({
        "grade_min": pa.Column(pl.UInt32),
        "thickness_min": pa.Column(pl.Float64),
        "length": pa.Column(pl.Float64),
        "cover": pa.Column(pl.Int32),
        "dia_horz0": pa.Column(pl.Int32),
        "dia_vert0": pa.Column(pl.Int32),
        "dia_vert1": pa.Column(pl.Int32),
        "dia_vert2": pa.Column(pl.Int32),
        "s_vert0": pa.Column(pl.Int32),
    }).validate(df)

    # Input
    columns = ["grade_min", "thickness_min", "length", "cover", "dia_horz0", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2"]
    df_unique = (
        df.select(columns)
        .unique()
    )
    cols = df_unique.select(columns).to_numpy()

    # Constant
    n_rows = df_unique.shape[0]

    p_m2_1_all, p_m2_2_all, p_m2_3_all, p_m2_4_all, p_m2_5_all = [], [], [], [], []
    m2_1_all, m2_2_all, m2_3_all, m2_4_all, m2_5_all = [], [], [], [], []
    p_m3_1_all, p_m3_2_all, p_m3_3_all, p_m3_4_all, p_m3_5_all = [], [], [], [], []
    m3_1_all, m3_2_all, m3_3_all, m3_4_all, m3_5_all = [], [], [], [], []
    
    pbar = tqdm(total=n_rows, desc="Calculating pm curve simplified")
    for f_ck, thickness, length, cover, dia_horz, s_vert, dia_vert0, dia_vert1, dia_vert2 in cols:
        f_ck, thickness, length, cover, dia_horz, s_vert, dia_vert0, dia_vert1, dia_vert2 = (
            int(f_ck), float(thickness), float(length), int(cover), int(dia_horz), int(s_vert), int(dia_vert0), int(dia_vert1), int(dia_vert2)
        )

        length_vert = length - 2 * cover - 2 * dia_horz - dia_vert0
        n_vert = np.ceil(length_vert / s_vert) + 1
        dia_reo2, d_reo2 = _generate_reo_top_and_bottom_3(thickness, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2)
        dia_reo3, d_reo3 = _generate_reo_side_3(length, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2)
        (
            p_m2_1, p_m2_2, p_m2_3, p_m2_4, p_m2_5,
            m2_1, m2_2, m2_3, m2_4, m2_5,
            p_m3_1, p_m3_2, p_m3_3, p_m3_4, p_m3_5,
            m3_1, m3_2, m3_3, m3_4, m3_5,
        ) = _calculate_pm_curve(f_ck, length, thickness, d_reo2, dia_reo2, d_reo3, dia_reo3)

        p_m2_1_all.append(p_m2_1)
        p_m2_2_all.append(p_m2_2)
        p_m2_3_all.append(p_m2_3)
        p_m2_4_all.append(p_m2_4)
        p_m2_5_all.append(p_m2_5)
        m2_1_all.append(m2_1)
        m2_2_all.append(m2_2)
        m2_3_all.append(m2_3)
        m2_4_all.append(m2_4)
        m2_5_all.append(m2_5)
        p_m3_1_all.append(p_m3_1)
        p_m3_2_all.append(p_m3_2)
        p_m3_3_all.append(p_m3_3)
        p_m3_4_all.append(p_m3_4)
        p_m3_5_all.append(p_m3_5)
        m3_1_all.append(m3_1)
        m3_2_all.append(m3_2)
        m3_3_all.append(m3_3)
        m3_4_all.append(m3_4)
        m3_5_all.append(m3_5)

        pbar.update(1)
    pbar.close()
    
    df_unique = df_unique.hstack(pl.DataFrame({
        "p_m2_1": p_m2_1_all, 
        "p_m2_2": p_m2_2_all, 
        "p_m2_3": p_m2_3_all, 
        "p_m2_4": p_m2_4_all, 
        "p_m2_5": p_m2_5_all,
        "m2_1": m2_1_all, 
        "m2_2": m2_2_all, 
        "m2_3": m2_3_all, 
        "m2_4": m2_4_all, 
        "m2_5": m2_5_all,
        "p_m3_1": p_m3_1_all, 
        "p_m3_2": p_m3_2_all, 
        "p_m3_3": p_m3_3_all, 
        "p_m3_4": p_m3_4_all, 
        "p_m3_5": p_m3_5_all,
        "m3_1": m3_1_all, 
        "m3_2": m3_2_all, 
        "m3_3": m3_3_all, 
        "m3_4": m3_4_all, 
        "m3_5": m3_5_all,
    }))

    return (
        df
        .with_columns(
            p_m2_1 = pl.lit(None).cast(pl.Float64), 
            p_m2_2 = pl.lit(None).cast(pl.Float64), 
            p_m2_3 = pl.lit(None).cast(pl.Float64), 
            p_m2_4 = pl.lit(None).cast(pl.Float64), 
            p_m2_5 = pl.lit(None).cast(pl.Float64), 
            m2_1 = pl.lit(None).cast(pl.Float64), 
            m2_2 = pl.lit(None).cast(pl.Float64), 
            m2_3 = pl.lit(None).cast(pl.Float64), 
            m2_4 = pl.lit(None).cast(pl.Float64), 
            m2_5 = pl.lit(None).cast(pl.Float64), 
            p_m3_1 = pl.lit(None).cast(pl.Float64), 
            p_m3_2 = pl.lit(None).cast(pl.Float64),  
            p_m3_3 = pl.lit(None).cast(pl.Float64), 
            p_m3_4 = pl.lit(None).cast(pl.Float64), 
            p_m3_5 = pl.lit(None).cast(pl.Float64), 
            m3_1 = pl.lit(None).cast(pl.Float64), 
            m3_2 = pl.lit(None).cast(pl.Float64), 
            m3_3 = pl.lit(None).cast(pl.Float64), 
            m3_4 = pl.lit(None).cast(pl.Float64), 
            m3_5 = pl.lit(None).cast(pl.Float64), 
        )
        .update(df_unique, on=columns, how="left")
    )

def calculate_pm_column_3(df):
    """Generate pm curve for wall with 3 layers of rebars
    """

    # Validation
    pa.DataFrameSchema({
        "grade": pa.Column(pl.UInt32),
        "t3": pa.Column(pl.Float64),
        "t2": pa.Column(pl.Float64),
        "cover": pa.Column(pl.Int32),
        "dia_horz0": pa.Column(pl.Int32),
        "n_vert0": pa.Column(pl.Int32),
        "dia_vert0": pa.Column(pl.Int32),
        "dia_vert1": pa.Column(pl.Int32),
        "dia_vert2": pa.Column(pl.Int32),
        "s_vert0": pa.Column(pl.Int32),
    }).validate(df)

    # Input
    columns = ["grade", "t3", "t2", "cover", "dia_horz0", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "s_vert0"]
    df_unique = (
        df.select(columns)
        .unique()
    )
    cols = df_unique.select(columns).to_numpy()

    # Constant
    n_rows = df_unique.shape[0]

    p_m2_1_all, p_m2_2_all, p_m2_3_all, p_m2_4_all, p_m2_5_all = [], [], [], [], []
    m2_1_all, m2_2_all, m2_3_all, m2_4_all, m2_5_all = [], [], [], [], []
    p_m3_1_all, p_m3_2_all, p_m3_3_all, p_m3_4_all, p_m3_5_all = [], [], [], [], []
    m3_1_all, m3_2_all, m3_3_all, m3_4_all, m3_5_all = [], [], [], [], []
    
    pbar = tqdm(total=n_rows, desc="Calculating pm curve simplified")
    for f_ck, t3, t2, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2, s_vert in cols:
        f_ck, t3, t2, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2, s_vert = (
            int(f_ck), float(t3), float(t2), int(cover), int(dia_horz), int(n_vert), int(dia_vert0), int(dia_vert1), int(dia_vert2), int(s_vert)
        )

        dia_reo2, d_reo2 = _generate_reo_column3(t2, t3, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2, s_vert)
        dia_reo3, d_reo3 = _generate_reo_column3(t2, t3, cover, dia_horz, n_vert, dia_vert0, dia_vert1, dia_vert2, s_vert)
        (
            p_m2_1, p_m2_2, p_m2_3, p_m2_4, p_m2_5,
            m2_1, m2_2, m2_3, m2_4, m2_5,
            p_m3_1, p_m3_2, p_m3_3, p_m3_4, p_m3_5,
            m3_1, m3_2, m3_3, m3_4, m3_5,
        ) = _calculate_pm_curve(f_ck, t2, t3, d_reo2, dia_reo2, d_reo3, dia_reo3)

        p_m2_1_all.append(p_m2_1)
        p_m2_2_all.append(p_m2_2)
        p_m2_3_all.append(p_m2_3)
        p_m2_4_all.append(p_m2_4)
        p_m2_5_all.append(p_m2_5)
        m2_1_all.append(m2_1)
        m2_2_all.append(m2_2)
        m2_3_all.append(m2_3)
        m2_4_all.append(m2_4)
        m2_5_all.append(m2_5)
        p_m3_1_all.append(p_m3_1)
        p_m3_2_all.append(p_m3_2)
        p_m3_3_all.append(p_m3_3)
        p_m3_4_all.append(p_m3_4)
        p_m3_5_all.append(p_m3_5)
        m3_1_all.append(m3_1)
        m3_2_all.append(m3_2)
        m3_3_all.append(m3_3)
        m3_4_all.append(m3_4)
        m3_5_all.append(m3_5)

        pbar.update(1)
    pbar.close()
    
    df_unique = df_unique.hstack(pl.DataFrame({
        "p_m2_1": p_m2_1_all, 
        "p_m2_2": p_m2_2_all, 
        "p_m2_3": p_m2_3_all, 
        "p_m2_4": p_m2_4_all, 
        "p_m2_5": p_m2_5_all,
        "m2_1": m2_1_all, 
        "m2_2": m2_2_all, 
        "m2_3": m2_3_all, 
        "m2_4": m2_4_all, 
        "m2_5": m2_5_all,
        "p_m3_1": p_m3_1_all, 
        "p_m3_2": p_m3_2_all, 
        "p_m3_3": p_m3_3_all, 
        "p_m3_4": p_m3_4_all, 
        "p_m3_5": p_m3_5_all,
        "m3_1": m3_1_all, 
        "m3_2": m3_2_all, 
        "m3_3": m3_3_all, 
        "m3_4": m3_4_all, 
        "m3_5": m3_5_all,
    }))

    return (
        df
        .with_columns(
            p_m2_1 = pl.lit(None).cast(pl.Float64), 
            p_m2_2 = pl.lit(None).cast(pl.Float64), 
            p_m2_3 = pl.lit(None).cast(pl.Float64), 
            p_m2_4 = pl.lit(None).cast(pl.Float64), 
            p_m2_5 = pl.lit(None).cast(pl.Float64), 
            m2_1 = pl.lit(None).cast(pl.Float64), 
            m2_2 = pl.lit(None).cast(pl.Float64), 
            m2_3 = pl.lit(None).cast(pl.Float64), 
            m2_4 = pl.lit(None).cast(pl.Float64), 
            m2_5 = pl.lit(None).cast(pl.Float64), 
            p_m3_1 = pl.lit(None).cast(pl.Float64), 
            p_m3_2 = pl.lit(None).cast(pl.Float64),  
            p_m3_3 = pl.lit(None).cast(pl.Float64), 
            p_m3_4 = pl.lit(None).cast(pl.Float64), 
            p_m3_5 = pl.lit(None).cast(pl.Float64), 
            m3_1 = pl.lit(None).cast(pl.Float64), 
            m3_2 = pl.lit(None).cast(pl.Float64), 
            m3_3 = pl.lit(None).cast(pl.Float64), 
            m3_4 = pl.lit(None).cast(pl.Float64), 
            m3_5 = pl.lit(None).cast(pl.Float64), 
        )
        .update(df_unique, on=columns, how="left")
    )

def _generate_reo_top_and_bottom_6(h, cover, dia_horz, n, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4, dia_vert5):
    """Return the diameters and depths of top and bottom rebars"""
    
    s_top = np.maximum(dia_vert0, SPACING_MIN)
    d = cover + dia_horz + dia_vert0 / 2
    dia_reo = list(np.full(int(n), dia_vert0))
    d_reo = list(np.full(int(n), d))
    if dia_vert1 > 0:
        d += dia_vert0 / 2 + s_top + dia_vert1 / 2
        dia_reo += list(np.full(int(n), dia_vert1))
        d_reo += list(np.full(int(n), d))
    if dia_vert2 > 0:
        d += dia_vert1 / 2 + s_top + dia_vert2 / 2
        dia_reo += list(np.full(int(n), dia_vert2))
        d_reo += list(np.full(int(n), d))
    if dia_vert3 > 0:
        d += dia_vert2 / 2 + s_top + dia_vert3 / 2
        dia_reo += list(np.full(int(n), dia_vert3))
        d_reo += list(np.full(int(n), d))
    if dia_vert4 > 0:
        d += dia_vert3 / 2 + s_top + dia_vert4 / 2
        dia_reo += list(np.full(int(n), dia_vert4))
        d_reo += list(np.full(int(n), d))
    if dia_vert5 > 0:
        d += dia_vert4 / 2 + s_top + dia_vert5 / 2
        dia_reo += list(np.full(int(n), dia_vert5))
        d_reo += list(np.full(int(n), d))
    
    s_bot = np.maximum(dia_vert0, 32)
    d = [h - (cover + dia_horz + dia_vert0 / 2)]
    dia_reo += list(np.full(int(n), dia_vert0))
    d_reo += list(np.full(int(n), d))
    if dia_vert1 > 0:
        d -= (dia_vert0 / 2 + s_bot + dia_vert1 / 2)
        dia_reo += list(np.full(int(n), dia_vert1))
        d_reo += list(np.full(int(n), d))
    if dia_vert2 > 0:
        d -= (dia_vert1 / 2 + s_bot + dia_vert2 / 2)
        dia_reo += list(np.full(int(n), dia_vert2))
        d_reo += list(np.full(int(n), d))
    if dia_vert3 > 0:
        d -= (dia_vert2 / 2 + s_bot + dia_vert3 / 2)
        dia_reo += list(np.full(int(n), dia_vert3))
        d_reo += list(np.full(int(n), d))
    if dia_vert4 > 0:
        d -= (dia_vert3 / 2 + s_bot + dia_vert4 / 2)
        dia_reo += list(np.full(int(n), dia_vert4))
        d_reo += list(np.full(int(n), d))
    if dia_vert5 > 0:
        d -= (dia_vert4 / 2 + s_bot + dia_vert5 / 2)
        dia_reo += list(np.full(int(n), dia_vert5))
        d_reo += list(np.full(int(n), d))

    return np.array(dia_reo), np.array(d_reo)

def _generate_reo_side_6(h, cover, dia_horz, n, dia0, dia1, dia2, dia3, dia4, dia5):
    """Return the diameters and depths of side rebars"""
    
    d_first = cover + dia_horz + dia0 / 2
    length_reo = h - 2 * d_first
    s_reo = length_reo / (n - 1)

    dia_reo = []
    d_reo = []
    dia_reo += [dia0] * int(n) * 2
    d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia1 > 0:
        dia_reo += [dia1] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia2 > 0:
        dia_reo += [dia2] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia3 > 0:
        dia_reo += [dia3] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia4 > 0:
        dia_reo += [dia4] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia5 > 0:
        dia_reo += [dia5] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2

    return np.array(dia_reo), np.array(d_reo)

def _generate_reo_top_and_bottom_5(h, cover, dia_horz, n, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4):
    """Return the diameters and depths of top and bottom rebars"""
    
    s_top = np.maximum(dia_vert0, SPACING_MIN)
    d = cover + dia_horz + dia_vert0 / 2
    dia_reo = list(np.full(int(n), dia_vert0))
    d_reo = list(np.full(int(n), d))
    if dia_vert1 > 0:
        d += dia_vert0 / 2 + s_top + dia_vert1 / 2
        dia_reo += list(np.full(int(n), dia_vert1))
        d_reo += list(np.full(int(n), d))
    if dia_vert2 > 0:
        d += dia_vert1 / 2 + s_top + dia_vert2 / 2
        dia_reo += list(np.full(int(n), dia_vert2))
        d_reo += list(np.full(int(n), d))
    if dia_vert3 > 0:
        d += dia_vert2 / 2 + s_top + dia_vert3 / 2
        dia_reo += list(np.full(int(n), dia_vert3))
        d_reo += list(np.full(int(n), d))
    if dia_vert4 > 0:
        d += dia_vert3 / 2 + s_top + dia_vert4 / 2
        dia_reo += list(np.full(int(n), dia_vert4))
        d_reo += list(np.full(int(n), d))
    
    s_bot = np.maximum(dia_vert0, 32)
    d = [h - (cover + dia_horz + dia_vert0 / 2)]
    dia_reo += list(np.full(int(n), dia_vert0))
    d_reo += list(np.full(int(n), d))
    if dia_vert1 > 0:
        d -= (dia_vert0 / 2 + s_bot + dia_vert1 / 2)
        dia_reo += list(np.full(int(n), dia_vert1))
        d_reo += list(np.full(int(n), d))
    if dia_vert2 > 0:
        d -= (dia_vert1 / 2 + s_bot + dia_vert2 / 2)
        dia_reo += list(np.full(int(n), dia_vert2))
        d_reo += list(np.full(int(n), d))
    if dia_vert3 > 0:
        d -= (dia_vert2 / 2 + s_bot + dia_vert3 / 2)
        dia_reo += list(np.full(int(n), dia_vert3))
        d_reo += list(np.full(int(n), d))
    if dia_vert4 > 0:
        d -= (dia_vert3 / 2 + s_bot + dia_vert4 / 2)
        dia_reo += list(np.full(int(n), dia_vert4))
        d_reo += list(np.full(int(n), d))

    return np.array(dia_reo), np.array(d_reo)

def _generate_reo_side_5(h, cover, dia_horz, n, dia0, dia1, dia2, dia3, dia4):
    """Return the diameters and depths of side rebars"""
    
    d_first = cover + dia_horz + dia0 / 2
    length_reo = h - 2 * d_first
    s_reo = length_reo / (n - 1)

    dia_reo = []
    d_reo = []
    dia_reo += [dia0] * int(n) * 2
    d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia1 > 0:
        dia_reo += [dia1] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia2 > 0:
        dia_reo += [dia2] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia3 > 0:
        dia_reo += [dia3] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia4 > 0:
        dia_reo += [dia4] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2

    return np.array(dia_reo), np.array(d_reo)

def _generate_reo_top_and_bottom_3(h, cover, dia_horz, n, dia_vert0, dia_vert1, dia_vert2):
    """Return the diameters and depths of top and bottom rebars"""
    
    s_top = np.maximum(dia_vert0, SPACING_MIN)
    d = cover + dia_horz + dia_vert0 / 2
    dia_reo = list(np.full(int(n), dia_vert0))
    d_reo = list(np.full(int(n), d))
    if dia_vert1 > 0:
        d += dia_vert0 / 2 + s_top + dia_vert1 / 2
        dia_reo += list(np.full(int(n), dia_vert1))
        d_reo += list(np.full(int(n), d))
    if dia_vert2 > 0:
        d += dia_vert1 / 2 + s_top + dia_vert2 / 2
        dia_reo += list(np.full(int(n), dia_vert2))
        d_reo += list(np.full(int(n), d))
    
    s_bot = np.maximum(dia_vert0, 32)
    d = [h - (cover + dia_horz + dia_vert0 / 2)]
    dia_reo += list(np.full(int(n), dia_vert0))
    d_reo += list(np.full(int(n), d))
    if dia_vert1 > 0:
        d -= (dia_vert0 / 2 + s_bot + dia_vert1 / 2)
        dia_reo += list(np.full(int(n), dia_vert1))
        d_reo += list(np.full(int(n), d))
    if dia_vert2 > 0:
        d -= (dia_vert1 / 2 + s_bot + dia_vert2 / 2)
        dia_reo += list(np.full(int(n), dia_vert2))
        d_reo += list(np.full(int(n), d))

    return np.array(dia_reo), np.array(d_reo)

def _generate_reo_side_3(h, cover, dia_horz, n, dia0, dia1, dia2):
    """Return the diameters and depths of side rebars"""
    
    d_first = cover + dia_horz + dia0 / 2
    length_reo = h - 2 * d_first
    s_reo = length_reo / (n - 1)

    dia_reo = []
    d_reo = []
    dia_reo += [dia0] * int(n) * 2
    d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia1 > 0:
        dia_reo += [dia1] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2
    if dia2 > 0:
        dia_reo += [dia2] * int(n) * 2
        d_reo += list(np.arange(d_first, h, s_reo)) * 2

    return np.array(dia_reo), np.array(d_reo)

def _generate_reo_column3(b, h, cover, dia_horz, n, dia0, dia1, dia2, s):
    """Return the diameters and depths of column rebars"""

    n_b = int(np.ceil((b - 2 * (cover + dia_horz) - dia0) / s)) + 1
    n_h = int(np.ceil((h - 2 * (cover + dia_horz) - dia0) / s)) + 1

    s_reo0 = float((h - 2 * (cover + dia_horz) - dia0) / (n_h - 1))
    d_first0 = (cover + dia_horz + dia0 / 2)
    d_reo = [d_first0] * (n_b - 2)
    d_reo += [h - d_first0] * (n_b - 2)
    d_reo += list(np.arange(d_first0, h - d_first0 + 1, s_reo0)) * 2
    dia_reo = np.full(n, dia0)

    if dia1 != 0:
        s_reo1 = float((h - 2 * (cover + dia_horz) - 2 * dia0 - 2 * np.maximum(dia0, SPACING_MIN) - dia1) / (n_h - 1))
        d_first1 = (cover + dia_horz + dia0 + np.maximum(dia0, SPACING_MIN) + dia1 / 2)
        d_reo += [d_first1] * (n_b - 2)
        d_reo += [h - d_first1] * (n_b - 2)
        d_reo += list(np.arange(d_first1, h - d_first1 + 1, s_reo1)) * 2
        dia_reo = np.concatenate((dia_reo, np.full(n, dia1)))
    
    if dia2 != 0:
        s_reo2 = float((h - 2 * (cover + dia_horz) - 2 * dia0 - 2 * np.maximum(dia0, SPACING_MIN) - 2 * dia1 - 2 * np.maximum(dia1, SPACING_MIN) - dia2) / (n_h - 1))
        d_first2 = (cover + dia_horz + dia0 + np.maximum(dia0, SPACING_MIN) + dia1 + np.maximum(dia1, SPACING_MIN) + dia2 / 2)
        d_reo += [d_first2] * (n_b - 2)
        d_reo += [h - d_first2] * (n_b - 2)
        d_reo += list(np.arange(d_first2, h - d_first2 + 1, s_reo2)) * 2
        dia_reo = np.concatenate((dia_reo, np.full(n, dia1)))

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