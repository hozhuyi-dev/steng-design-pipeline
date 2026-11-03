"""Functions for utilities handling section_line

Functions:
"""

import polars as pl


def generate_key(section_name, *args):
    """Generate key for storing data
    """

    key = []
    key.append(repr(section_name))
    for arg in args:
        key.append(repr(arg))

    return "_".join(key)

def generate_keys(section_name, df):
    """Generate key for storing data
    """

    keys = []
    
    for row in df.rows():
        row = [section_name] + list(row)
        for i, item in zip(range(len(row)), row):
            row[i] = repr(item)
        keys.append("_".join(row))

    return keys

def get_columns(section_name):
    """Get the input column names"""

    if section_name == "rcb_pile_1":
        return ["grade", "rebar_type", "dia", "cover", "dia_horz", "n_vert", "dia_vert"]
    elif section_name == "rc_wall_u_1":
        return ["grade_min", "rebar_type", "polyline", "angle", "thickness_min", "cover", "dia_horz0", "s_vert0", "dia_vert0"]
    elif section_name == "rc_wall_u_3":
        return ["grade_min", "rebar_type", "polyline", "angle", "thickness_min", "cover", "dia_horz0", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2"]
    elif section_name == "rc_wall_u_5":
        return ["grade_min", "rebar_type", "polyline", "angle", "thickness_min", "cover", "dia_horz0", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4"]
    elif section_name == "rc_wall_u_6":
        return [
            "grade_min", "rebar_type", "polyline", "angle", "thickness_min", "cover", "dia_horz0", "s_vert0", 
            "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5"
        ]
    elif section_name == "rc_column_u_1":
        return ["grade", "rebar_type", "t2", "t3", "cover", "dia_horz0", "n_vert0", "dia_vert0"]
    elif section_name == "rc_column_u_3":
        return ["grade", "rebar_type", "t2", "t3", "cover", "dia_horz0", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2"]
    
    return None

def generate_reo_vert(df, section_name):
    """Return a df with reo_vert and dia_vert
    """

    if section_name == "rcb_pile_1":
        return df.with_columns(reo_vert = pl.concat_str("n_vert", "dia_vert", separator="H"))
    elif section_name == "rc_wall_u_1":
        return df.with_columns(
            dia_vert = pl.col("dia_vert0"),
            reo_vert = pl.concat_str(pl.lit("H"), pl.col("dia_vert0"), pl.lit("-"), pl.col("s_vert0"))
        )
    elif section_name == "rc_wall_u_3":
        return (
            df
            .with_columns(dia_vert = pl.concat_str("dia_vert0", "dia_vert1", "dia_vert2", separator="+", ignore_nulls=True))
            .with_columns(reo_vert = pl.concat_str(pl.lit("H"), pl.col("dia_vert"), pl.lit("-"), pl.col("s_vert0")))
        )
    elif section_name == "rc_wall_u_5":
        return (
            df
            .with_columns(dia_vert = pl.concat_str("dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", separator="+", ignore_nulls=True))
            .with_columns(reo_vert = pl.concat_str(pl.lit("H"), pl.col("dia_vert"), pl.lit("-"), pl.col("s_vert0")))
        )
    elif section_name == "rc_wall_u_6":
        return (
            df
            .with_columns(dia_vert = pl.concat_str("dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5", separator="+", ignore_nulls=True))
            .with_columns(reo_vert = pl.concat_str(pl.lit("H"), pl.col("dia_vert"), pl.lit("-"), pl.col("s_vert0")))
        )
    elif section_name == "rc_column_u_1":
        return df.with_columns(
            dia_vert = pl.col("dia_vert0"),
            reo_vert = pl.concat_str("n_vert0", "dia_vert0", separator="H")
        )
    elif section_name == "rc_column_u_3":
        return (
            df
            .with_columns(dia_vert = pl.concat_str("dia_vert0", "dia_vert1", "dia_vert2", separator="+", ignore_nulls=True))
            .with_columns(reo_vert = pl.concat_str("n_vert0", "dia_vert", separator="H"))
        )
    
    return df

def parse_rectangle(section_line):
    """Return depth and width
    """

    words = section_line.split()

    return float(words[1]), float(words[2])