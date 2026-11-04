"""Section names and keys: the columns that define each section type, and a key for each section

A section type is "rcb_pile_1", or "rc_wall_u_<n>" or "rc_column_u_<n>" for walls and columns with up
to n layers of vertical bars, dia_vert0 the outermost.
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

def _split_section_name(section_name):
    """Return the kind of section and its number of layers of vertical bars"""

    kind, _, n_layers = section_name.rpartition("_")
    if not n_layers.isdigit():
        return section_name, None

    return kind, int(n_layers)

def get_columns(section_name):
    """Get the input column names"""

    if section_name == "rcb_pile_1":
        return ["grade", "rebar_type", "dia", "cover", "dia_horz", "n_vert", "dia_vert"]

    kind, n_layers = _split_section_name(section_name)
    dia_vert = [f"dia_vert{k}" for k in range(n_layers or 0)]
    if kind == "rc_wall_u":
        return ["grade_min", "rebar_type", "polyline", "angle", "thickness_min", "cover", "dia_horz0", "s_vert0", *dia_vert]
    elif kind == "rc_column_u":
        # AdSec's width and depth: t3 then t2, as the sections were built before t2 and t3 were read
        # the right way round. Its axes against ETABS's M2 and M3 are not checked here.
        return ["grade", "rebar_type", "t3", "t2", "cover", "dia_horz0", "n_vert0", *dia_vert]

    return None

def generate_reo_vert(df, section_name):
    """Return a df with dia_vert, the bar diameters of each layer, and reo_vert, the layout:
    "H32+20-200" for walls, bars at 200 mm on each face, and "20H32+20" for columns, 20 bars a layer
    """

    if section_name == "rcb_pile_1":
        return df.with_columns(reo_vert = pl.concat_str("n_vert", "dia_vert", separator="H"))

    kind, n_layers = _split_section_name(section_name)
    dia_vert = pl.concat_str([f"dia_vert{k}" for k in range(n_layers or 0)], separator="+", ignore_nulls=True)
    if kind == "rc_wall_u":
        return (
            df
            .with_columns(dia_vert = dia_vert)
            .with_columns(reo_vert = pl.concat_str(pl.lit("H"), pl.col("dia_vert"), pl.lit("-"), pl.col("s_vert0")))
        )
    elif kind == "rc_column_u":
        return (
            df
            .with_columns(dia_vert = dia_vert)
            .with_columns(reo_vert = pl.concat_str("n_vert0", "dia_vert", separator="H"))
        )

    return df

def parse_rectangle(section_line):
    """Return depth and width
    """

    words = section_line.split()

    return float(words[1]), float(words[2])
