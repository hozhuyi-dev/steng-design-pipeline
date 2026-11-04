"""Section geometry and trial reinforcement: wall polylines, trial layouts of vertical bars, and
horizontal bars and links sized for shear
"""

import math

import numpy as np
import pandera.polars as pa
import polars as pl

from steng.design._checks import _drop_outputs, bar_layers

COLUMN_A_VERT_MIN = 0.2
PILE_A_VERT_MIN = 0.2
WALL_A_VERT_MIN = 0.2
LINK_DIA_MIN = 6  # mm, 9.5.3(1)

def calculate_wall_dimensions(df):
    """Return the wall pier polyline, length and shape
    """

    # Validation
    pa.DataFrameSchema({
        "pier": pa.Column(pl.String),
        "story": pa.Column(pl.String),
        "x": pa.Column(pl.List(pl.List(pl.Float64))),
        "y": pa.Column(pl.List(pl.List(pl.Float64))),
        "grade_min": pa.Column(pl.UInt32),
        "angle": pa.Column(pl.Float64),
        "thickness_min": pa.Column(pl.Float64),
    }).validate(df)
    
    # Input
    piers = df["pier"].to_list()
    stories = df["story"].to_list()
    xs = df["x"].to_list()
    ys = df["y"].to_list()
    grade_min = df["grade_min"].to_list()
    axis_angles = df["angle"].to_list()
    thickness_min = df["thickness_min"].to_list()

    # Constant
    # n_rows = df.shape[0]

    # Calculation
    polylines = []

    for x, y, pier, story in zip(xs, ys, piers, stories):
        # print(pier, story)
        if pier is None:
            polylines.append(None)
        else:
            polylines.append(_simplify_polyline(_shift_to_origin(_convert_lines_to_polyline(x, y))))

    lengths = _calculate_polyline_length(polylines)

    shapes = []

    for grade_min, polyline, axis_angle, thickness_min in zip(grade_min, polylines, axis_angles, thickness_min):
        shape = [
            repr(grade_min),
            repr(polyline),
            repr(axis_angle),
            repr(thickness_min * 1000),
        ]
        shape = "_".join(shape)
        shapes.append(shape)

    # Output
    return (
        df
        .with_columns(
            pl.Series("polyline", polylines, dtype=pl.List(pl.List(pl.Float64))),
            pl.Series("length", lengths, dtype=pl.Float64),
            pl.Series("shape", shapes, dtype=pl.String),
        )
    )

def _convert_lines_to_polyline(lines_x, lines_y):
    """Convert lines to series of points representing polyline"""

    polyline = [[lines_x[0][0], lines_y[0][0]], [lines_x[0][1], lines_y[0][1]]]
    xqueue = lines_x[1:].copy()
    yqueue = lines_y[1:].copy()
    while len(xqueue) > 0:
        x = xqueue[0]
        y = yqueue[0]
        xqueue.pop(0)
        yqueue.pop(0)
        if (x[0] == polyline[0][0]) and (y[0] == polyline[0][1]):
            polyline.insert(0, [x[1], y[1]])
        elif (x[0] == polyline[-1][0]) and (y[0] == polyline[-1][1]):
            polyline.append([x[1], y[1]])
        elif (x[1] == polyline[0][0]) and (y[1] == polyline[0][1]):
            polyline.insert(0, [x[0], y[0]])
        elif (x[1] == polyline[-1][0]) and (y[1] == polyline[-1][1]):
            polyline.append([x[0], y[0]])
        else:
            xqueue.append(x)
            yqueue.append(y)
    
    return polyline

def _shift_to_origin(polyline):
    """Shift polyline to origin"""

    origin = polyline[0]

    shifted = []
    for i in range(len(polyline)):
        shifted.append([
            round(polyline[i][0] - origin[0], 4),
            round(polyline[i][1] - origin[1], 4)
        ])

    return shifted

def _is_collinear(p1, p2, p3, eps=1e-9):
    """Check if points p1, p2, p3 are collinear using the area of the triangle method."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    area = abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
    return area < eps    

def _simplify_polyline(polyline):
    """Simplify polyline by removing collinear points."""
    if len(polyline) < 3:
        return polyline

    simplified = [polyline[0]]
    for i in range(1, len(polyline) - 1):
        if not _is_collinear(polyline[i - 1], polyline[i], polyline[i + 1]):
            simplified.append(polyline[i])
    simplified.append(polyline[-1])

    return simplified

def _calculate_polyline_length(polylines):
    """Calculate the length of a given polyline"""

    lengths = []
    for polyline in polylines:
        if polyline is None:
            lengths.append(None)
        else:
            length = 0
            for i in range(1, len(polyline)):
                delta_x = polyline[i][0] - polyline[i - 1][0]
                delta_y = polyline[i][1] - polyline[i - 1][1]
                length += math.sqrt((delta_x ** 2) + (delta_y ** 2))
            lengths.append(length)
    
    return lengths

def generate_reo_column_horz(df, A_sw_s_name, direction, diameter, spacing, ur_limit):
    """Generate the links for columns given A_sw_s, for every layout of vertical bars
    """

    # Validation
    pa.DataFrameSchema({
        A_sw_s_name: pa.Column(pl.Float64),
        "dia_horz0": pa.Column(pl.Int32),
        "n" + direction + "_horz0": pa.Column(pl.Int32),
        "dia_vert0": pa.Column(pl.Int32),
    }).validate(df)

    # Input
    A_sw_s_req = df[A_sw_s_name].to_numpy()
    n_horz0 = df["n" + direction + "_horz0"].to_numpy()
    dia_vert = bar_layers(df)
    diameter = np.array(diameter)
    spacing = np.array(spacing)

    # Constant
    n_rows = df.shape[0]

    # Calculations
    layouts = np.unique(np.column_stack((n_horz0, *dia_vert)), axis=0)

    s_horz = np.full(n_rows, 0)
    dia_horz = np.full(n_rows, 0)
    A_sw_s_prov = np.full(n_rows, 0.0)
    ur_horz = np.full(n_rows, 0.0)

    diameter, spacing = np.meshgrid(diameter, spacing, indexing='ij')
    diameter = diameter.ravel()
    spacing = spacing.ravel()

    for n_current, *dia_current in layouts:
        n_current = int(n_current)
        indices_current = np.logical_and.reduce(
            [n_horz0 == n_current] + [layer == int(dia) for layer, dia in zip(dia_vert, dia_current)]
        )

        # 9.5.3(1): links at least 6 mm and a quarter of the largest vertical bar
        d_min = max(LINK_DIA_MIN, max(int(dia) for dia in dia_current) / 4)
        is_allowed = diameter >= d_min
        diameter_current = diameter[is_allowed]
        spacing_current = spacing[is_allowed]

        if len(diameter_current) == 0:
            s_horz[indices_current] = np.min(spacing)
            dia_horz[indices_current] = np.max(diameter)
            A_sw_s_prov[indices_current] = n_current * (np.max(diameter) ** 2) / 4 * np.pi / np.min(spacing)

            ur_horz[indices_current] = d_min / np.max(diameter)
        else:
            A_sw_s_current = n_current * (diameter_current ** 2) / 4 * np.pi / spacing_current

            indices = np.argsort(A_sw_s_current)
            diameter_current = diameter_current[indices]
            spacing_current = spacing_current[indices]
            A_sw_s_current = A_sw_s_current[indices]

            indices = np.searchsorted(A_sw_s_current, A_sw_s_req[indices_current] / ur_limit, side='right')
            indices[indices == len(A_sw_s_current)] -= 1

            s_horz[indices_current] = spacing_current[indices]
            dia_horz[indices_current] = diameter_current[indices]
            A_sw_s_prov[indices_current] = A_sw_s_current[indices]

            ur_horz[indices_current] = A_sw_s_req[indices_current] / A_sw_s_prov[indices_current]

    # Output
    return (
        df
        .with_columns(
            pl.Series("A_sw_s_prov" + direction, list(A_sw_s_prov), dtype=pl.Float64),
            pl.Series("s_horz0" + direction, list(s_horz), dtype=pl.Int32),
            pl.Series("dia_horz0" + direction, list(dia_horz), dtype=pl.Int32),
            pl.Series("ur_horz" + direction, list(ur_horz), dtype=pl.Float64)
        )
    )

def generate_reo_wall_horz(df, A_sw_s_name, diameter, spacing, ur_limit, n_layers=6):
    """Generate the horizontal bars for walls given A_sw_s, for trials of up to n_layers layers of vertical bars

    A trial with k layers of vertical bars gets up to k + 1 layers of horizontal bars on each face, of
    diameters that never grow inwards, and the innermost only once the one outside it is the largest.
    """

    outputs = ["s_horz", *[f"dia_horz{k}" for k in range(n_layers + 1)], "ur_horz"]
    df = _drop_outputs(df, outputs)

    # Validation
    pa.DataFrameSchema({
        A_sw_s_name: pa.Column(pl.Float64),
        **{f"dia_vert{k}": pa.Column(pl.Int32) for k in range(1, n_layers)},
    }).validate(df)

    # Input
    n_rows = df.shape[0]
    A_sw_s_req = df[A_sw_s_name].to_numpy()
    diameter = np.array(diameter)
    spacing = np.array(spacing)

    # Calculations
    # Number of layers of vertical bars: one more than the innermost layer with bars
    n_vert = np.ones(n_rows, dtype=int)
    for k in range(1, n_layers):
        n_vert[df[f"dia_vert{k}"].to_numpy() != 0] = k + 1

    s_horz = np.full(n_rows, 0)
    dia_horz = np.zeros((n_layers + 1, n_rows), dtype=int)
    A_sw_s_prov = np.full(n_rows, 0.0)

    for n in np.unique(n_vert):
        layers, spacing_n, A_sw_s = _get_reo_wall_horz_options(diameter, spacing, n + 1)

        is_n = n_vert == n
        indices = np.searchsorted(A_sw_s, A_sw_s_req[is_n] / ur_limit, side='right')
        indices[indices == len(A_sw_s)] -= 1

        is_zero = np.logical_and(is_n, A_sw_s_req == 0)
        s_horz[is_n] = spacing_n[indices]
        s_horz[is_zero] = 0
        dia_horz[:n + 1, is_n] = layers[:, indices]
        dia_horz[:, is_zero] = 0
        A_sw_s_prov[is_n] = A_sw_s[indices]

    ur_horz = A_sw_s_req / A_sw_s_prov

    # Output
    return (
        df
        .with_columns(
            pl.Series("s_horz", s_horz, dtype=pl.Int32),
            *[pl.Series(f"dia_horz{k}", dia_horz[k], dtype=pl.Int32) for k in range(n_layers + 1)],
            pl.Series("ur_horz", ur_horz, dtype=pl.Float64),
        )
    )

def _get_reo_wall_horz_options(diameter, spacing, n_layers):
    """Return the layouts of n_layers layers of horizontal bars on each face, sorted by A_sw_s

    The innermost layer may be empty, and is used only when the one outside it is the largest bar.
    """

    diameter_max = np.max(diameter)
    grids = np.meshgrid(*[diameter] * (n_layers - 1), np.insert(diameter, 0, 0), spacing, indexing='ij')
    layers = np.array([grid.ravel() for grid in grids[:-1]])
    spacing = grids[-1].ravel()

    is_kept = np.logical_and(
        # inner layers no larger than outer layers
        np.all(layers[:-1] >= layers[1:], axis=0),
        # innermost layer only when the others are maxed
        np.logical_not(np.logical_and(layers[-2] != diameter_max, layers[-1] > 0)),
    )
    unique_rows = np.unique(np.column_stack((*layers[:, is_kept], spacing[is_kept])), axis=0)
    layers, spacing = unique_rows[:, :-1].T, unique_rows[:, -1]

    A_sw_s = 2 * sum(layer ** 2 for layer in layers) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)

    return layers[:, indices], spacing[indices], A_sw_s[indices]

def generate_reo_wall_line_uniform_tie(df, A_sw_s_name, diameter, spacing_vert, spacing_horz, ur_limit):
    """Generate the horizontal reinforcement for walls given A_sw_s
    """
    
    outputs = ["sv_tie", "sh_tie", "dia_tie", "ur_tie"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    df.shape[0]
    if A_sw_s_name not in df.columns:
        print("Missing column", A_sw_s_name)
        return df
    
    A_sw_s_req = np.array(df.select(A_sw_s_name).to_series())
    diameter = np.array(diameter)
    spacing_vert = np.array(spacing_vert)
    spacing_horz = np.array(spacing_horz)
    
    diameter, spacing_vert, spacing_horz = np.meshgrid(diameter, spacing_vert, spacing_horz, indexing='ij')
    diameter = diameter.ravel()
    spacing_vert = spacing_vert.ravel()
    spacing_horz = spacing_horz.ravel()
    A_sw_s = 1000 / spacing_horz * (diameter ** 2) / 4 * np.pi / spacing_vert

    indices = np.argsort(A_sw_s)
    diameter = diameter[indices]
    spacing_vert = spacing_vert[indices]
    spacing_horz = spacing_horz[indices]
    A_sw_s = A_sw_s[indices]

    indices = np.searchsorted(A_sw_s, A_sw_s_req / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    sv_tie = spacing_vert[indices]
    sv_tie[A_sw_s_req == 0] = 0
    sh_tie = spacing_horz[indices]
    sh_tie[A_sw_s_req == 0] = 0
    dia_tie = diameter[indices]
    dia_tie[A_sw_s_req == 0] = 0

    ur_tie = A_sw_s_req / A_sw_s[indices]

    df = df.hstack(
        pl.DataFrame({
            "sv_tie": sv_tie,
            "sh_tie": sh_tie,
            "dia_tie": dia_tie,
            "ur_tie": ur_tie,
        })
    )

    return df

def populate_reo_column_vert(
        df, perc, diameter, spacing, n_layers=3,
        cover=30, dia_horz=16
    ):
    """Generate trial column reo of up to n_layers layers, the closest to each of the given percentages
    """

    dia_vert = [f"dia_vert{k}" for k in range(n_layers)]

    # Validation
    pa.DataFrameSchema({
        "t3": pa.Column(pl.Float64),
        "t2": pa.Column(pl.Float64),
    }).validate(df)

    # Input
    df_unique = df.select("t3", "t2").unique()
    t2 = list(df_unique["t2"].to_numpy() * 1000)
    t3 = list(df_unique["t3"].to_numpy() * 1000)

    # Calculation
    trials = {column: [] for column in [*dia_vert, "n_vert0", "n2_horz0", "n3_horz0", "s_vert0"]}
    for width, depth in zip(t2, t3):  # n2_horz0 counts the links across the width, for V2
        layout = _get_closest_reo_perc_column_vert(width, depth, perc, diameter, spacing, cover, dia_horz, n_layers)
        for column, values in zip(trials, layout):
            trials[column].append(values)

    df_reo = (
        pl.DataFrame({"t3": t3, "t2": t2, **trials})
        .explode(*trials)
        .unique()
        .with_columns(
            rebar_type = pl.lit("B"),
            t3 = (pl.col("t3") / 1000).round(3),
            t2 = (pl.col("t2") / 1000).round(3),
            cover = pl.lit(cover),
            dia_horz0 = pl.lit(dia_horz),
            A_s = sum(pl.col("n_vert0") * pl.col(column).pow(2) * np.pi / 4 for column in dia_vert),
        )
        .with_columns(
            perc_vert = pl.col("A_s") / pl.col("t3") / pl.col("t2") / 1e6 * 100,
        )
        .filter(
            pl.col("perc_vert") >= COLUMN_A_VERT_MIN,
        )
        .with_columns(
            pl.col(*dia_vert, "n_vert0", "n2_horz0", "n3_horz0", "s_vert0", "dia_horz0").cast(pl.Int32)
        )
    )

    # Output
    return (
        df
        .update(df_reo, on=["t3", "t2"], how="left")
        .with_columns(
            pl.col("t3", "t2") * 1000,
        )
    )

def _get_closest_reo_perc_column_vert(width, depth, perc, diameter, spacing, cover, dia_horz, n_layers):
    """Get the column reo closest to each of the given percentages

    Each layer has as many bars as the outermost, no larger than the layer outside it, and is used
    only when the layer outside it is the largest bar.
    """

    perc = np.array(perc)
    diameter = np.array(diameter)
    spacing = np.array(spacing)
    diameter_max = np.max(diameter)

    grids = np.meshgrid(diameter, *[np.insert(diameter, 0, 0)] * (n_layers - 1), spacing, indexing='ij')
    layers = np.array([grid.ravel() for grid in grids[:-1]])
    spacing = grids[-1].ravel()

    is_kept = np.logical_and(
        # inner layers no larger than outer layers
        np.all(layers[:-1] >= layers[1:], axis=0),
        # a layer only when the one outside it is maxed
        np.logical_not(np.any(np.logical_and(layers[:-1] != diameter_max, layers[1:] > 0), axis=0)),
    )
    layers = layers[:, is_kept]
    spacing = spacing[is_kept]

    n_b = np.ceil((width - 2 * (cover + dia_horz) - layers[0]) / spacing) + 1
    n_h = np.ceil((depth - 2 * (cover + dia_horz) - layers[0]) / spacing) + 1
    # Link legs in each direction: the perimeter link's two, or more for the bars between the corners
    number2 = np.maximum(np.floor((n_b - 2) / 2), 2).astype(int)
    number3 = np.maximum(np.floor((n_h - 2) / 2), 2).astype(int)
    number = (2 * (n_b + n_h) - 4).astype(int)

    unique_rows = np.unique(np.column_stack((*layers, number, number2, number3, spacing)), axis=0)
    layers = unique_rows[:, :n_layers].T
    number, number2, number3, spacing = unique_rows[:, n_layers:].T

    reo_area = sum((layer ** 2) / 4 * np.pi for layer in layers)
    reo_perc = (number * reo_area / width / depth) * 100

    indices = np.argsort(reo_perc)
    layers = layers[:, indices]
    number, number2, number3, spacing = number[indices], number2[indices], number3[indices], spacing[indices]
    reo_perc = reo_perc[indices]

    diff = np.abs(reo_perc[None, :] - perc[:, None])
    indices = diff.argmin(axis=1)

    return (*layers[:, indices], number[indices], number2[indices], number3[indices], spacing[indices])

def populate_reo_wall_vert(
        df, perc, diameter, spacing, n_layers=6,
        cover=30, dia_horz=16
    ):
    """Generate trial wall reo of up to n_layers layers on each face, the closest to each of the given
    percentages, for the PMM simplified calculation

    Notes:
    1. reo max = 4%
    2. reo min = 0.2%
    """

    dia_vert = [f"dia_vert{k}" for k in range(n_layers)]

    # Validation
    pa.DataFrameSchema({
        "thickness_min": pa.Column(pl.Float64),
    }).validate(df)

    # Input
    thickness_min = df["thickness_min"].to_list()   #m, kept as given to join the trials back on

    # Calculation
    trials = {column: [] for column in [*dia_vert, "s_vert0"]}
    for thickness in np.array(thickness_min) * 1000:  #mm
        layout = _get_closest_reo_perc_wall_vert(thickness, perc, diameter, spacing, n_layers, cover, dia_horz)
        for column, values in zip(trials, layout):
            trials[column].append(values)

    df_reo = (
        pl.DataFrame({"thickness_min": thickness_min, **trials})
        .explode(*trials)
        .unique()
        .with_columns(
            # A horizontal bar outside each layer of vertical bars, none inside the innermost
            *[
                pl.when(pl.col(f"dia_vert{k}") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)).alias(f"dia_horz{k}")
                for k in range(1, n_layers)
            ],
            pl.lit(0).alias(f"dia_horz{n_layers}"),
            rebar_type = pl.lit("B"),
            cover = pl.lit(cover),
            dia_horz0 = pl.lit(dia_horz),
            A_s = sum(pl.col(column).pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0") for column in dia_vert),
        )
        .with_columns(
            perc_vert = pl.col("A_s") / 1000 / pl.col("thickness_min") / 1000 * 100,
        )
        .filter(
            pl.col("perc_vert") >= WALL_A_VERT_MIN,
        )
        .with_columns(
            pl.col(*dia_vert, "s_vert0").cast(pl.Int32)
        )
    )

    # Output
    return (
        df
        .update(df_reo, on=["thickness_min"], how="left")
        .with_columns(
            pl.col("thickness_min", "length") * 1000,
        )
    )

def _get_closest_reo_perc_wall_vert(thickness, perc, diameter, spacing, n_layers, cover, dia_horz):
    """Get the wall reo closest to each of the given percentages, of the layouts that fit the thickness

    Layer k is tried with every diameter once the layers outside it are the largest bar. Layers are 25 mm
    or a bar apart (8.2(2)), and the innermost layers from the two faces must leave as much clear between them.
    """

    perc = np.array(perc)
    diameter = np.sort(np.array(diameter))
    spacing = np.sort(np.array(spacing))
    diameter_max = diameter[-1]

    layers = [np.array([]) for _ in range(n_layers)]
    spacing0 = np.array([])
    for s in spacing:
        for k in range(n_layers):
            for i in range(n_layers):
                if i < k:
                    layer = np.full(diameter.shape, diameter_max)
                elif i == k:
                    layer = diameter
                else:
                    layer = np.zeros(diameter.shape)
                layers[i] = np.concatenate([layers[i], layer])
            spacing0 = np.concatenate([spacing0, np.full(diameter.shape, s)])

    # Depth from each face to the innermost layer, and the clear distance left between the two faces' layers
    depth = cover + dia_horz + layers[0] / 2
    innermost = layers[0]
    for outer, inner in zip(layers, layers[1:]):
        depth = np.where(inner > 0, depth + outer / 2 + np.maximum(outer, 25) + inner / 2, depth)
        innermost = np.where(inner > 0, inner, innermost)
    is_fitting = thickness - 2 * depth - innermost >= np.maximum(innermost, 25)
    layers = [layer[is_fitting] for layer in layers]
    spacing0 = spacing0[is_fitting]

    reo_area = sum((layer ** 2) / 4 * np.pi for layer in layers)
    reo_perc = (reo_area * 2 * 1000 / spacing0 / 1000 / thickness) * 100

    indices = np.argsort(reo_perc)
    layers = [layer[indices] for layer in layers]
    spacing0 = spacing0[indices]
    reo_perc = reo_perc[indices]

    diff = np.abs(reo_perc[None, :] - perc[:, None])
    indices = diff.argmin(axis=1)

    return (*[layer[indices] for layer in layers], spacing0[indices])

def populate_reo_pile_vert_1(
        df, perc, diameter, spacing, 
        cover=75, dia_horz=16
    ):
    """Populate the vertical reo for each pile diameter
    """

    # Validation
    pa.DataFrameSchema({
        "dia": pa.Column(pl.Int32),
    }).validate(df)

    # Input
    dias = df.unique("dia").sort("dia")["dia"].to_list()

    # Calculation
    dia_vert_all, n_vert_all = [], []
    for dia in dias:
        dia_vert, n_vert = _get_closest_reo_perc_pile_vert_1(dia, perc, diameter, spacing, cover, dia_horz)
        dia_vert_all.append(dia_vert)
        n_vert_all.append(n_vert)
    
    df_reo = (
        pl.DataFrame({
            "dia": dias,
            "dia_vert": dia_vert_all,
            "n_vert": n_vert_all,
        })
        .explode("dia_vert", "n_vert")
        .unique()
        .cast({"dia": pl.Int32})
        .with_columns(
            A_s = (pl.col("n_vert") * pl.col("dia_vert").pow(2) / 4 * np.pi).cast(pl.Float64)
        )
        .with_columns(
            perc_vert = pl.col("A_s") / (pl.col("dia").cast(pl.Float64).pow(2) / 4 * np.pi) * 100
        )
    )
    
    return (
        df
        .with_columns(
            # Updated
            dia_vert = pl.lit(None).cast(pl.Int32),
            n_vert = pl.lit(None).cast(pl.Int32),
            A_s = pl.lit(None).cast(pl.Float64),
            perc_vert = pl.lit(None).cast(pl.Float64),

            # Output
            cover = pl.lit(cover),
            dia_horz = pl.lit(dia_horz),
            rebar_type = pl.lit("B"),
        )
        .update(df_reo, on="dia", how="left")
    )

def _get_closest_reo_perc_pile_vert_1(dia, perc, diameter, spacing, cover, dia_horz):
    """Return the closest diameter and number of rebars
    """

    perc = np.array(perc)
    diameter = np.array(diameter)
    spacing = np.array(spacing)

    diameter, spacing = np.meshgrid(diameter, spacing, indexing='ij')
    diameter = diameter.ravel()
    spacing = spacing.ravel()

    stacked = np.column_stack((diameter, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter, spacing = unique_rows.T

    reo_area = (diameter ** 2) / 4 * np.pi
    number = (np.floor((dia - 2 * cover - 2 * dia_horz - diameter) * np.pi / spacing)).astype(np.int32)
    reo_perc = ((number * reo_area) / (dia ** 2 / 4 * np.pi)) * 100

    indices = np.argsort(reo_perc)
    diameter = diameter[indices]
    number = number[indices]
    reo_perc = reo_perc[indices]

    diff = np.abs(reo_perc[None, :] - perc[:, None])
    indices = diff.argmin(axis=1)

    return diameter[indices], number[indices]