"""Functions for calculations around section

Functions:
assign_shape*: 
calculate*: 
initialise*:
generate_reo: Generate the reo based on required area
populate_reo: Populate with different rebars arrangement
"""

import math

import numpy as np
import pandera.polars as pa
import polars as pl

COLUMN_A_VERT_MIN = 0.2
PILE_A_VERT_MIN = 0.2
WALL_A_VERT_MIN = 0.2

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

def generate_reo_column_horz_1(df, A_sw_s_name, direction, diameter, spacing, ur_limit):
    """Generate the horizontal reinforcement for columns given A_sw_s
    """

    # Validation
    pa.DataFrameSchema({
        A_sw_s_name: pa.Column(pl.Float64),
        "dia_horz0": pa.Column(pl.Int32),
        "n" + direction + "_horz0": pa.Column(pl.Int32),
        "dia_vert0": pa.Column(pl.Int32),
        "dia_vert1": pa.Column(pl.Int32),
        "dia_vert2": pa.Column(pl.Int32),
    }).validate(df)

    # Input
    A_sw_s_req = df[A_sw_s_name].to_numpy()
    dia_horz = df["dia_horz0"].to_numpy()
    n_horz0 = df["n" + direction + "_horz0"].to_numpy()
    dia_vert0 = df["dia_vert0"].to_numpy()
    dia_vert1 = df["dia_vert1"].to_numpy()
    dia_vert2 = df["dia_vert2"].to_numpy()
    diameter = np.array(diameter)
    spacing = np.array(spacing)
    
    # Constant
    n_rows = df.shape[0]

    # Calculations
    cols = np.column_stack((n_horz0, dia_vert0, dia_vert1, dia_vert2))
    cols = np.unique(cols, axis=0)

    s_horz = np.full(n_rows, 0)
    dia_horz = np.full(n_rows, 0)
    A_sw_s_prov = np.full(n_rows, 0.0)
    ur_horz = np.full(n_rows, 0.0)

    diameter, spacing = np.meshgrid(diameter, spacing, indexing='ij')
    diameter = diameter.ravel()
    spacing = spacing.ravel()

    for n_current, d0_current, d1_current, d2_current in cols:
        n_current, d0_current, d1_current, d2_current = (
            int(n_current), int(d0_current), int(d1_current), int(d2_current)
        )
        indices_current = np.logical_and(
            np.logical_and(n_horz0 == n_current, dia_vert0 == d0_current),
            np.logical_and(dia_vert1 == d1_current, dia_vert2 == d2_current)
        )

        d_min = np.floor(((d0_current ** 2 + d1_current ** 2 + d2_current ** 2) ** 0.5) / 4)
        diameter_current = diameter[diameter >= d_min].copy()
        spacing_current = spacing[diameter >= d_min].copy()

        if len(diameter_current) == 0:
            s_horz[indices_current] = np.min(spacing)
            dia_horz[indices_current] = np.max(diameter)
            A_sw_s_prov[indices_current] = n_current * (np.max(diameter) ** 2) / 4 * np.pi / np.min(spacing)

            ur_horz[indices_current] = d_min / np.max(diameter)
        else:
            A_sw_s_current = n_current * (diameter ** 2) / 4 * np.pi / spacing

            indices = np.argsort(A_sw_s_current)
            diameter_current = diameter[indices]
            spacing_current = spacing[indices]
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

def generate_reo_wall_horz_6(df, A_sw_s_name, diameter, spacing, ur_limit):
    """Generate the horizontal reinforcement for walls given A_sw_s
    """
    
    outputs = ["s_horz", "dia_horz0", "dia_horz1", "dia_horz2", "dia_horz3", "dia_horz4", "dia_horz5", "dia_horz6", "ur_horz"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    # Validation
    pa.DataFrameSchema({
        A_sw_s_name: pa.Column(pl.Float64),
        "dia_vert1": pa.Column(pl.Int32),
        "dia_vert2": pa.Column(pl.Int32),
        "dia_vert3": pa.Column(pl.Int32),
        "dia_vert4": pa.Column(pl.Int32),
        "dia_vert5": pa.Column(pl.Int32),
    }).validate(df)
    
    n_rows = df.shape[0]
    A_sw_s_req = df[A_sw_s_name].to_numpy()
    dia_vert1 = df["dia_vert1"].to_numpy()
    dia_vert2 = df["dia_vert2"].to_numpy()
    dia_vert3 = df["dia_vert3"].to_numpy()
    dia_vert4 = df["dia_vert4"].to_numpy()
    dia_vert5 = df["dia_vert5"].to_numpy()

    diameter = np.array(diameter)
    spacing = np.array(spacing)
    diameter_max = np.max(diameter)

    s_horz = np.full(n_rows, 0)
    dia_horz0 = np.full(n_rows, 0)
    dia_horz1 = np.full(n_rows, 0)
    dia_horz2 = np.full(n_rows, 0)
    dia_horz3 = np.full(n_rows, 0)
    dia_horz4 = np.full(n_rows, 0)
    dia_horz5 = np.full(n_rows, 0)
    dia_horz6 = np.full(n_rows, 0)
    A_sw_s_prov = np.full(n_rows, 0.0)
    
    # 1 VERT LAYER
    diameter0, diameter1, spacing = np.meshgrid(diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            diameter0 < diameter1,
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter0 != diameter_max, diameter1 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_0 = dia_vert1 == 0
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_0] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_0, A_sw_s_req == 0)
    s_horz[is_0] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_0] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_0] = diameter1[indices]
    dia_horz1[is_zero] = 0
    A_sw_s_prov[is_0] = A_sw_s[indices]

    # 2 VERT LAYER
    diameter0, diameter1, diameter2, spacing = np.meshgrid(diameter, diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            np.logical_or(
                np.logical_or(diameter0 < diameter1, diameter1 < diameter2),
                diameter0 < diameter2
            ),
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter1 != diameter_max, diameter2 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, diameter2, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2 + diameter2 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_1 = np.logical_and(dia_vert1 != 0, dia_vert2 == 0)
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_1] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_1, A_sw_s_req == 0)
    s_horz[is_1] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_1] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_1] = diameter1[indices]
    dia_horz1[is_zero] = 0
    dia_horz2[is_1] = diameter2[indices]
    dia_horz2[is_zero] = 0
    A_sw_s_prov[is_1] = A_sw_s[indices]

    # 3 VERT LAYER
    diameter0, diameter1, diameter2, diameter3, spacing = np.meshgrid(diameter, diameter, diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    diameter3 = diameter3.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            np.logical_or(
                np.logical_or(
                    np.logical_or(diameter0 < diameter1, diameter1 < diameter2),
                    np.logical_or(diameter0 < diameter2, diameter0 < diameter3)
                ),
                np.logical_or(diameter1 < diameter3, diameter2 < diameter3)
            ),
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter2 != diameter_max, diameter3 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    diameter3 = diameter3[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, diameter2, diameter3, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, diameter3, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2 + diameter2 ** 2 + diameter3 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    diameter3 = diameter3[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_2 = dia_vert2 != 0
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_2] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_2, A_sw_s_req == 0)
    s_horz[is_2] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_2] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_2] = diameter1[indices]
    dia_horz1[is_zero] = 0
    dia_horz2[is_2] = diameter2[indices]
    dia_horz2[is_zero] = 0
    dia_horz3[is_2] = diameter3[indices]
    dia_horz3[is_zero] = 0
    A_sw_s_prov[is_2] = A_sw_s[indices]

    # 4 VERT LAYER
    diameter0, diameter1, diameter2, diameter3, diameter4, spacing = np.meshgrid(diameter, diameter, diameter, diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    diameter3 = diameter3.ravel()
    diameter4 = diameter4.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            np.logical_or(
                np.logical_or(
                    np.logical_or(diameter0 < diameter1, diameter1 < diameter2),
                    np.logical_or(diameter0 < diameter2, diameter0 < diameter3)
                ),
                np.logical_or(
                    np.logical_or(diameter1 < diameter3, diameter2 < diameter3),
                    np.logical_or(
                        np.logical_or(diameter0 < diameter4, diameter1 < diameter4),
                        np.logical_or(diameter2 < diameter4, diameter3 < diameter4)
                    ),
                ),
            ),
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter3 != diameter_max, diameter4 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    diameter3 = diameter3[boolean_exclude]
    diameter4 = diameter4[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, diameter2, diameter3, diameter4, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, diameter3, diameter4, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2 + diameter2 ** 2 + diameter3 ** 2 + diameter4 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    diameter3 = diameter3[indices]
    diameter4 = diameter4[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_3 = dia_vert3 != 0
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_3] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_3, A_sw_s_req == 0)
    s_horz[is_3] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_3] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_3] = diameter1[indices]
    dia_horz1[is_zero] = 0
    dia_horz2[is_3] = diameter2[indices]
    dia_horz2[is_zero] = 0
    dia_horz3[is_3] = diameter3[indices]
    dia_horz3[is_zero] = 0
    dia_horz4[is_3] = diameter4[indices]
    dia_horz4[is_zero] = 0
    A_sw_s_prov[is_3] = A_sw_s[indices]

    # 5 VERT LAYER
    diameter0, diameter1, diameter2, diameter3, diameter4, diameter5, spacing = np.meshgrid(diameter, diameter, diameter, diameter, diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    diameter3 = diameter3.ravel()
    diameter4 = diameter4.ravel()
    diameter5 = diameter5.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            np.logical_or(
                np.logical_or(
                    np.logical_or(diameter0 < diameter1, diameter1 < diameter2),
                    np.logical_or(diameter0 < diameter2, diameter0 < diameter3)
                ),
                np.logical_or(
                    np.logical_or(
                        np.logical_or(diameter1 < diameter3, diameter2 < diameter3),
                        np.logical_or(
                            np.logical_or(diameter0 < diameter4, diameter1 < diameter4),
                            np.logical_or(diameter2 < diameter4, diameter3 < diameter4)
                        ),
                    ),
                    np.logical_or(
                        np.logical_or(
                            np.logical_or(diameter0 < diameter5, diameter1 < diameter5),
                            np.logical_or(diameter2 < diameter5, diameter3 < diameter5)
                        ),
                        diameter4 < diameter5
                    ),
                ),
            ),
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter4 != diameter_max, diameter5 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    diameter3 = diameter3[boolean_exclude]
    diameter4 = diameter4[boolean_exclude]
    diameter5 = diameter5[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, diameter2, diameter3, diameter4, diameter5, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, diameter3, diameter4, diameter5, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2 + diameter2 ** 2 + diameter3 ** 2 + diameter4 ** 2 + diameter5 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    diameter3 = diameter3[indices]
    diameter4 = diameter4[indices]
    diameter5 = diameter5[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_4 = dia_vert4 != 0
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_4] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_4, A_sw_s_req == 0)
    s_horz[is_4] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_4] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_4] = diameter1[indices]
    dia_horz1[is_zero] = 0
    dia_horz2[is_4] = diameter2[indices]
    dia_horz2[is_zero] = 0
    dia_horz3[is_4] = diameter3[indices]
    dia_horz3[is_zero] = 0
    dia_horz4[is_4] = diameter4[indices]
    dia_horz4[is_zero] = 0
    dia_horz5[is_4] = diameter5[indices]
    dia_horz5[is_zero] = 0
    A_sw_s_prov[is_4] = A_sw_s[indices]

    # 6 VERT LAYER
    diameter0, diameter1, diameter2, diameter3, diameter4, diameter5, diameter6, spacing = np.meshgrid(diameter, diameter, diameter, diameter, diameter, diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    diameter3 = diameter3.ravel()
    diameter4 = diameter4.ravel()
    diameter5 = diameter5.ravel()
    diameter6 = diameter6.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            np.logical_or(
                np.logical_or(
                    np.logical_or(diameter0 < diameter1, diameter1 < diameter2),
                    np.logical_or(diameter0 < diameter2, diameter0 < diameter3)
                ),
                np.logical_or(
                    np.logical_or(
                        np.logical_or(diameter1 < diameter3, diameter2 < diameter3),
                        np.logical_or(
                            np.logical_or(diameter0 < diameter4, diameter1 < diameter4),
                            np.logical_or(diameter2 < diameter4, diameter3 < diameter4)
                        ),
                    ),
                    np.logical_or(
                        np.logical_or(
                            np.logical_or(
                                np.logical_or(diameter0 < diameter5, diameter1 < diameter5),
                                np.logical_or(diameter2 < diameter5, diameter3 < diameter5)
                            ),
                            diameter4 < diameter5
                        ),
                        np.logical_or(
                            np.logical_or(
                                np.logical_or(diameter0 < diameter6, diameter1 < diameter6),
                                np.logical_or(diameter2 < diameter6, diameter3 < diameter6)
                            ),
                            np.logical_or(diameter4 < diameter6, diameter5 < diameter6)
                        ),
                    ),
                ),
            ),
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter5 != diameter_max, diameter6 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    diameter3 = diameter3[boolean_exclude]
    diameter4 = diameter4[boolean_exclude]
    diameter5 = diameter5[boolean_exclude]
    diameter6 = diameter6[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, diameter2, diameter3, diameter4, diameter5, diameter6, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, diameter3, diameter4, diameter5, diameter6, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2 + diameter2 ** 2 + diameter3 ** 2 + diameter4 ** 2 + diameter5 ** 2 + diameter6 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    diameter3 = diameter3[indices]
    diameter4 = diameter4[indices]
    diameter5 = diameter5[indices]
    diameter6 = diameter6[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_5 = dia_vert5 != 0
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_5] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_5, A_sw_s_req == 0)
    s_horz[is_5] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_5] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_5] = diameter1[indices]
    dia_horz1[is_zero] = 0
    dia_horz2[is_5] = diameter2[indices]
    dia_horz2[is_zero] = 0
    dia_horz3[is_5] = diameter3[indices]
    dia_horz3[is_zero] = 0
    dia_horz4[is_5] = diameter4[indices]
    dia_horz4[is_zero] = 0
    dia_horz5[is_5] = diameter5[indices]
    dia_horz5[is_zero] = 0
    dia_horz6[is_5] = diameter6[indices]
    dia_horz6[is_zero] = 0
    A_sw_s_prov[is_5] = A_sw_s[indices]

    ur_horz = A_sw_s_req / A_sw_s_prov

    return(
        df
        .with_columns(
            pl.Series("s_horz", s_horz, dtype=pl.Int32),
            pl.Series("dia_horz0", dia_horz0, dtype=pl.Int32),
            pl.Series("dia_horz1", dia_horz1, dtype=pl.Int32),
            pl.Series("dia_horz2", dia_horz2, dtype=pl.Int32),
            pl.Series("dia_horz3", dia_horz3, dtype=pl.Int32),
            pl.Series("dia_horz4", dia_horz4, dtype=pl.Int32),
            pl.Series("dia_horz5", dia_horz5, dtype=pl.Int32),
            pl.Series("dia_horz6", dia_horz6, dtype=pl.Int32),
            pl.Series("ur_horz", ur_horz, dtype=pl.Float64),
        )
    )

def generate_reo_wall_horz_3(df, A_sw_s_name, diameter, spacing, ur_limit):
    """Generate the horizontal reinforcement for walls given A_sw_s
    """
    
    outputs = ["s_horz", "dia_horz0", "dia_horz1", "dia_horz2", "ur_horz"]
    selected = [item for item in outputs if item in df.columns]
    df = df.drop(selected)

    # Validation
    pa.DataFrameSchema({
        A_sw_s_name: pa.Column(pl.Float64),
        "dia_vert1": pa.Column(pl.Int32),
        "dia_vert2": pa.Column(pl.Int32),
    }).validate(df)
    
    n_rows = df.shape[0]
    A_sw_s_req = df[A_sw_s_name].to_numpy()
    dia_vert1 = df["dia_vert1"].to_numpy()
    dia_vert2 = df["dia_vert2"].to_numpy()

    diameter = np.array(diameter)
    spacing = np.array(spacing)
    diameter_max = np.max(diameter)

    s_horz = np.full(n_rows, 0)
    dia_horz0 = np.full(n_rows, 0)
    dia_horz1 = np.full(n_rows, 0)
    dia_horz2 = np.full(n_rows, 0)
    dia_horz3 = np.full(n_rows, 0)
    A_sw_s_prov = np.full(n_rows, 0.0)
    
    # 1 VERT LAYER
    diameter0, diameter1, spacing = np.meshgrid(diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            diameter0 < diameter1,
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter0 != diameter_max, diameter1 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_0 = dia_vert1 == 0
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_0] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_0, A_sw_s_req == 0)
    s_horz[is_0] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_0] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_0] = diameter1[indices]
    dia_horz1[is_zero] = 0
    A_sw_s_prov[is_0] = A_sw_s[indices]

    # 2 VERT LAYER
    diameter0, diameter1, diameter2, spacing = np.meshgrid(diameter, diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            np.logical_or(
                np.logical_or(diameter0 < diameter1, diameter1 < diameter2),
                diameter0 < diameter2
            ),
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter1 != diameter_max, diameter2 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, diameter2, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2 + diameter2 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_1 = np.logical_and(dia_vert1 != 0, dia_vert2 == 0)
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_1] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_1, A_sw_s_req == 0)
    s_horz[is_1] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_1] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_1] = diameter1[indices]
    dia_horz1[is_zero] = 0
    dia_horz2[is_1] = diameter2[indices]
    dia_horz2[is_zero] = 0
    A_sw_s_prov[is_1] = A_sw_s[indices]

    # 3 VERT LAYER
    diameter0, diameter1, diameter2, diameter3, spacing = np.meshgrid(diameter, diameter, diameter, np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    diameter3 = diameter3.ravel()
    spacing = spacing.ravel()

    boolean_exclude = np.logical_not(
        np.logical_or(
            # inner layer has larger rebar than outer layer
            np.logical_or(
                np.logical_or(
                    np.logical_or(diameter0 < diameter1, diameter1 < diameter2),
                    np.logical_or(diameter0 < diameter2, diameter0 < diameter3)
                ),
                np.logical_or(diameter1 < diameter3, diameter2 < diameter3)
            ),
            # innermost layer can be used only when other layers are maxed
            np.logical_and(diameter0 != diameter_max, diameter1 > 0)
        )
    )
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    diameter3 = diameter3[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, diameter2, diameter3, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, diameter3, spacing = unique_rows.T

    A_sw_s = 2 * (diameter0 ** 2 + diameter1 ** 2 + diameter2 ** 2 + diameter3 ** 2) / 4 * np.pi / spacing

    indices = np.argsort(A_sw_s)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    diameter3 = diameter3[indices]
    spacing = spacing[indices]
    A_sw_s = A_sw_s[indices]

    is_2 = dia_vert2 != 0
    indices = np.searchsorted(A_sw_s, A_sw_s_req[is_2] / ur_limit, side='right')
    indices[indices == len(A_sw_s)] -= 1

    is_zero = np.logical_and(is_2, A_sw_s_req == 0)
    s_horz[is_2] = spacing[indices]
    s_horz[is_zero] = 0
    dia_horz0[is_2] = diameter0[indices]
    dia_horz0[is_zero] = 0
    dia_horz1[is_2] = diameter1[indices]
    dia_horz1[is_zero] = 0
    dia_horz2[is_2] = diameter2[indices]
    dia_horz2[is_zero] = 0
    dia_horz3[is_2] = diameter3[indices]
    dia_horz3[is_zero] = 0
    A_sw_s_prov[is_2] = A_sw_s[indices]

    ur_horz = A_sw_s_req / A_sw_s_prov

    return(
        df
        .with_columns(
            pl.Series("s_horz", s_horz, dtype=pl.Int32),
            pl.Series("dia_horz0", dia_horz0, dtype=pl.Int32),
            pl.Series("dia_horz1", dia_horz1, dtype=pl.Int32),
            pl.Series("dia_horz2", dia_horz2, dtype=pl.Int32),
            pl.Series("ur_horz", ur_horz, dtype=pl.Float64),
        )
    )

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

def populate_reo_column_vert_3(
        df, perc, diameter, spacing, 
        cover=30, dia_horz=16
    ):
    """Generate the column reo up to 3 layers of reo
    """

    # Validation
    pa.DataFrameSchema({
        "t3": pa.Column(pl.Float64),
        "t2": pa.Column(pl.Float64),
    })

    # Input
    df_unique = df.select("t3", "t2").unique()
    t2 = list(df_unique["t2"].to_numpy() * 1000)
    t3 = list(df_unique["t3"].to_numpy() * 1000)

    # Calculation
    dia_vert0_all, dia_vert1_all, dia_vert2_all = [], [], []
    n_vert0_all = []
    n2_horz0_all, n3_horz0_all = [], []
    s_vert0_all = []
    for width, depth in zip(t3, t2):
        dia_vert0, dia_vert1, dia_vert2, n_vert0, n2_horz0, n3_horz0, s_vert0 = _get_closest_reo_perc_column_vert_3(width, depth, perc, diameter, spacing, cover, dia_horz)
        dia_vert0_all.append(dia_vert0)
        dia_vert1_all.append(dia_vert1)
        dia_vert2_all.append(dia_vert2)
        n_vert0_all.append(n_vert0)
        n2_horz0_all.append(n2_horz0)
        n3_horz0_all.append(n3_horz0)
        s_vert0_all.append(s_vert0)

    df_reo = (
        pl.DataFrame({
            "t3": t3,
            "t2": t2,
            "dia_vert0": dia_vert0_all,
            "dia_vert1": dia_vert1_all,
            "dia_vert2": dia_vert2_all,
            "n_vert0": n_vert0_all,
            "n2_horz0": n2_horz0_all,
            "n3_horz0": n3_horz0_all,
            "s_vert0": s_vert0_all,
        })
        .explode("dia_vert0", "dia_vert1", "dia_vert2", "n_vert0", "n2_horz0", "n3_horz0", "s_vert0")
        .unique()
        .with_columns(
            rebar_type = pl.lit("B"),
            t3 = (pl.col("t3") / 1000).round(3),
            t2 = (pl.col("t2") / 1000).round(3),
            cover = pl.lit(cover),
            dia_horz0 = pl.lit(dia_horz),
            A_s = (
                pl.col("n_vert0") * pl.col("dia_vert0").pow(2) * np.pi / 4
                + pl.col("n_vert0") * pl.col("dia_vert1").pow(2) * np.pi / 4
                + pl.col("n_vert0") * pl.col("dia_vert2").pow(2) * np.pi / 4
            ),
        )
        .with_columns(
            perc_vert = pl.col("A_s") / pl.col("t3") / pl.col("t2") / 1e6 * 100,
        )
        .filter(
            pl.col("perc_vert") >= COLUMN_A_VERT_MIN,
        )
        .with_columns(
            pl.col("dia_vert0", "dia_vert1", "dia_vert2", "n_vert0", "n2_horz0", "n3_horz0", "s_vert0", "dia_horz0").cast(pl.Int32)
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

def _get_closest_reo_perc_column_vert_3(width, depth, perc, diameter, spacing, cover, dia_horz):
    """Get the reo closest to the given list of percentages
    """

    perc = np.array(perc)
    diameter = np.array(diameter)
    spacing = np.array(spacing)
    diameter_max = np.max(diameter)

    diameter0, diameter1, diameter2, spacing = np.meshgrid(diameter, np.insert(diameter, 0, 0), np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    spacing = spacing.ravel()

    # Filter when inner layer has larger rebar than outer layer
    boolean_exclude = np.logical_not(np.logical_or(
        np.logical_or(
            np.logical_or(
                diameter0 < diameter1,
                diameter0 < diameter2
            ),
            diameter1 < diameter2
        ),
        np.logical_or(
            np.logical_and(
                diameter0 != diameter_max,
                diameter1 > 0
            ),
            np.logical_and(
                diameter1 != diameter_max,
                diameter2 > 0
            )
        )
    ))
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    spacing = spacing[boolean_exclude]

    n_b = np.ceil((width - 2 * (cover + dia_horz) - diameter0) / spacing) + 1
    n_h = np.ceil((depth - 2 * (cover + dia_horz) - diameter0) / spacing) + 1
    number2 = np.floor((n_b - 2) / 2)
    number2 = number2.astype(int)
    number3 = np.floor((n_h - 2) / 2)
    number3 = number3.astype(int)
    number = 2 * (n_b + n_h) - 4
    number = number.astype(int)

    stacked = np.column_stack((diameter0, diameter1, diameter2, number, number2, number3, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, number, number2, number3, spacing = unique_rows.T

    reo_area0 = (diameter0 ** 2) / 4 * np.pi
    reo_area1 = (diameter1 ** 2) / 4 * np.pi
    reo_area2 = (diameter2 ** 2) / 4 * np.pi
    reo_area = reo_area0 + reo_area1 + reo_area2
    reo_perc = (number * reo_area / width / depth) * 100

    indices = np.argsort(reo_perc)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    number = number[indices]
    number2 = number2[indices]
    number3 = number3[indices]
    spacing = spacing[indices]
    reo_perc = reo_perc[indices]

    diff = np.abs(reo_perc[None, :] - perc[:, None])
    indices = diff.argmin(axis=1)

    return diameter0[indices], diameter1[indices], diameter2[indices], number[indices], number2[indices], number3[indices], spacing[indices]

def populate_reo_wall_vert_6(
        df, perc, diameter, spacing, 
        cover=30, dia_horz=16
    ):
    """Generate reo for wall PMM simplified calculation

    Notes:
    1. reo max = 4%
    2. reo min = 0.2%
    """

    # Validation
    pa.DataFrameSchema({
        "thickness_min": pa.Column(pl.Float64),
    })

    # Input
    thickness_min = list(df["thickness_min"].to_numpy() * 1000)

    # Calculation
    dia_vert0_all, dia_vert1_all, dia_vert2_all, dia_vert3_all, dia_vert4_all, dia_vert5_all = [], [], [], [], [], []
    s_vert0_all = []
    for thickness in thickness_min:
        dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4, dia_vert5, s_vert0 = _get_closest_reo_perc_wall_vert_6(
            thickness, perc, diameter, spacing
        )
        dia_vert0_all.append(dia_vert0)
        dia_vert1_all.append(dia_vert1)
        dia_vert2_all.append(dia_vert2)
        dia_vert3_all.append(dia_vert3)
        dia_vert4_all.append(dia_vert4)
        dia_vert5_all.append(dia_vert5)
        s_vert0_all.append(s_vert0)

    df_reo = (
        pl.DataFrame({
            "thickness_min": thickness_min,
            "dia_vert0": dia_vert0_all,
            "dia_vert1": dia_vert1_all,
            "dia_vert2": dia_vert2_all,
            "dia_vert3": dia_vert3_all,
            "dia_vert4": dia_vert4_all,
            "dia_vert5": dia_vert5_all,
            "s_vert0": s_vert0_all,
        })
        .explode("dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5", "s_vert0")
        .unique()
        .with_columns(
            rebar_type = pl.lit("B"),
            thickness_min = pl.col("thickness_min") / 1000,
            cover = pl.lit(cover),
            dia_horz0 = pl.lit(dia_horz),
            dia_horz1 = pl.when(pl.col("dia_vert1") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz2 = pl.when(pl.col("dia_vert2") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz3 = pl.when(pl.col("dia_vert3") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz4 = pl.when(pl.col("dia_vert4") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz5 = pl.when(pl.col("dia_vert5") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz6 = pl.lit(0),
            A_s = (
                pl.col("dia_vert0").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert1").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert2").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert3").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert4").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert5").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
            ),
        )
        .with_columns(
            perc_vert = pl.col("A_s") / 1000 / pl.col("thickness_min") / 1000 * 100,
        )
        .filter(
            pl.col("perc_vert") >= WALL_A_VERT_MIN,
        )
        .with_columns(
            pl.col("dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5", "s_vert0").cast(pl.Int32)
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

def _get_closest_reo_perc_wall_vert_6(thickness, perc, diameter, spacing):
    """Get the reo closest to the given list of percentages
    """

    perc = np.array(perc)
    diameter = np.sort(np.array(diameter))
    spacing = np.sort(np.array(spacing))
    diameter_max = diameter[-1]

    diameter0 = np.array([])
    diameter1 = np.array([])
    diameter2 = np.array([])
    diameter3 = np.array([])
    diameter4 = np.array([])
    diameter5 = np.array([])
    spacing0 = np.array([])

    for s in spacing:
        diameter0 = np.concatenate([diameter0, diameter])
        diameter1 = np.concatenate([diameter1, np.zeros(diameter.shape)])
        diameter2 = np.concatenate([diameter2, np.zeros(diameter.shape)])
        diameter3 = np.concatenate([diameter3, np.zeros(diameter.shape)])
        diameter4 = np.concatenate([diameter4, np.zeros(diameter.shape)])
        diameter5 = np.concatenate([diameter5, np.zeros(diameter.shape)])
        spacing0 = np.concatenate([spacing0, np.full(diameter.shape, s)])
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, diameter])
        diameter2 = np.concatenate([diameter2, np.zeros(diameter.shape)])
        diameter3 = np.concatenate([diameter3, np.zeros(diameter.shape)])
        diameter4 = np.concatenate([diameter4, np.zeros(diameter.shape)])
        diameter5 = np.concatenate([diameter5, np.zeros(diameter.shape)])
        spacing0 = np.concatenate([spacing0, np.full(diameter.shape, s)])
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, np.full(diameter.shape, diameter_max)])
        diameter2 = np.concatenate([diameter2, diameter])
        diameter3 = np.concatenate([diameter3, np.zeros(diameter.shape)])
        diameter4 = np.concatenate([diameter4, np.zeros(diameter.shape)])
        diameter5 = np.concatenate([diameter5, np.zeros(diameter.shape)])
        spacing0 = np.concatenate([spacing0, np.full(diameter.shape, s)])
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, np.full(diameter.shape, diameter_max)])
        diameter2 = np.concatenate([diameter2, np.full(diameter.shape, diameter_max)])
        diameter3 = np.concatenate([diameter3, diameter])
        diameter4 = np.concatenate([diameter4, np.zeros(diameter.shape)])
        diameter5 = np.concatenate([diameter5, np.zeros(diameter.shape)])
        spacing0 = np.concatenate([spacing0, np.full(diameter.shape, s)])
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, np.full(diameter.shape, diameter_max)])
        diameter2 = np.concatenate([diameter2, np.full(diameter.shape, diameter_max)])
        diameter3 = np.concatenate([diameter3, np.full(diameter.shape, diameter_max)])
        diameter4 = np.concatenate([diameter4, diameter])
        diameter5 = np.concatenate([diameter5, np.zeros(diameter.shape)])
        spacing0 = np.concatenate([spacing0, np.full(diameter.shape, s)])
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, np.full(diameter.shape, diameter_max)])
        diameter2 = np.concatenate([diameter2, np.full(diameter.shape, diameter_max)])
        diameter3 = np.concatenate([diameter3, np.full(diameter.shape, diameter_max)])
        diameter4 = np.concatenate([diameter4, np.full(diameter.shape, diameter_max)])
        diameter5 = np.concatenate([diameter5, diameter])
        spacing0 = np.concatenate([spacing0, np.full(diameter.shape, s)])

    reo_area0 = (diameter0 ** 2) / 4 * np.pi
    reo_area1 = (diameter1 ** 2) / 4 * np.pi
    reo_area2 = (diameter2 ** 2) / 4 * np.pi
    reo_area3 = (diameter3 ** 2) / 4 * np.pi
    reo_area4 = (diameter4 ** 2) / 4 * np.pi
    reo_area5 = (diameter5 ** 2) / 4 * np.pi
    reo_area = reo_area0 + reo_area1 + reo_area2 + reo_area3 + reo_area4 + reo_area5
    reo_perc = (reo_area * 2 * 1000 / spacing0 / 1000 / thickness) * 100

    indices = np.argsort(reo_perc)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    diameter3 = diameter3[indices]
    diameter4 = diameter4[indices]
    diameter5 = diameter5[indices]
    spacing0 = spacing0[indices]
    reo_perc = reo_perc[indices]

    diff = np.abs(reo_perc[None, :] - perc[:, None])
    indices = diff.argmin(axis=1)

    return diameter0[indices], diameter1[indices], diameter2[indices], diameter3[indices], diameter4[indices], diameter5[indices], spacing0[indices]

def populate_reo_wall_vert_5(
        df, perc, diameter, spacing, 
        cover=30, dia_horz=16
    ):
    """Generate reo for wall PMM simplified calculation

    Notes:
    1. reo max = 4%
    2. reo min = 0.2%
    """

    # Validation
    pa.DataFrameSchema({
        "thickness_min": pa.Column(pl.Float64),
    })

    # Input
    thickness_min = list(df["thickness_min"].to_numpy() * 1000)

    # Calculation
    dia_vert0_all, dia_vert1_all, dia_vert2_all, dia_vert3_all, dia_vert4_all = [], [], [], [], []
    s_vert0_all = []
    for thickness in thickness_min:
        dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4, s_vert0 = _get_closest_reo_perc_wall_vert_5(
            thickness, perc, diameter, spacing
        )
        dia_vert0_all.append(dia_vert0)
        dia_vert1_all.append(dia_vert1)
        dia_vert2_all.append(dia_vert2)
        dia_vert3_all.append(dia_vert3)
        dia_vert4_all.append(dia_vert4)
        s_vert0_all.append(s_vert0)

    df_reo = (
        pl.DataFrame({
            "thickness_min": thickness_min,
            "dia_vert0": dia_vert0_all,
            "dia_vert1": dia_vert1_all,
            "dia_vert2": dia_vert2_all,
            "dia_vert3": dia_vert3_all,
            "dia_vert4": dia_vert4_all,
            "s_vert0": s_vert0_all,
        })
        .explode("dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "s_vert0")
        .unique()
        .with_columns(
            rebar_type = pl.lit("B"),
            thickness_min = pl.col("thickness_min") / 1000,
            cover = pl.lit(cover),
            dia_horz0 = pl.lit(dia_horz),
            dia_horz1 = pl.when(pl.col("dia_vert1") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz2 = pl.when(pl.col("dia_vert2") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz3 = pl.when(pl.col("dia_vert3") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz4 = pl.when(pl.col("dia_vert4") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz5 = pl.lit(0),
            A_s = (
                pl.col("dia_vert0").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert1").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert2").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert3").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert4").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
            ),
        )
        .with_columns(
            perc_vert = pl.col("A_s") / 1000 / pl.col("thickness_min") / 1000 * 100,
        )
        .filter(
            pl.col("perc_vert") >= WALL_A_VERT_MIN,
        )
        .with_columns(
            pl.col("dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "s_vert0").cast(pl.Int32)
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

def _get_closest_reo_perc_wall_vert_5(thickness, perc, diameter, spacing):
    """Get the reo closest to the given list of percentages
    """

    perc = np.array(perc)
    diameter = np.sort(np.array(diameter))
    spacing = np.sort(np.array(spacing))
    diameter_max = diameter[-1]

    for s in spacing:
        diameter0 = diameter.copy()
        diameter1 = np.zeros(diameter.shape)
        diameter2 = np.zeros(diameter.shape)
        diameter3 = np.zeros(diameter.shape)
        diameter4 = np.zeros(diameter.shape)
        spacing = np.full(diameter.shape, s)
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, diameter])
        diameter2 = np.concatenate([diameter2, np.zeros(diameter.shape)])
        diameter3 = np.concatenate([diameter3, np.zeros(diameter.shape)])
        diameter4 = np.concatenate([diameter4, np.zeros(diameter.shape)])
        spacing = np.concatenate([spacing, np.full(diameter.shape, s)])
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, np.full(diameter.shape, diameter_max)])
        diameter2 = np.concatenate([diameter2, diameter])
        diameter3 = np.concatenate([diameter3, np.zeros(diameter.shape)])
        diameter4 = np.concatenate([diameter4, np.zeros(diameter.shape)])
        spacing = np.concatenate([spacing, np.full(diameter.shape, s)])
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, np.full(diameter.shape, diameter_max)])
        diameter2 = np.concatenate([diameter2, np.full(diameter.shape, diameter_max)])
        diameter3 = np.concatenate([diameter3, diameter])
        diameter4 = np.concatenate([diameter4, np.zeros(diameter.shape)])
        spacing = np.concatenate([spacing, np.full(diameter.shape, s)])
        diameter0 = np.concatenate([diameter0, np.full(diameter.shape, diameter_max)])
        diameter1 = np.concatenate([diameter1, np.full(diameter.shape, diameter_max)])
        diameter2 = np.concatenate([diameter2, np.full(diameter.shape, diameter_max)])
        diameter3 = np.concatenate([diameter3, np.full(diameter.shape, diameter_max)])
        diameter4 = np.concatenate([diameter4, diameter])
        spacing = np.concatenate([spacing, np.full(diameter.shape, s)])

    reo_area0 = (diameter0 ** 2) / 4 * np.pi
    reo_area1 = (diameter1 ** 2) / 4 * np.pi
    reo_area2 = (diameter2 ** 2) / 4 * np.pi
    reo_area3 = (diameter3 ** 2) / 4 * np.pi
    reo_area4 = (diameter4 ** 2) / 4 * np.pi
    reo_area = reo_area0 + reo_area1 + reo_area2 + reo_area3 + reo_area4
    reo_perc = (reo_area * 2 * 1000 / spacing / 1000 / thickness) * 100

    indices = np.argsort(reo_perc)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    diameter3 = diameter3[indices]
    diameter4 = diameter4[indices]
    spacing = spacing[indices]
    reo_perc = reo_perc[indices]

    diff = np.abs(reo_perc[None, :] - perc[:, None])
    indices = diff.argmin(axis=1)

    return diameter0[indices], diameter1[indices], diameter2[indices], diameter3[indices], diameter4[indices], spacing[indices]

def populate_reo_wall_vert_3(
        df, perc, diameter, spacing, 
        cover=30, dia_horz=16
    ):
    """Generate reo for wall PMM simplified calculation

    Notes:
    1. reo max = 4%
    2. reo min = 0.2%
    """

    # Validation
    pa.DataFrameSchema({
        "thickness_min": pa.Column(pl.Float64),
    })

    # Input
    thickness_min = list(df["thickness_min"].to_numpy() * 1000)

    # Calculation
    dia_vert0_all, dia_vert1_all, dia_vert2_all = [], [], []
    s_vert0_all = []
    for thickness in thickness_min:
        dia_vert0, dia_vert1, dia_vert2, s_vert0 = _get_closest_reo_perc_wall_vert_3(
            thickness, perc, diameter, spacing
        )
        dia_vert0_all.append(dia_vert0)
        dia_vert1_all.append(dia_vert1)
        dia_vert2_all.append(dia_vert2)
        s_vert0_all.append(s_vert0)

    df_reo = (
        pl.DataFrame({
            "thickness_min": thickness_min,
            "dia_vert0": dia_vert0_all,
            "dia_vert1": dia_vert1_all,
            "dia_vert2": dia_vert2_all,
            "s_vert0": s_vert0_all,
        })
        .explode("dia_vert0", "dia_vert1", "dia_vert2", "s_vert0")
        .unique()
        .with_columns(
            rebar_type = pl.lit("B"),
            thickness_min = pl.col("thickness_min") / 1000,
            cover = pl.lit(cover),
            dia_horz0 = pl.lit(dia_horz),
            dia_horz1 = pl.when(pl.col("dia_vert1") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz2 = pl.when(pl.col("dia_vert2") > 0).then(pl.lit(dia_horz)).otherwise(pl.lit(0)),
            dia_horz3 = pl.lit(0),
            A_s = (
                pl.col("dia_vert0").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert1").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
                + pl.col("dia_vert2").pow(2) * np.pi / 4 * 2 * 1000 / pl.col("s_vert0")
            ),
        )
        .with_columns(
            perc_vert = pl.col("A_s") / 1000 / pl.col("thickness_min") / 1000 * 100,
        )
        .filter(
            pl.col("perc_vert") >= WALL_A_VERT_MIN,
        )
        .with_columns(
            pl.col("dia_vert0", "dia_vert1", "dia_vert2", "s_vert0").cast(pl.Int32)
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

def _get_closest_reo_perc_wall_vert_3(thickness, perc, diameter, spacing):
    """Get the reo closest to the given list of percentages
    """

    perc = np.array(perc)
    diameter = np.array(diameter)
    spacing = np.array(spacing)
    diameter_max = np.max(diameter)

    diameter0, diameter1, diameter2, spacing = np.meshgrid(diameter, np.insert(diameter, 0, 0), np.insert(diameter, 0, 0), spacing, indexing='ij')
    diameter0 = diameter0.ravel()
    diameter1 = diameter1.ravel()
    diameter2 = diameter2.ravel()
    spacing = spacing.ravel()

    # Filter when inner layer has larger rebar than outer layer
    boolean_exclude = np.logical_not(np.logical_or(
        np.logical_or(
            np.logical_or(
                diameter0 < diameter1,
                diameter0 < diameter2
            ),
            diameter1 < diameter2
        ),
        np.logical_or(
            np.logical_and(
                diameter0 != diameter_max,
                diameter1 > 0
            ),
            np.logical_and(
                diameter1 != diameter_max,
                diameter2 > 0
            )
        )
    ))
    diameter0 = diameter0[boolean_exclude]
    diameter1 = diameter1[boolean_exclude]
    diameter2 = diameter2[boolean_exclude]
    spacing = spacing[boolean_exclude]

    stacked = np.column_stack((diameter0, diameter1, diameter2, spacing))
    unique_rows = np.unique(stacked, axis=0)
    diameter0, diameter1, diameter2, spacing = unique_rows.T

    reo_area0 = (diameter0 ** 2) / 4 * np.pi
    reo_area1 = (diameter1 ** 2) / 4 * np.pi
    reo_area2 = (diameter2 ** 2) / 4 * np.pi
    reo_area = reo_area0 + reo_area1 + reo_area2
    reo_perc = (reo_area * 2 * 1000 / spacing / 1000 / thickness) * 100

    indices = np.argsort(reo_perc)
    diameter0 = diameter0[indices]
    diameter1 = diameter1[indices]
    diameter2 = diameter2[indices]
    spacing = spacing[indices]
    reo_perc = reo_perc[indices]

    diff = np.abs(reo_perc[None, :] - perc[:, None])
    indices = diff.argmin(axis=1)

    return diameter0[indices], diameter1[indices], diameter2[indices], spacing[indices]

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