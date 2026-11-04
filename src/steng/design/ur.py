"""Utilisation ratios of design forces against force-moment curves and failure surfaces
"""

import math

import numpy as np
import polars as pl
from tqdm import tqdm

from steng.geometry.rays import rays_triangles_intersection


def calculate_ur_force_moment(df_curve, unique_columns_curve, join_on, df_reaction, unique_columns_reaction):
    """Calculate the pile utilisation ratio given the design forces"""

    df_curve_segment = _convert_curve_segment(df_curve, "moment", "force", unique_columns_curve)

    df_reaction = (
        df_reaction
        .join(df_curve_segment, on=join_on, how="full")
        .drop(join_on + "_right")
    )

    return (
        df_reaction
        .hstack(_calculate_intersection_curve_segment(df_reaction, "moment", "force"))
        .filter(
            pl.col("t") > 0,
            pl.col("u") >= -0,
            pl.col("u") <= 1,
        )
        .with_columns(
            ur_adsec = 1 / pl.col("t")
        )
        .drop("t", "u")
        .group_by(unique_columns_reaction)
        .agg(
            pl.all().sort_by("ur_adsec").last()
        )
    )

def calculate_ur_failure_surface(df_surface, df_force, section_name):
    """Calculate the pile utilisation ratio given the design forces
    """

    df_unique = df_surface.unique(subset="key")
    keys = df_unique.select("key").to_series().to_list()

    schema = {
        section_name: pl.String, 
        "story": pl.String,
        "model": pl.String,
        "outputcase": pl.String, 
        "station": pl.Float64, 
        "key": pl.String, 
        "ur": pl.Float64,
    }
    df_force_ur = pl.DataFrame(schema=schema)

    pbar = tqdm(total=len(keys), desc="Calculating ur failure surface")
    for key in keys:
        
        df_force_surface = (
            df_force
            .clone()
            .filter(pl.col("key") == key)
            .join(df_surface.filter(pl.col("key") == key), on="key", how="left")
            .with_row_index()
            .with_columns(
                unique_force_surface = pl.concat_str(section_name, "story", "model", "outputcase", "station", "key"),
                cos_theta = ((pl.col("p") * pl.col("x")) + (pl.col("m2") * pl.col("yy")) + (pl.col("m3") * pl.col("zz"))) / pl.col("magnitude_force") / pl.col("magnitude_face"),
                intersected = pl.lit(False),
                t = pl.lit(None),
            )
        )

        is_first = True
        k = 20

        while not (
            df_force_surface
            .group_by("unique_force_surface")
            .agg(pl.col("intersected").any().alias("has_true"))
            .select(pl.col("has_true").all())
            .item()
        ):
        
            df_force_surface_filtered = df_force_surface.clone()
            
            if not is_first:
                df_force_surface_filtered = (
                    df_force_surface_filtered
                    .filter(
                        pl.col("unique_force_surface").is_in(
                            df_force_surface_filtered
                            .group_by("unique_force_surface")
                            .agg(pl.col("intersected").any().alias("has_true"))
                            .filter(~pl.col("has_true"))
                            .select("unique_force_surface")
                            .to_series()
                            .to_list()
                        ),
                        pl.col("t").is_null(),
                    )
                )
            is_first = False
            
            unique_force_surface = df_force_surface_filtered.select("unique_force_surface").unique().shape[0]
            k = min(math.floor(100000 / unique_force_surface), k * 5)

            df_force_surface_filtered = (
                df_force_surface_filtered
                .select(
                    pl.all()
                    .top_k_by("cos_theta", k=k)
                    .over("unique_force_surface", mapping_strategy="explode")
                )
            )

            index = np.array(df_force_surface_filtered.select("index").to_series().to_list())
            forces = np.array([
                df_force_surface_filtered.select("p").to_series().to_list(),
                df_force_surface_filtered.select("m2").to_series().to_list(),
                df_force_surface_filtered.select("m3").to_series().to_list(),
            ])
            forces = forces.T
            faces1 = list(zip(
                df_force_surface_filtered.select("x1").to_series().to_list(),
                df_force_surface_filtered.select("yy1").to_series().to_list(),
                df_force_surface_filtered.select("zz1").to_series().to_list(),
            ))
            faces2 = list(zip(
                df_force_surface_filtered.select("x2").to_series().to_list(),
                df_force_surface_filtered.select("yy2").to_series().to_list(),
                df_force_surface_filtered.select("zz2").to_series().to_list(),
            ))
            faces3 = list(zip(
                df_force_surface_filtered.select("x3").to_series().to_list(),
                df_force_surface_filtered.select("yy3").to_series().to_list(),
                df_force_surface_filtered.select("zz3").to_series().to_list(),
            ))
            faces = np.array(list(zip(faces1, faces2, faces3)))

            intersected, t = rays_triangles_intersection(forces, faces)

            df_force_surface_calculated = (
                pl.DataFrame({
                    "index": index,
                    "intersected": intersected,
                    "t": t,
                })
                .cast({"index": pl.UInt32})
            )

            df_force_surface = (
                df_force_surface
                .update(df_force_surface_calculated, on="index", how="left")
            )
        
        df_force_surface = (
            df_force_surface
            .filter(pl.col("intersected") == True)
            .with_columns(
                ur = 1 / pl.col("t"), 
            )
            .select(section_name, "story", "model", "outputcase", "station", "key", "ur")
            .group_by(section_name, "story", "model", "outputcase", "station", "key")
            .agg(pl.col("ur").max())
        )

        df_force_ur = (
            df_force_ur
            .vstack(df_force_surface)
        )

        pbar.update(1)
    pbar.close()

    return df_force_ur

def _convert_curve_segment(df_curve, x, y, unique_columns):
    """Convert to segments"""

    return (
        df_curve
        .select([x, y] + unique_columns)
        .rename({
            x: "x1",
            y: "y1",
        })
        .group_by(unique_columns)
        .agg(pl.all())
        .with_columns(
            x1 = pl.col("x1").list.head(pl.col("x1").list.len() - 1),
            y1 = pl.col("y1").list.head(pl.col("y1").list.len() - 1),
            x2 = pl.col("x1").list.shift(-1).list.head(pl.col("x1").list.len() - 1),
            y2 = pl.col("y1").list.shift(-1).list.head(pl.col("y1").list.len() - 1),
        )
        .with_columns(
            x2 = pl.col("x2") - pl.col("x1"),
            y2 = pl.col("y2") - pl.col("y1"),
        )
        .explode("x1", "y1", "x2", "y2")
    )

def _calculate_intersection_curve_segment(df, x, y):
    rx = np.array(df.select(x).to_series().to_list())
    ry = np.array(df.select(y).to_series().to_list())
    qx = np.array(df.select("x1").to_series().to_list())
    qy = np.array(df.select("y1").to_series().to_list())
    sx = np.array(df.select("x2").to_series().to_list())
    sy = np.array(df.select("y2").to_series().to_list())

    cross_r_s = (rx * sy) - (ry * sx)

    sx_cross = sx / cross_r_s
    sy_cross = sy / cross_r_s
    t = (qx * sy_cross) - (qy * sx_cross)

    rx_cross = rx / cross_r_s
    ry_cross = ry / cross_r_s
    u = (qx * ry_cross) - (qy * rx_cross)

    return (
        pl.DataFrame({
            "t": t,
            "u": u
        })
    )