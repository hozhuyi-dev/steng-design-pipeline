"""RhinoMixin: converts Rhino and Grasshopper geometry to the standard model
"""

import polars as pl

from steng.grasshopper import input_to_list, list_to_tree


class RhinoMixin:
    def __init__(self):
        """Instance of the RhinoMixin class
        """
        pass

    def _from_rhino(self, attribute, x, y, z, keys, values):
        """Read the coordinates and key:value pairs from Rhino
        """

        x, y, z, keys, values = input_to_list(x, y, z, keys, values)

        if attribute in ["story",]:
            x, y, z = x[0], y[0], z[0]
            records = []
            for zi, klist, vlist in zip(z, keys, values):
                row = {"elev": zi}
                row.update(dict(zip(klist, vlist)))
                records.append(row)
            
            df = (
                pl.DataFrame(records)
                .with_columns(pl.col("elev").cast(pl.Float64) / 1000)
            )
        elif attribute in ["point",]:
            x, y, z = x[0], y[0], z[0]
            records = []
            for xi, yi, zi, klist, vlist in zip(x, y, z, keys, values):
                row = {"x": xi, "y": yi, "z": zi}
                row.update(dict(zip(klist, vlist)))
                records.append(row)
            
            df = (
                pl.DataFrame(records)
                .with_columns(pl.col("x", "y", "z").cast(pl.Float64) / 1000)
            )
        else:
            records = []
            for xi, yi, zi, klist, vlist in zip(x, y, z, keys, values):
                row = {"xs": xi, "ys": yi, "zs": zi}
                row.update(dict(zip(klist, vlist)))
                records.append(row)
            
            df = (
                pl.DataFrame(records)
                .with_columns(pl.col("xs", "ys", "zs").list.eval(pl.element() / 1000))
            )

        return df

    def to_rhino(self, attribute):
        """Return the coordinates, layer and key:value pairs to be baked in Rhino
        """

        x_min, y_min, _ = self._get_base_point() # from class_session

        if attribute in ["story",]:
            df_geometry = (
                getattr(self, attribute).clone()
                .rename({"elev": "z"})
                .with_columns(
                    pl.lit(x_min * 1000).alias("x"),
                    pl.lit(y_min * 1000 - 10000).alias("y"),
                    pl.col("z") * 1000,
                    pl.lit(attribute).alias("layer"),
                )
            )
        elif attribute in ["point",]:
            df_geometry = (
                getattr(self, attribute).clone()
                .filter(
                    pl.any_horizontal(pl.col(["restraint_ux", "restraint_uy", "restraint_uz", "restraint_rxx", "restraint_ryy", "restraint_rzz"]))
                    | pl.col("section_line").is_not_null()
                )
                .with_columns(
                    pl.col("x", "y", "z") * 1000,
                    pl.format(attribute).alias("layer"),
                )
            )
        else:
            df_geometry = (
                getattr(self, attribute).clone()
                .rename({"xs": "x", "ys": "y", "zs": "z"})
                .with_columns(
                    pl.col("x", "y", "z").list.eval(pl.element() * 1000),
                    pl.format(attribute).alias("layer"),
                )
            )
        
        keys = list(df_geometry.columns)
        keys.remove("x")
        keys.remove("y")
        keys.remove("z")
        keys.remove("layer")

        return (
            list_to_tree(df_geometry["x"].to_list()),
            list_to_tree(df_geometry["y"].to_list()),
            list_to_tree(df_geometry["z"].to_list()),
            list_to_tree(df_geometry["layer"].to_list()),
            list_to_tree(keys),
            list_to_tree(df_geometry.select(pl.concat_list(keys).alias("value"))["value"].to_list()),
        )