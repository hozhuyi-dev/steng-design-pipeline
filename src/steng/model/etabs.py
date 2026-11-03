"""An interface to convert ETABS to standardised model

Interfaces:
EtabsMixin -- A class for interfacing with ETABS

Functions:
"""

import polars as pl

DEFAULT_UZ = 0.0125

class EtabsMixin:
    def __init__(self):
        """Instance of the EtabsMixin class
        """
        pass

    def to_etabs_story(self, story, pid=None, is_add_point=False):
        """Push a story to etabs
        """

        from steng.connectors.etabs import Etabs

        story = str(story).strip()

        with Etabs(pid=int(pid)) as etabs:
            grades = self._get_grade_unique()
            for grade in grades:
                etabs = etabs.add_material(grade)

            section_lines, fzs, uzs = self._get_pointspringprop_unique()
            for section_line, fz, uz in zip(section_lines, fzs, uzs):
                etabs = etabs.add_prop_point_spring(section_line, fz, uz)

            section_lines, grades = self._get_lineprop_unique("beam")
            for section_line, grade in zip(section_lines, grades):
                etabs = etabs.add_prop_beam(section_line, grade)

            section_lines, grades = self._get_lineprop_unique("brace")
            for section_line, grade in zip(section_lines, grades):
                etabs = etabs.add_prop_brace(section_line, grade)

            section_lines, grades = self._get_lineprop_unique("column")
            for section_line, grade in zip(section_lines, grades):
                etabs = etabs.add_prop_column(section_line, grade)
            
            thicknesses, grades = self._get_areaprop_unique("floor")
            for thickness, grade in zip(thicknesses, grades):
                etabs = etabs.add_prop_slab(thickness, grade)
            
            thicknesses, grades = self._get_areaprop_unique("wall")
            for thickness, grade in zip(thicknesses, grades):
                etabs = etabs.add_prop_wall(thickness, grade)
            
            df_beam = (
                self.beam.clone().filter(pl.col("story") == story)
                .with_columns(
                    xi = pl.col("xs").list.get(0),
                    yi = pl.col("ys").list.get(0),
                    zi = pl.col("zs").list.get(0),
                    xj = pl.col("xs").list.get(1),
                    yj = pl.col("ys").list.get(1),
                    zj = pl.col("zs").list.get(1),
                )
            )
            xis = df_beam["xi"].to_list()
            yis = df_beam["yi"].to_list()
            zis = df_beam["zi"].to_list()
            xjs = df_beam["xj"].to_list()
            yjs = df_beam["yj"].to_list()
            zjs = df_beam["zj"].to_list()
            section_lines = df_beam["section_line"].to_list()
            grades = df_beam["grade"].to_list()
            for xi, yi, zi, xj, yj, zj, section_line, grade in zip(xis, yis, zis, xjs, yjs, zjs, section_lines, grades):
                etabs = etabs.add_frame_obj_by_coord(xi, yi, zi, xj, yj, zj, section_line, grade)
            
            df_brace = (
                self.brace.clone().filter(pl.col("story") == story)
                .with_columns(
                    xi = pl.col("xs").list.get(0),
                    yi = pl.col("ys").list.get(0),
                    zi = pl.col("zs").list.get(0),
                    xj = pl.col("xs").list.get(1),
                    yj = pl.col("ys").list.get(1),
                    zj = pl.col("zs").list.get(1),
                )
            )
            xis = df_brace["xi"].to_list()
            yis = df_brace["yi"].to_list()
            zis = df_brace["zi"].to_list()
            xjs = df_brace["xj"].to_list()
            yjs = df_brace["yj"].to_list()
            zjs = df_brace["zj"].to_list()
            section_lines = df_brace["section_line"].to_list()
            grades = df_brace["grade"].to_list()
            for xi, yi, zi, xj, yj, zj, section_line, grade in zip(xis, yis, zis, xjs, yjs, zjs, section_lines, grades):
                etabs = etabs.add_frame_obj_by_coord(xi, yi, zi, xj, yj, zj, section_line, grade)
            
            df_column = (
                self.column.clone().filter(pl.col("story") == story)
                .with_columns(
                    xi = pl.col("xs").list.get(0),
                    yi = pl.col("ys").list.get(0),
                    zi = pl.col("zs").list.get(0),
                    xj = pl.col("xs").list.get(1),
                    yj = pl.col("ys").list.get(1),
                    zj = pl.col("zs").list.get(1),
                )
            )
            xis = df_column["xi"].to_list()
            yis = df_column["yi"].to_list()
            zis = df_column["zi"].to_list()
            xjs = df_column["xj"].to_list()
            yjs = df_column["yj"].to_list()
            zjs = df_column["zj"].to_list()
            section_lines = df_column["section_line"].to_list()
            grades = df_column["grade"].to_list()
            for xi, yi, zi, xj, yj, zj, section_line, grade in zip(xis, yis, zis, xjs, yjs, zjs, section_lines, grades):
                etabs = etabs.add_frame_obj_by_coord(xi, yi, zi, xj, yj, zj, section_line, grade)
            
            df_floor = self.floor.clone().filter(pl.col("story") == story)
            xs = df_floor["xs"].to_list()
            ys = df_floor["ys"].to_list()
            zs = df_floor["zs"].to_list()
            thicknesses = df_floor["thickness"].to_list()
            grades = df_floor["grade"].to_list()
            for x, y, z, thickness, grade in zip(xs, ys, zs, thicknesses, grades):
                etabs = etabs.add_slab_obj_by_coord(x, y, z, thickness, grade)
            
            df_wall = self.wall.clone().filter(pl.col("story") == story)
            xs = df_wall["xs"].to_list()
            ys = df_wall["ys"].to_list()
            zs = df_wall["zs"].to_list()
            thicknesses = df_wall["thickness"].to_list()
            grades = df_wall["grade"].to_list()
            for x, y, z, thickness, grade in zip(xs, ys, zs, thicknesses, grades):
                etabs = etabs.add_wall_obj_by_coord(x, y, z, thickness, grade)
            
            df_point = self.point.clone().filter(pl.col("story") == story)
            xs = df_point["x"].to_list()
            ys = df_point["y"].to_list()
            zs = df_point["z"].to_list()
            restraint_uxs = df_point["restraint_ux"].to_list()
            restraint_uys = df_point["restraint_uy"].to_list()
            restraint_uzs = df_point["restraint_uz"].to_list()
            restraint_rxxs = df_point["restraint_rxx"].to_list()
            restraint_ryys = df_point["restraint_ryy"].to_list()
            restraint_rzzs = df_point["restraint_rzz"].to_list()
            section_lines = df_point["section_line"].to_list()
            fzs = df_point["fz"].to_list()
            uzs = df_point["uz"].to_list()
            for (
                x, y, z, 
                restraint_ux, restraint_uy, restraint_uz, restraint_rxx, restraint_ryy, restraint_rzz, 
                section_line, fz, uz
            ) in zip(
                xs, ys, zs, 
                restraint_uxs, restraint_uys, restraint_uzs, restraint_rxxs, restraint_ryys, restraint_rzzs, 
                section_lines, fzs, uzs
            ):
                if restraint_ux is not None:
                    etabs = etabs.add_restraint(x, y, z, restraint_ux, restraint_uy, restraint_uz, restraint_rxx, restraint_ryy, restraint_rzz)
                if section_line is not None:
                    etabs = etabs.add_point_spring(x, y, z, section_line, fz, uz)
                if is_add_point:
                    etabs = etabs._add_point_obj_by_coord(x, y, z)

            etabs = etabs.refresh_view()

        return self
    
    # ----- geometry -----

    def _expand_story(self):
        """Return a df of stories with elevations"""

        if (
            (self.etabs["tower_and_base_story_definitions"] is None) or
            (self.etabs["story_definitions"] is None)
        ):
            return None

        df_base_story = (
            self.etabs["tower_and_base_story_definitions"]
            .select("bsname", "bselev")
            .rename({
                "bsname": "story",
                "bselev": "elev",
            })
            .with_columns(
                pl.col("elev").cast(pl.Float64),
                height = pl.lit(None).cast(pl.Float64),
            )
            .select("story", "height", "elev")
        )

        return (
            self.etabs["story_definitions"]
            .select("story", "height")
            .cast({
                "height": pl.Float64
            })
            .with_columns(
                elev = pl.lit(None).cast(pl.Float64),
            )
            .vstack(df_base_story)
            .with_columns(
                elev = pl.col("elev").fill_null(strategy="backward") + pl.col("height").cum_sum(reverse=True).fill_null(0)
            )
            .with_columns(
                story_below = pl.col("story").shift(-1),
                elev_below = pl.col("elev").shift(-1),
            )
            .sort("elev", descending=True)
        )
    
    def _expand_point(self):
        """Return a df of points"""

        if self.etabs["point_object_connectivity"] is None:
            return None
        
        df_point = (
            self.etabs["point_object_connectivity"]
            .select("uniquename", "story", "pointbay", "x", "y", "z")
            .rename({
                "uniquename": "unique_point",
                "pointbay": "point",
            })
            .cast({
                "unique_point": pl.UInt32,
                "point": pl.UInt32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            })
            .sort("story", "point")
        )

        if (
            (self.etabs["spring_property_definitions_-_point_springs"] is None) or
            (self.etabs["joint_assignments_-_springs"] is None)
        ):
            df_point = df_point.with_columns(
                section_line = pl.lit(None).cast(pl.String),
                fz = pl.lit(None).cast(pl.Float64),
                uz = pl.lit(None).cast(pl.Float64),
            )
        else:
            df_spring = (
                self.etabs["spring_property_definitions_-_point_springs"]
                .select(
                    pl.col("name").alias("springprop"),
                    (pl.col("stiffuz").cast(pl.Float64) * pl.lit(DEFAULT_UZ)).alias("fz")
                )
                .with_columns(
                    section_line = pl.col("springprop"),
                    uz = pl.lit(DEFAULT_UZ)
                )
            )

            df_springassign = (
                self.etabs["joint_assignments_-_springs"]
                .select(
                    pl.col("label").cast(pl.UInt32).alias("point"),
                    pl.col("story"),
                    pl.col("springprop"),
                )
                .join(df_spring, on="springprop", how="left")
            )

            df_point = (
                df_point
                .join(df_springassign, on=["point", "story"], how="left")
            )

        if self.etabs["joint_assignments_-_restraints"] is None:
            df_point = df_point.with_columns(
                restraint_ux = pl.lit(False),
                restraint_uy = pl.lit(False),
                restraint_uz = pl.lit(False),
                restraint_rxx = pl.lit(False),
                restraint_ryy = pl.lit(False),
                restraint_rzz = pl.lit(False),
            )
        else:
            df_restraint = (
                self.etabs["joint_assignments_-_restraints"]
                .select(
                    pl.col("label").cast(pl.UInt32).alias("point"),
                    pl.col("story"),
                    pl.col("ux"),
                    pl.col("uy"),
                    pl.col("uz"),
                    pl.col("rx"),
                    pl.col("ry"),
                    pl.col("rz"),
                )
                .with_columns(
                    restraint_ux = pl.when(pl.col("ux") == "Yes").then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_uy = pl.when(pl.col("uy") == "Yes").then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_uz = pl.when(pl.col("uz") == "Yes").then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_rxx = pl.when(pl.col("rx") == "Yes").then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_ryy = pl.when(pl.col("ry") == "Yes").then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_rzz = pl.when(pl.col("rz") == "Yes").then(pl.lit(True)).otherwise(pl.lit(False)),
                )
            )

            df_point = (
                df_point
                .join(df_restraint, on=["point", "story"], how="left")
            )

        return df_point
    
    def _expand_beam(self):
        """Return a df of beams"""

        if self.etabs["beam_object_connectivity"] is None:
            return None

        df_point = (
            self.etabs["point_object_connectivity"]
            .select("uniquename", "story", "pointbay", "x", "y", "z")
            .rename({
                "uniquename": "unique_point",
                "pointbay": "point",
            })
            .cast({
                "unique_point": pl.UInt32,
                "point": pl.UInt32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            })
            .sort("story", "point")
        )

        df_beam = (
            self.etabs["beam_object_connectivity"]
            .rename({
                "uniquename": "unique_line",
                "beambay": "beam",
                "uniquepti": "unique_pointj",
                "uniqueptj": "unique_pointi",
            })
            .cast({
                "unique_line": pl.UInt32,
                "unique_pointi": pl.UInt32,
                "unique_pointj": pl.UInt32,
                "length": pl.Float64,
            })
            .with_columns(
                unique_points = pl.concat_list("unique_pointi", "unique_pointj")
            )
            .explode("unique_points")
            .join(df_point.select("unique_point", "x", "y", "z"), left_on=["unique_points"], right_on=["unique_point"], how="left")
            .group_by("beam", "story")
            .agg(
                pl.all().exclude("unique_points", "x", "y", "z").first(),
                pl.col("unique_points", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .sort("story", "beam")
        )

        if self.etabs["frame_assignments_-_local_axes"] is None:
            df_beam = df_beam.with_columns(angle = pl.lit(None).cast(pl.Float64))
        else:
            df_angle = (
                self.etabs["frame_assignments_-_local_axes"]
                .select("story", "label", "angle")
                .rename({
                    "label": "beam",
                })
                .cast({"angle": pl.Float64})
            )

            df_beam = (
                df_beam
                .join(df_angle, on=["story", "beam"], how="left")
            )

        if (
            (self.etabs["frame_assignments_-_section_properties"] is None) or 
            (self.etabs["frame_section_property_definitions_-_concrete_rectangular"] is None)
        ):
            df_beam = (
                df_beam
                .with_columns(
                    grade = pl.lit(None).cast(pl.UInt32),
                    section_line = pl.lit("None").cast(pl.String),
                )
            )
        else:
            df_frameprop = (
                self.etabs["frame_assignments_-_section_properties"]
                .select("story", "label", "shape", "sectprop")
                .rename({
                    "label": "beam",
                })
            )

            df_frameproperty = (
                self.etabs["frame_section_property_definitions_-_concrete_rectangular"]
                .select("name", "material", "t3", "t2")
                .rename({
                    "name": "sectprop",
                })
                .with_columns(
                    grade = pl.col("material").str.extract(r"([0-9]+)", 1).cast(pl.Float64),
                    section_line = pl.format("R(m) {} {}", pl.col("t3"), pl.col("t2")),
                )
            )

            df_beam = (
                df_beam
                .join(df_frameprop, on=["story", "beam"], how="left")
                .join(df_frameproperty, on="sectprop", how="left")
            )

        return df_beam
    
    def _expand_brace(self):
        """Return a df of braces"""

        if self.etabs["brace_object_connectivity"] is None:
            return None
        
        df_point = (
            self.etabs["point_object_connectivity"]
            .select("uniquename", "story", "pointbay", "x", "y", "z")
            .rename({
                "uniquename": "unique_point",
                "pointbay": "point",
            })
            .cast({
                "unique_point": pl.UInt32,
                "point": pl.UInt32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            })
            .sort("story", "point")
        )

        df_brace = (
            self.etabs["brace_object_connectivity"]
            .rename({
                "uniquename": "unique_line",
                "bracebay": "brace",
                "uniquepti": "unique_pointj",
                "uniqueptj": "unique_pointi",
            })
            .cast({
                "unique_line": pl.UInt32,
                "unique_pointi": pl.UInt32,
                "unique_pointj": pl.UInt32,
                "length": pl.Float64,
            })
            .with_columns(
                unique_points = pl.concat_list("unique_pointi", "unique_pointj")
            )
            .explode("unique_points")
            .join(df_point.select("unique_point", "x", "y", "z"), left_on=["unique_points"], right_on=["unique_point"], how="left")
            .group_by("brace", "story")
            .agg(
                pl.all().exclude("unique_points", "x", "y", "z").first(),
                pl.col("unique_points", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .sort("story", "brace")
        )

        if self.etabs["frame_assignments_-_local_axes"] is None:
            df_brace = df_brace.with_columns(angle = pl.lit(None).cast(pl.Float64))
        else:
            df_angle = (
                self.etabs["frame_assignments_-_local_axes"]
                .select("story", "label", "angle")
                .rename({
                    "label": "brace",
                })
                .cast({"angle": pl.Float64})
            )

            df_brace = (
                df_brace
                .join(df_angle, on=["story", "brace"], how="left")
            )

        if (
            (self.etabs["frame_assignments_-_section_properties"] is None) or 
            (self.etabs["frame_section_property_definitions_-_concrete_rectangular"] is None)
        ):
            df_brace = (
                df_brace
                .with_columns(
                    grade = pl.lit(None).cast(pl.UInt32),
                    section_line = pl.lit("None").cast(pl.String),
                )
            )
        else:
            df_frameprop = (
                self.etabs["frame_assignments_-_section_properties"]
                .select("story", "label", "shape", "sectprop")
                .rename({
                    "label": "brace",
                })
            )

            df_frameproperty = (
                self.etabs["frame_section_property_definitions_-_concrete_rectangular"]
                .select("name", "material", "t3", "t2")
                .rename({
                    "name": "sectprop",
                })
                .with_columns(
                    grade = pl.col("material").str.extract(r"([0-9]+)", 1).cast(pl.Float64),
                    section_line = pl.format("R(m) {} {}", pl.col("t3"), pl.col("t2")),
                )
            )

            df_brace = (
                df_brace
                .join(df_frameprop, on=["story", "brace"], how="left")
                .join(df_frameproperty, on="sectprop", how="left")
            )
        
        return df_brace
    
    def _expand_column(self):
        """Expand the column object connectivity
        """

        if self.etabs["column_object_connectivity"] is None:
            return None
        
        df_point = (
            self.etabs["point_object_connectivity"]
            .select("uniquename", "story", "pointbay", "x", "y", "z")
            .rename({
                "uniquename": "unique_point",
                "pointbay": "point",
            })
            .cast({
                "unique_point": pl.UInt32,
                "point": pl.UInt32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            })
            .sort("story", "point")
        )
        
        df_column = (
            self.etabs["column_object_connectivity"]
            .rename({
                "uniquename": "unique_line",
                "columnbay": "column",
                "uniquepti": "unique_pointi",
                "uniqueptj": "unique_pointj",
            })
            .cast({
                "unique_line": pl.UInt32,
                "unique_pointi": pl.UInt32,
                "unique_pointj": pl.UInt32,
                "length": pl.Float64,
            })
            .with_columns(
                unique_points = pl.concat_list("unique_pointi", "unique_pointj")
            )
            .explode("unique_points")
            .join(df_point.select("unique_point", "x", "y", "z"), left_on=["unique_points"], right_on=["unique_point"], how="left")
            .group_by("column", "story")
            .agg(
                pl.all().exclude("unique_points", "x", "y", "z").first(),
                pl.col("unique_points", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .sort("story", "column")
        )

        if self.etabs["frame_assignments_-_local_axes"] is None:
            df_column = df_column.with_columns(angle = pl.lit(None).cast(pl.Float64))
        else:
            df_angle = (
                self.etabs["frame_assignments_-_local_axes"]
                .select("story", "label", "angle")
                .rename({
                    "label": "column",
                })
                .cast({"angle": pl.Float64})
            )

            df_column = (
                df_column
                .join(df_angle, on=["story", "column"], how="left")
            )

        if (
            (self.etabs["frame_assignments_-_section_properties"] is None) or 
            (self.etabs["frame_section_property_definitions_-_concrete_rectangular"] is None)
        ):
            df_column = (
                df_column
                .with_columns(
                    grade = pl.lit(None).cast(pl.UInt32),
                    section_line = pl.lit("None").cast(pl.String),
                )
            )
        else:
            df_frameprop = (
                self.etabs["frame_assignments_-_section_properties"]
                .select("story", "label", "shape", "sectprop")
                .rename({
                    "label": "column",
                })
            )

            df_frameproperty = (
                self.etabs["frame_section_property_definitions_-_concrete_rectangular"]
                .select("name", "material", "t3", "t2")
                .rename({
                    "name": "sectprop",
                })
                .with_columns(
                    grade = pl.col("material").str.extract(r"([0-9]+)", 1).cast(pl.Float64),
                    section_line = pl.format("R(m) {} {}", pl.col("t3"), pl.col("t2")),
                )
            )

            df_column = (
                df_column
                .join(df_frameprop, on=["story", "column"], how="left")
                .join(df_frameproperty, on="sectprop", how="left")
            )
        
        return df_column
    
    def _expand_line(self):
        """Expand the line object connectivity
        """

        if self.etabs["null_line_object_connectivity"] is None:
            return None
        
        df_point = (
            self.etabs["point_object_connectivity"]
            .select("uniquename", "story", "pointbay", "x", "y", "z")
            .rename({
                "uniquename": "unique_point",
                "pointbay": "point",
            })
            .cast({
                "unique_point": pl.UInt32,
                "point": pl.UInt32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            })
            .sort("story", "point")
        )
        
        df_line = (
            self.etabs["null_line_object_connectivity"]
            .rename({
                "uniquename": "unique_line",
                "linebay": "line",
                "uniquepti": "unique_pointi",
                "uniqueptj": "unique_pointj",
            })
            .cast({
                "unique_line": pl.UInt32,
                "unique_pointi": pl.UInt32,
                "unique_pointj": pl.UInt32,
                "length": pl.Float64,
            })
            .with_columns(
                unique_points = pl.concat_list("unique_pointi", "unique_pointj")
            )
            .explode("unique_points")
            .join(df_point.select("unique_point", "x", "y", "z"), left_on=["unique_points"], right_on=["unique_point"], how="left")
            .group_by("line", "story")
            .agg(
                pl.all().exclude("unique_points", "x", "y", "z").first(),
                pl.col("unique_points", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .sort("story", "line")
        )

        return df_line
    
    def _expand_area(self):
        """Return a df of areas"""

        if self.etabs["null_area_object_connectivity"] is None:
            return None
        
        df_point = (
            self.etabs["point_object_connectivity"]
            .select("uniquename", "story", "pointbay", "x", "y", "z")
            .rename({
                "uniquename": "unique_point",
                "pointbay": "point",
            })
            .cast({
                "unique_point": pl.UInt32,
                "point": pl.UInt32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            })
            .sort("story", "point")
        )

        df_area = (
            self.etabs["null_area_object_connectivity"]
            .drop("area")
            .rename({
                "uniquename": "unique_area",
                "areabay": "area",
                "uniquept1": "unique_point1",
                "uniquept2": "unique_point2",
                "uniquept3": "unique_point3",
                "uniquept4": "unique_point4",
            })
            .cast({
                "unique_area": pl.UInt32,
                "unique_point1": pl.UInt32,
                "unique_point2": pl.UInt32,
                "unique_point3": pl.UInt32,
                "unique_point4": pl.UInt32,
                "perimeter": pl.Float64,
            })
            .with_columns(
                unique_points = pl.concat_list("unique_point1", "unique_point2", "unique_point3", "unique_point4")
            )
            .explode("unique_points")
            .filter(pl.col("unique_points").is_not_null())
            .join(df_point.select("unique_point", "x", "y", "z"), left_on=["unique_points"], right_on=["unique_point"], how="left")
            .group_by("area", "story")
            .agg(
                pl.all().exclude("unique_points", "x", "y", "z").first(),
                pl.col("unique_points", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .sort("story", "area")
        )

        return df_area
    
    def _expand_floor(self):
        """Return a df of floors"""

        if self.etabs["floor_object_connectivity"] is None:
            return None
        
        df_point = (
            self.etabs["point_object_connectivity"]
            .select("uniquename", "story", "pointbay", "x", "y", "z")
            .rename({
                "uniquename": "unique_point",
                "pointbay": "point",
            })
            .cast({
                "unique_point": pl.UInt32,
                "point": pl.UInt32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            })
            .sort("story", "point")
        )

        df_floor = (
            self.etabs["floor_object_connectivity"]
            .rename({
                "uniquename": "unique_area",
                "floorbay": "floor",
                "uniquept1": "unique_point1",
                "uniquept2": "unique_point2",
                "uniquept3": "unique_point3",
                "uniquept4": "unique_point4",
            })
            .cast({
                "unique_area": pl.UInt32,
                "unique_point1": pl.UInt32,
                "unique_point2": pl.UInt32,
                "unique_point3": pl.UInt32,
                "unique_point4": pl.UInt32,
                "perimeter": pl.Float64,
                "area": pl.Float64,
            })
            .with_columns(
                unique_points = pl.concat_list("unique_point1", "unique_point2", "unique_point3", "unique_point4")
            )
            .explode("unique_points")
            .filter(pl.col("unique_points").is_not_null())
            .join(df_point.select("unique_point", "x", "y", "z"), left_on=["unique_points"], right_on=["unique_point"], how="left")
            .group_by("floor", "story")
            .agg(
                pl.all().exclude("unique_points", "x", "y", "z").first(),
                pl.col("unique_points", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .sort("story", "floor")
        )

        if (
            (self.etabs["slab_property_definitions"] is None) or
            (self.etabs["area_assignments_-_section_properties"] is None)
        ):
            df_floor = (
                df_floor
                .with_columns(
                    grade = pl.lit(None).cast(pl.Float64),
                    thickness = pl.lit(None).cast(pl.Float64),
                )
            )
        else:
            df_slabprop = (
                self.etabs["slab_property_definitions"]
                .select("name", "material", "slab_thickness")
                .rename({
                    "name": "sectprop",
                    "slab_thickness": "thickness",
                })
                .cast({
                    "thickness": pl.Float64,
                })
                .with_columns(
                    grade = pl.col("material").str.extract(r"([0-9]+)", 1).cast(pl.Float64),
                )
                .drop("material")
            )

            df_areaassign = (
                self.etabs["area_assignments_-_section_properties"]
                .select("uniquename", "sectprop")
                .rename({
                    "uniquename": "unique_area",
                })
                .cast({
                    "unique_area": pl.UInt32,
                })
                .join(df_slabprop, on="sectprop", how="left")
            )

            df_floor = (
                df_floor
                .join(df_areaassign, on="unique_area", how="left")
            )

        return df_floor
    
    def _expand_wall(self):
        """Return a df of walls"""

        if self.etabs["wall_object_connectivity"] is None:
            return None
        
        df_point = (
            self.etabs["point_object_connectivity"]
            .select("uniquename", "story", "pointbay", "x", "y", "z")
            .rename({
                "uniquename": "unique_point",
                "pointbay": "point",
            })
            .cast({
                "unique_point": pl.UInt32,
                "point": pl.UInt32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
            })
            .sort("story", "point")
        )

        df_wall = (
            self.etabs["wall_object_connectivity"]
            .rename({
                "uniquename": "unique_area",
                "wallbay": "wall",
                "uniquept1": "unique_point1",
                "uniquept2": "unique_point2",
                "uniquept3": "unique_point3",
                "uniquept4": "unique_point4",
            })
            .cast({
                "unique_area": pl.UInt32,
                "unique_point1": pl.UInt32,
                "unique_point2": pl.UInt32,
                "unique_point3": pl.UInt32,
                "unique_point4": pl.UInt32,
                "perimeter": pl.Float64,
                "area": pl.Float64,
            })
            .with_columns(
                unique_points = pl.concat_list("unique_point1", "unique_point2", "unique_point3", "unique_point4")
            )
            .explode("unique_points")
            .filter(pl.col("unique_points").is_not_null())
            .join(df_point.select("unique_point", "x", "y", "z"), left_on=["unique_points"], right_on=["unique_point"], how="left")
            .group_by("wall", "story")
            .agg(
                pl.all().exclude("unique_points", "x", "y", "z").first(),
                pl.col("unique_points", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .sort("story", "wall")
        )

        if (
            (self.etabs["wall_property_definitions_-_specified"] is None) or
            (self.etabs["area_assignments_-_section_properties"] is None)
        ):
            df_wall = (
                df_wall
                .with_columns(
                    grade = pl.lit(None).cast(pl.Float64),
                    thickness = pl.lit(None).cast(pl.Float64),
                )
            )
        else:
            df_wallprop = (
                self.etabs["wall_property_definitions_-_specified"]
                .select("name", "material", "thickness")
                .rename({
                    "name": "sectprop",
                })
                .cast({
                    "thickness": pl.Float64,
                })
                .with_columns(
                    grade = pl.col("material").str.extract(r"([0-9]+)", 1).cast(pl.Float64),
                )
                .drop("material")
            )

            df_areaassign = (
                self.etabs["area_assignments_-_section_properties"]
                .select("uniquename", "sectprop")
                .rename({
                    "uniquename": "unique_area",
                })
                .cast({
                    "unique_area": pl.UInt32,
                })
                .join(df_wallprop, on="sectprop", how="left")
            )

            df_wall = (
                df_wall
                .join(df_areaassign, on="unique_area", how="left")
            )

        if self.etabs["area_assignments_-_pier_labels"] is None:
            df_wall = df_wall.with_columns(pier = pl.lit(None).cast(pl.String))
        else:
            df_pierassign = (
                self.etabs["area_assignments_-_pier_labels"]
                .select("uniquename", "piername")
                .rename({
                    "uniquename": "unique_area",
                    "piername": "pier"
                })
                .cast({
                    "unique_area": pl.UInt32,
                })
            )

            df_wall = (
                df_wall
                .join(df_pierassign, on="unique_area", how="left")
            )
        
        df_wall = df_wall.with_columns(angle = pl.lit(None).cast(pl.Float64))
        if self.etabs["pier_section_properties"] is not None:
            df_angle = (
                self.etabs["pier_section_properties"]
                .select("pier", "story", "axisangle")
                .rename({
                    "axisangle": "angle",
                })
                .cast({
                    "angle": pl.Float64,
                })
            )

            df_wall = (
                df_wall
                .update(df_angle, on=["pier", "story"], how="left")
            )

        if self.etabs["area_assignments_-_spandrel_labels"] is None:
            df_wall = df_wall.with_columns(spandrel = pl.lit(None).cast(pl.String))
        else:
            df_spandrelassign = (
                self.etabs["area_assignments_-_spandrel_labels"]
                .select("uniquename", "spandname")
                .rename({
                    "uniquename": "unique_area",
                    "spandname": "spandrel"
                })
                .cast({
                    "unique_area": pl.UInt32,
                })
            )

            df_wall = (
                df_wall
                .join(df_spandrelassign, on="unique_area", how="left")
            )

        return df_wall
    
    # ----- global -----
    
    def _expand_base_reaction(self):
        """Return a df of base reactions
        """

        if self.etabs["base_reactions"] is None:
            return None
        
        if "steptype" in self.etabs["base_reactions"].columns:
            df_reaction = (
                self.etabs["base_reactions"]
                .select("outputcase", "casetype", "steptype", "stepnumber", "fx", "fy", "fz", "mx", "my", "mz")
                .rename({
                    "casetype": "outputcasetype",
                    "mx": "mxx",
                    "my": "myy", 
                    "mz": "mzz",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
            )
        else:
            df_reaction = (
                self.etabs["base_reactions"]
                .select("outputcase", "casetype", "fx", "fy", "fz", "mx", "my", "mz")
                .rename({
                    "casetype": "outputcasetype",
                    "mx": "mxx",
                    "my": "myy", 
                    "mz": "mzz",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                )
                .cast({
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
            )

        return df_reaction
    
    def _expand_story_mass(self):
        """Return a df of story mass
        """

        if self.etabs["mass_summary_by_story"] is None:
            return None

        return (
            self.etabs["mass_summary_by_story"]
            .rename({
                "ux": "mx",
                "uy": "my",
                "uz": "mz",
            })
            .cast({
                "mx": pl.Float64,
                "my": pl.Float64,
                "mz": pl.Float64,
            })
        )
    
    def _expand_story_force(self):
        """Return a df of story force
        """

        if self.etabs["story_forces"] is None:
            return None
        
        if "stepnumber" in self.etabs["story_forces"].columns:
            df_story_force = (
                self.etabs["story_forces"]
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "vx": "fy", 
                    "vy": "fz",
                    "t": "mxx",
                    "mx": "myy",
                    "my": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Top").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
            )
        elif "steptype" in self.etabs["story_forces"].columns:
            df_story_force = (
                self.etabs["story_forces"]
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "vx": "fy", 
                    "vy": "fz",
                    "t": "mxx",
                    "mx": "myy",
                    "my": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Top").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = pl.col("steptype"),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
            )
        else:
            df_story_force = (
                self.etabs["story_forces"]
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "vx": "fy", 
                    "vy": "fz",
                    "t": "mxx",
                    "mx": "myy",
                    "my": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Top").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
            )

        return df_story_force
    
    def _expand_story_displacement(self):
        """Return a df of story displacement
        """

        if self.etabs["story_max_over_avg_displacements"] is None:
            return None
        
        if "stepnumber" in self.etabs["story_max_over_avg_displacements"].columns:
            df_story_displacement = (
                self.etabs["story_max_over_avg_displacements"]
                .rename({
                    "casetype": "outputcasetype",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .pivot("direction", index=["story", "outputcase", "outputcasetype", "steptype", "stepnumber"])
                .rename({
                    "maximum_X": "ux_max",
                    "maximum_Y": "uy_max",
                    "average_X": "ux_avg",
                    "average_Y": "uy_avg",
                })
                .cast({
                    "ux_max": pl.Float64,
                    "uy_max": pl.Float64,
                    "ux_avg": pl.Float64,
                    "uy_avg": pl.Float64,
                })
            )
        elif "steptype" in self.etabs["story_max_over_avg_displacements"].columns:
            df_story_displacement = (
                self.etabs["story_max_over_avg_displacements"]
                .rename({
                    "casetype": "outputcasetype",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = pl.col("steptype"),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .pivot("direction", index=["story", "outputcase", "outputcasetype", "steptype"])
                .rename({
                    "maximum_X": "ux_max",
                    "maximum_Y": "uy_max",
                    "average_X": "ux_avg",
                    "average_Y": "uy_avg",
                })
                .cast({
                    "ux_max": pl.Float64,
                    "uy_max": pl.Float64,
                    "ux_avg": pl.Float64,
                    "uy_avg": pl.Float64,
                })
            )
        else:
            df_story_displacement = (
                self.etabs["story_max_over_avg_displacements"]
                .rename({
                    "casetype": "outputcasetype",
                })
                .pivot("direction", index=["story", "outputcase", "outputcasetype"])
                .rename({
                    "maximum_X": "ux_max",
                    "maximum_Y": "uy_max",
                    "average_X": "ux_avg",
                    "average_Y": "uy_avg",
                })
                .cast({
                    "ux_max": pl.Float64,
                    "uy_max": pl.Float64,
                    "ux_avg": pl.Float64,
                    "uy_avg": pl.Float64,
                })
            )
        
        return df_story_displacement
    
    def _expand_story_drift(self):
        """Return a df of story drift
        """

        if self.etabs["story_max_over_avg_drifts"] is None:
            return None

        if "stepnumber" in self.etabs["story_max_over_avg_drifts"].columns:
            df_story_drift = (
                self.etabs["story_max_over_avg_drifts"]
                .rename({
                    "casetype": "outputcasetype",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = pl.col("steptype"),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .pivot("direction", index=["story", "outputcase", "outputcasetype", "steptype", "stepnumber"])
                .rename({
                    "max_drift_X": "ux_max",
                    "max_drift_Y": "uy_max",
                    "avg_drift_X": "ux_avg",
                    "avg_drift_Y": "uy_avg",
                })
                .cast({
                    "ux_max": pl.Float64,
                    "uy_max": pl.Float64,
                    "ux_avg": pl.Float64,
                    "uy_avg": pl.Float64,
                })
            )
        elif "steptype" in self.etabs["story_max_over_avg_drifts"].columns:
            df_story_drift = (
                self.etabs["story_max_over_avg_drifts"]
                .rename({
                    "casetype": "outputcasetype",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .pivot("direction", index=["story", "outputcase", "outputcasetype", "steptype"])
                .rename({
                    "max_drift_X": "ux_max",
                    "max_drift_Y": "uy_max",
                    "avg_drift_X": "ux_avg",
                    "avg_drift_Y": "uy_avg",
                })
                .cast({
                    "ux_max": pl.Float64,
                    "uy_max": pl.Float64,
                    "ux_avg": pl.Float64,
                    "uy_avg": pl.Float64,
                })
            )
        else:
            df_story_drift = (
                self.etabs["story_max_over_avg_drifts"]
                .rename({
                    "casetype": "outputcasetype",
                })
                .pivot("direction", index=["story", "outputcase", "outputcasetype"])
                .rename({
                    "max_drift_X": "ux_max",
                    "max_drift_Y": "uy_max",
                    "avg_drift_X": "ux_avg",
                    "avg_drift_Y": "uy_avg",
                })
                .cast({
                    "ux_max": pl.Float64,
                    "uy_max": pl.Float64,
                    "ux_avg": pl.Float64,
                    "uy_avg": pl.Float64,
                })
            )

        return df_story_drift
    
    def _expand_modal(self):
        """Return a df of story drift
        """

        if self.etabs["modal_participating_mass_ratios"] is None:
            return None

        return (
            self.etabs["modal_participating_mass_ratios"]
            .rename({
                "case": "outputcase",
                "rx": "rxx",
                "ry": "ryy",
                "rz": "rzz",
            })
            .with_columns(
                outputcase = pl.concat_str("outputcase", "mode", separator="_", ignore_nulls=True),
                outputcasetype = pl.lit("modal"),
            )
        )
    
    # ----- result joint -----

    def _expand_joint_displacement(self):
        """Return a df of joint displacement
        """

        if self.etabs["joint_displacements"] is None:
            return None
        
        if "stepnumber" in self.etabs["joint_displacements"].columns:
            df_displacement = (
                self.etabs["joint_displacements"]
                .select("uniquename", "story", "outputcase", "casetype", "steptype", "stepnumber", "ux", "uy", "uz", "rx", "ry", "rz")
                .rename({
                    "uniquename": "unique_point",
                    "casetype": "outputcasetype",
                    "rx": "rxx",
                    "ry": "ryy", 
                    "rz": "rzz",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "ux": pl.Float64,
                    "uy": pl.Float64,
                    "uz": pl.Float64,
                    "rxx": pl.Float64,
                    "ryy": pl.Float64,
                    "rzz": pl.Float64,
                })
            )
        elif "steptype" in self.etabs["joint_displacements"].columns:
            df_displacement = (
                self.etabs["joint_displacements"]
                .select("uniquename", "story", "outputcase", "casetype", "steptype", "ux", "uy", "uz", "rx", "ry", "rz")
                .rename({
                    "uniquename": "unique_point",
                    "casetype": "outputcasetype",
                    "rx": "rxx",
                    "ry": "ryy", 
                    "rz": "rzz",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.col("steptype").str.to_lowercase().str.replace_all(" ", "") == "").then(pl.lit(None))
                        .otherwise(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "ux": pl.Float64,
                    "uy": pl.Float64,
                    "uz": pl.Float64,
                    "rxx": pl.Float64,
                    "ryy": pl.Float64,
                    "rzz": pl.Float64,
                })
            )
        else:
            df_displacement = (
                self.etabs["joint_displacements"]
                .select("uniquename", "story", "outputcase", "casetype", "ux", "uy", "uz", "rx", "ry", "rz")
                .rename({
                    "uniquename": "unique_point",
                    "casetype": "outputcasetype",
                    "rx": "rxx",
                    "ry": "ryy", 
                    "rz": "rzz",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                )
                .cast({
                    "ux": pl.Float64,
                    "uy": pl.Float64,
                    "uz": pl.Float64,
                    "rxx": pl.Float64,
                    "ryy": pl.Float64,
                    "rzz": pl.Float64,
                })
            )

        return df_displacement

    def _expand_joint_reaction(self):
        """Return a df of joint reaction
        """

        if self.etabs["joint_reactions"] is None:
            return None
        
        if "stepnumber" in self.etabs["joint_reactions"].columns:
            df_reaction =  (
                self.etabs["joint_reactions"]
                .select("uniquename", "story", "outputcase", "casetype", "steptype", "stepnumber", "fx", "fy", "fz", "mx", "my", "mz")
                .rename({
                    "uniquename": "unique_point",
                    "casetype": "outputcasetype",
                    "mx": "mxx",
                    "my": "myy", 
                    "mz": "mzz",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
            )
        elif "steptype" in self.etabs["joint_reactions"].columns:
            df_reaction =  (
                self.etabs["joint_reactions"]
                .select("uniquename", "story", "outputcase", "casetype", "steptype", "fx", "fy", "fz", "mx", "my", "mz")
                .rename({
                    "uniquename": "unique_point",
                    "casetype": "outputcasetype",
                    "mx": "mxx",
                    "my": "myy", 
                    "mz": "mzz",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.col("steptype").str.to_lowercase().str.replace_all(" ", "") == "").then(pl.lit(None))
                        .otherwise(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
            )
        else:
            df_reaction =  (
                self.etabs["joint_reactions"]
                .select("uniquename", "story", "outputcase", "casetype", "fx", "fy", "fz", "mx", "my", "mz")
                .rename({
                    "uniquename": "unique_point",
                    "casetype": "outputcasetype",
                    "mx": "mxx",
                    "my": "myy", 
                    "mz": "mzz",
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                )
                .cast({
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
            )
        
        return df_reaction

    
    # ----- result frame pier -----

    def _expand_force_column(self):
        """Return a df of column force
        """

        if self.etabs["element_forces_-_columns"] is None:
            return None
        
        if "stepnumber" in self.etabs["element_forces_-_columns"].columns:
            df_column_force = (
                self.etabs["element_forces_-_columns"]
                .select("uniquename", "story", "outputcase", "casetype", "steptype", "stepnumber", "station", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "uniquename": "unique_line",
                    "casetype": "outputcasetype",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .select("unique_line", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        elif "steptype" in self.etabs["element_forces_-_columns"].columns:
            df_column_force = (
                self.etabs["element_forces_-_columns"]
                .select("uniquename", "story", "outputcase", "casetype", "steptype", "station", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "uniquename": "unique_line",
                    "casetype": "outputcasetype",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .with_columns(
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = pl.col("steptype"),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .select("unique_line", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        else:
            df_column_force = (
                self.etabs["element_forces_-_columns"]
                .select("uniquename", "story", "outputcase", "casetype", "station", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "uniquename": "unique_line",
                    "casetype": "outputcasetype",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("unique_line", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        
        if self.etabs["design_forces_-_columns"] is None:
            pass
        else:
            df_column_force = df_column_force.vstack(
                self.etabs["design_forces_-_columns"]
                .select("uniquename", "story", "combo", "station", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "uniquename": "unique_line",
                    "combo": "outputcase",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    outputcasetype = pl.lit("combination"),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("unique_line", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        
        return df_column_force
    
    def _expand_force_pier(self):
        """Return a df of pier force
        """

        if self.etabs["pier_forces"] is None:
            return None
        
        if "stepnumber" in self.etabs["pier_forces"].columns:
            df_pier_force = (
                self.etabs["pier_forces"]
                .select("pier", "story", "outputcase", "casetype", "steptype", "stepnumber", "location", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Top").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("pier", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        elif "steptype" in self.etabs["pier_forces"].columns:
            df_pier_force = (
                self.etabs["pier_forces"]
                .select("pier", "story", "outputcase", "casetype", "steptype", "location", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Top").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = pl.col("steptype"),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("pier", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        else:
            df_pier_force = (
                self.etabs["pier_forces"]
                .select("pier", "story", "outputcase", "casetype", "location", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Top").then(pl.lit(1.0)).otherwise(pl.lit(0.0))
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("pier", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        
        if self.etabs["design_forces_-_piers"] is None:
            pass
        else:
            df_pier_force = df_pier_force.vstack(
                self.etabs["design_forces_-_piers"]
                .select("pier", "story", "combo", "location", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "combo": "outputcase",
                    "location": "station",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Top").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                    outputcasetype = pl.lit("combination"),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("pier", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        
        return df_pier_force
    
    def _expand_force_spandrel(self):
        """Return a df of spandrel force
        """

        if self.etabs["spandrel_forces"] is None:
            return None
        
        if "stepnumber" in self.etabs["spandrel_forces"].columns:
            df_spandrel_force = (
                self.etabs["spandrel_forces"]
                .select("spandrel", "story", "outputcase", "casetype", "steptype", "stepnumber", "location", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Right").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = (
                        pl.when(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True) == "").then(pl.lit(None))
                        .otherwise(pl.concat_str(pl.col("steptype").str.to_lowercase().str.replace_all(" ", ""), pl.col("stepnumber"), ignore_nulls=True))
                    ),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("spandrel", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        elif "steptype" in self.etabs["spandrel_forces"].columns:
            df_spandrel_force = (
                self.etabs["spandrel_forces"]
                .select("spandrel", "story", "outputcase", "casetype", "steptype", "location", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Right").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                    outputcasetype = (
                        pl.col("outputcasetype")
                        .str.replace("LinStatic", "static")
                        .str.replace("LinModEigen", "modal")
                        .str.replace("LinRespSpec", "response_spectrum")
                        .str.replace("Combination", "combination")
                    ),
                    outputcasestep = pl.col("steptype"),
                )
                .with_columns(
                    outputcase = pl.concat_str("outputcase", "outputcasestep", separator="_", ignore_nulls=True),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("spandrel", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        else:
            df_spandrel_force = (
                self.etabs["spandrel_forces"]
                .select("spandrel", "story", "outputcase", "casetype", "location", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "casetype": "outputcasetype",
                    "location": "station",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Right").then(pl.lit(1.0)).otherwise(pl.lit(0.0))
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("spandrel", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        
        if self.etabs["design_forces_-_spandrels"] is None:
            pass
        else:
            df_spandrel_force = df_spandrel_force.vstack(
                self.etabs["design_forces_-_spandrels"]
                .select("spandrel", "story", "combo", "location", "p", "v2", "v3", "t", "m2", "m3")
                .rename({
                    "combo": "outputcase",
                    "location": "station",
                    "p": "fx",
                    "v2": "fy",
                    "v3": "fz",
                    "t": "mxx",
                    "m2": "myy",
                    "m3": "mzz",
                })
                .with_columns(
                    station = pl.when(pl.col("station") == "Right").then(pl.lit(1.0)).otherwise(pl.lit(0.0)),
                    outputcasetype = pl.lit("combination"),
                )
                .cast({
                    "station": pl.Float64,
                    "fx": pl.Float64,
                    "fy": pl.Float64,
                    "fz": pl.Float64,
                    "mxx": pl.Float64,
                    "myy": pl.Float64,
                    "mzz": pl.Float64,
                })
                .select("spandrel", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz")
            )
        
        return df_spandrel_force