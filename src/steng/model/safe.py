"""SafeMixin: pushes a story of the standard model to SAFE
"""

import polars as pl


class SafeMixin:
    def __init__(self):
        """Instance of the SafeMixin class
        """
        pass

    def to_safe_story(self, story, pid=None):
        """Push a story of the standardised model to SAFE
        """

        from steng.connectors.safe import Safe

        story = str(story).strip()

        with Safe(pid=int(pid)) as safe:

            df_story = self._get_story(story)
            DatumName = df_story["story"].item()
            StryNmAbove = df_story["story_above"].item()
            if StryNmAbove is None: StryNmAbove = "_story_above"
            StryNmBelow = df_story["story_below"].item()
            if StryNmBelow is None: StryNmBelow = "_story_below"
            ModelDatum = df_story["elev"].item()
            StryHtAbove = df_story["height_above"].item()
            if StryHtAbove is None: StryHtAbove = 3
            StryHtBelow = df_story["height_below"].item()
            if StryHtBelow is None: StryHtBelow = 3
            safe = safe.update_story(DatumName, StryNmAbove, StryNmBelow, ModelDatum, StryHtAbove, StryHtBelow)

            grades = self._get_grade_unique()
            for grade in grades:
                safe = safe.add_material(grade)

            section_lines, fzs, uzs = self._get_pointspringprop_unique()
            for section_line, fz, uz in zip(section_lines, fzs, uzs):
                safe = safe.add_prop_point_spring(section_line, fz, uz)

            section_lines, grades = self._get_lineprop_unique("beam")
            for section_line, grade in zip(section_lines, grades):
                safe = safe.add_prop_beam(section_line, grade)

            section_lines, grades = self._get_lineprop_unique("brace")
            for section_line, grade in zip(section_lines, grades):
                safe = safe.add_prop_brace(section_line, grade)

            section_lines, grades = self._get_lineprop_unique("column")
            for section_line, grade in zip(section_lines, grades):
                safe = safe.add_prop_column(section_line, grade)
            
            thicknesses, grades = self._get_areaprop_unique("floor")
            for thickness, grade in zip(thicknesses, grades):
                safe = safe.add_prop_slab(thickness, grade)
            
            thicknesses, grades = self._get_areaprop_unique("wall")
            for thickness, grade in zip(thicknesses, grades):
                safe = safe.add_prop_wall(thickness, grade)

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
                safe = safe.add_beam_obj_by_coord(xi, yi, zi, xj, yj, zj, section_line, grade)
            
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
                safe = safe.add_brace_obj_by_coord(xi, yi, zi, xj, yj, zj, section_line, grade)
            
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
                safe = safe.add_column_obj_by_coord(xi, yi, zi, xj, yj, zj, section_line, grade)
            
            df_floor = self.floor.clone().filter(pl.col("story") == story)
            xs = df_floor["xs"].to_list()
            ys = df_floor["ys"].to_list()
            zs = df_floor["zs"].to_list()
            thicknesses = df_floor["thickness"].to_list()
            grades = df_floor["grade"].to_list()
            for x, y, z, thickness, grade in zip(xs, ys, zs, thicknesses, grades):
                safe = safe.add_slab_obj_by_coord(x, y, z, thickness, grade)
            
            df_wall = self.wall.clone().filter(pl.col("story") == story)
            xs = df_wall["xs"].to_list()
            ys = df_wall["ys"].to_list()
            zs = df_wall["zs"].to_list()
            thicknesses = df_wall["thickness"].to_list()
            grades = df_wall["grade"].to_list()
            for x, y, z, thickness, grade in zip(xs, ys, zs, thicknesses, grades):
                safe = safe.add_wall_obj_by_coord(x, y, z, thickness, grade)
            
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
                    safe = safe.add_restraint(x, y, z, restraint_ux, restraint_uy, restraint_uz, restraint_rxx, restraint_ryy, restraint_rzz)
                if section_line is not None:
                    safe = safe.add_point_spring(x, y, z, section_line, fz, uz)

            safe = safe.refresh_view()

        return self