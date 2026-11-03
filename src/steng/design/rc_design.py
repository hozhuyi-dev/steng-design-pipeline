"""An interface with methods to design RC elemets

Classes:
RcDesignMixin -- An interface for designing RC elements

Functions:
"""

import numpy as np
import polars as pl
import scipy as sp
from tqdm import tqdm

from steng.design import pm as pm_c
from steng.design import rc as rc_c
from steng.design import reo as reo_c
from steng.design import section as sec_c
from steng.design import ur as ur_c
from steng.geometry.section_line import generate_keys, generate_reo_vert, get_columns
from steng.model.adsec import AdSecMixin
from steng.utils import check_folder, generate_full_path

# RC PILE
PILE_SERVICE_STRESS = 11.5 #MPa

# RC WALL
V2_EFFECTIVE_DEPTH_FACTOR = 0.9
V3_EFFECTIVE_DEPTH_FACTOR = 1.0

class RcDesignMixin(AdSecMixin):
    def __init__(self):
        """Initialise instance
        """
        pass

    # ----- rc column -----

    def read_rc_column(self, model, format="parquet"):
        """read state from folder
        """

        model = generate_full_path(model)
        check_folder(model)

        if format == "parquet":
            self.column = pl.read_parquet(model + "column." + format)
            self.force_column = pl.read_parquet(model + "force_column." + format)
            self.design_rc_column_column = pl.read_parquet(model + "design_rc_column_column." + format)
            self.design_rc_column_force_pmm = pl.read_parquet(model + "design_rc_column_force_pmm." + format)
            self.design_rc_column_force_v2 = pl.read_parquet(model + "design_rc_column_force_v2." + format)
            self.design_rc_column_force_v3 = pl.read_parquet(model + "design_rc_column_force_v3." + format)
            self.design_rc_column_force_pmm_adsec = pl.read_parquet(model + "design_rc_column_force_pmm_adsec." + format)
        elif format == "csv":
            self.column = pl.read_csv(model + "column." + format)
            self.force_column = pl.read_csv(model + "force_column." + format)
            self.design_rc_column_column = pl.read_csv(model + "design_rc_column_column." + format)
            self.design_rc_column_force_pmm = pl.read_csv(model + "design_rc_column_force_pmm." + format)
            self.design_rc_column_force_v2 = pl.read_csv(model + "design_rc_column_force_v2." + format)
            self.design_rc_column_force_v3 = pl.read_csv(model + "design_rc_column_force_v3." + format)
            self.design_rc_column_force_pmm_adsec = pl.read_csv(model + "design_rc_column_force_pmm_adsec." + format)

        return self
    
    def write_rc_column(self, model, format="parquet"):
        """Write the current state to folder
        """

        model = generate_full_path(model)
        check_folder(model)

        if format == "parquet":
            self.column.write_parquet(model + "column." + format)
            self.force_column.write_parquet(model + "force_column." + format)
            self.design_rc_column_column.write_parquet(model + "design_rc_column_column." + format)
            self.design_rc_column_force_pmm.write_parquet(model + "design_rc_column_force_pmm." + format)
            self.design_rc_column_force_v2.write_parquet(model + "design_rc_column_force_v2." + format)
            self.design_rc_column_force_v3.write_parquet(model + "design_rc_column_force_v3." + format)
            self.design_rc_column_force_pmm_adsec.write_parquet(model + "design_rc_column_force_pmm_adsec." + format)
        elif format == "csv":
            self.column.write_csv(model + "column." + format)
            self.force_column.write_csv(model + "force_column." + format)
            self.design_rc_column_column.write_csv(model + "design_rc_column_column." + format)
            self.design_rc_column_force_pmm.write_csv(model + "design_rc_column_force_pmm." + format)
            self.design_rc_column_force_v2.write_csv(model + "design_rc_column_force_v2." + format)
            self.design_rc_column_force_v3.write_csv(model + "design_rc_column_force_v3." + format)
            self.design_rc_column_force_pmm_adsec.write_csv(model + "design_rc_column_force_pmm_adsec." + format)

        return self

    def create_rc_column(self):
        """Create from the children models, geometry to get from the latest model, forces to envelope all children models
        """

        self.column = (
            self.get_df_head("column", 1).drop("datetime", "model")
            .with_columns(
                # Input
                k2 = pl.lit(1.0),
                k3 = pl.lit(1.0),
                p_add = pl.lit(0.0),
                m2_add = pl.lit(0.0),
                m3_add = pl.lit(0.0),
                v2_add = pl.lit(0.0),
                v3_add = pl.lit(0.0),

                # Output
                cover = pl.lit(None).cast(pl.Int32),
                rebar_type = pl.lit("B"),

                model_vert = pl.lit(None).cast(pl.String),
                outputcase_vert = pl.lit(None).cast(pl.String),
                station_vert = pl.lit(None).cast(pl.Float64),
                ur_pmm_vert = pl.lit(None).cast(pl.Float64),
                ur_adsec_vert = pl.lit(None).cast(pl.Float64),
                A_s = pl.lit(None).cast(pl.Float64),
                perc_vert = pl.lit(None).cast(pl.Float64),
                dia_vert0 = pl.lit(None).cast(pl.Int32),
                dia_vert1 = pl.lit(None).cast(pl.Int32),
                dia_vert2 = pl.lit(None).cast(pl.Int32),
                n_vert0 = pl.lit(None).cast(pl.Int32),
                s_vert0 = pl.lit(None).cast(pl.Int32),
                n2_horz0 = pl.lit(None).cast(pl.Int32),
                n3_horz0 = pl.lit(None).cast(pl.Int32),

                model_horz2 = pl.lit(None).cast(pl.String),
                outputcase_horz2 = pl.lit(None).cast(pl.String),
                station_horz2 = pl.lit(None).cast(pl.Float64),
                ur_vc_horz2 = pl.lit(None).cast(pl.Float64),
                ur_vmax_horz2 = pl.lit(None).cast(pl.Float64),
                ur_horz2 = pl.lit(None).cast(pl.Float64),
                A_sw_s2 = pl.lit(None).cast(pl.Float64),

                model_horz3 = pl.lit(None).cast(pl.String),
                outputcase_horz3 = pl.lit(None).cast(pl.String),
                station_horz3 = pl.lit(None).cast(pl.Float64),
                ur_vc_horz3 = pl.lit(None).cast(pl.Float64),
                ur_vmax_horz3 = pl.lit(None).cast(pl.Float64),
                ur_horz3 = pl.lit(None).cast(pl.Float64),
                A_sw_s3 = pl.lit(None).cast(pl.Float64),
                
                dia_horz0 = pl.lit(None).cast(pl.Int32),
                s_horz0 = pl.lit(None).cast(pl.Int32),
            )
        )

        self.force_column = (
            self.get_df("force_column").drop("datetime")
            .with_columns(
                # Output
            )
        )

        return self
    
    def design_rc_column(
            self,
            contain_column="", not_contain_column="!",
            contain_story="", not_contain_story="!",
            contain_outputcase="", not_contain_outputcase="!",
            ur_limit_pmm=0.9, ur_limit_v=0.9,
            perc_vert=np.array([1.0, 2.0, 3.0, 4.0]), dia_vert=np.array([10, 12, 16, 20, 25, 32]), s_vert=np.array([150,]),
            dia_horz=np.array([10, 12, 16]), s_horz=np.array([200,]),
            n=1,
        ):
        """Design rc columns based on the following methodology: 

        1. Design using the simplified pmm curve + v2 and v3 (Eurocode 2 simplified method)
        2. Get the governing cases for design and use adsec for pmm design
        3. Select the most optimise reo
        """

        return (
            self
            ._design_rc_column_initialise(
                contain_column, not_contain_column,
                contain_story, not_contain_story,
                contain_outputcase, not_contain_outputcase,
            )
            ._design_rc_column_populate_vert(perc_vert, dia_vert, s_vert)
            ._design_rc_column_calculate_pm_column_3()
            ._design_rc_column_calculate_force_pmm_eccentricity_3()
            ._design_rc_column_calculate_ur_pmm_simplified_biaxial()
            ._design_rc_column_calculate_force_v2_1()
            ._design_rc_column_calculate_force_v3_1()
            ._design_rc_column_generate_reo_horz_1(dia_horz, s_horz, ur_limit_v)
            ._design_rc_column_set_ur_simplified_3()
            .adsec_calculate_pmm("rc_column_u_3", self.design_rc_column_column)
            ._design_rc_column_calculate_force_pmm_adsec(n=n)
            ._design_rc_column_calculate_ur_pmm_adsec()
            ._design_rc_column_adsec_set_ur()
            # TODO to set the data back to column so that this can be exported to other software
        )
    
    def get_rc_column(self, ur_limit, group_column=pl.col("column"), group_story=pl.col("elev")):
        """Return the column table based on a single ur column
        """

        df_max = (
            self.design_rc_column_column
            .select(
                "story", "column", "elev", "grade", "t2", "t3",
                "model_vert", "outputcase_vert", "station_vert",
                "ur_pmm_vert", "ur_adsec_vert",
                "key", "reo_vert", "dia_vert", "perc_vert", "A_s", 
                "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "n2_horz0", "n3_horz0",
                "model_horz2", "outputcase_horz2", "station_horz2",
                "ur_vc_horz2", "ur_vmax_horz2", "ur_horz2", 
                "A_sw_s2", 
                "model_horz3", "outputcase_horz3", "station_horz3",
                "ur_vc_horz3", "ur_vmax_horz3", "ur_horz3", 
                "A_sw_s3", 
                "s_horz0", "dia_horz0", 
            )
            .group_by("story", "column")
            .agg(pl.all().sort_by("ur_adsec_vert").first())
        )

        df_design = (
            self.design_rc_column_column
            .select(
                "story", "column", "elev", "grade", "t2", "t3",
                "model_vert", "outputcase_vert", "station_vert",
                "ur_pmm_vert", "ur_adsec_vert",
                "key", "reo_vert", "dia_vert", "perc_vert", "A_s", 
                "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "n2_horz0", "n3_horz0",
                "model_horz2", "outputcase_horz2", "station_horz2",
                "ur_vc_horz2", "ur_vmax_horz2", "ur_horz2", 
                "A_sw_s2", 
                "model_horz3", "outputcase_horz3", "station_horz3",
                "ur_vc_horz3", "ur_vmax_horz3", "ur_horz3", 
                "A_sw_s3", 
                "s_horz0", "dia_horz0", 
            )
            .filter(pl.col("ur_adsec_vert") <= ur_limit)
            .group_by("story", "column")
            .agg(
                pl.all().sort_by("A_s").first(),
            )
        )

        df_column = (
            df_max
            .update(df_design, on=["column", "story"], how="left")
            .with_columns(
                columns = pl.col("column"),
                column = group_column,
            )
            .with_columns(
                pl.col("t2", "t3").cast(pl.UInt32)
            )
            .group_by("story", "column")
            .agg(
                pl.col("columns"),
                pl.col("t2").min().alias("t2_min"),
                pl.col("t2").max().alias("t2_max"),
                pl.col("t3").min().alias("t3_min"),
                pl.col("t3").max().alias("t3_max"),
                pl.col("grade").min().alias("grade_min"),
                pl.col("elev").first(),
                pl.col(
                    "model_vert", "outputcase_vert", "station_vert",
                    "ur_pmm_vert", "ur_adsec_vert",
                    "key", "reo_vert", "dia_vert", "perc_vert", "A_s", 
                    "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "n2_horz0", "n3_horz0",
                ).sort_by("A_s").last(),
                pl.col(
                    "model_horz2", "outputcase_horz2", "station_horz2",
                    "ur_vc_horz2", "ur_vmax_horz2", "ur_horz2", 
                    "A_sw_s2", 
                    "model_horz3", "outputcase_horz3", "station_horz3",
                    "ur_vc_horz3", "ur_vmax_horz3", "ur_horz3", 
                    "A_sw_s3", 
                    "s_horz0", "dia_horz0", 
                ).sort_by("A_sw_s2").last(),
            )
        )

        df_group = (
            df_column
            .join(
                self.get_df_head("story", 1)
                .select("story", "elev"),
                on="story", how="left"
            )
            .with_columns(
                story_below = pl.col("story").shift(-1),
                elev_below = pl.col("elev").shift(-1),
            )
            .with_columns(
                group = (
                    group_story
                ),
            )
            .group_by("group", "column", "t2_min", "t2_max", "t3_min", "t3_max")
            .agg(
                pl.col("story", "elev").sort_by("elev").last(),
                pl.col("columns", "story_below", "elev_below").sort_by("elev_below").first(),
                pl.col("grade_min").min(),
                pl.col(
                    "model_vert", "outputcase_vert", "station_vert",
                    "ur_pmm_vert", "ur_adsec_vert",
                    "key", "reo_vert", "dia_vert", "perc_vert", "A_s", 
                    "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "n2_horz0", "n3_horz0",
                ).sort_by("A_s").last(),
                pl.col(
                    "model_horz2", "outputcase_horz2", "station_horz2",
                    "ur_vc_horz2", "ur_vmax_horz2", "ur_horz2", 
                    "A_sw_s2", 
                    "model_horz3", "outputcase_horz3", "station_horz3",
                    "ur_vc_horz3", "ur_vmax_horz3", "ur_horz3", 
                    "A_sw_s3", 
                    "s_horz0", "dia_horz0", 
                ).sort_by("A_sw_s2").last(),
            )
            .sort("elev")
            .sort("column")
        )

        return (
            df_group
        )
    
    def _design_rc_column_initialise(
            self,
            contain_column, not_contain_column,
            contain_story, not_contain_story,
            contain_outputcase, not_contain_outputcase,
        ):
        """Initialise the 4 dataframes for design: Column, Force_PMM, Force_V2, Force_V3
        """

        self.design_rc_column_column = (
            self.column
            .filter(
                pl.col("column").str.contains(contain_column),
                ~pl.col("column").str.contains(not_contain_column),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
            )
            .join(
                self.get_df_head("story", 1).select("story", "elev", "height"), 
                on="story", how="left"
            )
            .with_columns(
                x = pl.col("xs").list.first(),
                y = pl.col("ys").list.first(),
                t2 = pl.col("section_line").str.extract(r"(\d+(?:\.\d+)?)", 1).cast(pl.Float64),
                t3 = pl.col("section_line").str.extract_all(r"(\d+(?:\.\d+)?)").list.get(1).cast(pl.Float64),
            )
        )

        self.design_rc_column_force_pmm = (
            self.force_column
            .select("model", "unique_line", "station", "outputcase", "fx", "myy", "mzz")
            .rename({
                "fx": "p_base",
                "myy": "m2_base",
                "mzz": "m3_base"
            })
            .join(
                self.design_rc_column_column
                .select(
                    "unique_line", "column", "story", "height", "grade", "t3", "t2", 
                    "k2", "k3", "p_add", "m2_add", "m3_add"
                )
                .rename({
                    "grade": "f_ck",
                    "t3": "h",
                    "t2": "b",
                    "height": "l"
                }),
                on=["unique_line"], how="left"
            )
            .filter(
                pl.col("column").str.contains(contain_column),
                ~pl.col("column").str.contains(not_contain_column),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
            .with_columns(
                p_base = pl.col("p_base") * -1,
                m2_base = pl.col("m2_base").abs(),
                m3_base = pl.col("m3_base").abs(),
                b = pl.col("b") * 1000, #mm
                h = pl.col("h") * 1000, #mm
                l = pl.col("l") * 1000, #mm
            )
        )

        self.design_rc_column_force_v2 = (
            self.force_column
            .select("model", "unique_line", "station", "outputcase", "fx", "fy")
            .rename({
                "fx": "p_base",
                "fy": "v2_base",
            })
            .join(
                self.design_rc_column_column
                .select(
                    "unique_line", "column", "story", "grade", "t3", "t2", 
                    "p_add", "v2_add"
                ),
                on=["unique_line"],
                how="left"
            )
            .rename({
                "grade": "f_ck",
                "t3": "h",
                "t2": "b_w",
            })
            .with_columns(
                p_base = pl.col("p_base") * -1,
                v2_base = pl.col("v2_base").abs(),
                h = pl.col("h") * 1000,     #mm
                b_w = pl.col("b_w") * 1000  #mm
            )
            .with_columns(
                effective_depth_factor = pl.lit(0.9)
            )
            .filter(
                pl.col("column").str.contains(contain_column),
                ~pl.col("column").str.contains(not_contain_column),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
        )

        self.design_rc_column_force_v3 = (
            self.force_column
            .select("model", "unique_line", "station", "outputcase", "fx", "fz")
            .rename({
                "fx": "p_base",
                "fz": "v3_base",
            })
            .join(
                self.design_rc_column_column
                .select(
                    "unique_line", "column", "story", "grade", "t3", "t2", 
                    "p_add", "v3_add"
                ),
                on=["unique_line"],
                how="left"
            )
            .rename({
                "grade": "f_ck",
                "t2": "h",
                "t3": "b_w",
            })
            .with_columns(
                p_base = pl.col("p_base") * -1,
                v3_base = pl.col("v3_base").abs(),
                h = pl.col("h") * 1000,     #mm
                b_w = pl.col("b_w") * 1000  #mm
            )
            .with_columns(
                effective_depth_factor = pl.lit(0.9)
            )
            .filter(
                pl.col("column").str.contains(contain_column),
                ~pl.col("column").str.contains(not_contain_column),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
        )

        return self
    
    def _design_rc_column_populate_vert(self, perc_vert, dia_vert, s_vert, cover=30, dia_horz=16):
        """Generate vertical reo
        """

        self.design_rc_column_column = sec_c.populate_reo_column_vert_3(self.design_rc_column_column, perc_vert, dia_vert, s_vert, cover=cover, dia_horz=dia_horz)

        self.design_rc_column_column = (
            self.design_rc_column_column
            .with_columns(
                pl.Series("key", generate_keys("rc_column_u_3", self.design_rc_column_column.select(get_columns("rc_column_u_3"))), dtype=pl.String)
            )
        )

        self.design_rc_column_column = generate_reo_vert(self.design_rc_column_column, "rc_column_u_3")

        self.design_rc_column_force_v2 = (
            self.design_rc_column_force_v2
            .join(self.design_rc_column_column, on=["column", "story"], how="left")
        )

        self.design_rc_column_force_v3 = (
            self.design_rc_column_force_v3
            .join(self.design_rc_column_column, on=["column", "story"], how="left")
        )

        return self
    
    def _design_rc_column_calculate_pm_column_3(self):
        """Calculate the pmm simplified curve
        """

        self.design_rc_column_column = pm_c.calculate_pm_column_3(self.design_rc_column_column)

        self.design_rc_column_force_pmm = (
            self.design_rc_column_force_pmm
            .join(self.design_rc_column_column, on=["column", "story"], how="left")
        )

        return self
    
    def _design_rc_column_calculate_force_pmm_eccentricity_3(self):
        """Calculate the pmm check with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=8, desc="Calculating force pmm eccentricity")
        self.design_rc_column_force_pmm = rc_c.calculate_pmm_design(self.design_rc_column_force_pmm)
        pbar.update(1)
        self.design_rc_column_force_pmm = rc_c.calculate_material(self.design_rc_column_force_pmm)
        pbar.update(1)
        self.design_rc_column_force_pmm = rc_c.calculate_effective_length_column(self.design_rc_column_force_pmm)
        self.design_rc_column_force_pmm = rc_c.calculate_effective_depth_3(self.design_rc_column_force_pmm)
        pbar.update(1)
        self.design_rc_column_force_pmm = rc_c.calculate_eccentricity_0_i(self.design_rc_column_force_pmm)
        pbar.update(1)
        self.design_rc_column_force_pmm = rc_c.calculate_creep(self.design_rc_column_force_pmm)
        pbar.update(1)
        self.design_rc_column_force_pmm = rc_c.calculate_slenderness_3(self.design_rc_column_force_pmm)
        pbar.update(1)
        self.design_rc_column_force_pmm = rc_c.calculate_eccentricity_2(self.design_rc_column_force_pmm)
        pbar.update(1)
        self.design_rc_column_force_pmm = rc_c.calculate_design_moment(self.design_rc_column_force_pmm)
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_column_calculate_ur_pmm_simplified_biaxial(self):
        """Calculate the pmm ur with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=2, desc="Calculating ur pmm simplified biaxial")
        self.design_rc_column_force_pmm = rc_c.calculate_moment_capacity_simplified_biaxial(self.design_rc_column_force_pmm)
        pbar.update(1)
        self.design_rc_column_force_pmm = rc_c.calculate_ur_biaxial(self.design_rc_column_force_pmm)
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_column_calculate_force_v2_1(self):
        """Calculate the v2 shear check with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=5, desc="Calculating force v2")
        self.design_rc_column_force_v2 = rc_c.calculate_pv2_design(self.design_rc_column_force_v2)
        pbar.update(1)
        self.design_rc_column_force_v2 = rc_c.calculate_material(self.design_rc_column_force_v2)
        pbar.update(1)
        self.design_rc_column_force_v2 = rc_c.calculate_capacity_shear_concrete_3(self.design_rc_column_force_v2)
        pbar.update(1)
        self.design_rc_column_force_v2 = rc_c.calculate_capacity_shear_max(self.design_rc_column_force_v2)
        pbar.update(1)
        self.design_rc_column_force_v2 = rc_c.calculate_required_shear_reo_single_layer_column(self.design_rc_column_force_v2, "2")
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_column_calculate_force_v3_1(self):
        """Calculate the v3 shear check with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=5, desc="Calculating force v3")
        self.design_rc_column_force_v3 = rc_c.calculate_pv3_design(self.design_rc_column_force_v3)
        pbar.update(1)
        self.design_rc_column_force_v3 = rc_c.calculate_material(self.design_rc_column_force_v3)
        pbar.update(1)
        self.design_rc_column_force_v3 = rc_c.calculate_capacity_shear_concrete_3(self.design_rc_column_force_v3)
        pbar.update(1)
        self.design_rc_column_force_v3 = rc_c.calculate_capacity_shear_max(self.design_rc_column_force_v3)
        pbar.update(1)
        self.design_rc_column_force_v3 = rc_c.calculate_required_shear_reo_single_layer_column(self.design_rc_column_force_v3, "3")
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_column_generate_reo_horz_1(self, diameter, spacing, ur_limit):
        """Generate horz reo
        """

        self.design_rc_column_force_v2 = sec_c.generate_reo_column_horz_1(self.design_rc_column_force_v2, "A_sw_s_req2", "2", diameter, spacing, ur_limit)
        self.design_rc_column_force_v3 = sec_c.generate_reo_column_horz_1(self.design_rc_column_force_v3, "A_sw_s_req3", "3", diameter, spacing, ur_limit)

        return self
    
    def _design_rc_column_set_ur_simplified_3(self):
        """Update the pier table with the utilisation ratios"""

        df_force_pmm = (
            self.design_rc_column_force_pmm
            .select(
                "column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "s_vert0",
                "ur_m2", "ur_m3", "ur_biaxial"
            )
            .group_by("column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "s_vert0")
            .agg(pl.all().max())
            .with_columns(ur_pmm_vert = pl.max_horizontal("ur_m2", "ur_m3", "ur_biaxial"))
        )

        df_force_v2 = (
            self.design_rc_column_force_v2
            .select(
                "column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "s_vert0",
                "model", "outputcase", "station",
                "ur_vc", "ur_vmax", "ur_horz2",
                "A_sw_s_prov2", "s_horz02", "dia_horz02"
            )
            .rename({
                "model": "model_horz2",
                "outputcase": "outputcase_horz2",
                "station": "station_horz2",
                "ur_vc": "ur_vc_horz2",
                "ur_vmax": "ur_vmax_horz2",
            })
            .group_by("column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "s_vert0")
            .agg(
                pl.col("ur_vc_horz2", "ur_vmax_horz2").max(),
                pl.all().exclude("ur_vc_horz2", "ur_vmax_horz2").sort_by("A_sw_s_prov2").last()
            )
        )

        df_force_v = (
            self.design_rc_column_force_v3
            .select(
                "column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "s_vert0",
                "model", "outputcase", "station",
                "ur_vc", "ur_vmax", "ur_horz3",
                "A_sw_s_prov3", "s_horz03", "dia_horz03"
            )
            .rename({
                "model": "model_horz3",
                "outputcase": "outputcase_horz3",
                "station": "station_horz3",
                "ur_vc": "ur_vc_horz3",
                "ur_vmax": "ur_vmax_horz3",
            })
            .group_by("column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "s_vert0")
            .agg(
                pl.col("ur_vc_horz3", "ur_vmax_horz3").max(),
                pl.all().exclude("ur_vc_horz3", "ur_vmax_horz3").sort_by("A_sw_s_prov3").last()
            )
            .join(df_force_v2, on=["column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "s_vert0"], how="left")
            .with_columns(
                dia_horz0 = (
                    pl.when((pl.col("dia_horz03").pow(2) / pl.col("s_horz03")) > (pl.col("dia_horz02").pow(2) / pl.col("s_horz02")))
                    .then(pl.col("dia_horz03")).otherwise(pl.col("dia_horz02"))
                ),
                s_horz0 = (
                    pl.when((pl.col("dia_horz03").pow(2) / pl.col("s_horz03")) > (pl.col("dia_horz02").pow(2) / pl.col("s_horz02")))
                    .then(pl.col("s_horz03")).otherwise(pl.col("s_horz02"))
                ),
                A_sw_s2 = (
                    pl.when((pl.col("dia_horz03").pow(2) / pl.col("s_horz03")) > (pl.col("dia_horz02").pow(2) / pl.col("s_horz02")))
                    .then(
                        pl.col("A_sw_s_prov2") / (pl.col("dia_horz02").pow(2) / pl.col("s_horz02")) * (pl.col("dia_horz03").pow(2) / pl.col("s_horz03"))
                    ).otherwise(pl.col("A_sw_s_prov2"))
                ),
                A_sw_s3 = (
                    pl.when((pl.col("dia_horz03").pow(2) / pl.col("s_horz03")) > (pl.col("dia_horz02").pow(2) / pl.col("s_horz02")))
                    .then(pl.col("A_sw_s_prov3"))
                    .otherwise(
                        pl.col("A_sw_s_prov3") / (pl.col("dia_horz03").pow(2) / pl.col("s_horz03")) * (pl.col("dia_horz02").pow(2) / pl.col("s_horz02"))
                    )
                ),
            )
        )

        self.design_rc_column_column = (
            self.design_rc_column_column
            .update(df_force_pmm, on=["column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2"], how="left")
            .update(df_force_v, on=["column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2"], how="left")
        )

        self.design_rc_column_column = (
            self.design_rc_column_column
            .with_columns(
                pl.Series("key", generate_keys("rc_column_u_3", self.design_rc_column_column.select(get_columns("rc_column_u_3"))), dtype=pl.String)
            )
        )

        self.design_rc_column_force_pmm = (
            self.design_rc_column_force_pmm
            .update(
                self.design_rc_column_column.select("column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "key"),
                on=["column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2"], how="left"
            )
        )

        self.design_rc_column_force_v2 = (
            self.design_rc_column_force_v2
            .update(
                self.design_rc_column_column.select("column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "key"),
                on=["column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2"], how="left"
            )
        )

        self.design_rc_column_force_v3 = (
            self.design_rc_column_force_v3
            .update(
                self.design_rc_column_column.select("column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "key"),
                on=["column", "story", "n_vert0", "dia_vert0", "dia_vert1", "dia_vert2"], how="left"
            )
        )

        return self
    
    def _design_rc_column_calculate_force_pmm_adsec(self, n=1):
        """Initialise the force pmm for adsec check filter based on simplified
        """

        self.design_rc_column_force_pmm_adsec = (
            self.design_rc_column_force_pmm
            .select(
                "column", "story", "model", "outputcase", "station", "key",
                pl.col("p") * -1, pl.col("M_Edtotal2").alias("m2"), pl.col("M_Edtotal3").alias("m3"), pl.max_horizontal("ur_m2", "ur_m3", "ur_biaxial").alias("ur_simplified")
            )
            .group_by("column", "story", "key")
            .agg(
                pl.all().sort_by("ur_simplified", descending=True).head(n)
            )
            .explode("model", "outputcase", "station", "p", "m2", "m3", "ur_simplified")
            .with_columns(
                magnitude_force = (pl.col("p").pow(2) + pl.col("m2").pow(2) + pl.col("m3").pow(2)).sqrt(),
                ur_adsec = pl.lit(None).cast(pl.Float64),
            )
        )

        return self
    
    def _design_rc_column_calculate_ur_pmm_adsec(self):
        """Calculate the utilisation ratio
        """

        df_force_ur = (
            ur_c.calculate_ur_failure_surface(self.surface, self.design_rc_column_force_pmm_adsec, "column")
            .rename({"ur": "ur_adsec"})
        )

        self.design_rc_column_force_pmm_adsec = (
            self.design_rc_column_force_pmm_adsec
            .update(df_force_ur, on=["column", "story", "model", "outputcase", "station", "key"], how="left")
        )

        return self
    
    def _design_rc_column_adsec_set_ur(self):
        """Update the pier table with the utilisation ratio"""

        df_force = (
            self.design_rc_column_force_pmm_adsec
            .select(
                "column", "story", "key", 
                "model", "outputcase", "station", "ur_adsec"
            )
            .rename({
                "model": "model_vert",
                "outputcase": "outputcase_vert",
                "station": "station_vert",
                "ur_adsec": "ur_adsec_vert",
            })
            .group_by("column", "story", "key")
            .agg(pl.all().sort_by("ur_adsec_vert").last())
        )

        self.design_rc_column_column = (
            self.design_rc_column_column
            .update(df_force, on=["column", "story", "key"], how="left")
        )

        return self

    # ----- rc coupling beam -----

    def read_rc_coupling_beam(self, model, format="parquet"):
        """read state from folder
        """

        model = generate_full_path(model)
        check_folder(model)

        if format == "parquet":
            self.wall = pl.read_parquet(model + "wall." + format)
            self.force_spandrel = pl.read_parquet(model + "force_spandrel." + format)
            self.design_rc_coupling_beam_spandrel = pl.read_parquet(model + "design_rc_coupling_beam_spandrel." + format)
            self.design_rc_coupling_beam_force = pl.read_parquet(model + "design_rc_coupling_beam_force." + format)
        elif format == "csv":
            self.wall = pl.read_csv(model + "wall." + format)
            self.force_spandrel = pl.read_csv(model + "force_spandrel." + format)
            self.design_rc_coupling_beam_spandrel = pl.read_csv(model + "design_rc_coupling_beam_spandrel." + format)
            self.design_rc_coupling_beam_force = pl.read_csv(model + "design_rc_coupling_beam_force." + format)
        return self
    
    def write_rc_coupling_beam(self, model, format="parquet"):
        """Write the current state to folder
        """

        model = generate_full_path(model)
        check_folder(model)

        if format == "parquet":
            self.wall.write_parquet(model + "wall." + format)
            self.force_spandrel.write_parquet(model + "force_spandrel." + format)
            self.design_rc_coupling_beam_spandrel.write_parquet(model + "design_rc_coupling_beam_spandrel." + format)
            self.design_rc_coupling_beam_force.write_parquet(model + "design_rc_coupling_beam_force." + format)
        elif format == "csv":
            self.wall.write_csv(model + "wall." + format)
            self.force_spandrel.write_csv(model + "force_spandrel." + format)
            self.design_rc_coupling_beam_spandrel.write_csv(model + "design_rc_coupling_beam_spandrel." + format)
            self.design_rc_coupling_beam_force.write_csv(model + "design_rc_coupling_beam_force." + format)

        return self
    
    def create_rc_coupling_beam(self):
        """Create from the children models, geometry to get from the latest model, forces to envelope all children models
        """

        self.wall = (
            self.get_df_head("wall", 1).drop("datetime", "model")
            .with_columns(
                # Input
                p_add = pl.lit(0.0),
                m3_add = pl.lit(0.0),
                v2_add = pl.lit(0.0),

                # Output
                cover = pl.lit(None).cast(pl.Int32),
                rebar_type = pl.lit("B"),  

                model_flexure = pl.lit(None).cast(pl.String),
                outputcase_flexure = pl.lit(None).cast(pl.String),
                station_flexure = pl.lit(None).cast(pl.Float64),
                ur_flexure = pl.lit(None).cast(pl.Float64),
                ur_top = pl.lit(None).cast(pl.Float64), ur_bot = pl.lit(None).cast(pl.Float64), 
                A_top = pl.lit(None).cast(pl.Float64), A_bot = pl.lit(None).cast(pl.Float64),
                perc_top = pl.lit(None).cast(pl.Float64), perc_bot = pl.lit(None).cast(pl.Float64),
                dia_top00 = pl.lit(None).cast(pl.Int32), dia_bot00 = pl.lit(None).cast(pl.Int32), n_top00 = pl.lit(None).cast(pl.Int32), n_bot00 = pl.lit(None).cast(pl.Int32),
                dia_top01 = pl.lit(None).cast(pl.Int32), dia_bot01 = pl.lit(None).cast(pl.Int32), n_top01 = pl.lit(None).cast(pl.Int32), n_bot01 = pl.lit(None).cast(pl.Int32),
                dia_top02 = pl.lit(None).cast(pl.Int32), dia_bot02 = pl.lit(None).cast(pl.Int32), n_top02 = pl.lit(None).cast(pl.Int32), n_bot02 = pl.lit(None).cast(pl.Int32),
                dia_top03 = pl.lit(None).cast(pl.Int32), dia_bot03 = pl.lit(None).cast(pl.Int32), n_top03 = pl.lit(None).cast(pl.Int32), n_bot03 = pl.lit(None).cast(pl.Int32),
                dia_top04 = pl.lit(None).cast(pl.Int32), dia_bot04 = pl.lit(None).cast(pl.Int32), n_top04 = pl.lit(None).cast(pl.Int32), n_bot04 = pl.lit(None).cast(pl.Int32),
                dia_top05 = pl.lit(None).cast(pl.Int32), dia_bot05 = pl.lit(None).cast(pl.Int32), n_top05 = pl.lit(None).cast(pl.Int32), n_bot05 = pl.lit(None).cast(pl.Int32),
                dia_top06 = pl.lit(None).cast(pl.Int32), dia_bot06 = pl.lit(None).cast(pl.Int32), n_top06 = pl.lit(None).cast(pl.Int32), n_bot06 = pl.lit(None).cast(pl.Int32),
                dia_top07 = pl.lit(None).cast(pl.Int32), dia_bot07 = pl.lit(None).cast(pl.Int32), n_top07 = pl.lit(None).cast(pl.Int32), n_bot07 = pl.lit(None).cast(pl.Int32),
                dia_top08 = pl.lit(None).cast(pl.Int32), dia_bot08 = pl.lit(None).cast(pl.Int32), n_top08 = pl.lit(None).cast(pl.Int32), n_bot08 = pl.lit(None).cast(pl.Int32),
                dia_top09 = pl.lit(None).cast(pl.Int32), dia_bot09 = pl.lit(None).cast(pl.Int32), n_top09 = pl.lit(None).cast(pl.Int32), n_bot09 = pl.lit(None).cast(pl.Int32),

                model_link = pl.lit(None).cast(pl.String),
                outputcase_link = pl.lit(None).cast(pl.String),
                station_link = pl.lit(None).cast(pl.Float64),
                ur_vc_link = pl.lit(None).cast(pl.Float64),
                ur_vmax_link = pl.lit(None).cast(pl.Float64),
                ur_link = pl.lit(None).cast(pl.Float64),
                A_sw_s2 = pl.lit(None).cast(pl.Float64),
                dia_link = pl.lit(None).cast(pl.Int32),
                n_link = pl.lit(None).cast(pl.Int32),
                s_link = pl.lit(None).cast(pl.Int32),
            )
        )

        self.force_spandrel = (
            self.get_df_with_story("force_spandrel").drop("datetime")
            .with_columns(
                # Output
            )
        )

        return self
    
    def design_rc_coupling_beam(
            self,
            contain_spandrel="", not_contain_spandrel="!",
            contain_story="", not_contain_story="!",
            contain_outputcase="", not_contain_outputcase="!",
            ur_limit=0.9,  reduce_depth=0,
            diameter_flexure=np.array([10, 12, 16, 20, 25, 32]),
            diameter_shear=np.array([10, 12, 16, 20, 25, 32]),
            spacing_shear=np.array([100, 150, 200, 250, 300]),
            n=1,
        ):
        """Design rc coupling beams based on the following methodology: 

        TODO
        1. 
        """

        return (
            self
            ._design_rc_coupling_beam_initialise(
                reduce_depth,
                contain_spandrel, not_contain_spandrel,
                contain_story, not_contain_story,
                contain_outputcase, not_contain_outputcase,
            )
            ._design_rc_coupling_beam_generate_spandrel_reo()
            ._design_rc_coupling_beam_calculate_beam_reo(ur_limit, diameter_flexure, diameter_shear, spacing_shear, n)
            ._design_rc_coupling_beam_set_ur_reo()
            # TODO to set the data back to wall so that this can be exported to other software
        )
    
    def _design_rc_coupling_beam_initialise(
            self, reduce_depth,
            contain_spandrel, not_contain_spandrel,
            contain_story, not_contain_story,
            contain_outputcase, not_contain_outputcase,
        ):
        """Initialise the 2 dataframes for design: Spandrel, Force
        """

        self.design_rc_coupling_beam_spandrel = (
            self.sessions[list(self.sessions.keys())[-1]].etabs["spandrel_section_properties"]
            .select("spandrel", "story", "length", "depthlt", "thicklt", "material")
            .rename({
                "depthlt": "depth",
                "thicklt": "thick",
            })
            .cast({
                "length": pl.Float64,
                "depth": pl.Float64,
                "thick": pl.Float64,
            })
            .with_columns(
                pl.col("depth").alias("depth_base"),
                pl.col("depth") - (reduce_depth / 1000),
                pl.col("material").str.strip_chars_start("C").str.split("/").list.get(0).cast(pl.UInt16).alias("cylinder_strength"),
            )
            .drop("material")
            .with_columns(
                p_add = pl.lit(0.0),
                m3_add = pl.lit(0.0),
                v2_add = pl.lit(0.0),
            )
            .join(self.get_df_head("story", 1).select("story", "elev"), on="story", how="left")
            .filter(
                pl.col("spandrel").str.contains(contain_spandrel),
                ~pl.col("spandrel").str.contains(not_contain_spandrel),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
            )
        )

        self.design_rc_coupling_beam_force = (
            self.force_spandrel
            .select(
                spandrel = pl.col("spandrel"),
                story = pl.col("story"),
                outputcase = pl.col("outputcase"),
                station = pl.col("station"),
                p_base = pl.col("fx").cast(pl.Float64) * -1,
                m3_base = pl.col("mzz").cast(pl.Float64).abs(),
                v2_base = pl.col("fy").cast(pl.Float64).abs(),
            )
            .join(
                self.design_rc_coupling_beam_spandrel.select("spandrel", "story", "cylinder_strength", "length", "depth", "thick", "p_add", "m3_add", "v2_add")
                .rename({
                    "cylinder_strength": "f_ck",
                    "thick": "b",
                    "depth": "h",
                    "length": "l",
                }),
                on=["spandrel", "story"], how="left"
            )
            .with_columns(
                b = pl.col("b") * 1000, #mm
                h = pl.col("h") * 1000, #mm
                l = pl.col("l") * 1000, #mm
            )
            .filter(
                pl.col("spandrel").str.contains(contain_spandrel),
                ~pl.col("spandrel").str.contains(not_contain_spandrel),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
        )

        return self
    
    def _design_rc_coupling_beam_generate_spandrel_reo(self):
        """Assign cover and initial link dia
        """

        self.design_rc_coupling_beam_spandrel = (
            self.design_rc_coupling_beam_spandrel
            .with_columns(
                cover = pl.lit(45),
                dia_link = pl.lit(10),
                n_link = pl.lit(2),
                s_link = pl.lit(300),
                dia_top00 = pl.lit(10), dia_bot00 = pl.lit(10), n_top00 = pl.lit(2), n_bot00 = pl.lit(2),
                dia_top01 = pl.lit(0), dia_bot01 = pl.lit(0), n_top01 = pl.lit(0), n_bot01 = pl.lit(0),
                dia_top02 = pl.lit(0), dia_bot02 = pl.lit(0), n_top02 = pl.lit(0), n_bot02 = pl.lit(0),
                dia_top03 = pl.lit(0), dia_bot03 = pl.lit(0), n_top03 = pl.lit(0), n_bot03 = pl.lit(0),
                dia_top04 = pl.lit(0), dia_bot04 = pl.lit(0), n_top04 = pl.lit(0), n_bot04 = pl.lit(0),
                dia_top05 = pl.lit(0), dia_bot05 = pl.lit(0), n_top05 = pl.lit(0), n_bot05 = pl.lit(0),
                dia_top06 = pl.lit(0), dia_bot06 = pl.lit(0), n_top06 = pl.lit(0), n_bot06 = pl.lit(0),
                dia_top07 = pl.lit(0), dia_bot07 = pl.lit(0), n_top07 = pl.lit(0), n_bot07 = pl.lit(0),
                dia_top08 = pl.lit(0), dia_bot08 = pl.lit(0), n_top08 = pl.lit(0), n_bot08 = pl.lit(0),
                dia_top09 = pl.lit(0), dia_bot09 = pl.lit(0), n_top09 = pl.lit(0), n_bot09 = pl.lit(0),
            )
            .with_columns(
                A_top_prov = pl.col("n_top00") * pl.col("dia_top00").pow(2) * np.pi / 4,
                A_bot_prov = pl.col("n_bot00") * pl.col("dia_bot00").pow(2) * np.pi / 4,
                A_top_req = pl.lit(0.0),
                A_bot_req = pl.lit(0.0),
            )
            .with_columns(
                ur_top = pl.lit(0.0),
                ur_bot = pl.lit(0.0),
                ur_vc = pl.lit(0.0),
                ur_vmax = pl.lit(0.0),
                ur_link = pl.lit(0.0),
            )
            .with_columns(
                ur_max = pl.max_horizontal("ur_top", "ur_bot", "ur_vmax", "ur_link"),
                top = (
                    pl.concat_str(
                        pl.when(pl.col("n_top00") > 0).then(pl.concat_str("n_top00", "dia_top00", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top01") > 0).then(pl.concat_str("n_top01", "dia_top01", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top02") > 0).then(pl.concat_str("n_top02", "dia_top02", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top03") > 0).then(pl.concat_str("n_top03", "dia_top03", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top04") > 0).then(pl.concat_str("n_top04", "dia_top04", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top05") > 0).then(pl.concat_str("n_top05", "dia_top05", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top06") > 0).then(pl.concat_str("n_top06", "dia_top06", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top07") > 0).then(pl.concat_str("n_top07", "dia_top07", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top08") > 0).then(pl.concat_str("n_top08", "dia_top08", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top09") > 0).then(pl.concat_str("n_top09", "dia_top09", separator="-H")).otherwise(pl.lit(None)),
                        separator="+",
                        ignore_nulls=True,
                    )
                ),
                bot = (
                    pl.concat_str(
                        pl.when(pl.col("n_bot00") > 0).then(pl.concat_str("n_bot00", "dia_bot00", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot01") > 0).then(pl.concat_str("n_bot01", "dia_bot01", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot02") > 0).then(pl.concat_str("n_bot02", "dia_bot02", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot03") > 0).then(pl.concat_str("n_bot03", "dia_bot03", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot04") > 0).then(pl.concat_str("n_bot04", "dia_bot04", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot05") > 0).then(pl.concat_str("n_bot05", "dia_bot05", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot06") > 0).then(pl.concat_str("n_bot06", "dia_bot06", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot07") > 0).then(pl.concat_str("n_bot07", "dia_bot07", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot08") > 0).then(pl.concat_str("n_bot08", "dia_bot08", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot09") > 0).then(pl.concat_str("n_bot09", "dia_bot09", separator="-H")).otherwise(pl.lit(None)),
                        separator="+",
                        ignore_nulls=True,
                    )
                ),
                link = (
                    pl.concat_str(
                        pl.lit("H"),
                        pl.col("dia_link"),
                        pl.lit("-"),
                        pl.col("s_link"),
                        pl.lit(" ("),
                        pl.col("n_link"),
                        pl.lit(" legs)")
                    )
                ),
            )
            .with_columns(
                flexure = pl.when(pl.col("A_bot_prov") >= pl.col("A_top_prov")).then(pl.col("bot")).otherwise(pl.col("top"))
            )
        )

        self.design_rc_coupling_beam_force = (
            self.design_rc_coupling_beam_force
            .with_columns(
                cover = pl.lit(45),
                dia_link = pl.lit(10),
                n_link = pl.lit(2),
                s_link = pl.lit(300),
                dia_top00 = pl.lit(10), dia_bot00 = pl.lit(10), n_top00 = pl.lit(2), n_bot00 = pl.lit(2),
                dia_top01 = pl.lit(0), dia_bot01 = pl.lit(0), n_top01 = pl.lit(0), n_bot01 = pl.lit(0),
                dia_top02 = pl.lit(0), dia_bot02 = pl.lit(0), n_top02 = pl.lit(0), n_bot02 = pl.lit(0),
                dia_top03 = pl.lit(0), dia_bot03 = pl.lit(0), n_top03 = pl.lit(0), n_bot03 = pl.lit(0),
                dia_top04 = pl.lit(0), dia_bot04 = pl.lit(0), n_top04 = pl.lit(0), n_bot04 = pl.lit(0),
                dia_top05 = pl.lit(0), dia_bot05 = pl.lit(0), n_top05 = pl.lit(0), n_bot05 = pl.lit(0),
                dia_top06 = pl.lit(0), dia_bot06 = pl.lit(0), n_top06 = pl.lit(0), n_bot06 = pl.lit(0),
                dia_top07 = pl.lit(0), dia_bot07 = pl.lit(0), n_top07 = pl.lit(0), n_bot07 = pl.lit(0),
                dia_top08 = pl.lit(0), dia_bot08 = pl.lit(0), n_top08 = pl.lit(0), n_bot08 = pl.lit(0),
                dia_top09 = pl.lit(0), dia_bot09 = pl.lit(0), n_top09 = pl.lit(0), n_bot09 = pl.lit(0),
            )
            .with_columns(
                A_top_prov = pl.col("n_top00") * pl.col("dia_top00").pow(2) * np.pi / 4,
                A_bot_prov = pl.col("n_bot00") * pl.col("dia_bot00").pow(2) * np.pi / 4,
                A_top_req = pl.lit(0.0),
                A_bot_req = pl.lit(0.0),
                is_converged = pl.lit(False),
            )
            .with_columns(
                ur_top = pl.lit(0.0),
                ur_bot = pl.lit(0.0),
                ur_vc = pl.lit(0.0),
                ur_vmax = pl.lit(0.0),
                ur_link = pl.lit(0.0),
            )
        )

        return self
    
    def _design_rc_coupling_beam_calculate_beam_reo(
            self, ur_limit, 
            diameter_flexure, diameter_shear, spacing_shear,
            n,
        ):
        """Calculate the required flexure reo
        """
        
        pbar = tqdm(total=n, desc="Iterating for flexure design")
        self.design_rc_coupling_beam_force = rc_c.calculate_material(self.design_rc_coupling_beam_force)
        self.design_rc_coupling_beam_force = rc_c.calculate_beam_design(self.design_rc_coupling_beam_force)
        self.design_rc_coupling_beam_force = rc_c.calculate_x_max(self.design_rc_coupling_beam_force)
        self.design_rc_coupling_beam_force = rc_c.calculate_beam_reo_centroid(self.design_rc_coupling_beam_force)
        run = 1
        while run <= n and not (self.design_rc_coupling_beam_force.select(pl.col("is_converged").all()).item()):
            self.design_rc_coupling_beam_force = rc_c.calculate_beam_flexure_reo(self.design_rc_coupling_beam_force)
            self.design_rc_coupling_beam_force = reo_c.generate_reo_beam_flexure(self.design_rc_coupling_beam_force, ur_limit, diameter_flexure)
            self.design_rc_coupling_beam_force = rc_c.calculate_beam_reo_centroid(self.design_rc_coupling_beam_force)
            self.design_rc_coupling_beam_force = rc_c.calculate_beam_shear_concrete(self.design_rc_coupling_beam_force)
            self.design_rc_coupling_beam_force = rc_c.calculate_beam_shear_max(self.design_rc_coupling_beam_force)
            self.design_rc_coupling_beam_force = rc_c.calculate_beam_shear_reo(self.design_rc_coupling_beam_force)
            self.design_rc_coupling_beam_force = reo_c.generate_reo_beam_shear(self.design_rc_coupling_beam_force, ur_limit, diameter_shear, spacing_shear)
            pbar.update(1)
            run += 1
        pbar.close()

        return self
    
    def _design_rc_coupling_beam_set_ur_reo(self):
        """Update the spandrel table with the utilisation ratio
        """

        df_force = (
            self.design_rc_coupling_beam_force
            .select(
                "spandrel", "story", 
                "dia_link", "s_link", "n_link",
                "dia_top00", "dia_bot00", "n_top00", "n_bot00",
                "dia_top01", "dia_bot01", "n_top01", "n_bot01",
                "dia_top02", "dia_bot02", "n_top02", "n_bot02",
                "dia_top03", "dia_bot03", "n_top03", "n_bot03",
                "dia_top04", "dia_bot04", "n_top04", "n_bot04",
                "dia_top05", "dia_bot05", "n_top05", "n_bot05",
                "dia_top06", "dia_bot06", "n_top06", "n_bot06",
                "dia_top07", "dia_bot07", "n_top07", "n_bot07",
                "dia_top08", "dia_bot08", "n_top08", "n_bot08",
                "dia_top09", "dia_bot09", "n_top09", "n_bot09",
                "A_top_prov", "A_bot_prov", "A_top_req", "A_bot_req",
                "A_sw_s_req", "A_sw_s_prov",
                "ur_top", "ur_bot", "ur_vc", "ur_vmax", "ur_link"
            )
            .group_by("spandrel", "story")
            .agg(
                pl.col("dia_link", "s_link", "n_link", "A_sw_s_req", "A_sw_s_prov", "ur_link").sort_by("A_sw_s_prov", "ur_link").last(),
                pl.col(
                    "dia_top00", "dia_top01", "dia_top02", "dia_top03", "dia_top04", 
                    "dia_top05", "dia_top06", "dia_top07", "dia_top08", "dia_top09",
                    "n_top00", "n_top01", "n_top02", "n_top03", "n_top04",
                    "n_top05", "n_top06", "n_top07", "n_top08", "n_top09",
                    "A_top_req", "A_top_prov", "ur_top"
                ).sort_by("A_top_prov", "ur_top").last(),
                pl.col(
                    "dia_bot00", "dia_bot01", "dia_bot02", "dia_bot03", "dia_bot04", 
                    "dia_bot05", "dia_bot06", "dia_bot07", "dia_bot08", "dia_bot09",
                    "n_bot00", "n_bot01", "n_bot02", "n_bot03", "n_bot04",
                    "n_bot05", "n_bot06", "n_bot07", "n_bot08", "n_bot09",
                    "A_bot_req", "A_bot_prov", "ur_bot"
                ).sort_by("A_bot_prov", "ur_bot").last(),
                pl.col("ur_vc", "ur_vmax").max()
            )
            .with_columns(
                ur_max = pl.max_horizontal("ur_top", "ur_bot", "ur_vmax", "ur_link"),
                top = (
                    pl.concat_str(
                        pl.when(pl.col("n_top00") > 0).then(pl.concat_str("n_top00", "dia_top00", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top01") > 0).then(pl.concat_str("n_top01", "dia_top01", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top02") > 0).then(pl.concat_str("n_top02", "dia_top02", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top03") > 0).then(pl.concat_str("n_top03", "dia_top03", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top04") > 0).then(pl.concat_str("n_top04", "dia_top04", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top05") > 0).then(pl.concat_str("n_top05", "dia_top05", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top06") > 0).then(pl.concat_str("n_top06", "dia_top06", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top07") > 0).then(pl.concat_str("n_top07", "dia_top07", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top08") > 0).then(pl.concat_str("n_top08", "dia_top08", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_top09") > 0).then(pl.concat_str("n_top09", "dia_top09", separator="-H")).otherwise(pl.lit(None)),
                        separator="+",
                        ignore_nulls=True,
                    )
                ),
                bot = (
                    pl.concat_str(
                        pl.when(pl.col("n_bot00") > 0).then(pl.concat_str("n_bot00", "dia_bot00", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot01") > 0).then(pl.concat_str("n_bot01", "dia_bot01", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot02") > 0).then(pl.concat_str("n_bot02", "dia_bot02", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot03") > 0).then(pl.concat_str("n_bot03", "dia_bot03", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot04") > 0).then(pl.concat_str("n_bot04", "dia_bot04", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot05") > 0).then(pl.concat_str("n_bot05", "dia_bot05", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot06") > 0).then(pl.concat_str("n_bot06", "dia_bot06", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot07") > 0).then(pl.concat_str("n_bot07", "dia_bot07", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot08") > 0).then(pl.concat_str("n_bot08", "dia_bot08", separator="-H")).otherwise(pl.lit(None)),
                        pl.when(pl.col("n_bot09") > 0).then(pl.concat_str("n_bot09", "dia_bot09", separator="-H")).otherwise(pl.lit(None)),
                        separator="+",
                        ignore_nulls=True,
                    )
                ),
                link = (
                    pl.concat_str(
                        pl.lit("H"),
                        pl.col("dia_link"),
                        pl.lit("-"),
                        pl.col("s_link"),
                        pl.lit(" ("),
                        pl.col("n_link"),
                        pl.lit(" legs)")
                    )
                ),
            )
            .with_columns(
                flexure = pl.when(pl.col("A_bot_prov") >= pl.col("A_top_prov")).then(pl.col("bot")).otherwise(pl.col("top"))
            )
        )

        self.design_rc_coupling_beam_spandrel = (
            self.design_rc_coupling_beam_spandrel
            .update(df_force, on=["spandrel", "story"], how="left")
        )

        return self

    # ----- rc pile -----

    def create_rc_pile(self):
        """Create from the children models, geometry to get from the latest model, forces to envelope all children models
        """

        self.point = (
            self.get_df_head("point", 1).drop("datetime", "model")
            .filter(pl.col("section_line").is_not_null())
            .with_columns(
                dia = (pl.col("section_line").str.split(" ").list.get(1).cast(pl.Float64) * 1000).cast(pl.Int32),
                grade = (pl.col("section_line").str.split(" ").list.get(2).cast(pl.Float64)).cast(pl.Int32),

                # Output
                model_compression_geo = pl.lit(None).cast(pl.String),
                outputcase_compression_geo = pl.lit(None).cast(pl.String),
                ur_compression_geo = pl.lit(None).cast(pl.Float64),
                model_tension_geo = pl.lit(None).cast(pl.String),
                outputcase_tension_geo = pl.lit(None).cast(pl.String),
                ur_tension_geo = pl.lit(None).cast(pl.Float64),

                model_vert = pl.lit(None).cast(pl.String),
                outputcase_vert = pl.lit(None).cast(pl.String),
                force = pl.lit(None).cast(pl.Float64),
                moment = pl.lit(None).cast(pl.Float64),
                ur_adsec_vert = pl.lit(None).cast(pl.Float64),
                key = pl.lit(None).cast(pl.String),
                A_s = pl.lit(None).cast(pl.Float64),
                perc_vert = pl.lit(None).cast(pl.Float64),
                dia_vert = pl.lit(None).cast(pl.Int32),
                n_vert = pl.lit(None).cast(pl.Int32),
            )
        )

        self.joint_displacement = (
            self.get_df("joint_displacement").drop("datetime")
        )

        self.joint_reaction = (
            self.get_df("joint_reaction").drop("datetime")
        )

        return self
    
    def design_rc_pile(
            self,
            contain_geo="", not_contain_geo="!",
            contain_str="", not_contain_str="!",
            ur_limit_pm=0.9, ur_limit_v=0.9,
            perc_vert=[1.0, 2.0, 3.0, 4.0], dia_vert=[10, 12, 16, 20, 25, 32], s_vert=[150,],
            fxy_scale=75.0/1500,
            df_capacity_geo=None, df_mxy_scale=None
        ):
        """Design RC piles based on the following methodology:

        1. Check geo combination against pile geo capacity
        2. Check PM against PM curve from Adsec
        3. Check V for spiral links
        """

        if df_mxy_scale is None:
            self.design_rc_pile_mxy_scale = (
                pl.DataFrame({"dia": self.point.unique("dia").sort("dia")["dia"].to_list()})
                .cast({"dia": pl.Int32})
                .with_columns(
                    fxy = pl.lit(100.0),
                    mxy = pl.lit(300.0),
                )
            )
        else:
            self.design_rc_pile_mxy_scale = df_mxy_scale.cast({"dia": pl.Int32})
        
        if df_capacity_geo is None:
            self.design_rc_pile_capacity_geo = (
                pl.DataFrame({"dia": self.point.unique("dia").sort("dia")["dia"].to_list()})
                .cast({"dia": pl.Int32})
                .with_columns(
                    compression_capacity_geo = (pl.col("dia").pow(2) / 4 * np.pi) * 11.5 / 1000
                )
                .with_columns(
                    tension_capacity_geo = pl.col("compression_capacity_geo") / 2
                )
            )
        else:
            self.design_rc_pile_capacity_geo = df_capacity_geo.cast({"dia": pl.Int32})

        return (
            self
            ._design_rc_pile_initialise(
                contain_geo=contain_geo, not_contain_geo=not_contain_geo,
                contain_str=contain_str, not_contain_str=not_contain_str,
            )
            ._design_rc_pile_populate_vert(perc_vert=perc_vert, dia_vert=dia_vert, s_vert=s_vert)
            ._design_rc_pile_geo()
            ._design_rc_pile_pm_initialise(fxy_scale=fxy_scale)
            .adsec_calculate_pm(0.0, "rcb_pile_1", self.design_rc_pile_point)
            ._design_rc_pile_calculate_pm_adsec()
            ._design_rc_pile_set_ur_adsec()
            ._design_rc_pile_set_point(ur_limit_pm)
            # TODO Check shear for spiral links
        )
    
    def get_rc_pile_fz_max(self, contain_outputcase, not_contain_outputcase="!"):
        """Return a df of fz
        """

        return (
            self.joint_reaction
            .filter(
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
            .group_by("unique_point")
            .agg(pl.all().sort_by("fz").last())
            .join(self.point.select("unique_point", "x", "y", "z"), on="unique_point", how="left")
        )
    
    def get_rc_pile_fz_min(self, contain_outputcase, not_contain_outputcase="!"):
        """Return a df of fz
        """

        return (
            self.joint_reaction
            .filter(
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
            .group_by("unique_point")
            .agg(pl.all().sort_by("fz").first())
            .with_columns(pl.when(pl.col("fz") < 0).then(pl.col("fz")).otherwise(pl.lit(0.0)))
            .join(self.point.select("unique_point", "x", "y", "z"), on="unique_point", how="left")
        )
    
    def get_rc_pile_uz_min(self, contain_outputcase, not_contain_outputcase="!"):
        """Return a df of uz min
        """

        return (
            self.joint_displacement
            .filter(
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
            .group_by("unique_point")
            .agg(pl.all().sort_by("uz").first())
            .join(self.point.select("unique_point", "x", "y", "z"), on="unique_point", how="left")
        )
    
    def _design_rc_pile_initialise(
            self,
            contain_geo="", not_contain_geo="!",
            contain_str="", not_contain_str="!",
        ):
        """Initialise the 2 dataframes for design: Reaction_Geo, Reaction_PMM, Reaction_V
        """

        self.design_rc_pile_point = self.point.clone()

        self.design_rc_pile_reaction_geo = (
            self.joint_reaction
            .select("unique_point", "story", "model", "outputcase", "fz")
            .join(self.point.select("unique_point", "dia"), on="unique_point", how="left")
            .filter(
                pl.col("outputcase").str.contains(contain_geo),
                ~pl.col("outputcase").str.contains(not_contain_geo),
            )
        )

        self.design_rc_pile_reaction_pm = (
            self.joint_reaction
            .select("unique_point", "story", "model", "outputcase", "fx", "fy", "fz")
            .join(self.point.select("unique_point", "dia"), on="unique_point", how="left")
            .filter(
                pl.col("outputcase").str.contains(contain_str),
                ~pl.col("outputcase").str.contains(not_contain_str),
            )
        )

        self.design_rc_pile_reaction_v = (
            self.joint_reaction
            .select("unique_point", "story", "model", "outputcase", "fx", "fy", "fz")
            .join(self.point.select("unique_point", "dia"), on="unique_point", how="left")
            .filter(
                pl.col("outputcase").str.contains(contain_str),
                ~pl.col("outputcase").str.contains(not_contain_str),
            )
        )

        return self
    
    def _design_rc_pile_populate_vert(self, perc_vert, dia_vert, s_vert, cover=75, dia_horz=16):
        """Populate with vert reo
        """

        self.design_rc_pile_point = sec_c.populate_reo_pile_vert_1(self.design_rc_pile_point, perc=perc_vert, diameter=dia_vert, spacing=s_vert, cover=cover, dia_horz=dia_horz)

        self.design_rc_pile_point = (
            self.design_rc_pile_point
            .with_columns(
                pl.Series("key", generate_keys("rcb_pile_1", self.design_rc_pile_point.select(get_columns("rcb_pile_1"))), dtype=pl.String)
            )
        )

        self.design_rc_pile_reaction_pm = (
            self.design_rc_pile_reaction_pm
            .join(self.design_rc_pile_point.select("unique_point", "dia_vert", "n_vert", "dia_horz", "cover", "rebar_type", "key"), on="unique_point", how="left")
        )

        self.design_rc_pile_reaction_v = (
            self.design_rc_pile_reaction_v
            .join(self.design_rc_pile_point.select("unique_point", "dia_vert", "n_vert", "dia_horz", "cover", "rebar_type", "key"), on="unique_point", how="left")
        )

        return self
    
    def _design_rc_pile_geo(self):
        """Check the reactions agains the geotechnical capacity
        """

        self.design_rc_pile_reaction_geo = (
            self.design_rc_pile_reaction_geo
            .join(self.design_rc_pile_capacity_geo, on="dia", how="left")
            .with_columns(
                ur_compression_geo = pl.when(pl.col("fz") >= 0).then(pl.col("fz")).otherwise(pl.lit(0.0)) / pl.col("compression_capacity_geo"),
                ur_tension_geo = pl.when(pl.col("fz") <= 0).then(pl.col("fz").abs()).otherwise(pl.lit(0.0)) / pl.col("tension_capacity_geo"),
            )
        )

        self.design_rc_pile_point = (
            self.design_rc_pile_point
            .update((
                self.design_rc_pile_reaction_geo
                .select("unique_point", "model", "outputcase", "ur_compression_geo")
                .rename({
                    "model": "model_compression_geo",
                    "outputcase": "outputcase_compression_geo",
                })
                .group_by("unique_point")
                .agg(pl.all().sort_by("ur_compression_geo").last())
            ), on="unique_point", how="left")
            .update((
                self.design_rc_pile_reaction_geo
                .select("unique_point", "model", "outputcase", "ur_tension_geo")
                .rename({
                    "model": "model_tension_geo",
                    "outputcase": "outputcase_tension_geo",
                })
                .group_by("unique_point")
                .agg(pl.all().sort_by("ur_tension_geo").last())
            ), on="unique_point", how="left")
        )

        self.point = (
            self.point
            .update((
                self.design_rc_pile_reaction_geo
                .select("unique_point", "model", "outputcase", "ur_compression_geo")
                .rename({
                    "model": "model_compression_geo",
                    "outputcase": "outputcase_compression_geo",
                })
                .group_by("unique_point")
                .agg(pl.all().sort_by("ur_compression_geo").last())
            ), on="unique_point", how="left")
            .update((
                self.design_rc_pile_reaction_geo
                .select("unique_point", "model", "outputcase", "ur_tension_geo")
                .rename({
                    "model": "model_tension_geo",
                    "outputcase": "outputcase_tension_geo",
                })
                .group_by("unique_point")
                .agg(pl.all().sort_by("ur_tension_geo").last())
            ), on="unique_point", how="left")
        )

        return self
    
    def _design_rc_pile_pm_initialise(self, fxy_scale=75.0/1500):
        """Calculate the design fz and mxy
        """

        self.design_rc_pile_reaction_pm = (
            self.design_rc_pile_reaction_pm
            .with_columns(
                fxy_base = (pl.col("fx").pow(2) + pl.col("fy").pow(2)).pow(0.5),
                fxy_scale = pl.col("fz").abs() * fxy_scale,
            )
            .with_columns(
                fxy = pl.col("fxy_base") + pl.col("fxy_scale"),
                mxy = pl.lit(None).cast(pl.Float64),
            )
        )

        dias = self.design_rc_pile_reaction_pm.unique("dia").sort("dia")["dia"].to_list()

        for dia in dias:
            df_mxy = (
                self.design_rc_pile_reaction_pm
                .filter(pl.col("dia") == dia)
                .select("unique_point", "model", "outputcase", "dia_vert", "n_vert", "fxy")
            )

            fxy_line = np.array(self.design_rc_pile_mxy_scale.filter(pl.col("dia") == dia).sort("fxy")["fxy"].to_list())
            mxy_line = np.array(self.design_rc_pile_mxy_scale.filter(pl.col("dia") == dia).sort("fxy")["mxy"].to_list())
            fxy = np.array(df_mxy["fxy"].to_list())
            mxy = sp.interpolate.interp1d(fxy_line, mxy_line, kind="linear", fill_value="extrapolate")(fxy)

            df_mxy = df_mxy.with_columns(mxy = mxy)

            self.design_rc_pile_reaction_pm = self.design_rc_pile_reaction_pm.update(df_mxy.drop("fxy"), on=["unique_point", "model", "outputcase", "dia_vert", "n_vert"], how="left")

        self.design_rc_pile_reaction_pm = self.design_rc_pile_reaction_pm.with_columns(fz = pl.col("fz") * -1)

        return self
    
    def _design_rc_pile_calculate_pm_adsec(self):
        """Calculate ur for pm
        """

        self.design_rc_pile_reaction_pm = ur_c.calculate_ur_force_moment(
            self.curve, ["key"], "key", 
            self.design_rc_pile_reaction_pm.rename({"mxy": "moment", "fz": "force"}), ["unique_point", "story", "model", "outputcase"])

        return self
    
    def _design_rc_pile_set_ur_adsec(self):
        """Update design_point with largest ur_adsec
        """

        df_force = (
            self.design_rc_pile_reaction_pm
            .select(
                "unique_point", "story", "key", 
                "model", "outputcase", "force", "moment", "ur_adsec"
            )
            .rename({
                "model": "model_vert",
                "outputcase": "outputcase_vert",
                "ur_adsec": "ur_adsec_vert",
            })
            .group_by("unique_point", "story", "key")
            .agg(pl.all().sort_by("ur_adsec_vert").last())
        )

        self.design_rc_pile_point = (
            self.design_rc_pile_point
            .update(df_force, on=["unique_point", "story", "key"], how="left")
        )

        return self
    
    def _design_rc_pile_set_point(self, ur_limit_pm):
        """Update point with selected reo
        """

        df_max = (
            self.design_rc_pile_point
            .select(
                "unique_point", 
                "model_vert", "outputcase_vert", "force", "moment",
                "ur_adsec_vert",
                "key", "A_s", "perc_vert", "n_vert", "dia_vert",
            )
            .group_by("unique_point")
            .agg(pl.all().sort_by("ur_adsec_vert").first())
        )

        df_design = (
            self.design_rc_pile_point
            .select(
                "unique_point", 
                "model_vert", "outputcase_vert", "force", "moment",
                "ur_adsec_vert",
                "key", "A_s", "perc_vert", "n_vert", "dia_vert",
            )
            .filter(pl.col("ur_adsec_vert") <= ur_limit_pm)
            .group_by("unique_point")
            .agg(
                pl.all().sort_by("A_s").first(),
            )
        )

        df_final = (
            df_max
            .update(df_design, on=["unique_point"], how="left")
        )

        self.point = (
            self.point
            .update(df_final, on="unique_point", how="left")
        )

        return self

    # ----- rc wall -----

    def read_rc_wall(self, model, format="parquet"):
        """read state from folder
        """

        model = generate_full_path(model)
        check_folder(model)

        if format == "parquet":
            self.wall = pl.read_parquet(model + "wall." + format)
            self.force_pier = pl.read_parquet(model + "force_pier." + format)
            self.design_rc_wall_pier = pl.read_parquet(model + "design_rc_wall_pier." + format)
            self.design_rc_wall_force_pmm = pl.read_parquet(model + "design_rc_wall_force_pmm." + format)
            self.design_rc_wall_force_v2 = pl.read_parquet(model + "design_rc_wall_force_v2." + format)
            self.design_rc_wall_force_v3 = pl.read_parquet(model + "design_rc_wall_force_v3." + format)
            self.design_rc_wall_force_pmm_adsec = pl.read_parquet(model + "design_rc_wall_force_pmm_adsec." + format)
        elif format == "csv":
            self.wall = pl.read_csv(model + "wall." + format)
            self.force_pier = pl.read_csv(model + "force_pier." + format)
            self.design_rc_wall_pier = pl.read_csv(model + "design_rc_wall_pier." + format)
            self.design_rc_wall_force_pmm = pl.read_csv(model + "design_rc_wall_force_pmm." + format)
            self.design_rc_wall_force_v2 = pl.read_csv(model + "design_rc_wall_force_v2." + format)
            self.design_rc_wall_force_v3 = pl.read_csv(model + "design_rc_wall_force_v3." + format)
            self.design_rc_wall_force_pmm_adsec = pl.read_csv(model + "design_rc_wall_force_pmm_adsec." + format)

        return self
    
    def write_rc_wall(self, model, format="parquet"):
        """Write the current state to folder
        """

        model = generate_full_path(model)
        check_folder(model)

        if format == "parquet":
            self.wall.write_parquet(model + "wall." + format)
            self.force_pier.write_parquet(model + "force_pier." + format)
            self.design_rc_wall_pier.write_parquet(model + "design_rc_wall_pier." + format)
            self.design_rc_wall_force_pmm.write_parquet(model + "design_rc_wall_force_pmm." + format)
            self.design_rc_wall_force_v2.write_parquet(model + "design_rc_wall_force_v2." + format)
            self.design_rc_wall_force_v3.write_parquet(model + "design_rc_wall_force_v3." + format)
            self.design_rc_wall_force_pmm_adsec.write_parquet(model + "design_rc_wall_force_pmm_adsec." + format)
        elif format == "csv":
            self.wall.write_csv(model + "wall." + format)
            self.force_pier.write_csv(model + "force_pier." + format)
            self.design_rc_wall_pier.write_csv(model + "design_rc_wall_pier." + format)
            self.design_rc_wall_force_pmm.write_csv(model + "design_rc_wall_force_pmm." + format)
            self.design_rc_wall_force_v2.write_csv(model + "design_rc_wall_force_v2." + format)
            self.design_rc_wall_force_v3.write_csv(model + "design_rc_wall_force_v3." + format)
            self.design_rc_wall_force_pmm_adsec.write_csv(model + "design_rc_wall_force_pmm_adsec." + format)

        return self

    def create_rc_wall(self):
        """Create from the children models, geometry to get from the latest model, forces to envelope all children models
        """

        self.wall = (
            self.get_df_head("wall", 1).drop("datetime", "model")
            .with_columns(
                # Input
                k2 = pl.lit(1.0),
                k3 = pl.lit(1.0),
                n_side_restrained = pl.lit(2),
                p_add_m = pl.lit(0.0),
                m2_add_m = pl.lit(0.0),
                m3_add_m = pl.lit(0.0),
                v2_add_m = pl.lit(0.0),
                v3_add_m = pl.lit(0.0),

                # Output
                cover = pl.lit(None).cast(pl.Int32),
                rebar_type = pl.lit("B"),

                model_vert = pl.lit(None).cast(pl.String),
                outputcase_vert = pl.lit(None).cast(pl.String),
                station_vert = pl.lit(None).cast(pl.Float64),
                ur_pmm_vert = pl.lit(None).cast(pl.Float64),
                ur_adsec_vert = pl.lit(None).cast(pl.Float64),
                A_s = pl.lit(None).cast(pl.Float64),
                perc_vert = pl.lit(None).cast(pl.Float64),
                dia_vert0 = pl.lit(None).cast(pl.Int32),
                dia_vert1 = pl.lit(None).cast(pl.Int32),
                dia_vert2 = pl.lit(None).cast(pl.Int32),
                dia_vert3 = pl.lit(None).cast(pl.Int32),
                dia_vert4 = pl.lit(None).cast(pl.Int32),
                dia_vert5 = pl.lit(None).cast(pl.Int32),
                dia_vert6 = pl.lit(None).cast(pl.Int32),
                dia_vert7 = pl.lit(None).cast(pl.Int32),
                dia_vert8 = pl.lit(None).cast(pl.Int32),
                dia_vert9 = pl.lit(None).cast(pl.Int32),
                s_vert0 = pl.lit(None).cast(pl.Int32),

                model_horz = pl.lit(None).cast(pl.String),
                outputcase_horz = pl.lit(None).cast(pl.String),
                station_horz = pl.lit(None).cast(pl.Float64),
                ur_vc_horz = pl.lit(None).cast(pl.Float64),
                ur_vmax_horz = pl.lit(None).cast(pl.Float64),
                ur_horz = pl.lit(None).cast(pl.Float64),
                A_sw_s2 = pl.lit(None).cast(pl.Float64),
                dia_horz0 = pl.lit(None).cast(pl.Int32),
                dia_horz1 = pl.lit(None).cast(pl.Int32),
                dia_horz2 = pl.lit(None).cast(pl.Int32),
                dia_horz3 = pl.lit(None).cast(pl.Int32),
                dia_horz4 = pl.lit(None).cast(pl.Int32),
                dia_horz5 = pl.lit(None).cast(pl.Int32),
                dia_horz6 = pl.lit(None).cast(pl.Int32),
                dia_horz7 = pl.lit(None).cast(pl.Int32),
                dia_horz8 = pl.lit(None).cast(pl.Int32),
                dia_horz9 = pl.lit(None).cast(pl.Int32),
                s_horz = pl.lit(None).cast(pl.Int32),

                model_tie = pl.lit(None).cast(pl.String),
                outputcase_tie = pl.lit(None).cast(pl.String),
                station_tie = pl.lit(None).cast(pl.Float64),
                ur_vc_tie = pl.lit(None).cast(pl.Float64),
                ur_vmax_tie = pl.lit(None).cast(pl.Float64),
                ur_tie = pl.lit(None).cast(pl.Float64),
                A_sw_s3 = pl.lit(None).cast(pl.Float64),
                dia_tie = pl.lit(None).cast(pl.Int32),
                sh_tie = pl.lit(None).cast(pl.Int32),
                sv_tie = pl.lit(None).cast(pl.Int32),
            )
        )

        self.force_pier = (
            self.get_df_with_story("force_pier").drop("datetime")
            .with_columns(
                # Output
            )
        )

        return self
    
    def design_rc_wall(
            self,
            contain_pier="", not_contain_pier="!",
            contain_story="", not_contain_story="!",
            contain_outputcase="", not_contain_outputcase="!",
            ur_limit_pmm=0.9, ur_limit_v=0.9,
            perc_vert=np.array([1.0, 2.0, 3.0, 4.0]), dia_vert=np.array([10, 12, 16, 20, 25, 32]), s_vert=np.array([200,]),
            dia_horz=np.array([10, 12, 16, 20, 25]), s_horz=np.array([200,]),
            dia_tie=np.array([10, 12, 16]),
            n=1,
        ):
        """Design rc walls based on the following methodology: 

        1. Design using the simplified pmm curve + v2 and v3 (Eurocode 2 simplified method)
        2. Get the governing cases for design and use adsec for pmm design
        3. Select the most optimise reo
        """

        return (
            self
            ._design_rc_wall_initialise(
                contain_pier, not_contain_pier,
                contain_story, not_contain_story,
                contain_outputcase, not_contain_outputcase,
            )
            ._design_rc_wall_populate_vert(perc_vert, dia_vert, s_vert)
            ._design_rc_wall_calculate_pm_wall_6()
            ._design_rc_wall_calculate_force_pmm_eccentricity()
            ._design_rc_wall_calculate_ur_pmm_eccentricity_simplified_biaxial()
            ._design_rc_wall_calculate_force_v2()
            ._design_rc_wall_generate_reo_horz_6(dia_horz, s_horz, ur_limit_v)
            ._design_rc_wall_calculate_force_v3(ur_limit_v)
            ._design_rc_wall_generate_reo_tie_1(dia_tie, s_vert, s_horz, ur_limit_v)
            ._design_rc_wall_set_ur_simplified()
            .adsec_calculate_pmm("rc_wall_u_6", self.design_rc_wall_pier)
            ._design_rc_wall_calculate_force_pmm_adsec(n=n)
            ._design_rc_wall_calculate_adsec_ur_pmm()
            ._design_rc_wall_adsec_set_ur()
            # TODO to set the data back to wall so that this can be exported to other software
        )
    
    def get_rc_wall(self, ur_limit, group_pier=pl.col("pier"), group_story=pl.col("elev")):
        """Return the pier table based on a single ur column
        """

        df_pier_max = (
            self.design_rc_wall_pier
            .select(
                "story", "pier", "elev", "thickness_min", "grade_min",
                "model_vert", "outputcase_vert", "station_vert",
                "ur_pmm_vert", "ur_adsec_vert",
                "key", "reo_vert", "dia_vert", "perc_vert", "A_s", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",
                "model_horz", "outputcase_horz", "station_horz",
                "ur_vc_horz", "ur_vmax_horz", "ur_horz", 
                "A_sw_s2", "s_horz", "dia_horz0", "dia_horz1", "dia_horz2", "dia_horz3", "dia_horz4", "dia_horz5", "dia_horz6",
                "model_tie", "outputcase_tie", "station_tie",
                "ur_vc_tie", "ur_vmax_tie", "ur_tie", 
                "A_sw_s3", "sh_tie", "sv_tie", "dia_tie",
            )
            .group_by("story", "pier")
            .agg(pl.all().sort_by("ur_adsec_vert").first())
        )

        df_pier_design = (
            self.design_rc_wall_pier
            .select(
                "story", "pier", "elev", "thickness_min", "grade_min",
                "model_vert", "outputcase_vert", "station_vert",
                "ur_pmm_vert", "ur_adsec_vert",
                "key", "reo_vert", "dia_vert", "perc_vert", "A_s", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",
                "model_horz", "outputcase_horz", "station_horz",
                "ur_vc_horz", "ur_vmax_horz", "ur_horz", 
                "A_sw_s2", "s_horz", "dia_horz0", "dia_horz1", "dia_horz2", "dia_horz3", "dia_horz4", "dia_horz5", "dia_horz6",
                "model_tie", "outputcase_tie", "station_tie",
                "ur_vc_tie", "ur_vmax_tie", "ur_tie", 
                "A_sw_s3", "sh_tie", "sv_tie", "dia_tie",
            )
            .filter(pl.col("ur_adsec_vert") <= ur_limit)
            .group_by("story", "pier")
            .agg(
                pl.all().sort_by("A_s").first(),
            )
        )

        df_pier = (
            df_pier_max
            .update(df_pier_design, on=["pier", "story"], how="left")
            .with_columns(
                piers = pl.col("pier"),
                pier = group_pier,
            )
            .with_columns(
                thickness_min = pl.col("thickness_min").cast(pl.UInt32)
            )
            .group_by("story", "pier")
            .agg(
                pl.col("piers"),
                pl.col("thickness_min").min().alias("thickness_min"),
                pl.col("thickness_min").max().alias("thickness_max"),
                pl.col("elev", "grade_min").first(),
                pl.col(
                    "model_vert", "outputcase_vert", "station_vert",
                    "ur_pmm_vert", "ur_adsec_vert",
                    "key", "reo_vert", "dia_vert", "perc_vert", "A_s", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",
                ).sort_by("A_s").last(),
                pl.col(
                    "model_horz", "outputcase_horz", "station_horz",
                    "ur_vc_horz", "ur_vmax_horz", "ur_horz", 
                    "A_sw_s2", "s_horz", "dia_horz0", "dia_horz1", "dia_horz2", "dia_horz3", "dia_horz4", "dia_horz5", "dia_horz6",
                ).sort_by("A_sw_s2").last(),
                pl.col(
                    "model_tie", "outputcase_tie", "station_tie",
                    "ur_vc_tie", "ur_vmax_tie", "ur_tie", 
                    "A_sw_s3", "sh_tie", "sv_tie", "dia_tie",
                ).sort_by("A_sw_s3").last(),
            )
        )

        df_pier_group = (
            df_pier
            .join(
                self.get_df_head("story", 1)
                .select("story", "elev")
                .with_columns(
                    story_below = pl.col("story").shift(-1),
                    elev_below = pl.col("elev").shift(-1),
                ),
                on="story", how="left"
            )
            .with_columns(
                group = (
                    group_story
                ),
            )
            .group_by("group", "pier", "thickness_min", "thickness_max")
            .agg(
                pl.col("story", "elev").sort_by("elev").last(),
                pl.col("piers", "story_below", "elev_below").sort_by("elev_below").first(),
                pl.col("grade_min").min(),
                pl.col(
                    "model_vert", "outputcase_vert", "station_vert",
                    "ur_pmm_vert", "ur_adsec_vert",
                    "key", "reo_vert", "dia_vert", "perc_vert", "A_s", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",
                ).sort_by("A_s").last(),
                pl.col(
                    "model_horz", "outputcase_horz", "station_horz",
                    "ur_vc_horz", "ur_vmax_horz", "ur_horz", 
                    "A_sw_s2", "s_horz", "dia_horz0", "dia_horz1", "dia_horz2", "dia_horz3", "dia_horz4", "dia_horz5", "dia_horz6",
                ).sort_by("A_sw_s2").last(),
                pl.col(
                    "model_tie", "outputcase_tie", "station_tie",
                    "ur_vc_tie", "ur_vmax_tie", "ur_tie", 
                    "A_sw_s3", "sh_tie", "sv_tie", "dia_tie",
                ).sort_by("A_sw_s3").last(),
            )
            .sort("elev")
            .sort("pier")
        )

        return (
            df_pier_group
        )
    
    def _design_rc_wall_initialise(
            self,
            contain_pier, not_contain_pier,
            contain_story, not_contain_story,
            contain_outputcase, not_contain_outputcase,
        ):
        """Initialise the 4 dataframes for design: Pier, Force_PMM, Force_V2, Force_V3
        """

        self.design_rc_wall_pier = (
            self.wall
            .filter(pl.col("pier").is_not_null())
            .explode("xs", "ys")
            .rename({
                "xs": "x",
                "ys": "y",
            })
            .unique(["unique_area", "x", "y"])
            .sort(["x", "y"])
            .group_by("unique_area", maintain_order=True)
            .agg(
                pl.col("x", "y"),
                pl.all().exclude("x", "y").first(),
            )
            .drop("unique_area")
            .unique(["pier", "story", "x", "y"])
            .group_by(["pier", "story"])
            .agg(
                pl.col("grade").min().alias("grade_min"),
                pl.col("thickness").min().alias("thickness_min"),
                pl.col("wall", "x", "y"),
                pl.all().exclude("wall", "x", "y", "grade", "thickness").first()
            )
            .join(
                self.get_df_head("story", 1).select("story", "elev", "height"), 
                on="story", how="left"
            )
            .filter(
                pl.col("pier").str.contains(contain_pier),
                ~pl.col("pier").str.contains(not_contain_pier),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
            )
        )

        self.design_rc_wall_pier = (
            sec_c.calculate_wall_dimensions(self.design_rc_wall_pier)
            .drop("x", "y")
        )

        self.design_rc_wall_force_pmm = (
            self.force_pier
            .select("model", "pier", "story", "station", "outputcase", "fx", "myy", "mzz")
            .rename({
                "fx": "p_base",
                "myy": "m2_base",
                "mzz": "m3_base"
            })
            .filter(
                pl.col("pier").str.contains(contain_pier),
                ~pl.col("pier").str.contains(not_contain_pier),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
            .join(
                self.design_rc_wall_pier
                .select(
                    "pier", "story", "grade_min", "thickness_min", "length",
                    "k2", "k3", "n_side_restrained", "p_add_m", "m2_add_m", "m3_add_m"
                )
                .rename({
                    "grade_min": "f_ck",
                    "thickness_min": "b",
                    "length": "h",
                }),
                on=["pier", "story"], how="left"
            )
            .join(
                self.get_df_head("story", 1).select("story", "height")
                .rename({"height": "l"}), 
                on="story", how="left"
            )
            .with_columns(
                p_base = pl.col("p_base") * -1,
                m2_base = pl.col("m2_base").abs(),
                m3_base = pl.col("m3_base").abs(),
                b = pl.col("b") * 1000, #mm
                h = pl.col("h") * 1000, #mm
                l = pl.col("l") * 1000, #mm 
            )
        )

        self.design_rc_wall_force_v2 = (
            self.force_pier
            .select("model", "pier", "story", "station", "outputcase", "fx", "fy")
            .rename({
                "fx": "p_base",
                "fy": "v2_base",
            })
            .filter(
                pl.col("pier").str.contains(contain_pier),
                ~pl.col("pier").str.contains(not_contain_pier),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
            .join(
                self.design_rc_wall_pier
                .select(
                    "pier", "story", "grade_min", "thickness_min", "length",
                    "p_add_m", "v2_add_m"
                )
                .rename({
                    "grade_min": "f_ck",
                    "thickness_min": "b_w",
                    "length": "h",
                }),
                on=["pier", "story"], how="left"
            )
            .with_columns(
                effective_depth_factor = pl.lit(V2_EFFECTIVE_DEPTH_FACTOR),
            )
            .with_columns(
                p_base = pl.col("p_base") * -1,
                v2_base = pl.col("v2_base").abs(),
                b_w = pl.col("b_w") * 1000, #mm
                h = pl.col("h") * 1000, #mm
            )
        )

        self.design_rc_wall_force_v3 = (
            self.force_pier
            .select("model", "pier", "story", "station", "outputcase", "fx", "fz")
            .rename({
                "fx": "p_base",
                "fz": "v3_base",
            })
            .filter(
                pl.col("pier").str.contains(contain_pier),
                ~pl.col("pier").str.contains(not_contain_pier),
                pl.col("story").str.contains(contain_story),
                ~pl.col("story").str.contains(not_contain_story),
                pl.col("outputcase").str.contains(contain_outputcase),
                ~pl.col("outputcase").str.contains(not_contain_outputcase),
            )
            .join(
                self.design_rc_wall_pier.select(
                    "pier", "story", "grade_min", "thickness_min", "length",
                    "p_add_m", "v3_add_m"
                )
                .rename({
                    "grade_min": "f_ck",
                    "thickness_min": "h",
                    "length": "b_w",
                }),
                on=["pier", "story"], how="left"
            )
            .with_columns(
                effective_depth_factor = pl.lit(V3_EFFECTIVE_DEPTH_FACTOR),
            )
            .with_columns(
                p_base = pl.col("p_base") * -1,
                v3_base = pl.col("v3_base").abs(),
                b_w = pl.col("b_w") * 1000, #mm
                h = pl.col("h") * 1000, #mm
            )
        )

        return self

    def _design_rc_wall_populate_vert(self, perc_vert, dia_vert, s_vert, cover=30, dia_horz=16):
        """Generate the reo table given the list of wall thickness_min applicable to all piers
        """

        self.design_rc_wall_pier = sec_c.populate_reo_wall_vert_6(self.design_rc_wall_pier, perc_vert, dia_vert, s_vert, cover=cover, dia_horz=dia_horz)

        self.design_rc_wall_pier = (
            self.design_rc_wall_pier
            .with_columns(
                pl.Series("key", generate_keys("rc_wall_u_6", self.design_rc_wall_pier.select(get_columns("rc_wall_u_6"))), dtype=pl.String)
            )
        )

        self.design_rc_wall_pier = generate_reo_vert(self.design_rc_wall_pier, "rc_wall_u_6")

        self.design_rc_wall_force_v2 = (
            self.design_rc_wall_force_v2
            .join(self.design_rc_wall_pier, on=["pier", "story"], how="left")
        )

        self.design_rc_wall_force_v3 = (
            self.design_rc_wall_force_v3
            .join(self.design_rc_wall_pier, on=["pier", "story"], how="left")
        )

        return self
    
    def _design_rc_wall_calculate_pm_wall_6(self):
        """Calculate the pmm simplified curve
        """

        self.design_rc_wall_pier = pm_c.calculate_pm_wall_6(self.design_rc_wall_pier)

        self.design_rc_wall_force_pmm = (
            self.design_rc_wall_force_pmm
            .join(self.design_rc_wall_pier, on=["pier", "story"], how="left")
        )

        return self
    
    def _design_rc_wall_calculate_force_pmm_eccentricity(self):
        """Calculate the pmm check with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=8, desc="Calculating force pmm eccentricity")
        self.design_rc_wall_force_pmm = rc_c.calculate_pmm_add(self.design_rc_wall_force_pmm)
        self.design_rc_wall_force_pmm = rc_c.calculate_pmm_design(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_material(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_effective_length_wall(self.design_rc_wall_force_pmm)
        self.design_rc_wall_force_pmm = rc_c.calculate_effective_depth_wall_6(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_eccentricity_0_i(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_creep(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_slenderness_3(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_eccentricity_2(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_design_moment(self.design_rc_wall_force_pmm)
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_wall_calculate_ur_pmm_simplified_biaxial(self):
        """Calculate the pmm ur with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=2, desc="Calculating ur pmm simplified biaxial")
        self.design_rc_wall_force_pmm = rc_c.calculate_moment_capacity_simplified_biaxial(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_ur_biaxial(self.design_rc_wall_force_pmm)
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_wall_calculate_ur_pmm_eccentricity_simplified_biaxial(self):
        """Calculate the pmm ur with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=2, desc="Calculating ur pmm eccentricity simplified biaxial")
        self.design_rc_wall_force_pmm = rc_c.calculate_moment_capacity_simplified_biaxial(self.design_rc_wall_force_pmm)
        pbar.update(1)
        self.design_rc_wall_force_pmm = rc_c.calculate_ur_biaxial_eccentricity(self.design_rc_wall_force_pmm)
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_wall_calculate_force_v2(self):
        """Calculate the v2 shear check with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=6, desc="Calculating force v2")
        self.design_rc_wall_force_v2 = rc_c.calculate_pv2_add(self.design_rc_wall_force_v2)
        self.design_rc_wall_force_v2 = rc_c.calculate_pv2_design(self.design_rc_wall_force_v2)
        pbar.update(1)
        self.design_rc_wall_force_v2 = rc_c.calculate_material(self.design_rc_wall_force_v2)
        pbar.update(1)
        self.design_rc_wall_force_v2 = rc_c.calculate_capacity_shear_concrete_6(self.design_rc_wall_force_v2)
        pbar.update(1)
        self.design_rc_wall_force_v2 = rc_c.calculate_capacity_shear_max(self.design_rc_wall_force_v2)
        pbar.update(1)
        self.design_rc_wall_force_v2 = rc_c.calculate_required_shear_reo_wall6(self.design_rc_wall_force_v2)
        pbar.update(1)
        self.design_rc_wall_force_v2 = rc_c.calculate_required_shear_reo2(self.design_rc_wall_force_v2)
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_wall_generate_reo_horz_6(self, dia_horz, s_horz, ur_limit_v):
        """Generate horz reo
        """

        self.design_rc_wall_force_v2 = sec_c.generate_reo_wall_horz_6(self.design_rc_wall_force_v2, "A_sw_s2", dia_horz, s_horz, ur_limit_v)

        return self
    
    def _design_rc_wall_calculate_force_v3(self, ur_limit_v):
        """Calculate the v3 shear check with the Eurocode 2 simplified method
        """

        pbar = tqdm(total=6, desc="Calculating force v3")
        self.design_rc_wall_force_v3 = rc_c.calculate_pv3_add(self.design_rc_wall_force_v3)
        self.design_rc_wall_force_v3 = rc_c.calculate_pv3_design(self.design_rc_wall_force_v3)
        pbar.update(1)
        self.design_rc_wall_force_v3 = rc_c.calculate_material(self.design_rc_wall_force_v3)
        pbar.update(1)
        self.design_rc_wall_force_v3 = rc_c.calculate_capacity_shear_concrete_6(self.design_rc_wall_force_v3)
        pbar.update(1)
        self.design_rc_wall_force_v3 = rc_c.calculate_capacity_shear_max(self.design_rc_wall_force_v3)
        pbar.update(1)
        self.design_rc_wall_force_v3 = rc_c.calculate_required_shear_reo_wall6(self.design_rc_wall_force_v3)
        pbar.update(1)
        self.design_rc_wall_force_v3 = rc_c.calculate_required_shear_reo3(self.design_rc_wall_force_v3, ur_limit_v)
        pbar.update(1)
        pbar.close()

        return self
    
    def _design_rc_wall_generate_reo_tie_1(self, dia_tie, s_vert, s_horz, ur_limit_v):
        """Generate tie reo
        """

        self.design_rc_wall_force_v3 = sec_c.generate_reo_wall_line_uniform_tie(self.design_rc_wall_force_v3, "A_sw_s3", dia_tie, s_horz, s_vert, ur_limit_v)

        return self
    
    def _design_rc_wall_set_ur_simplified(self):
        """Update the pier table with the utilisation ratios"""

        df_force_pmm = (
            self.design_rc_wall_force_pmm
            .select(
                "pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",
                "ur_m2", "ur_m3", "ur_biaxial"
            )
            .group_by("pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",)
            .agg(pl.all().max())
            .with_columns(ur_pmm_vert = pl.max_horizontal("ur_m2", "ur_m3", "ur_biaxial"))
        )

        df_force_v2 = (
            self.design_rc_wall_force_v2
            .select(
                "pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",
                "model", "outputcase", "station",
                "ur_vc", "ur_vmax", "ur_horz",
                "A_sw_s2", "s_horz", "dia_horz0", "dia_horz1", "dia_horz2", "dia_horz3", "dia_horz4", "dia_horz5", "dia_horz6",
            )
            .rename({
                "model": "model_horz",
                "outputcase": "outputcase_horz",
                "station": "station_horz",
                "ur_vc": "ur_vc_horz",
                "ur_vmax": "ur_vmax_horz",
            })
            .group_by("pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5")
            .agg(
                pl.col("ur_vc_horz", "ur_vmax_horz").max(),
                pl.all().exclude("ur_vc_horz", "ur_vmax_horz").sort_by("A_sw_s2").last())
        )

        df_force_v3 = (
            self.design_rc_wall_force_v3
            .select(
                "pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",
                "model", "outputcase", "station",
                "ur_vc", "ur_vmax", "ur_tie",
                "A_sw_s3", "sv_tie", "sh_tie", "dia_tie", 
            )
            .rename({
                "model": "model_tie",
                "outputcase": "outputcase_tie",
                "station": "station_tie",
                "ur_vc": "ur_vc_tie",
                "ur_vmax": "ur_vmax_tie",
            })
            .group_by("pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",)
            .agg(
                pl.col("ur_vc_tie", "ur_vmax_tie").max(),
                pl.all().exclude("ur_vc_tie", "ur_vmax_tie").sort_by("A_sw_s3").last(),
            )
        )

        self.design_rc_wall_pier = (
            self.design_rc_wall_pier
            .update(df_force_pmm, on=["pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",], how="left")
            .update(df_force_v2, on=["pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",], how="left")
            .update(df_force_v3, on=["pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",], how="left")
        )

        self.design_rc_wall_pier = (
            self.design_rc_wall_pier
            .with_columns(
                pl.Series("key", generate_keys("rc_wall_u_6", self.design_rc_wall_pier.select(get_columns("rc_wall_u_6"))), dtype=pl.String)
            )
        )

        self.design_rc_wall_force_pmm = (
            self.design_rc_wall_force_pmm
            .update(
                self.design_rc_wall_pier.select("pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5", "key"), 
                on=["pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",], how="left"
            )
        )

        self.design_rc_wall_force_v2 = (
            self.design_rc_wall_force_v2
            .update(
                self.design_rc_wall_pier.select("pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5", "key"), 
                on=["pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",], how="left"
            )
        )

        self.design_rc_wall_force_v3 = (
            self.design_rc_wall_force_v3
            .update(
                self.design_rc_wall_pier.select("pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5", "key"), 
                on=["pier", "story", "s_vert0", "dia_vert0", "dia_vert1", "dia_vert2", "dia_vert3", "dia_vert4", "dia_vert5",], how="left"
            )
        )

        return self
    
    def _design_rc_wall_calculate_force_pmm_adsec(self, n=1):
        """Initialise the force pmm for adsec check filter based on simplified
        """

        self.design_rc_wall_force_pmm_adsec = (
            self.design_rc_wall_force_pmm
            .select(
                "model", "pier", "story", "outputcase", "station", "key",
                pl.col("p") * -1, pl.col("M_Edtotal2").alias("m2"), pl.col("M_Edtotal3").alias("m3"), pl.max_horizontal("ur_m2", "ur_m3", "ur_biaxial").alias("ur_simplified")
            )
            .group_by("pier", "story", "key")
            .agg(
                pl.all().sort_by("ur_simplified", descending=True).head(n)
            )
            .explode("model", "outputcase", "station", "p", "m2", "m3", "ur_simplified")
            .with_columns(
                magnitude_force = (pl.col("p").pow(2) + pl.col("m2").pow(2) + pl.col("m3").pow(2)).sqrt(),
                ur_adsec = pl.lit(None).cast(pl.Float64),
            )
        )

        return self
    
    def _design_rc_wall_calculate_adsec_ur_pmm(self):
        """Calculate the utilisation ratio
        """

        df_force_ur = (
            ur_c.calculate_ur_failure_surface(self.surface, self.design_rc_wall_force_pmm_adsec, "pier")
            .rename({"ur": "ur_adsec"})
        )

        self.design_rc_wall_force_pmm_adsec = (
            self.design_rc_wall_force_pmm_adsec
            .update(df_force_ur, on=["pier", "story", "model", "outputcase", "station", "key"], how="left")
        )

        return self
    
    def _design_rc_wall_adsec_set_ur(self):
        """Update the pier table with the utilisation ratio"""

        df_force = (
            self.design_rc_wall_force_pmm_adsec
            .select(
                "pier", "story", "key", 
                "model", "outputcase", "station", "ur_adsec"
            )
            .rename({
                "model": "model_vert",
                "outputcase": "outputcase_vert",
                "station": "station_vert",
                "ur_adsec": "ur_adsec_vert",
            })
            .group_by("pier", "story", "key")
            .agg(pl.all().sort_by("ur_adsec_vert").last())
        )

        self.design_rc_wall_pier = (
            self.design_rc_wall_pier
            .update(df_force, on=["pier", "story", "key"], how="left")
        )

        return self
