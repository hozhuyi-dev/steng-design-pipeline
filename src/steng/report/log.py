"""LogMixin: charts that track a model's results across its revisions
"""

import altair as alt
import polars as pl


class LogMixin:
    """Charts that track a model across its revisions, for a Session

    Each chart reads the session's child sessions through get_df: the charts over time show every
    revision, and those up the building show the latest. Each returns an Altair chart to show in a
    notebook or save.
    """

    def __init__(self) -> None:
        """Nothing to set up: the charts read the session's attributes"""
        pass

    # ----- numbers -----
    
    def log_members_vs_time(self) -> alt.TopLevelMixin:
        """Line chart of the number of each kind of member in each revision: columns, beams, walls,
        floors, restraints, and piles by section
        """

        df_restraint = (
            self.get_df("point")
            .filter(pl.any_horizontal(pl.col(["restraint_ux", "restraint_uy", "restraint_uz", "restraint_rxx", "restraint_ryy", "restraint_rzz"])))
            .with_columns(pl.lit("Restraint").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("point").count().alias("number"))
        )

        df_pile = (
            self.get_df("point")
            .filter(pl.col("section_line").is_not_null())
            .with_columns(pl.lit("Pile - ").alias("type_head"))
            .with_columns(pl.col("section_line").str.extract(r"^(\S+\s+\S+\s+\S+)").alias("type_body"))
            .with_columns(pl.concat_str("type_head", "type_body").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("point").count().alias("number"))
        )

        df_beam = (
            self.get_df("beam")
            .with_columns(pl.lit("Beam").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("beam").count().alias("number"))
        )

        df_brace = (
            self.get_df("brace")
            .with_columns(pl.lit("Brace").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("brace").count().alias("number"))
        )

        df_column = (
            self.get_df("column")
            .with_columns(pl.lit("Column").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("column").count().alias("number"))
        )

        df_line = (
            self.get_df("line")
            .with_columns(pl.lit("Line").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("line").count().alias("number"))
        )

        df_area = (
            self.get_df("area")
            .with_columns(pl.lit("Area").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("area").count().alias("number"))
        )

        df_floor = (
            self.get_df("floor")
            .with_columns(pl.lit("Floor").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("floor").count().alias("number"))
        )

        df_wall = (
            self.get_df("wall")
            .with_columns(pl.lit("Wall").alias("type"))
            .group_by("datetime", "model", "type")
            .agg(pl.col("wall").count().alias("number"))
        )
    
        df_numbers = pl.concat([df_restraint, df_pile, df_beam, df_brace, df_column, df_line, df_area, df_floor, df_wall])

        return (
            (
                (
                    alt.Chart(
                        df_numbers
                    ).mark_circle().encode(
                        alt.X("datetime:T").title("Date Time"),
                        alt.Y("number:Q").title("Number").scale(alt.Scale(type="log")),
                        alt.Color("type:N"),
                        alt.Tooltip(["model", "number"]),
                    )
                )
                +
                (
                    alt.Chart(
                        df_numbers
                    ).mark_line().encode(
                        alt.X("datetime:T").title("Date Time"),
                        alt.Y("number:Q").title("Number").scale(alt.Scale(type="log")),
                        alt.Color("type:N"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Number of Members against Time"))
            .configure_scale(zero=False)
        )
    
    # ----- basereactions -----

    def log_latest_base_fz(self) -> alt.TopLevelMixin:
        """Pie chart of the latest revision's vertical base reaction by load case, for the static and
        response spectrum cases over 10 kN
        """

        return (
            alt.Chart(
                self.get_df_head("base_reaction", 1)
                .filter(
                    pl.col("outputcasetype").is_in(["static", "response_spectrum"]),
                    pl.col("fz") > 10,
                )
            ).mark_arc().encode(
                alt.Theta("fz:Q"),
                alt.Color("outputcase:N"),
                alt.Tooltip(["model", "outputcase", "fz"]),
            )
            .properties(width=1200, title=(self.name + " - Base FZ Breakdown"))
            .configure_scale(zero=False)
        )
    
    def log_base_reaction_vs_time(self) -> alt.TopLevelMixin:
        """Line chart of each static and response spectrum case's base reactions in each revision
        """

        df_fz = (
            self.get_df("base_reaction")
            .filter(
                pl.col("outputcasetype").is_in(["static", "response_spectrum"]),
                pl.col("fz") > 1,
                ~pl.col("outputcase").str.contains("stepbystep"),
            )
            .rename({"fz": "reaction"})
            .with_columns(outputcase = pl.format("fz_{}", pl.col("outputcase")))
            .select("datetime", "model", "outputcase", "reaction")
        )

        df_fx = (
            self.get_df("base_reaction")
            .with_columns(
                fx_abs = pl.col("fx").abs()
            )
            .filter(
                pl.col("outputcasetype").is_in(["static", "response_spectrum"]),
                pl.col("fx").abs() > 1,
                ~pl.col("outputcase").str.contains("stepbystep"),
            )
            .rename({"fx_abs": "reaction"})
            .with_columns(outputcase = pl.format("fx_{}", pl.col("outputcase")))
            .select("datetime", "model", "outputcase", "reaction")
        )

        df_fy = (
            self.get_df("base_reaction")
            .with_columns(
                fy_abs = pl.col("fy").abs()
            )
            .filter(
                pl.col("outputcasetype").is_in(["static", "response_spectrum"]),
                pl.col("fy").abs() > 1,
                ~pl.col("outputcase").str.contains("stepbystep"),
            )
            .rename({"fy_abs": "reaction"})
            .with_columns(outputcase = pl.format("fy_{}", pl.col("outputcase")))
            .select("datetime", "model", "outputcase", "reaction")
        )

        df_reactions = pl.concat([df_fz, df_fx, df_fy])

        return (
            (
                (
                    alt.Chart(
                        df_reactions
                    ).mark_circle().encode(
                        alt.X("datetime:T").title("Date Time"),
                        alt.Y("reaction:Q").title("Reaction [kN]").scale(alt.Scale(type="log")),
                        alt.Color("outputcase:N"),
                        alt.Tooltip(["model", "outputcase", "reaction"]),
                    )
                )
                +
                (
                    alt.Chart(
                        df_reactions
                    ).mark_line().encode(
                        alt.X("datetime:T").title("Date Time"),
                        alt.Y("reaction:Q").title("Reaction [kN]").scale(alt.Scale(type="log")),
                        alt.Color("outputcase:N"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Base Reactions against Time"))
            .configure_scale(zero=False)
        )
    
    # ----- reactions -----

    # ----- storymass -----

    def log_story_mass(self, head: int) -> alt.TopLevelMixin:
        """Line chart of the story mass up the building, for each of the latest head revisions
        """

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_mass", head)
                    ).mark_circle().encode(
                        alt.X("mx:Q").title("Mass"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("model:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "mx"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_mass", head)
                    ).mark_line().encode(
                        alt.X("mx:Q").title("Mass"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("model:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "mx"]),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Mass"))
            .configure_scale(zero=False)
        )

    # ----- storyforce -----
    
    def log_story_force_vx(self, outputcases: list[str]) -> alt.TopLevelMixin:
        """Line chart of the story shear in X up the latest revision, for the given cases

        The shear carried by the columns is drawn too, where they share the pier label "_columns",
        as Etabs.assign_pier_columns sets. Cases ending in X are drawn with their sign flipped.
        """

        df_pier_columns = (
            self.get_df_head_with_story("force_pier", 1)
            .filter(
                pl.col("outputcase").is_in(outputcases),
                pl.col("pier") == "_columns",
            )
            .select("model", "story", "outputcase", "fy", "elev")
            .with_columns(
                pl.concat_str(pl.col("outputcase"), pl.lit("_columns"))
            )
            .rename({"fy": "fy"})
        )

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_force", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .select("model", "story", "outputcase", "fy", "elev")
                        .vstack(df_pier_columns)
                        .with_columns(
                            fy = pl.when(pl.col("outputcase").str.contains("^W*X$")).then(- pl.col("fy")).otherwise(pl.col("fy"))
                        )
                    ).mark_circle().encode(
                        alt.X("fy:Q").title("Shear Force [kN]"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "fy"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_force", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .select("model", "story", "outputcase", "fy", "elev")
                        .vstack(df_pier_columns)
                        .with_columns(
                            fy = pl.when(pl.col("outputcase").str.contains("^W*X$")).then(- pl.col("fy")).otherwise(pl.col("fy"))
                        )
                    ).mark_line().encode(
                        alt.X("fy:Q").title("Shear Force [kN]"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "fy"]),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Force VX"))
            .configure_scale(zero=False)
        )
    
    def log_story_force_vy(self, outputcases: list[str]) -> alt.TopLevelMixin:
        """Line chart of the story shear in Y up the latest revision, for the given cases

        The shear carried by the columns is drawn too, where they share the pier label "_columns",
        as Etabs.assign_pier_columns sets. Cases ending in X are drawn with their sign flipped.
        """

        df_pier_columns = (
            self.get_df_head_with_story("force_pier", 1)
            .filter(
                pl.col("outputcase").is_in(outputcases),
                pl.col("pier") == "_columns",
            )
            .select("model", "story", "outputcase", "fz", "elev")
            .with_columns(
                pl.concat_str(pl.col("outputcase"), pl.lit("_columns"))
            )
            .rename({"fz": "fz"})
        )

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_force", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .select("model", "story", "outputcase", "fz", "elev")
                        .vstack(df_pier_columns)
                        .with_columns(
                            fz = pl.when(pl.col("outputcase").str.contains("^W*Y$")).then(- pl.col("fz")).otherwise(pl.col("fz"))
                        )
                    ).mark_circle().encode(
                        alt.X("fz:Q").title("Shear Force [kN]"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "fz"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_force", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .select("model", "story", "outputcase", "fz", "elev")
                        .vstack(df_pier_columns)
                        .with_columns(
                            fz = pl.when(pl.col("outputcase").str.contains("^W*Y$")).then(- pl.col("fz")).otherwise(pl.col("fz"))
                        )
                    ).mark_line().encode(
                        alt.X("fz:Q").title("Shear Force [kN]"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "fz"]),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Force VY"))
            .configure_scale(zero=False)
        )
    
    def log_story_force_mxx(self, outputcases: list[str]) -> alt.TopLevelMixin:
        """Line chart of the overturning moment about X up the latest revision, for the given cases

        The moment carried by the columns is drawn too, where they share the pier label "_columns",
        as Etabs.assign_pier_columns sets. Cases ending in X are drawn with their sign flipped.
        """

        df_pier_columns = (
            self.get_df_head_with_story("force_pier", 1)
            .filter(
                pl.col("outputcase").is_in(outputcases),
                pl.col("pier") == "_columns",
            )
            .select("model", "story", "outputcase", "myy", "elev")
            .with_columns(
                pl.concat_str(pl.col("outputcase"), pl.lit("_columns"))
            )
            .rename({"myy": "myy"})
        )

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_force", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .select("model", "story", "outputcase", "myy", "elev")
                        .vstack(df_pier_columns)
                    ).mark_circle().encode(
                        alt.X("myy:Q").title("Moment [kNm]"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "myy"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_force", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .select("model", "story", "outputcase", "myy", "elev")
                        .vstack(df_pier_columns)
                    ).mark_line().encode(
                        alt.X("myy:Q").title("Moment [kNm]"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Force MXX"))
            .configure_scale(zero=False)
        )
    
    def log_story_force_myy(self, outputcases: list[str]) -> alt.TopLevelMixin:
        """Line chart of the overturning moment about Y up the latest revision, for the given cases

        The moment carried by the columns is drawn too, where they share the pier label "_columns",
        as Etabs.assign_pier_columns sets. Cases ending in X are drawn with their sign flipped.
        """

        df_pier_columns = (
            self.get_df_head_with_story("force_pier", 1)
            .filter(
                pl.col("outputcase").is_in(outputcases),
                pl.col("pier") == "_columns",
            )
            .select("model", "story", "outputcase", "mzz", "elev")
            .with_columns(
                pl.concat_str(pl.col("outputcase"), pl.lit("_columns"))
            )
            .rename({"mzz": "mzz"})
        )

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_force", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .select("model", "story", "outputcase", "mzz", "elev")
                        .vstack(df_pier_columns)
                        .with_columns(
                            mzz = pl.when(pl.col("outputcase").str.contains("^W*X$")).then(- pl.col("mzz")).otherwise(pl.col("mzz"))
                        )
                    ).mark_circle().encode(
                        alt.X("mzz:Q").title("Moment [kNm]"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "mzz"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_force", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .select("model", "story", "outputcase", "mzz", "elev")
                        .vstack(df_pier_columns)
                        .with_columns(
                            mzz = pl.when(pl.col("outputcase").str.contains("^W*X$")).then(- pl.col("mzz")).otherwise(pl.col("mzz"))
                        )
                    ).mark_line().encode(
                        alt.X("mzz:Q").title("Moment [kNm]"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Force MYY"))
            .configure_scale(zero=False)
        )
    
    # ----- storydisplacement -----

    def log_story_displacement_ux_max(
            self, outputcases: list[str], elev_min: float | None = None, elev_max: float | None = None,
            displacement_limit: float = 500,
        ) -> alt.TopLevelMixin:
        """Line chart of the maximum story displacement in X up the latest revision, against height / displacement_limit

        Args:
            outputcases: The load cases and combinations to plot.
            elev_min, elev_max: The elevations the limit line spans; the lowest and highest stories if None.
            displacement_limit: The limit as a fraction of the height: 500 for H/500.
        """

        if elev_min is None:
            elev_min = self.get_df("story").select(pl.col("elev").min()).item()
        if elev_max is None:
            elev_max = self.get_df("story").select(pl.col("elev").max()).item()

        df_line = (
            pl.DataFrame({
                "y": [elev_min, elev_max]
            })
            .with_columns(
                x = pl.col("y") / displacement_limit
            )
        )

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_displacement", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                    ).mark_circle().encode(
                        alt.X("ux_max:Q").title("ux_max"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "ux_max"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_displacement", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                    ).mark_line().encode(
                        alt.X("ux_max:Q").title("ux_max"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "ux_max"]),
                    )
                )
                +
                (
                    alt.Chart(
                        df_line
                    ).mark_line().encode(
                        alt.X("x:Q").title("ux_max"),
                        alt.Y("y:Q").title("Elevation [m]"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Displacement UX Max"))
            .configure_scale(zero=False)
        )
    
    def log_story_displacement_uy_max(
            self, outputcases: list[str], elev_min: float | None = None, elev_max: float | None = None,
            displacement_limit: float = 500,
        ) -> alt.TopLevelMixin:
        """Line chart of the maximum story displacement in Y up the latest revision, against height / displacement_limit

        Args:
            outputcases: The load cases and combinations to plot.
            elev_min, elev_max: The elevations the limit line spans; the lowest and highest stories if None.
            displacement_limit: The limit as a fraction of the height: 500 for H/500.
        """

        if elev_min is None:
            elev_min = self.get_df("story").select(pl.col("elev").min()).item()
        if elev_max is None:
            elev_max = self.get_df("story").select(pl.col("elev").max()).item()

        df_line = (
            pl.DataFrame({
                "y": [elev_min, elev_max]
            })
            .with_columns(
                x = pl.col("y") / displacement_limit
            )
        )

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_displacement", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                    ).mark_circle().encode(
                        alt.X("uy_max:Q").title("uy_max"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "elev", "uy_max", "outputcase"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_displacement", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                    ).mark_line().encode(
                        alt.X("uy_max:Q").title("uy_max"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                    )
                )
                +
                (
                    alt.Chart(
                        df_line
                    ).mark_line().encode(
                        alt.X("x:Q").title("uy_max"),
                        alt.Y("y:Q").title("Elevation [m]"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Displacement UY Max"))
            .configure_scale(zero=False)
        )
    
    # ----- storydrift -----

    def log_story_drift_ux_max(
            self, outputcases: list[str], elev_min: float | None = None, elev_max: float | None = None,
            drift_limit: float = 300,
        ) -> alt.TopLevelMixin:
        """Line chart of the maximum interstorey drift ratio in X up the latest revision, against 1 / drift_limit

        Args:
            outputcases: The load cases and combinations to plot.
            elev_min, elev_max: The elevations the limit line spans; the lowest and highest stories if None.
            drift_limit: The limit as a fraction of the storey height: 300 for h/300.
        """

        if elev_min is None:
            elev_min = self.get_df("story").select(pl.col("elev").min()).item()
        if elev_max is None:
            elev_max = self.get_df("story").select(pl.col("elev").max()).item()

        df_line = (
            pl.DataFrame({
                "y": [elev_min, elev_max]
            })
            .with_columns(
                x = pl.lit(1.0/drift_limit)
            )
        )

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_drift", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .with_columns(
                            ux_max = pl.col("ux_max") / pl.col("height")
                        )
                    ).mark_circle().encode(
                        alt.X("ux_max:Q").title("ux_max"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "elev", "ux_max"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_drift", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .with_columns(
                            ux_max = pl.col("ux_max") / pl.col("height")
                        )
                    ).mark_line().encode(
                        alt.X("ux_max:Q").title("ux_max"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                    )
                )
                +
                (
                    alt.Chart(
                        df_line
                    ).mark_line().encode(
                        alt.X("x:Q").title("ux_max"),
                        alt.Y("y:Q").title("Elevation [m]"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Drift UX Max"))
            .configure_scale(zero=False)
        )
    
    def log_story_drift_uy_max(
            self, outputcases: list[str], elev_min: float | None = None, elev_max: float | None = None,
            drift_limit: float = 300,
        ) -> alt.TopLevelMixin:
        """Line chart of the maximum interstorey drift ratio in Y up the latest revision, against 1 / drift_limit

        Args:
            outputcases: The load cases and combinations to plot.
            elev_min, elev_max: The elevations the limit line spans; the lowest and highest stories if None.
            drift_limit: The limit as a fraction of the storey height: 300 for h/300.
        """

        if elev_min is None:
            elev_min = self.get_df("story").select(pl.col("elev").min()).item()
        if elev_max is None:
            elev_max = self.get_df("story").select(pl.col("elev").max()).item()

        df_line = (
            pl.DataFrame({
                "y": [elev_min, elev_max]
            })
            .with_columns(
                x = pl.lit(1.0/drift_limit)
            )
        )

        return (
            (
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_drift", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .with_columns(
                            uy_max = pl.col("uy_max") / pl.col("height")
                        )
                    ).mark_circle().encode(
                        alt.X("uy_max:Q").title("uy_max"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["model", "story", "elev", "uy_max"]),
                    )
                )
                +
                (
                    alt.Chart(
                        self.get_df_head_with_story("story_drift", 1)
                        .filter(pl.col("outputcase").is_in(outputcases))
                        .with_columns(
                            uy_max = pl.col("uy_max") / pl.col("height")
                        )
                    ).mark_line().encode(
                        alt.X("uy_max:Q").title("uy_max"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                    )
                )
                +
                (
                    alt.Chart(
                        df_line
                    ).mark_line().encode(
                        alt.X("x:Q").title("uy_max"),
                        alt.Y("y:Q").title("Elevation [m]"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Story Drift UY Max"))
            .configure_scale(zero=False)
        )
    
    # ----- modal -----

    def log_modal_period_vs_time(self) -> alt.TopLevelMixin:
        """Line chart of the period of the mode with the most mass participation in UX, UY and RZ,
        in each revision
        """

        df_modal = (
            self.get_df("modal")
            .group_by("datetime", "model")
            .agg(
                pl.col("outputcase", "period", "ux").sort_by("ux").last(),
                pl.col("ux").sort_by("ux").last().alias("modal_participation"),
                pl.lit("ux_max").alias("type"),
            )
            .drop("ux")
            .vstack(
                self.get_df("modal")
                .group_by("datetime", "model")
                .agg(
                    pl.col("outputcase", "period", "uy").sort_by("uy").last(),
                    pl.col("uy").sort_by("uy").last().alias("modal_participation"),
                    pl.lit("uy_max").alias("type"),
                )
                .drop("uy")
            )
            .vstack(
                self.get_df("modal")
                .group_by("datetime", "model")
                .agg(
                    pl.col("outputcase", "period", "rzz").sort_by("rzz").last(),
                    pl.col("rzz").sort_by("rzz").last().alias("modal_participation"),
                    pl.lit("rzz_max").alias("type"),
                )
                .drop("rzz")
            )
            .sort("datetime", "outputcase")
        )

        return (
            (
                (
                    alt.Chart(
                        df_modal
                    ).mark_circle().encode(
                        alt.X("datetime:T").title("Date Time"),
                        alt.Y("period:Q").title("Period [s]"),
                        alt.Color("type:N"),
                        alt.Tooltip(["model", "period", "modal_participation"]),
                    )
                )
                +
                (
                    alt.Chart(
                        df_modal
                    ).mark_line().encode(
                        alt.X("datetime:T").title("Date Time"),
                        alt.Y("period:Q").title("Period [s]"),
                        alt.Color("type:N"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Modal Periods vs Time"))
            .configure_scale(zero=False)
        )
    
    def log_modal_participation_vs_time(self) -> alt.TopLevelMixin:
        """Line chart of the cumulative mass participation in each direction, over all the modes,
        in each revision
        """

        df_modal = (
            self.get_df("modal")
            .group_by("datetime", "model")
            .agg(pl.col("ux", "uy", "uz", "rxx", "ryy", "rzz").sum())
            .unpivot(on=["ux", "uy", "uz", "rxx", "ryy", "rzz"], index=["datetime", "model"], variable_name="direction", value_name="cumulative_sum")
        )

        return (
            (
                (
                    alt.Chart(
                        df_modal
                    ).mark_circle().encode(
                        alt.X("datetime:T").title("Date Time"),
                        alt.Y("cumulative_sum:Q").title("Cumulative mass participation"),
                        alt.Color("direction:N"),
                        alt.Tooltip(["model", "direction", "cumulative_sum"]),
                    )
                )
                +
                (
                    alt.Chart(
                        df_modal
                    ).mark_line().encode(
                        alt.X("datetime:T").title("Date Time"),
                        alt.Y("cumulative_sum:Q").title("Cumulative mass participation"),
                        alt.Color("direction:N"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Modal Cumulative Participation Ratios vs Time"))
            .configure_scale(zero=False)
        )
    
    # ----- seismic -----

    def log_seismic_sensitivity_x(
            self, contains_outputcase: str, elev_min: float | None = None, elev_max: float | None = None,
            sensitivity_limit: float = 0.3,
        ) -> alt.TopLevelMixin:
        """Line chart of the interstorey drift sensitivity coefficient in X up the latest revision

        theta = P d / (V h), EN 1998-1 4.4.2.2(2): P the axial force and V the shear at the top of each
        story, d its drift and h its height. Second-order effects may be neglected below 0.1, and
        theta may not exceed 0.3, 4.4.2.2(4).

        Args:
            contains_outputcase: Regular expression the seismic cases match.
            elev_min, elev_max: The elevations the limit line spans; the lowest and highest stories if None.
            sensitivity_limit: The limit drawn.
        """

        if elev_min is None:
            elev_min = self.get_df("story").select(pl.col("elev").min()).item()
        if elev_max is None:
            elev_max = self.get_df("story").select(pl.col("elev").max()).item()

        df_line = (
            pl.DataFrame({
                "y": [elev_min, elev_max]
            })
            .with_columns(
                x = pl.lit(sensitivity_limit)
            )
        )

        df_story_force = (
            self.get_df_head_with_story("story_force", 1)
            .filter(
                pl.col("outputcase").str.contains(contains_outputcase),
                pl.col("station") == 1,
            )
            .select("outputcase", "story", "station", "elev", "height", "fx", "fy", "fz")
        )

        df_drift = (
            self.get_df_head("story_drift", 1)
            .filter(
                pl.col("outputcase").str.contains(contains_outputcase),
            )
            .select("outputcase", "story", "ux_max", "uy_max")
        )

        df_seismic_sensitivity = (
            df_story_force
            .join(df_drift, on=["story", "outputcase"], how="left")
            .with_columns(
                sensitivity_x = (pl.col("fx") / pl.col("fy") * pl.col("ux_max") / pl.col("height")).abs(),
                sensitivity_y = (pl.col("fx") / pl.col("fz") * pl.col("uy_max") / pl.col("height")).abs(),
            )
        )

        return (
            (
                (
                    alt.Chart(
                        df_seismic_sensitivity
                    ).mark_circle().encode(
                        alt.X("sensitivity_x:Q").title("sensitivity_x"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["outputcase", "sensitivity_x"])
                    )
                )
                +
                (
                    alt.Chart(
                        df_seismic_sensitivity
                    ).mark_line().encode(
                        alt.X("sensitivity_x:Q").title("sensitivity_x"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                    )
                )
                +
                (
                    alt.Chart(
                        df_line
                    ).mark_line().encode(
                        alt.X("x:Q").title("sensitivity_x"),
                        alt.Y("y:Q").title("Elevation [m]"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Seismic Sensitivity X"))
            .configure_scale(zero=False)
        )
    
    def log_seismic_sensitivity_y(
            self, contains_outputcase: str, elev_min: float | None = None, elev_max: float | None = None,
            sensitivity_limit: float = 0.3,
        ) -> alt.TopLevelMixin:
        """Line chart of the interstorey drift sensitivity coefficient in Y up the latest revision

        theta = P d / (V h), EN 1998-1 4.4.2.2(2): P the axial force and V the shear at the top of each
        story, d its drift and h its height. Second-order effects may be neglected below 0.1, and
        theta may not exceed 0.3, 4.4.2.2(4).

        Args:
            contains_outputcase: Regular expression the seismic cases match.
            elev_min, elev_max: The elevations the limit line spans; the lowest and highest stories if None.
            sensitivity_limit: The limit drawn.
        """

        if elev_min is None:
            elev_min = self.get_df("story").select(pl.col("elev").min()).item()
        if elev_max is None:
            elev_max = self.get_df("story").select(pl.col("elev").max()).item()

        df_line = (
            pl.DataFrame({
                "y": [elev_min, elev_max]
            })
            .with_columns(
                x = pl.lit(sensitivity_limit)
            )
        )

        df_story_force = (
            self.get_df_head_with_story("story_force", 1)
            .filter(
                pl.col("outputcase").str.contains(contains_outputcase),
                pl.col("station") == 1,
            )
            .select("outputcase", "story", "station", "elev", "height", "fx", "fy", "fz")
        )

        df_drift = (
            self.get_df_head("story_drift", 1)
            .filter(
                pl.col("outputcase").str.contains(contains_outputcase),
            )
            .select("outputcase", "story", "ux_max", "uy_max")
        )

        df_seismic_sensitivity = (
            df_story_force
            .join(df_drift, on=["story", "outputcase"], how="left")
            .with_columns(
                sensitivity_x = (pl.col("fx") / pl.col("fy") * pl.col("ux_max") / pl.col("height")).abs(),
                sensitivity_y = (pl.col("fx") / pl.col("fz") * pl.col("uy_max") / pl.col("height")).abs(),
            )
        )

        return (
            (
                (
                    alt.Chart(
                        df_seismic_sensitivity
                    ).mark_circle().encode(
                        alt.X("sensitivity_y:Q").title("sensitivity_y"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                        alt.Tooltip(["outputcase", "sensitivity_y"])
                    )
                )
                +
                (
                    alt.Chart(
                        df_seismic_sensitivity
                    ).mark_line().encode(
                        alt.X("sensitivity_y:Q").title("sensitivity_y"),
                        alt.Y("elev:Q").title("Elevation [m]"),
                        alt.Color("outputcase:N"),
                        alt.Order("elev:Q"),
                    )
                )
                +
                (
                    alt.Chart(
                        df_line
                    ).mark_line().encode(
                        alt.X("x:Q").title("sensitivity_y"),
                        alt.Y("y:Q").title("Elevation [m]"),
                    )
                )
            )
            .properties(width=1200, title=(self.name + " - Seismic Sensitivity Y"))
            .configure_scale(zero=False)
        )