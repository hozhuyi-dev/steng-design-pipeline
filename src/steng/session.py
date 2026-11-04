"""Session: the standard model, one Polars DataFrame per attribute, with a child session per model revision
"""

import datetime
import os
import re
from importlib.resources import files
from typing import TypeVar

import numpy as np
import polars as pl

from steng.model.det import DetMixin
from steng.model.etabs import EtabsMixin
from steng.model.rhino import RhinoMixin
from steng.model.safe import SafeMixin
from steng.report.log import LogMixin
from steng.utils import check_folder, convert_to_list

TYPES = {
    "angle": pl.Float64,
    "area": pl.String,
    "beam": pl.String,
    "brace": pl.String,
    "column": pl.String,
    "elev": pl.Float64,
    "floor": pl.String,
    "fx": pl.Float64,
    "fy": pl.Float64,
    "fz": pl.Float64,
    "grade": pl.UInt32,
    "height": pl.Float64,
    "line": pl.String,
    "mx": pl.Float64,
    "my": pl.Float64,
    "mz": pl.Float64,
    "mxx": pl.Float64,
    "myy": pl.Float64,
    "mzz": pl.Float64,
    "outputcase": pl.String,
    "outputcasetype": pl.String, # ("static", "modal", "response_spectrum", "combination")
    "outputstep": pl.UInt32,
    "period": pl.Float64,
    "pier": pl.String,
    "point": pl.UInt32,
    "restraint_ux": pl.Boolean,
    "restraint_uy": pl.Boolean,
    "restraint_uz": pl.Boolean,
    "restraint_rxx": pl.Boolean,
    "restraint_ryy": pl.Boolean,
    "restraint_rzz": pl.Boolean,
    "rxx": pl.Float64,
    "ryy": pl.Float64,
    "rzz": pl.Float64,
    "section_line": pl.String,
    "spandrel": pl.String,
    "station": pl.Float64,
    "story": pl.String,
    "story_below": pl.String,
    "thickness": pl.Float64,
    "unique_area": pl.UInt32,
    "unique_line": pl.UInt32,
    "unique_point": pl.UInt32,
    "ux": pl.Float64,
    "ux_avg": pl.Float64,
    "ux_max": pl.Float64,
    "uy": pl.Float64,
    "uy_avg": pl.Float64,
    "uy_max": pl.Float64,
    "uz": pl.Float64,
    "wall": pl.String,
    "x": pl.Float64,
    "xs": pl.List(pl.Float64),
    "y": pl.Float64,
    "ys": pl.List(pl.Float64),
    "z": pl.Float64,
    "zs": pl.List(pl.Float64),
}

S = TypeVar("S", bound="Session")

SCHEMA = {
    # GEOMETRY
    # TODO property modifier, end releases, shelltype, group
    "story": {i: TYPES[i] for i in ["story", "elev"] if i in TYPES},
    "point": {i: TYPES[i] for i in [
        "unique_point", "point", "story", "x", "y", "z",
        "restraint_ux", "restraint_uy", "restraint_uz", "restraint_rxx", "restraint_ryy", "restraint_rzz", 
        "section_line", "fz", "uz",
    ] if i in TYPES},
    "beam": {i: TYPES[i] for i in ["unique_line", "beam", "story", "xs", "ys", "zs", "grade", "section_line", "angle"] if i in TYPES},
    "brace": {i: TYPES[i] for i in ["unique_line", "brace", "story", "xs", "ys", "zs", "grade", "section_line", "angle"] if i in TYPES},
    "column": {i: TYPES[i] for i in ["unique_line", "column", "story", "xs", "ys", "zs", "grade", "section_line", "angle"] if i in TYPES},
    "line": {i: TYPES[i] for i in ["unique_line", "line", "story", "xs", "ys", "zs"] if i in TYPES},
    "area": {i: TYPES[i] for i in ["unique_area", "area", "story", "xs", "ys", "zs"] if i in TYPES},
    "floor": {i: TYPES[i] for i in ["unique_area", "floor", "story", "xs", "ys", "zs", "grade", "thickness"] if i in TYPES},
    "wall": {i: TYPES[i] for i in ["unique_area", "wall", "story", "xs", "ys", "zs", "grade", "thickness", "pier", "angle", "spandrel"] if i in TYPES},

    # LOAD
    # TODO "load_case": {i: TYPES[i] for i in [] if i in TYPES},
    # TODO "load_combination": {i: TYPES[i] for i in [] if i in TYPES},
    # TODO "uniform_load_set": {i: TYPES[i] for i in [] if i in TYPES},

    # RESULT GLOBAL
    "base_reaction": {i: TYPES[i] for i in ["outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz"] if i in TYPES},
    "story_mass": {i: TYPES[i] for i in ["story", "mx", "my", "mz"] if i in TYPES},
    "story_force": {i: TYPES[i] for i in ["story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz"] if i in TYPES},
    "story_displacement": {i: TYPES[i] for i in ["story", "outputcase", "outputcasetype", "ux_max", "ux_avg", "uy_max", "uy_avg"] if i in TYPES},
    "story_drift": {i: TYPES[i] for i in ["story", "outputcase", "outputcasetype", "ux_max", "ux_avg", "uy_max", "uy_avg"] if i in TYPES},
    "modal": {i: TYPES[i] for i in ["outputcase", "outputcasetype", "period", "ux", "uy", "uz", "rxx", "ryy", "rzz"] if i in TYPES},

    # RESULT JOINT
    "joint_displacement": {i: TYPES[i] for i in ["unique_point", "outputcase", "outputcasetype", "ux", "uy", "uz", "rxx", "ryy", "rzz"] if i in TYPES},
    "joint_reaction": {i: TYPES[i] for i in ["unique_point", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz"] if i in TYPES},
    # TODO "joint_mass": {i: TYPES[i] for i in ["unique_point", "ux", "uy", "uz", "rxx", "ryy", "rzz"] if i in TYPES},

    # RESULT FRAME or PIER or SPANDREL
    "force_column": {i: TYPES[i] for i in ["unique_line", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz"] if i in TYPES},
    "force_pier": {i: TYPES[i] for i in ["pier", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz"] if i in TYPES},
    "force_spandrel": {i: TYPES[i] for i in ["spandrel", "story", "station", "outputcase", "outputcasetype", "fx", "fy", "fz", "mxx", "myy", "mzz"] if i in TYPES},
}

class Session(DetMixin, EtabsMixin, RhinoMixin, SafeMixin, LogMixin):
    """The standard model of a structure: one Polars DataFrame per attribute in SCHEMA

    A session read from several models holds each as a child session in sessions, keyed by model name,
    and get_df returns an attribute across all of them, stamped with each model's datetime and name.

    Attributes:
        name: The model name, from its file or export folder.
        datetime: When the model was saved or exported, as "YYYYmmdd-HHMMSS".
        sessions: Child sessions by model name.
        story, point, beam, ..., force_spandrel: One DataFrame per attribute in SCHEMA, in kN and m.
    """

    def __init__(self, name: str | None = None) -> None:
        """Start an empty session, every attribute an empty DataFrame of its SCHEMA columns"""

        self.datetime = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
        self.name = name
        self.unit_force, self.unit_length, self.unit_temperature = 1.0, 1.0, 1.0
        
        self = self._initialise_attributes(list(SCHEMA.keys()))

        self.sessions = {}

        DetMixin.__init__(self)
        EtabsMixin.__init__(self)
        RhinoMixin.__init__(self)
        SafeMixin.__init__(self)

        LogMixin.__init__(self)

    def _initialise_attributes(self, attributes):
        """Initialise self with empty attributes df
        """

        attributes = convert_to_list(attributes)

        for attribute in attributes:
            setattr(
                self, 
                attribute, 
                _cast_df(None, SCHEMA[attribute])
            )

        return self
    
    # ----- create -----

    def read_models(
            self: S, files: str | list[str], abs_path: str | None = None, format: str = "parquet", purpose: str = "all",
        ) -> S:
        """Read each model as a child session, keyed by its model name

        Args:
            files: ETABS text files ("tower_rev01.$et") under abs_path/etabs, or ETABS table export
                folders named <datetime>_<model>_<purpose>_etabs, as Etabs.write_tables writes them.
            abs_path: The folder holding the etabs folder; the working directory if None.
            format: The table exports' format, "parquet" or "csv".
            purpose: The column of data/etabs_tables.csv that lists the tables to read.
        """

        files = convert_to_list(files)

        for file in files:
            if file.endswith(".$et") or file.endswith(".e2k"):
                name = file.split(".")[0]
                self.sessions[name] = Session(name).read_det(file, abs_path=abs_path)
            elif file.endswith("_etabs"):
                name = _split_export_name(file)[1]
                self.sessions[name] = Session(name).read_etabs(file, abs_path=abs_path, format=format, purpose=purpose)

        return self
    
    def read_model(self: S, folder: str, format: str = "parquet") -> S:
        """Read a standard model written by write_model, one file per attribute

        Args:
            folder: The folder write_model wrote, <datetime>_<name>_model.
            format: "parquet" or "csv".
        """

        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)
        self.datetime, self.name = _split_export_name(os.path.basename(os.path.normpath(folder)))[:2]

        for attribute, schema in SCHEMA.items():
            path = os.path.join(folder, attribute + "." + format)
            if not os.path.exists(path):
                continue
            df = pl.read_parquet(path) if format == "parquet" else pl.read_csv(path)
            setattr(self, attribute, _cast_df(df, schema, include_custom=True))

        return self
    
    def read_det(self: S, file: str, abs_path: str | None = None, folder: str = r"./etabs") -> S:
        """Read the geometry of an ETABS text file (.$et) into this session, dated when it was saved

        Args:
            file: The file name, which becomes the session's name without its extension.
            abs_path: The folder holding folder; the working directory if None.
            folder: The folder holding the file.
        """
        
        path = _append_abs_path(abs_path, folder + os.sep + file)
        self.datetime = _read_saved_datetime(path) or datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
        self.name = file.split(".")[0]
        self.det = DetMixin._read_det(self, path)

        self.unit_force, self.unit_length, self.unit_temperature = _convert_unit_to_multiplier(*DetMixin._expand_unit(self))

        self.story = _cast_df(DetMixin._expand_story(self), SCHEMA["story"])
        self.point = _cast_df(DetMixin._expand_point(self), SCHEMA["point"])
        self.beam, self.brace, self.column, self.line = DetMixin._expand_line(self)
        self.beam = _cast_df(self.beam, SCHEMA["beam"])
        self.brace = _cast_df(self.brace, SCHEMA["brace"])
        self.column = _cast_df(self.column, SCHEMA["column"])
        self.line = _cast_df(self.line, SCHEMA["line"])
        self.area, self.floor, self.wall = DetMixin._expand_area(self)
        self.area = _cast_df(self.area, SCHEMA["area"])
        self.floor = _cast_df(self.floor, SCHEMA["floor"])
        self.wall = _cast_df(self.wall, SCHEMA["wall"])

        self = self._calculate_secondary_attributes()

        return self
    
    def read_etabs(
            self: S, subfolder: str, abs_path: str | None = None, folder: str = r"./etabs",
            format: str = "parquet", purpose: str = "all",
        ) -> S:
        """Read an ETABS table export into this session: geometry, and results where exported

        Args:
            subfolder: The export folder, <datetime>_<model>_<purpose>_etabs.
            abs_path: The folder holding folder; the working directory if None.
            folder: The folder holding the export folders.
            format: The tables' format, "parquet" or "csv".
            purpose: The column of data/etabs_tables.csv that lists the tables to read.
        """

        self.datetime, self.name = _split_export_name(subfolder)[:2]
        self.etabs = _read_dfs(
            abs_path, folder, subfolder, 
            pl.read_csv(files("steng") / "data" / "etabs_tables.csv").filter(pl.col(purpose) == 1)["table"].to_list(), 
            format
        )

        self.story = _cast_df(EtabsMixin._expand_story(self), SCHEMA["story"])
        self.point = _cast_df(EtabsMixin._expand_point(self), SCHEMA["point"])
        self.beam = _cast_df(EtabsMixin._expand_beam(self), SCHEMA["beam"])
        self.brace = _cast_df(EtabsMixin._expand_brace(self), SCHEMA["brace"])
        self.column = _cast_df(EtabsMixin._expand_column(self), SCHEMA["column"])
        self.line = _cast_df(EtabsMixin._expand_line(self), SCHEMA["line"])
        self.area = _cast_df(EtabsMixin._expand_area(self), SCHEMA["area"])
        self.floor = _cast_df(EtabsMixin._expand_floor(self), SCHEMA["floor"])
        self.wall = _cast_df(EtabsMixin._expand_wall(self), SCHEMA["wall"])

        self.base_reaction = _cast_df(EtabsMixin._expand_base_reaction(self), SCHEMA["base_reaction"])
        self.story_mass = _cast_df(EtabsMixin._expand_story_mass(self), SCHEMA["story_mass"])
        self.story_force = _cast_df(EtabsMixin._expand_story_force(self), SCHEMA["story_force"])
        self.story_displacement = _cast_df(EtabsMixin._expand_story_displacement(self), SCHEMA["story_displacement"])
        self.story_drift = _cast_df(EtabsMixin._expand_story_drift(self), SCHEMA["story_drift"])
        self.modal = _cast_df(EtabsMixin._expand_modal(self), SCHEMA["modal"])

        self.joint_displacement = _cast_df(EtabsMixin._expand_joint_displacement(self), SCHEMA["joint_displacement"])
        self.joint_reaction = _cast_df(EtabsMixin._expand_joint_reaction(self), SCHEMA["joint_reaction"])

        self.force_column = _cast_df(EtabsMixin._expand_force_column(self), SCHEMA["force_column"])
        self.force_pier = _cast_df(EtabsMixin._expand_force_pier(self), SCHEMA["force_pier"])
        self.force_spandrel = _cast_df(EtabsMixin._expand_force_spandrel(self), SCHEMA["force_spandrel"])

        self = self._calculate_secondary_attributes()

        return self
    
    def from_rhino(self: S, attribute: str, x: list, y: list, z: list, keys: list[str], values: list) -> S:
        """Set an attribute from Rhino geometry, as a Grasshopper component passes it

        Args:
            attribute: The attribute to set, "point", "column", "wall", ...
            x, y, z: The coordinates of each object's vertices [m].
            keys, values: Each object's other columns and their values, such as "story" and "grade".
        """

        self.__setattr__(attribute, _cast_df(
            RhinoMixin._from_rhino(self, attribute, x, y, z, keys, values), SCHEMA[attribute],
            include_custom=True
        ))

        self = self._calculate_secondary_attributes()

        return self

    def create_latest_model(self: S) -> S:
        """Copy the latest child session's attributes and units into this session

        The latest is the child session with the most recent datetime. The children stay in sessions,
        so get_df still returns every revision.
        """

        self.name = self.get_df_head("story", 1).select("model").unique().item()
        latest = self.sessions[self.name]
        self.datetime = latest.datetime
        self.unit_force, self.unit_length, self.unit_temperature = latest.unit_force, latest.unit_length, latest.unit_temperature

        self = self._clone_attributes(self.name, list(SCHEMA.keys()))

        return self
    
    def _clone_attributes(self, name, attributes):
        """Clone from a model to the current model
        """

        attributes = convert_to_list(attributes)

        for attribute in attributes:
            setattr(
                self, 
                attribute, 
                (
                    getattr(self.sessions[name], attribute)
                    .clone()
                )
            )

        return self

    # ----- read -----

    def get_df(self, attribute: str) -> pl.DataFrame | None:
        """Return an attribute of every child session stacked, with datetime and model columns first

        Args:
            attribute: The attribute, "story", "wall", "story_drift", ...

        Returns:
            None if there are no child sessions.
        """

        df = None

        for key in list(self.sessions.keys()):
            if getattr(self.sessions[key], attribute) is None:
                continue
            
            df_current = getattr(self.sessions[key], attribute)
            columns = list(df_current.columns)
            df_current = (
                df_current
                .with_columns(
                    datetime = pl.lit(self.sessions[key].datetime).str.to_datetime("%Y%m%d-%H%M%S"),
                    model = pl.lit(key)
                )
                .select(["datetime", "model"] + columns)
            )

            if df is None:
                df = df_current
            else:
                df = df.vstack(df_current)
        
        return df

    def get_df_with_story(self, attribute: str) -> pl.DataFrame:
        """Return get_df(attribute) joined to each model's stories, adding elev and height
        """

        return (
            self.get_df(attribute)
            .join(self.get_df("story"), on=["datetime", "model", "story"])
        )
    
    def get_df_head(self, attribute: str, number: int) -> pl.DataFrame:
        """Return get_df(attribute) for the latest number of models only, by datetime
        """

        datetimes = self.get_df(attribute).select("datetime").unique().sort("datetime", descending=True).head(number).to_series().to_list()
        
        return self.get_df(attribute).filter(pl.col("datetime").is_in(datetimes))
    
    def get_df_head_with_story(self, attribute: str, number: int) -> pl.DataFrame:
        """Return get_df_head(attribute, number) joined to each model's stories, adding elev and height
        """
        
        return (
            self.get_df_head(attribute, number)
            .join(self.get_df("story"), on=["datetime", "model", "story"])
        )
    
    def _get_base_point(self):
        """Return the smallest x, y and z
        """

        df_coordinates = (
            self.point.select("x", "y", "z")
            .vstack(self.beam.select("xs", "ys", "zs").rename({"xs": "x", "ys": "y", "zs": "z"}).explode("x", "y", "z"))
            .vstack(self.brace.select("xs", "ys", "zs").rename({"xs": "x", "ys": "y", "zs": "z"}).explode("x", "y", "z"))
            .vstack(self.column.select("xs", "ys", "zs").rename({"xs": "x", "ys": "y", "zs": "z"}).explode("x", "y", "z"))
            .vstack(self.line.select("xs", "ys", "zs").rename({"xs": "x", "ys": "y", "zs": "z"}).explode("x", "y", "z"))
            .vstack(self.area.select("xs", "ys", "zs").rename({"xs": "x", "ys": "y", "zs": "z"}).explode("x", "y", "z"))
            .vstack(self.floor.select("xs", "ys", "zs").rename({"xs": "x", "ys": "y", "zs": "z"}).explode("x", "y", "z"))
            .vstack(self.wall.select("xs", "ys", "zs").rename({"xs": "x", "ys": "y", "zs": "z"}).explode("x", "y", "z"))
            .select(
                pl.col("x").min().alias("x_min"),
                pl.col("y").min().alias("y_min"),
                pl.col("z").min().alias("z_min"),
            )
        )

        return (
            df_coordinates.select("x_min").item(),
            df_coordinates.select("y_min").item(),
            df_coordinates.select("z_min").item(),
        )
    
    def _get_story(self, story):
        """Return a df of the info on a story
        """

        return (
            self.story.clone()
            .with_columns(
                story_above = pl.col("story").shift(1),
                story_below = pl.col("story").shift(-1),
                height_above = pl.col("elev").shift(1) - pl.col("elev"),
                height_below = pl.col("elev") - pl.col("elev").shift(-1),
            )
            .filter(pl.col("story") == story)
        )
    
    def _get_grade_unique(self):
        """Return a list of the unique materials
        """

        return (
            self.beam["grade"]
            .append(self.brace["grade"])
            .append(self.column["grade"])
            .append(self.floor["grade"])
            .append(self.wall["grade"])
            .unique()
            .sort()
            .to_list()
        )
    
    def _get_pointspringprop_unique(self):
        """Return the unique point springs
        """

        df = (
            self.point
            .select("section_line", "fz", "uz")
            .unique()
            .filter(
                pl.col("section_line").is_not_null(),
                pl.col("fz").is_not_null(),
                pl.col("uz").is_not_null(),
            )
            .sort("section_line", "fz", "uz")
        )

        return (
            df["section_line"].to_list(),
            df["fz"].to_list(),
            df["uz"].to_list()
        )
    
    def _get_lineprop_unique(self, attribute):
        """Return 2 lists of unique section_lines and grades
        """

        df = (
            getattr(self, attribute)
            .select("grade", "section_line")
            .unique()
            .sort("grade", "section_line")
        )

        return (
            df["section_line"].to_list(),
            df["grade"].to_list()
        )
    
    def _get_areaprop_unique(self, attribute):
        """Return 2 lists of unique thickness and grades
        """

        df = (
            getattr(self, attribute)
            .select("grade", "thickness")
            .unique()
            .sort("grade", "thickness")
        )

        return (
            df["thickness"].to_list(),
            df["grade"].to_list()
        )
    
    # ----- update -----
    
    def update(self: S, attributes: str | list[str], df: pl.DataFrame, on: str | list[str]) -> S:
        """Overwrite values of one or more attributes from df, matching rows on the given columns

        Nulls in df leave the values as they were, as polars DataFrame.update does. Use it to set design
        inputs per member, such as effective length factors:

            session.update("wall", df_k, on=["pier", "story"])
        """

        attributes = convert_to_list(attributes)

        for attribute in attributes:
            setattr(
                self, 
                attribute, 
                (
                    getattr(self, attribute)
                    .update(df, on=on, how="left")
                )
            )

        return self
    
    def _calculate_secondary_attributes(self):
        """Calculate secondary attributes from the primary attributes
        """

        self.story = (
            self.story
            .sort("elev", descending=True)
            .with_columns(
                height = pl.col("elev") - pl.col("elev").shift(-1),
            )
        )

        areas = [_polygon_area_3d_np(xs, ys, zs) for xs, ys, zs in self.area.select("xs", "ys", "zs").iter_rows()]
        self.area = self.area.with_columns(pl.Series("area2", areas))

        areas = [_polygon_area_3d_np(xs, ys, zs) for xs, ys, zs in self.floor.select("xs", "ys", "zs").iter_rows()]
        self.floor = self.floor.with_columns(pl.Series("area", areas))

        areas = [_polygon_area_3d_np(xs, ys, zs) for xs, ys, zs in self.wall.select("xs", "ys", "zs").iter_rows()]
        self.wall = self.wall.with_columns(pl.Series("area", areas))

        return self
    
    # ----- delete -----

    # ----- out -----

    def write_model(self: S, folder: str = R".\model", format: str = "parquet") -> S:
        """Write every attribute to its own file in folder/<datetime>_<name>_model, for read_model

        Args:
            folder: The folder to write the model folder in.
            format: "parquet" or "csv".
        """

        folder = folder + os.sep + self.datetime + "_" + self.name + "_model"
        check_folder(folder)

        for attribute in list(SCHEMA.keys()):
            path = folder + os.sep + attribute + "." + format

            if format == "parquet":
                getattr(self, attribute).write_parquet(path)
            elif format == "csv":
                getattr(self, attribute).write_csv(path, separator=",")

        return self
    
def _read_saved_datetime(path):
    """Return when an ETABS text file was saved, from its "$ File ... saved" line, or None

    ETABS writes the date month first, with a 12-hour clock: "saved 5/6/2025 10:15:00 AM".
    """

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = re.match(r"\$ File .* saved (\S+ \S+(?: [AP]M)?)\s*$", line)
            if match:
                for format in ["%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"]:
                    try:
                        return datetime.datetime.strptime(match.group(1), format).strftime("%Y%m%d-%H%M%S")
                    except ValueError:
                        continue
            if line.strip() and not line.startswith("$ File"):
                return None

    return None

def _split_export_name(folder):
    """Return the datetime, model name and purpose of an export folder's name

    ETABS table exports are <datetime>_<model>_<purpose>_etabs and write_model writes
    <datetime>_<model>_model, without a purpose. Model names may contain underscores.
    """

    parts = folder.split("_")
    if parts[-1] == "etabs":
        return parts[0], "_".join(parts[1:-2]), parts[-2]

    return parts[0], "_".join(parts[1:-1]), None

def _read_df(abs_path, folder, subfolder, file, format):
    """Read df and format
    """

    file = _append_abs_path(abs_path, folder + os.sep + subfolder + os.sep + file + "." + format)

    try:
        if format == "parquet":
            df = pl.read_parquet(file)
        elif format == "csv":
            df = pl.read_csv(file)
        else:
            # print(f"Format not support: {format}")
            return None
    except FileNotFoundError:
        # print(f"File not found: {file}")
        return None

    return df

def _read_dfs(abs_path, folder, subfolder, files, format):
    """Read and format
    """

    model = {}

    for file in files:
        model[file] = _read_df(abs_path, folder, subfolder, file, format)
    
    for key in model.keys():
        if model[key] is not None:
            model[key] = (
                model[key]
                .with_columns([
                    pl.when(pl.col(col) == "None").then(None).otherwise(pl.col(col)).alias(col)
                    for col in model[key].columns
                ])
            )
    
    return model

def _cast_df(df, schema, include_custom=False):
    """Typical logic to cast df to the schema
    """

    if df is None:
        return pl.DataFrame(schema=schema)
    
    for column in list(schema.keys()):
        if column not in df.columns:
            df = df.with_columns(pl.lit(None).alias(column))
    
    if include_custom:
        columns = list(schema.keys()) + sorted([x for x in df.columns if x not in list(schema.keys())])
    else:
        columns = list(schema.keys())

    return df.select(columns).cast(schema)

def _convert_unit_to_multiplier(unit_force, unit_length, unit_temperature):
    """Convert units to multiplier to standardise to kN, m, C
    """

    if unit_force == "KN":
        unit_force = 1
    elif unit_force == "N":
        unit_force = 0.001
    if unit_length == "M":
        unit_length = 1
    elif unit_length == "MM":
        unit_length = 0.001
    if unit_temperature == "C":
        unit_temperature = 1
    
    return unit_force, unit_length, unit_temperature

def _append_abs_path(abs_path, rel_file):
    """Return the absolute path
    """

    if abs_path:
        abs_file = os.path.join(abs_path, rel_file)
    else:
        abs_file = rel_file
    
    return abs_file

def _polygon_area_3d_np(x, y, z):
    """Calculate area from lists of xs, ys, zs
    """

    points = np.column_stack((x, y, z))  # shape (n, 3), n is number of points
    shifted = np.roll(points, -1, axis=0)
    cross_prod = np.cross(points, shifted)
    area_vector = cross_prod.sum(axis=0)
    return 0.5 * np.linalg.norm(area_vector)