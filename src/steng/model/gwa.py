"""Gwa: writes the standard model to GSA text files (.gwa)
"""

import copy
import difflib

import polars as pl

from steng.utils import check_folder


class Gwa:
    def __init__(self):
        """Initialise gwa formatter."""

        self.data = {}
        self.data["header"] = _initialise_header()
        self.um_node = {}
        self.um_memb = {}
        self.um_elem = {}

    def from_det(self, det):
        """Read the data from det."""
        
        new_self = copy.deepcopy(self)

        # TODO
        keywords = [
            # "title_1",
            # "spec_steel_design_1",
            # "spec_concrete_design_1",
            # "spec_load_design_1",
            "unit_data_1",
            "grid_plane_3",
            "mat_anal",
            # "mat_concrete_17",
            # "mat_steel_3",
            # "section_7",
            # "section_anal_4",
            "prop_sec_2", # modifier
            "prop_2d_8", # modifier
            "node_3",
            "memb_8.alt_1d.automatic", # rls_1, rls_2
            "memb_8.alt_2d",
            # "el_4", # TODO
            # "el_2", # Implemented but not used
            # "load_title_2",
            # "load_gravity_4",
            # "load_beam_udl_3",
            # "load_2d_face_3",
            # "combination_1",
            # "list_1",
        ]
        
        for keyword in keywords:
            df = det.to_gwa(keyword)
            if df is not None:
                new_self.data[keyword] = _clean_df(df, keyword)

        return new_self
    
    def write_data_to_gwa(self, path):
        """Write to gwa file all data."""

        check_folder(r"./gsa")

        keywords = self.data.keys()

        with open(path, 'w') as gwa:
            for keyword in keywords:
                lines = _format_lines(self.data[keyword], keyword)
                for line in lines:
                    gwa.write(line)
                gwa.write('\n')

        return self
    
    def write_user_mod_node_from_det_to_gwa(self, det, name, point, story, value, path, dimension=""):
        """Write user module node to gwa file"""

        df_user_mod_node_title = pl.DataFrame({
            "module": 0,
            "name": name,
            "dimension": "m"
        })

        df_det_to_gwa_mapping = det.to_gwa("user_mod_node")

        df_user_mod_node = (
            pl.DataFrame({
                "point": point,
                "story": story,
                "value": value,
            })
            .cast({"point": pl.UInt32})
            .with_columns(
                module = pl.lit(0)
            )
            .join(df_det_to_gwa_mapping, on=["point", "story"], how="left")
            .drop("point", "story")
        )

        with open(path, 'w') as gwa:
            lines = _format_lines(df_user_mod_node_title, "user_mod_node_title_1")
            for line in lines:
                gwa.write(line)
            gwa.write('\n')

            lines = _format_lines(df_user_mod_node, "user_mod_node")
            for line in lines:
                gwa.write(line)
            gwa.write('\n')

        return self

def _initialise_header():
    """Initialise headers as a list of texts to be printed at the beginning of gwa"""

    header = pl.DataFrame({
        "header": [
            r"! This file is written by GWA."
        ]
    })

    return header

def _clean_df(df, keyword):
    """Clean data frame by finding closest match and add missing column"""

    existing_columns = df.columns
    options = _get_options(keyword)
    columns = options.keys()
    for column in columns:
        if column in existing_columns:
            df = (
                df
                .with_columns(
                    pl.col(column)
                    .map_elements(
                        _generate_func_match(options[column]),
                        return_dtype=pl.String
                    )
                    .alias(column)
                )
            )

    defaults = _get_defaults(keyword)
    columns = defaults.keys()
    for column in columns:
        if column not in existing_columns:
            df = (
                df
                .with_columns(
                    pl.lit(defaults[column]).alias(column)
                )
            )
    
    existing_columns = df.columns
    format = _get_format(keyword)[1:]
    for column in format:
        if column not in existing_columns:
            df = (
                df
                .with_columns(
                    pl.lit("").alias(column)
                )
            )
    
    return df

def _get_options(keyword):
    """Return a dictionary of options to be matched against"""

    if keyword == "unit_data_1":
        options = {
            "option": [
                "FORCE", "LENGTH", "DISP", "SECTION", "MASS", "TIME",
                "TEMP", "STRESS", "ACCEL", "ENERGY", "STRAIN"
            ],
            "name": [
                "N", "kN", "MN", 
                "m", "cm", "mm", 
                "°C", "°F"
            ],
        }
    elif keyword == "grid_plane_3":
        options = {
            "type": ["GENERAL", "STOREY"]
        }
    elif keyword == "prop_sec_2":
        options = {
            "type": ["GENERIC", "BEAM", "COLUMN"]
        }
    elif keyword == "prop_2d_8":
        options = {
            "type": ["STRESS", "SHELL"]
        }
    elif keyword == "memb_8.alt_1d.automatic":
        options = {
            "type (1D)": ["BEAM", "COLUMN", "1D_GENERIC"],
            "exposure": ["ALL", "THREE", "TOP_BOT", "SIDES", "ONE", "NONE"],
            "is_intersector": ["YES", "NO"],
            "analysis_type": ["BEAM"],
            "dummy": ["ACTIVE", "DUMMY"],
            "AUTOMATIC": ["AUTOMATIC"],
            "load_ref": ["SHR_CENTRE", "TOP_FLANGE", "BOT_FLANGE"],
            "is_off": ["OFF", "NO_OFF"],
            "has_ltb_factor": ["LTB_FACTOR", "NO_LTB_FACTOR"],
            "has_moment_amplification_factor_strong_axis": [
                "MOMENT_AMPLIFICATION_FACTOR_STRONG_AXIS", 
                "NO_MOMENT_AMPLIFICATION_FACTOR_STRONG_AXIS"
            ],
            "has_moment_amplification_factor_weak_axis": [
                "MOMENT_AMPLIFICATION_FACTOR_WEAK_AXIS",
                "NO_MOMENT_AMPLIFICATION_FACTOR_WEAK_AXIS"
            ],
        }
    elif keyword == "memb_8.alt_2d":
        options = {
            "type (2D)": ["SLAB", "WALL", "2D_GENERIC"],
            "exposure": ["ALL", "THREE", "TOP_BOT", "SIDES", "ONE", "NONE"],
            "is_intersector": ["YES", "NO"],
            "analysis_type": ["LINEAR"],
            "dummy": ["ACTIVE", "DUMMY"],
            "off_auto_internal": ["YES", "NO"],
            "reinforcement2d[1]": ["REBAR_2D.1"],
            "mesh_mode": ["TRI", "MIXED", "QUAD"],
        }
    else:
        options = {}

    return options

def _get_defaults(keyword):
    """Return a dictionary of defaults"""

    if keyword == "grid_plane_3":
        defaults = {
            "type": "STOREY",
            "axis": 0,
            "below": 0,
            "above": 0,
        }
    elif keyword == "mat_anal":
        defaults = {
            "MAT_ELAS_ISO": "MAT_ELAS_ISO",
            "colour": "NO_RGB",
            "6": 6,
            "G": 0,
            "damp": 0,
        }
    elif keyword == "prop_sec_2":
        defaults = {
            "colour": "NO_RGB",
            "type": "GENERIC",
        }
    elif keyword == "prop_2d_8":
        defaults = {
            "colour": "NO_RGB",
            "axis": "GLOBAL",
            "design": 0,
            "ref_pt": "CENTROID",
            "ref_z": 0,
            "mass": 0,
            "flex": "100%",
            "shear": "100%",
            "inplane": "100%",
            "weight": "100%",
        }
    elif keyword == "node_3":
        defaults = {
            "colour": "NO_RGB",
        }
    elif keyword == "memb_8.alt_1d.automatic":
        defaults = {
            "type (1D)": "1D_GENERIC",
            "colour": "NO_RGB",
            "exposure": "ALL",
            "prop": 1,
            "group": 1,
            "node": 0,
            "mesh_size": 0,
            "is_intersector": "YES",
            "analysis_type": "BEAM",
            "fire": 0,
            "limiting_temperature": 0,
            "time[1]": 0,
            "time[2]": 0,
            "time[3]": 0,
            "time[4]": 0,
            "dummy": "ACTIVE",
            "rls_1": "FFFFFF",
            "rls_2": "FFFFFF",
            "restraint_end_1": "Free",
            "restraint_end_2": "Free",
            "AUTOMATIC": "AUTOMATIC",
            "load_height": 0,
            "load_ref": "SHR_CENTRE",
            "is_off": "NO_OFF",
            "has_ltb_factor": "NO_LTB_FACTOR",
            "has_moment_amplification_factor_strong_axis": "NO_MOMENT_AMPLIFICATION_FACTOR_STRONG_AXIS",
            "has_moment_amplification_factor_weak_axis": "NO_MOMENT_AMPLIFICATION_FACTOR_STRONG_AXIS",
        }
    elif keyword == "memb_8.alt_2d":
        defaults = {
            "colour": "NO_RGB",
            "exposure": "ALL",
            "prop": 1,
            "group": 1,
            "node": 0,
            "angle": 0,
            "mesh_size": 0,
            "is_intersector": "YES",
            "analysis_type": "LINEAR",
            "fire": 0,
            "limiting_temperature": 0,
            "time[1]": 0,
            "time[2]": 0,
            "time[3]": 0,
            "time[4]": 0,
            "dummy": "ACTIVE",
            "off_z": 0,
            "off_auto_internal": "NO",
            "reinforcement2d[1]": "REBAR_2D.1",
            "reinforcement2d[2]": 0.03,
            "reinforcement2d[3]": 0.03,
            "reinforcement2d[4]": 0,
            "mesh_mode": "MIXED",
        }
    elif keyword == "el_2":
        defaults = {
            "colour": "NO_RGB",
            "type": "BEAM",
        }
    else:
        defaults = {}
    
    return defaults

def _get_format(keyword):
    """Return the format string for a given keyword"""

    if keyword == "header": 
        format = ["header"]
    elif keyword == "unit_data_1": 
        format = ["keyword", "option", "name", "factor"]
    elif keyword == "grid_plane_3":
        format = [
            "keyword", "num", "name", "type", "axis", "elev", "below", 
            "above"
        ]
    elif keyword == "mat_anal":
        format = [
            "keyword", "num", "MAT_ELAS_ISO", "name", "colour", "6", "E", 
            "nu", "rho", "alpha", "G", "damp"
        ]
    elif keyword == "prop_sec_2":
        format = [
            "keyword", "num", "name", "colour", "type", "desc", "anal"
        ]
    elif keyword == "prop_2d_8":
        format = [
            "keyword", "num", "name", "colour", "type", "axis", "mat", 
            "mat_type", "grade", "design", "profile", "ref_pt", "ref_z", 
            "mass", "flex", "shear", "inplane", "weight"
        ]
    elif keyword == "node_3":
        format = [
            "keyword", "num", "name", "colour", "x", "y", "z"
        ]
    elif keyword == "memb_8.alt_1d.automatic":
        format = [
            "keyword", "num", "name", "colour", "type (1D)", "exposure", 
            "prop", "group", "topology", "node", "angle", "mesh_size", 
            "is_intersector", "analysis_type", "fire", "limiting_temperature", 
            "time[1]", "time[2]", "time[3]", "time[4]", "dummy", "rls_1", 
            "rls_2", "restraint_end_1", "restraint_end_2", "AUTOMATIC", 
            "load_height", "load_ref", "is_off", 
            "has_ltb_factor", 
            "has_moment_amplification_factor_strong_axis", 
            "has_moment_amplification_factor_weak_axis"
        ]
    elif keyword == "memb_8.alt_2d":
        format = [
            "keyword", "num", "name", "colour", "type (2D)", "exposure", "prop", 
            "group", "topology", "node", "angle", "mesh_size", "is_intersector", 
            "analysis_type", "fire", "limiting_temperature", 
            "time[1]", "time[2]", "time[3]", "time[4]", "dummy", "off_z", 
            "off_auto_internal", "reinforcement2d[1]", "reinforcement2d[2]", 
            "reinforcement2d[3]", "reinforcement2d[4]", "mesh_mode"
        ]
    elif keyword == "el_2":
        format = [
            "keyword", "num", "name", "colour", "type", "prop", "group", 
            "topo1", "topo2"
        ]
    elif keyword == "user_mod_node_title_1":
        format = [
            "keyword", "module", "name", "dimension"
        ]
    elif keyword == "user_mod_node":
        format = [
            "keyword", "module", "node", "value"
        ]
    else:
        format = []

    return format

def _generate_func_match(options, cutoff=0.2):
    """Generate a function to get the closest match"""

    def func(word):
        word = word.lower()
        options_lower = [s.lower() for s in options]
        match = difflib.get_close_matches(word, options_lower, n=1, cutoff=cutoff)
        if match:
            match = match[0]
            index = options_lower.index(match)
            return options[index]
        else:
            return None

    return func

def _format_lines(df, keyword):
    """Format dataframes into list of strings"""

    format = _get_format(keyword)

    keyword = keyword.split(".")[0]
    keyword = keyword.upper()
    if keyword[-1].isnumeric():
        keyword = list(keyword)
        keyword[-2] = '.'
        keyword = "".join(keyword)

    lines = (
        df
        .with_columns(keyword = pl.lit(keyword))
        .select(format)
        .with_columns(end = pl.lit('\n'))
        .with_row_index()
        .unpivot(index="index")
        .group_by("index")
        .agg(pl.col("value"))
        .sort("index")
        .select(
            line = (
                pl.col("value")
                .list.join('\t')
            )
        )
        .to_series()
        .to_list()
    )

    return lines