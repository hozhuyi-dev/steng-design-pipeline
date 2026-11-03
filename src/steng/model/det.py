""" A library that allows det parsing and formatting.

Interfaces:
DET -- A class with etabs $et data parsed as polars DataFrame and methods for
    post-processing.

Functions:
"""

import re

import polars as pl

DEFAULT_SPRING_UZ = 0.0125

class DetMixin:
    def __init__(self):
        """Initialise DET object.
        """
        pass

    def _read_det(self, path, groups=None):
        """
        """

        data = _parse_etabs_file(path, groups)
        data = _convert_tree_to_dataframe(data)

        return data
    
    def _expand_unit(self):
        """Expand units
        """
        
        df_unit = (
            self.det["controls"]["units"]
            .with_columns(
                pl.col("units")
                .list.to_struct(fields=["force", "length", "temperature"])
            )
            .unnest("units")
        )
        
        return (
            df_unit.select("force").item(),
            df_unit.select("length").item(),
            df_unit.select("temperature").item()
        )

    def _expand_story(self):
        """Expand stories with elevations from their heights.
        """

        return (
            self.det["stories_-_in_sequence_from_top"]["story"]
            .with_columns(
                elev = pl.col("elev").fill_null(strategy="backward") + pl.col("height").cum_sum(reverse=True).fill_null(0)
            )
            .with_columns(
                pl.col("elev") * self.unit_length,
                pl.col("height") * self.unit_length,
            )
        )
    
    def _expand_point(self):
        """Expand points to their IDs and coordinates.
        """

        df_story = self.story.with_row_index()

        df_point = (
            self.det["point_coordinates"]["point"]
            .with_columns(
                pl.col('point').list.concat(
                    pl.lit(0).repeat_by(
                        4 - pl.col('point').list.len()
                    )
                )
            )
            .with_columns(pl.col("point").list.to_struct(fields=["point", "x", "y", "z_offset"]))
            .unnest("point")
            .with_columns(
                point = pl.col("point").cast(pl.UInt32),
                x = pl.col("x").cast(pl.Float64),
                y = pl.col("y").cast(pl.Float64),
                z_offset = pl.col("z_offset").cast(pl.Float64).fill_null(0),
            )
        )
        
        df_line = (
            self.det["line_connectivities"]["line"]
            .unpivot(index="line", variable_name="line_type", value_name="points")
            .drop_nulls(subset="points")
            .with_columns(
                line_type = (
                    pl.col("line_type")
                    .str.strip_chars('_')
                )
            )
            .with_columns(
                points = (
                    pl.col("points")
                    .list.to_struct(fields=["point1", "point2", "offset1"])
                )
            )
            .unnest("points")
            .with_columns(
                point1 = pl.col("point1").cast(pl.UInt32),
                point2 = pl.col("point2").cast(pl.UInt32),
                offset1 = pl.col("offset1").cast(pl.UInt8),
                offset2 = pl.lit(0).cast(pl.UInt8)
            )
            .with_columns(
                point = pl.concat_list("point1", "point2"),
                offset = pl.concat_list("offset1", "offset2")
            )
            .drop(["point1", "point2", "offset1", "offset2"])
        )

        df_area = (
            self.det["area_connectivities"]["area"]
            .unpivot(index="area", variable_name="area_type", value_name="points")
            .drop_nulls(subset="points")
            .with_columns(
                area_type = (
                    pl.col("area_type")
                    .str.strip_chars('_')
                )
            )
            .with_columns(
                n_point = (
                    pl.col("points")
                    .list.first().cast(pl.UInt8)
                ),
                point = (
                    pl.col("points")
                    .list.slice(1, pl.col("points").list.first().cast(pl.UInt8))
                    .cast(pl.List(pl.UInt32))
                ),
                offset = (
                    pl.col("points")
                    .list.slice(pl.col("points").list.first().cast(pl.UInt8) + 1, pl.col("points").list.first().cast(pl.UInt8))
                    .cast(pl.List(pl.UInt8))
                )
            )
            .drop("points")
        )

        df_pointassign = (
            self.det["point_assigns"]["pointassign"]
            .with_columns(pl.col("pointassign").list.to_struct(fields=["point", "story"]))
            .unnest("pointassign")
            .with_columns(point = pl.col("point").cast(pl.UInt32))
        )

        if "restraint" in df_pointassign.columns:
            df_pointassign = (
                df_pointassign
                .with_columns(
                    restraint_ux = pl.when(pl.col("restraint").str.contains("UX")).then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_uy = pl.when(pl.col("restraint").str.contains("UY")).then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_uz = pl.when(pl.col("restraint").str.contains("UZ")).then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_rxx = pl.when(pl.col("restraint").str.contains("RX")).then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_ryy = pl.when(pl.col("restraint").str.contains("RY")).then(pl.lit(True)).otherwise(pl.lit(False)),
                    restraint_rzz = pl.when(pl.col("restraint").str.contains("RZ")).then(pl.lit(True)).otherwise(pl.lit(False)),
                )
            )
        else:
            df_pointassign = (
                df_pointassign
                .with_columns(
                    restraint_ux = pl.lit(False),
                    restraint_uy = pl.lit(False),
                    restraint_uz = pl.lit(False),
                    restraint_rxx = pl.lit(False),
                    restraint_ryy = pl.lit(False),
                    restraint_rzz = pl.lit(False),
                )
            )
        
        if "springprop" in df_pointassign.columns:
            df_springprop = (
                self.det["point_spring_properties"]["pointspring"]
                .rename({
                    "pointspring": "springprop",
                    "uz": "spring_z"
                })
                .with_columns(
                    fz = pl.col("spring_z") * self.unit_force / self.unit_length * DEFAULT_SPRING_UZ,
                    uz = pl.lit(DEFAULT_SPRING_UZ)
                )
            )

            df_pointassign = (
                df_pointassign
                .with_columns(
                    pl.format(
                        "STD C(m) {}", 
                        pl.col("springprop").str.extract(r"([0-9]+)", 1).cast(pl.Float64) / 1000
                    ).alias("section_line")
                )
                .join(df_springprop, on="springprop", how="left")
            )
        else:
            df_pointassign = df_pointassign.with_columns(
                section_line = pl.lit(None).cast(pl.String),
                fz = pl.lit(None).cast(pl.Float64),
                uz = pl.lit(None).cast(pl.Float64),
            )
        
        df_lineassign = (
            self.det["line_assigns"]["lineassign"]
            .select(pl.col("lineassign").list.to_struct(fields=["line", "story"]))
            .unnest("lineassign")
            .join(df_line, on="line", how="left")
            .explode(["point", "offset"])
            .join(df_story.select("story", "index"), on="story", how="left")
            .with_columns(pl.col("index") + pl.col("offset"))
            .drop("story")
            .join(df_story.select("story", "index"), on="index", how="left")
        )

        df_areaassign = (
            self.det["area_assigns"]["areaassign"]
            .select(pl.col("areaassign").list.to_struct(fields=["area", "story"]))
            .unnest("areaassign")
            .join(df_area, on="area", how="left")
            .explode(["point", "offset"])
            .join(df_story.select("story", "index"), on="story", how="left")
            .with_columns(pl.col("index") + pl.col("offset"))
            .drop("story")
            .join(df_story.select("story", "index"), on="index", how="left")
        )

        return (
            df_pointassign.select("point", "story")
            .vstack(df_lineassign.select("point", "story"))
            .vstack(df_areaassign.select("point", "story"))
            .unique()
            .join(df_point.select("point", "x", "y", "z_offset"), on="point", how="left")
            .join(df_story.select("story", "elev"), on="story", how="left")
            .with_columns(z = pl.col("elev") - pl.col("z_offset"))
            .drop("elev", "z_offset")
            .sort("z", "y", "x")
            .with_row_index(name="unique_point", offset=1)
            .join(df_pointassign, on=["point", "story"], how="left")
            .with_columns(
                pl.col("x") * self.unit_length,
                pl.col("y") * self.unit_length,
            )
        )
    
    def _expand_line(self):
        """Expand lines
        """

        df_story = self.story.with_row_index()

        # TODO grade other than fc and fy
        df_material = (
            self.det["material_properties"]["material"]
            .group_by("material")
            .agg(pl.all())
            .with_columns(
                pl.exclude("material").list.drop_nulls().list.first(),
            )
            .with_columns(
                grade = pl.when(pl.col("type") == "Concrete").then(pl.col("fc")).otherwise(pl.col("fy"))
            )
            .with_columns(
                pl.col("grade") * self.unit_force / (self.unit_length ** 2) / 1e3
            )
        )

        # TODO shape other than Concrete Rectangular
        df_framesection = (
            self.det["frame_sections"]["framesection"]
            .group_by("framesection")
            .agg(pl.all())
            .with_columns(
                pl.exclude("framesection").list.drop_nulls().list.first(),
            )
            .with_columns(
                pl.col("d") * self.unit_length,
                pl.col("b") * self.unit_length,
            )
            .with_columns(
                section_line = (
                    pl
                    .when(pl.col("shape") == "Concrete Rectangular").then(pl.format("R(m) {} {}", pl.col("d"), pl.col("b")))
                    .otherwise(pl.lit(None))
                )
            )
        )

        df_line = (
            self.det["line_connectivities"]["line"]
            .unpivot(index="line", variable_name="line_type", value_name="points")
            .drop_nulls(subset="points")
            .with_columns(
                line_type = (
                    pl.col("line_type")
                    .str.strip_chars('_')
                )
            )
            .with_columns(
                points = (
                    pl.col("points")
                    .list.to_struct(fields=["point1", "point2", "offset1"])
                )
            )
            .unnest("points")
            .with_columns(
                point1 = pl.col("point1").cast(pl.UInt32),
                point2 = pl.col("point2").cast(pl.UInt32),
                offset1 = pl.col("offset1").cast(pl.UInt8),
                offset2 = pl.lit(0).cast(pl.UInt8)
            )
            .with_columns(
                point = pl.concat_list("point1", "point2"),
                offset = pl.concat_list("offset1", "offset2")
            )
            .drop(["point1", "point2", "offset1", "offset2"])
        )

        df_lineassign = (
            self.det["line_assigns"]["lineassign"]
            .with_columns(
                lineassign = (
                    pl.col("lineassign")
                    .list.to_struct(fields=["line", "story"])
                )
            )
            .unnest("lineassign")
            .join(df_line, on="line", how="left")
            .explode("point", "offset")
            .join(df_story.select("story", "index"), on="story", how="left")
            .with_columns(pl.col("index") + pl.col("offset"))
            .join(df_story.select("story", "index").rename({"story": "story_point"}), on="index", how="left")
            .join(self._expand_point().select("point", "story", "x", "y", "z"), left_on=["point", "story_point"], right_on=["point", "story"], how="left")
            .group_by("line", "story")
            .agg(
                pl.all().exclude("point", "offset", "index", "story_point", "x", "y", "z").first(),
                pl.col("point", "offset", "index", "story_point", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .join(df_framesection, left_on="section", right_on="framesection", how="left")
            .join(df_material, on="material", how="left")
            .sort("story", "line")
            .with_row_index(name="unique_line", offset=1)
        )

        # TODO check angle if not working
        if "angle" not in df_lineassign.columns:
            df_lineassign = df_lineassign.with_columns(angle = pl.lit(0.0))

        return (
            df_lineassign.filter(pl.col("line_type") == "beam").rename({"line": "beam"}),
            df_lineassign.filter(pl.col("line_type") == "brace").rename({"line": "brace"}),
            df_lineassign.filter(pl.col("line_type") == "column").rename({"line": "column"}),
            df_lineassign.filter(pl.col("line_type") == "line").rename({"line": "line"}),
        )
    
    def _expand_area(self):
        """Expand areas
        """

        df_story = self.story.with_row_index()

        # TODO grade other than fc and fy
        df_material = (
            self.det["material_properties"]["material"]
            .group_by("material")
            .agg(pl.all())
            .with_columns(
                pl.exclude("material").list.drop_nulls().list.first(),
            )
            .with_columns(
                grade = pl.when(pl.col("type") == "Concrete").then(pl.col("fc")).otherwise(pl.col("fy"))
            )
            .with_columns(
                pl.col("grade") * self.unit_force / (self.unit_length ** 2) / 1e3
            )
        )

        # TODO what about area shellprop
        df_slab_shellprop = (
            self.det["slab_properties"]["shellprop"]
            .group_by("shellprop")
            .agg(pl.all())
            .with_columns(
                pl.exclude("shellprop")
                .list.drop_nulls()
                .list.first()
            )
            .with_columns(
                thickness = pl.col("slabthickness") * self.unit_length,
            )
        )

        df_wall_shellprop = (
            self.det["wall_properties"]["shellprop"]
            .group_by("shellprop")
            .agg(pl.all())
            .with_columns(
                pl.exclude("shellprop")
                .list.drop_nulls()
                .list.first()
            )
            .with_columns(
                thickness = pl.col("wallthickness") * self.unit_length,
            )
        )

        df_area = (
            self.det["area_connectivities"]["area"]
            .unpivot(index="area", variable_name="area_type", value_name="points")
            .drop_nulls(subset="points")
            .with_columns(
                area_type = (
                    pl.col("area_type")
                    .str.strip_chars('_')
                    .replace({"panel": "wall"})
                )
            )
            .with_columns(
                n_point = (
                    pl.col("points")
                    .list.first().cast(pl.UInt8)
                ),
                point = (
                    pl.col("points")
                    .list.slice(1, pl.col("points").list.first().cast(pl.UInt8))
                    .cast(pl.List(pl.UInt32))
                ),
                offset = (
                    pl.col("points")
                    .list.slice(pl.col("points").list.first().cast(pl.UInt8) + 1, pl.col("points").list.first().cast(pl.UInt8))
                    .cast(pl.List(pl.UInt8))
                )
            )
            .drop("points")
        )

        df_areaassign = (
            self.det["area_assigns"]["areaassign"]
            .with_columns(
                pl.col("areaassign")
                .list.to_struct(fields=["area", "story"])
            )
            .unnest("areaassign")
            .join(df_area, on="area", how="left")
            .explode("point", "offset")
            .join(df_story.select("story", "index"), on="story", how="left")
            .with_columns(pl.col("index") + pl.col("offset"))
            .join(df_story.select("story", "index").rename({"story": "story_point"}), on="index", how="left")
            .join(self._expand_point().select("point", "story", "x", "y", "z"), left_on=["point", "story_point"], right_on=["point", "story"], how="left")
            .group_by("area", "story")
            .agg(
                pl.all().exclude("point", "offset", "index", "story_point", "x", "y", "z").first(),
                pl.col("point", "offset", "index", "story_point", "x", "y", "z"),
            )
            .rename({
                "x": "xs",
                "y": "ys",
                "z": "zs",
            })
            .join(df_slab_shellprop, left_on="section", right_on="shellprop", how="left")
            .update(df_wall_shellprop, left_on="section", right_on="shellprop", how="left")
            .join(df_material, on="material", how="left")
            .sort("story", "area")
            .with_row_index(name="unique_area", offset=1)
        )

        # TODO check angle if not working
        if "angle" not in df_areaassign.columns:
            df_areaassign = df_areaassign.with_columns(angle = pl.lit(None).cast(pl.Float64))

        return (
            df_areaassign.filter(pl.col("area_type") == "area").rename({"area": "area"}),
            df_areaassign.filter(pl.col("area_type") == "floor").rename({"area": "floor"}),
            df_areaassign.filter(pl.col("area_type") == "wall").rename({"area": "wall"}),
        )

def _parse_etabs_file(path, groups):
    """Parse .$et files as a dictionary tree as a representation of the .$et file."""

    # Initialise variables
    data = {}
    group = ""
    is_in_groups = False

    # Main logic for parsing file
    with open(path, "r") as lines:
        for line in lines:
            if _is_file_line(line):
                data["file"] = {"file": _parse_file_line(line)}
            elif _is_group(line):
                group = _get_group(line)
                if (not groups) or (group in groups):
                    is_in_groups = True
                    data[group] = {}
                else:
                    is_in_groups = False
            elif is_in_groups and (not line.isspace()):
                parsed_data = _parse_content_line(line)
                data[group] = _append_content_line(data[group], parsed_data)
    return data

def _is_file_line(line):
    """Check if the line is a header line."""
    return line.startswith(r"$ File")

def _parse_file_line(line):
    """Parse the first .$et file line. """
    matcher = re.compile(r"File (.*) saved (.*)")
    matchlist = re.findall(matcher, line)
    data = {
        "save_path": matchlist[0][0],
        "save_datetime": matchlist[0][1]
    }
    return data

def _is_group(line):
    """Check if the line is a group line."""
    return line.startswith(r"$ ")

def _get_group(line):
    """Get the group."""
    return line[2:].lower().replace(" ", "_").replace("\n", "")

def _parse_content_line(line):
    """Parse the content line into dictionary."""
    parsed_data = {}
    existing_keys = []
    is_first = True
    is_string = False
    for word in line.split():
        if is_first:
            table = word.lower()
            key = table
            existing_keys.append(key)
            parsed_data[table] = {}
            parsed_data[table][key] = []
            is_first = False
        elif (not is_string) and word.isalnum() and word.isupper():
            key = word.lower()
            while key in existing_keys:
                key += "_"
            existing_keys.append(key)
            parsed_data[table][key] = []
        elif (word[0] == "\"") and (word[-1] == "\""):
            word = word[1:-1]
            parsed_data[table][key].append(word)
        elif (word[0] == "\""):
            words = word[1:]
            is_string = True
        elif is_string:
            words += " " + word
            if (words[-1] == "\""):
                words = words[:-1]
                parsed_data[table][key].append(words)
                is_string = False
        else:
            word = _convert_string_to_number(word)
            parsed_data[table][key].append(word)
    
    # Flatten if only single element
    keys = list(parsed_data[table].keys())
    for key in keys:
        if len(parsed_data[table][key]) == 0:
            parsed_data[table][key] = None
        elif len(parsed_data[table][key]) == 1:
            parsed_data[table][key] = parsed_data[table][key][0]
    return parsed_data

def _convert_string_to_number(word):
    """Convert a string to float if possible, else keep as string."""
    try:
        word = float(word)
        if word % 1 == 0:
            word = int(word)
    except ValueError:
        pass
    return word

def _append_content_line(dict, parsed_data):
    """Append the parsed data into the dictionary."""
    
    # First layer - tables
    table = list(parsed_data.keys())[0]

    if table not in list(dict.keys()):
        dict[table] = {}
        current_len = 0
    else:
        current_len = len(dict[table][next(iter(dict[table]))])

    # Second layer - keys
    keys_adding = list(parsed_data[table].keys())
    keys_existing = list(dict[table].keys())

    for key in keys_adding:
        if key in keys_existing:
            dict[table][key].append(parsed_data[table][key])
        else:
            dict[table][key] = [None] * current_len
            dict[table][key].append(parsed_data[table][key])

    for key in keys_existing:
        if key not in keys_adding:
            dict[table][key].append(None)
    return dict

def _convert_tree_to_dataframe(data):
    """Convert dictionary tree to polars DataFrame"""

    groups = list(data.keys())
    for group in groups:
        tables = list(data[group].keys())
        for table in tables:
            data[group][table] = pl.DataFrame(data[group][table], strict=False)
    return data

def _format_shape_gwa(dimension):
    """Format"""

    shape, d, b, tw, tf, t = dimension

    if shape is None:
            return "EXPLICIT"
    elif (d is None) and (b is None) and (tw is None) and (tf is None) and (t is None):
        sec_name = shape.replace('X','x')
        if sum(tt.isnumeric() for tt in sec_name[1:].split('x') if sec_name.startswith('W')) == 2:
            return ('CAT W ' if float(sec_name[1:].split('x')[0]) < 45 else 'CAT CA-W ') + sec_name
        elif sum(tt.isnumeric() for tt in sec_name[2:].split('x') if sec_name.startswith('UB')) == 3:
            return 'CAT UB ' + sec_name
        elif sum(tt.isnumeric() for tt in sec_name[3:].split('x') if sec_name.startswith('UKB')) == 3:
            return 'CAT BS-UB ' + sec_name.replace('UKB','')
        elif sum(tt.isnumeric() for tt in sec_name[2:].split('x') if sec_name.startswith('UC')) == 3:
            return 'CAT UC ' + sec_name
        elif sum(tt.isnumeric() for tt in sec_name[3:].split('x') if sec_name.startswith('UKC')) == 3:
            return 'CAT BS-UC ' + sec_name.replace('UKC','')
        else:
            return "EXPLICIT"
    elif shape in ['Steel I/Wide Flange', 'I/Wide Flange']:
        return 'STD I(m) {} {} {} {}'.format(d, b, tw, tf)
    elif shape in ['Steel Pipe', 'Pipe']:
        return 'STD CHS(m) {} {}'.format(d, t)
    elif shape in ['Steel Tube', 'Tube']:
        return 'STD RHS(m) {} {} {} {}'.format(d, b, tw, tf)
    elif shape in ['Steel Angle', 'Angle']:
        return 'STD A(m) {} {} {} {}'.format(d, b, tw, tf)
    elif shape in ['Steel Double Angle', 'Double Angle']:
        return 'STD D(m) {} {} {} {}'.format(d, str(0.5*float(b)), tw, tf)
    elif shape in ['Steel Plate']:
        return 'STD R(m) {} {}'.format(d, b)
    elif shape in ['Concrete Rectangular', 'Rectangle']:
        return 'STD R(m) {} {}'.format(d, b)
    elif shape in ['Concrete Circle']:
        return 'STD C(m) {}'.format(d)
    else:
        return "EXPLICIT"