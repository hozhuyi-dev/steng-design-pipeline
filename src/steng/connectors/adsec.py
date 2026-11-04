"""Wrapper over Oasys AdSec: builds wall, column and pile sections and returns their
force-moment curves and failure surfaces

AdSec loads its .NET assemblies on import, so only steng.model.adsec imports this module, when used.
"""

import math
import os

import oasys.adsec  # noqa: F401  loads the .NET assemblies imported below
import polars as pl
import shapely as sl
from Oasys.AdSec import IAdSec, ISection  # type: ignore
from Oasys.AdSec.DesignCode import EN1992  # type: ignore
from Oasys.AdSec.IO.Graphics.Section import SectionImageBuilder  # type: ignore
from Oasys.AdSec.Reinforcement import IBarBundle, ICover  # type: ignore
from Oasys.AdSec.Reinforcement.Groups import ILinkGroup, IPerimeterGroup  # type: ignore
from Oasys.AdSec.Reinforcement.Layers import ILayerByBarCount, ILayerByBarPitch  # type: ignore
from Oasys.AdSec.StandardMaterials import Concrete, Reinforcement  # type: ignore
from Oasys.Profiles import ICircleProfile, IPerimeterProfile, IPoint, IPolygon, IRectangleProfile  # type: ignore
from OasysUnits import Angle, Length  # type: ignore
from OasysUnits.Units import AngleUnit, LengthUnit  # type: ignore
from tqdm import tqdm

from steng.geometry.section_line import generate_key, generate_keys, get_columns

DESIGN_CODE = EN1992.Part1_1.Edition_2004.NationalAnnex.SG.Edition_2008
DESIGN_CODE_CONCRETE = Concrete.EN1992.Part1_1.Edition_2004.NationalAnnex.SG.Edition_2008
DESIGN_CODE_REBAR = Reinforcement.Steel.EN1992.Part1_1.Edition_2004.NationalAnnex.SG.Edition_2008

FOLDER_CURVE = R".\adsec-curve"
FOLDER_SURFACE = R".\adsec-surface"

class AdSec:
    def __init__(self):
        """Initialise the ADSEC instance"""

        self.section = {}
        self.solution = {}
        self.curve = {}
        self.surface = {}

    def _get_standard_concrete(self, cylinder_strength):
        """Return the standard concrete object"""

        if cylinder_strength < 12:
            return None
        elif cylinder_strength < 16:
            return DESIGN_CODE_CONCRETE.C12_15
        elif cylinder_strength < 20:
            return DESIGN_CODE_CONCRETE.C16_20
        elif cylinder_strength < 25:
            return DESIGN_CODE_CONCRETE.C20_25
        elif cylinder_strength < 30:
            return DESIGN_CODE_CONCRETE.C25_30
        elif cylinder_strength < 32:
            return DESIGN_CODE_CONCRETE.C30_37
        elif cylinder_strength < 35:
            return DESIGN_CODE_CONCRETE.C32_40
        elif cylinder_strength < 40:
            return DESIGN_CODE_CONCRETE.C35_45
        elif cylinder_strength < 45:
            return DESIGN_CODE_CONCRETE.C40_50
        elif cylinder_strength < 50:
            return DESIGN_CODE_CONCRETE.C45_55
        elif cylinder_strength < 55:
            return DESIGN_CODE_CONCRETE.C50_60
        elif cylinder_strength < 60:
            return DESIGN_CODE_CONCRETE.C55_67
        elif cylinder_strength < 70:
            return DESIGN_CODE_CONCRETE.C60_75
        elif cylinder_strength < 80:
            return DESIGN_CODE_CONCRETE.C70_85
        elif cylinder_strength < 90:
            return DESIGN_CODE_CONCRETE.C80_95
        else:
            return DESIGN_CODE_CONCRETE.C90_105
    
    def _get_standard_rebar(self, rebar_type):
        """Return a standard rebar material"""

        if rebar_type.lower() == "a":
            return DESIGN_CODE_REBAR.S500A
        elif rebar_type.lower() == "b":
            return DESIGN_CODE_REBAR.S500B
        elif rebar_type.lower() == "c":
            return DESIGN_CODE_REBAR.S500C
        else:
            return None
    
    def _get_section(self, section_type, *args):
        """Return the section generated"""

        key = generate_key(section_type, *args)

        if key not in list(self.section.keys()):
            return None
        
        return self.section[key]
    
    def _get_solution(self, section_type, *args):
        """Return the section generated"""

        key = generate_key(section_type, *args)

        if key not in list(self.solution.keys()):
            return None
        
        return self.solution[key]
    
    def _generate_section_rcb_pile_1(self, grade, rebar_type, dia, cover, dia_horz, n_vert, dia_vert):
        """Generate an adsec section of reinforced concrete piles"""

        key = generate_key("rcb_pile_1", grade, rebar_type, dia, cover, dia_horz, n_vert, dia_vert)

        mat_concrete = self._get_standard_concrete(grade)
        mat_rebar = self._get_standard_rebar(rebar_type)

        profile = ICircleProfile.Create(Length(float(dia), LengthUnit.Millimeter))
        section = ISection.Create(profile, mat_concrete)

        section.Cover = ICover.Create(Length(float(cover), LengthUnit.Millimeter))

        link_bar = IBarBundle.Create(mat_rebar, Length(float(dia_horz), LengthUnit.Millimeter), 1)
        link_group = ILinkGroup.Create(link_bar)
        section.ReinforcementGroups.Add(link_group)

        vert_bar = IBarBundle.Create(mat_rebar, Length(float(dia_vert), LengthUnit.Millimeter), 1)
        vert_layer = ILayerByBarCount.Create(n_vert, vert_bar)
        vert_group = IPerimeterGroup.Create()
        vert_group.Layers.Add(vert_layer)
        section.ReinforcementGroups.Add(vert_group)

        self.section[key] = section

        return self

    def _generate_section_rc_wall(self, section_type, cylinder_strength, rebar_type, polyline, axis_angle, thickness, cover, dia_horz, s_vert, *dia_vert):
        """Generate an adsec section of a reinforced concrete wall given its polyline, with a layer of
        vertical bars at s_vert for each non-zero diameter in dia_vert, the first the outermost"""

        key = generate_key(section_type, cylinder_strength, rebar_type, polyline, axis_angle, thickness, cover, dia_horz, s_vert, *dia_vert)

        mat_concrete = self._get_standard_concrete(cylinder_strength)
        mat_rebar = self._get_standard_rebar(rebar_type)

        polyline = _rotate_by_axis_angle(polyline, axis_angle)
        vertices = _convert_polyline_to_vertices(polyline, thickness / 1000)
        polygon = IPolygon.Create()
        for vertex in vertices:
            polygon.Points.Add(IPoint.Create(
                Length(float(vertex[0]), LengthUnit.Meter),
                Length(float(vertex[1]), LengthUnit.Meter)
            ))
        profile = IPerimeterProfile.Create()
        profile.SolidPolygon = polygon
        section = ISection.Create(profile, mat_concrete)

        section.Cover = ICover.Create(Length(float(cover), LengthUnit.Millimeter))

        link_bar = IBarBundle.Create(mat_rebar, Length(float(dia_horz), LengthUnit.Millimeter), 1)
        link_group = ILinkGroup.Create(link_bar)
        section.ReinforcementGroups.Add(link_group)

        vert_group = IPerimeterGroup.Create()
        for k, dia in enumerate(dia_vert):
            if k == 0 or dia != 0:
                vert_bar = IBarBundle.Create(mat_rebar, Length(float(dia), LengthUnit.Millimeter), 1)
                vert_layer = ILayerByBarPitch.Create(vert_bar, Length(float(s_vert), LengthUnit.Millimeter))
                vert_group.Layers.Add(vert_layer)
        section.ReinforcementGroups.Add(vert_group)

        self.section[key] = section

        return self
    
    def _generate_section_rc_column(self, section_type, cylinder_strength, rebar_type, width, depth, cover, dia_horz, n_vert, *dia_vert):
        """Generate an adsec section of a rectangular reinforced concrete column, with a layer of n_vert
        bars around the perimeter for each non-zero diameter in dia_vert, the first the outermost"""

        key = generate_key(section_type, cylinder_strength, rebar_type, width, depth, cover, dia_horz, n_vert, *dia_vert)

        mat_concrete = self._get_standard_concrete(cylinder_strength)
        mat_rebar = self._get_standard_rebar(rebar_type)
        
        profile = IRectangleProfile.Create(Length(float(width), LengthUnit.Millimeter), Length(float(depth), LengthUnit.Millimeter))
        section = ISection.Create(profile, mat_concrete)

        section.Cover = ICover.Create(Length(float(cover), LengthUnit.Millimeter))

        link_bar = IBarBundle.Create(mat_rebar, Length(float(dia_horz), LengthUnit.Millimeter), 1)
        link_group = ILinkGroup.Create(link_bar)
        section.ReinforcementGroups.Add(link_group)

        vert_group = IPerimeterGroup.Create()
        for k, dia in enumerate(dia_vert):
            if k == 0 or dia != 0:
                vert_bar = IBarBundle.Create(mat_rebar, Length(float(dia), LengthUnit.Millimeter), 1)
                vert_layer = ILayerByBarCount.Create(n_vert, vert_bar)
                vert_group.Layers.Add(vert_layer)
        section.ReinforcementGroups.Add(vert_group)

        self.section[key] = section

        return self
    
    def _analyse_section(self, section_type, df):
        """Analyse section"""

        columns = get_columns(section_type)
        if not columns:
            return self

        keys = generate_keys(section_type, df.select(columns))

        adsec = IAdSec.Create(DESIGN_CODE)

        for i, key in zip(range(len(keys)), keys):
            if key not in list(self.section.keys()):
                if section_type == "rcb_pile_1":
                    self = (self._generate_section_rcb_pile_1(*list(df.select(columns).row(i))))
                elif section_type.startswith("rc_wall_u_"):
                    self = (self._generate_section_rc_wall(section_type, *list(df.select(columns).row(i))))
                elif section_type.startswith("rc_column_u_"):
                    self = (self._generate_section_rc_column(section_type, *list(df.select(columns).row(i))))
                else:
                    continue
            
            if key not in list(self.solution.keys()):
                self.solution[key] = adsec.Analyse(self.section[key])

        return self
    
    def get_force_moment_curve(self, angle, section_type, df):
        """Return the force moment interaction curve given the angle"""

        columns = get_columns(section_type)
        if not columns:
            return None
        
        schema = {
            "key": pl.String,
            "force": pl.Float64,
            "moment": pl.Float64,
        }
        df_curve = pl.DataFrame(schema=schema)

        keys = generate_keys(section_type, df.select(columns))
        existing_keys = []

        pbar = tqdm(total=len(keys), desc="Getting adsec failure curve")
        for i, key in zip(range(len(keys)), keys):
            if key in existing_keys:
                pbar.update(1)
                continue
            else:
                existing_keys.append(key)

            df_current = _read_curve(section_type, key)

            if df_current is not None:
                self.curve[key] = df_current

            else:
                self = self._analyse_section(section_type, df.slice(i, 1))
                
                curve_list = self.solution[key].Strength.GetForceMomentInteractionCurve(Angle(float(angle), AngleUnit.Degree))

                # Determine the number of points
                n_points = curve_list.get_Item(0).Points.Count

                # Initialise an empty numpy array
                force = []
                moment = []

                # TODO moment calculation for angle other than 0
                # Get each point and input to array
                for j in range(n_points):
                    force.append(curve_list.get_Item(0).Points.get_Item(j).X.Kilonewtons)
                    moment.append(curve_list.get_Item(0).Points.get_Item(j).YY.KilonewtonMeters)

                df_current = (
                    pl.DataFrame({
                        "key": key,
                        "force": force,
                        "moment": moment,
                    })
                    .cast(schema)
                )

                self.curve[key] = df_current
                _write_curve(section_type, key, df_current)

            df_curve = df_curve.vstack(df_current)

            pbar.update(1)
        pbar.close()

        print("Number of curve loaded", len(existing_keys))
        
        return df_curve
    
    def get_failure_surface(self, section_type, df):
        """Return the failure surface"""

        columns = get_columns(section_type)
        if not columns:
            return None

        schema = {
            "key": pl.String,
            "x1": pl.Float64,
            "yy1": pl.Float64,
            "zz1": pl.Float64,
            "x2": pl.Float64,
            "yy2": pl.Float64,
            "zz2": pl.Float64,
            "x3": pl.Float64,
            "yy3": pl.Float64,
            "zz3": pl.Float64,
        }
        df_surface = pl.DataFrame(schema=schema)

        keys = generate_keys(section_type, df.select(columns))
        existing_keys = []

        pbar = tqdm(total=len(keys), desc="Getting adsec failure surface")
        for i, key in zip(range(len(keys)), keys):
            if key in existing_keys:
                pbar.update(1)
                continue
            else:
                existing_keys.append(key)

            df_face = _read_surface(section_type, key)

            if df_face is not None:
                self.surface[key] = df_face

            else:
                self = self._analyse_section(section_type, df.slice(i, 1))

                load_surface_list = self.solution[key].Strength.GetFailureSurface()

                # Determine the number of output
                n_faces = load_surface_list.Faces.Count
                n_vertices = load_surface_list.Vertices.Count

                # Initialise an empty array
                face_vertex1 = []
                face_vertex2 = []
                face_vertex3 = []
                vertex_x = []
                vertex_yy = []
                vertex_zz = []

                # Get each point and input to array
                for j in range(n_faces):
                    face_vertex1.append(load_surface_list.Faces.get_Item(j).Vertex1)
                    face_vertex2.append(load_surface_list.Faces.get_Item(j).Vertex2)
                    face_vertex3.append(load_surface_list.Faces.get_Item(j).Vertex3)
                
                for j in range(n_vertices):
                    vertex_x.append(load_surface_list.Vertices.get_Item(j).X.Kilonewtons)
                    vertex_yy.append(load_surface_list.Vertices.get_Item(j).YY.KilonewtonMeters)
                    vertex_zz.append(load_surface_list.Vertices.get_Item(j).ZZ.KilonewtonMeters)

                df_vertex = (
                    pl.DataFrame({
                        "x": vertex_x,
                        "yy": vertex_yy,
                        "zz": vertex_zz,
                    })
                    .with_row_index()
                )
                
                df_face = (
                    pl.DataFrame({
                        "vertex1": face_vertex1,
                        "vertex2": face_vertex2,
                        "vertex3": face_vertex3,
                    })
                    .with_columns(
                        key = pl.lit(key),
                    )
                    .cast({"vertex1": pl.UInt32, "vertex2": pl.UInt32, "vertex3": pl.UInt32})
                    .join(df_vertex.rename({"index": "vertex1"}), on="vertex1", how="left")
                    .rename({"x": "x1", "yy": "yy1", "zz": "zz1"})
                    .join(df_vertex.rename({"index": "vertex2"}), on="vertex2", how="left")
                    .rename({"x": "x2", "yy": "yy2", "zz": "zz2"})
                    .join(df_vertex.rename({"index": "vertex3"}), on="vertex3", how="left")
                    .rename({"x": "x3", "yy": "yy3", "zz": "zz3"})
                    .drop("vertex1", "vertex2", "vertex3")
                    .cast(schema)
                )

                self.surface[key] = df_face
                _write_surface(section_type, key, df_face)

            df_surface = df_surface.vstack(df_face)

            pbar.update(1)
        pbar.close()
        
        df_surface = (
            df_surface
            .with_columns(
                x = pl.mean_horizontal("x1", "x2", "x3"),
                yy = pl.mean_horizontal("yy1", "yy2", "yy3"),
                zz = pl.mean_horizontal("zz1", "zz2", "zz3"),
            )
            .with_columns(
                magnitude_face = (pl.col("x").pow(2) + pl.col("yy").pow(2) + pl.col("zz").pow(2)).sqrt(),
            )
        )

        print("Number of surface loaded", len(existing_keys))

        return df_surface

    def get_svg_section(self, section_type, df):

        columns = get_columns(section_type)
        if not columns:
            return None

        keys = generate_keys(section_type, df.select(columns))

        adsec = IAdSec.Create(DESIGN_CODE)

        svg_strings = []
        for i, key in zip(range(len(keys)), keys):
            if key in list(self.section.keys()):
                flattened_sections = adsec.Flatten(self.section[key])
                svg_strings.append(SectionImageBuilder(flattened_sections).Svg())

        return svg_strings

def _rotate_by_axis_angle(polyline, axis_angle):
    """Rotate polyline at origin by axis_angle"""

    angle_radians = - math.radians(axis_angle)
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)

    rotated = []
    for x, y in polyline:
        x_new = x * cos_theta - y * sin_theta
        y_new = x * sin_theta + y * cos_theta
        rotated.append((x_new, y_new))

    return rotated

def _convert_polyline_to_vertices(polyline, thickness):
    """Generate perimeter of polyline with a given thickness"""

    linestring = sl.geometry.LineString(polyline)
    linestring = linestring.buffer(thickness/2, cap_style="flat", join_style="mitre")
    polygon = sl.geometry.Polygon(linestring)
    x, y = polygon.exterior.xy

    return list(zip(list(x), list(y)))

def _read_curve(section_type, key):
    """Read a curve parquet file if the file exists"""
    
    _check_folder(FOLDER_CURVE + os.sep + section_type)
    files = _list_files_os_listdir(FOLDER_CURVE + os.sep + section_type)

    for file in files:
        if file == (key + ".parquet"):
            file = FOLDER_CURVE + os.sep + section_type + os.sep + file
            return pl.read_parquet(file, glob=False)
    
    return None

def _write_curve(section_type, key, df):
    """Write a curve to a parquet file"""

    _check_folder(FOLDER_CURVE + os.sep + section_type)
    file = FOLDER_CURVE + os.sep + section_type + os.sep + key + ".parquet"
    df.write_parquet(file)

def _read_surface(section_type, key):
    """Read a surface parquet file if the file exists"""
    
    _check_folder(FOLDER_SURFACE + os.sep + section_type)
    files = _list_files_os_listdir(FOLDER_SURFACE + os.sep + section_type)

    for file in files:
        if file == (key + ".parquet"):
            file = FOLDER_SURFACE + os.sep + section_type + os.sep + file
            return pl.read_parquet(file, glob=False)
    
    return None

def _write_surface(section_type, key, df):
    """Write a surface to a parquet file"""

    _check_folder(FOLDER_SURFACE + os.sep + section_type)
    file = FOLDER_SURFACE + os.sep + section_type + os.sep + key + ".parquet"
    df.write_parquet(file)

def _check_folder(folder):
    """Create a folder if it does not exist"""

    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass

def _list_files_os_listdir(folder):
    """Lists all files and directories in the given path.
    """
    return os.listdir(folder)

# def rebarposition_couplingBeam(b, h, cover, d_link, n_rebar: list, d_rebar, spacing):
#     '''
#     The function is to generate the starting and ending positions of rebar line group
#     '''

#     # Initialise variables
#     y_start = []
#     z_start = []
#     y_end = []
#     z_end = []
#     z_top = h/2 - cover - d_link
#     z_bottom = -h/2 + cover + d_link

#     for i in range(len(n_rebar)):
#         if i == 0:
#             z_top -= d_rebar[i]/2
#             z_bottom += d_rebar[i]/2
#         else:
#             z_top = z_top - d_rebar[i-1]/2 - spacing - d_rebar[i]/2
#             z_bottom = z_bottom + d_rebar[i-1]/2 + spacing + d_rebar[i]/2

#         # Top rebars
#         y_start.append(-b/2 + cover + d_link + d_rebar[i]/2)
#         z_start.append(z_top)
#         y_end.append(b/2 - cover - d_link - d_rebar[i]/2)
#         z_end.append(z_top)

#         # Bottom rebars
#         y_start.append(-b/2 + cover + d_link + d_rebar[i]/2)
#         z_start.append(z_bottom)
#         y_end.append(b/2 - cover - d_link - d_rebar[i]/2)
#         z_end.append(z_bottom)

#     return y_start, z_start, y_end, z_end

# def defineSection_couplingBeam(b = 300, h = 600, cover = 50, d_link = 20, n_rebar = [5,], d_rebar = [25,], spacing = 25, 
#                                 t_plate = None, h_plate = None, offset_plate = None, 
#                                 mat_concrete = Concrete.EN1992.Part1_1.Edition_2004.NationalAnnex.GB.Edition_2014.C40_50, 
#                                 mat_rebar = Reinforcement.Steel.EN1992.Part1_1.Edition_2004.NationalAnnex.GB.Edition_2014.S500B, 
#                                 mat_steel = Steel.EN1993.Edition_2005.S355):
#     '''
#     The function is to define a coupling beam section
#     - Same top and bottom rebars
#     - Embedded steel plate with offset

#     Input:
#         b - Width of beam
#         h - Depth of beam
#         cover - Cover to links
#         d_link - Diameter of links
#         n-rebar - List of number of rebars in each layer
#         d_rebar - List of diameter of rebars in each layer
#         t_plate - Thickness of plate
#         h_plate - Depth of plate
#         offset_plate - Offset of plate (coordinates)
#         materials
#     '''

#     # Check that n_rebar and d_rebar are of the same length
#     if len(n_rebar) != len(d_rebar):
#         print('Error: n_rebar and d_rebar should be lists of the same length.')
#         return 0

#     # Determine rebar position
#     y_start, z_start, y_end, z_end = rebarposition_couplingBeam(b, h, cover, d_link, n_rebar, d_rebar, spacing)

#     # Create rectangular concrete section
#     b = Length(float(b), LengthUnit.Millimeter)
#     h = Length(float(h), LengthUnit.Millimeter)
#     profile = IRectangleProfile.Create(h, b)
#     section = ISection.Create(profile, mat_concrete)

#     # Define reinforcements
#     for i in range(len(n_rebar)):
#         # Top rebar
#         d_rebar[i] = Length(float(d_rebar[i]), LengthUnit.Millimeter)
#         barBundle = IBarBundle.Create(mat_rebar, d_rebar[i], 1)
#         layer = ILayerByBarCount.Create(n_rebar[i], barBundle)
#         y1 = Length(float(y_start[(2*i)]), LengthUnit.Millimeter)
#         z1 = Length(float(z_start[(2*i)]), LengthUnit.Millimeter)
#         y2 = Length(float(y_end[(2*i)]), LengthUnit.Millimeter)
#         z2 = Length(float(z_end[(2*i)]), LengthUnit.Millimeter)
#         group = ILineGroup.Create(IPoint.Create(y1, z1), IPoint.Create(y2, z2), layer)
#         section.ReinforcementGroups.Add(group)

#         # Bottom rebar
#         y1 = Length(float(y_start[(2*i)+1]), LengthUnit.Millimeter)
#         z1 = Length(float(z_start[(2*i)+1]), LengthUnit.Millimeter)
#         y2 = Length(float(y_end[(2*i)+1]), LengthUnit.Millimeter)
#         z2 = Length(float(z_end[(2*i)+1]), LengthUnit.Millimeter)
#         group = ILineGroup.Create(IPoint.Create(y1, z1), IPoint.Create(y2, z2), layer)
#         section.ReinforcementGroups.Add(group)

#     # Define steel plate with offset
#     if not ((t_plate is None) or (h_plate is None) or (offset_plate is None)):
#         # Create steel plate section
#         t_plate = Length(float(t_plate), LengthUnit.Millimeter)
#         h_plate = Length(float(h_plate), LengthUnit.Millimeter)
#         profile = IRectangleProfile.Create(h_plate, t_plate)
#         section_steelPlate = ISection.Create(profile, mat_steel)
        
#         # Sub compoent
#         for i in range(len(offset_plate)):
#             offset_plate[i] = Length(float(offset_plate[i]), LengthUnit.Millimeter)
#         subComponent = ISubComponent.Create(section_steelPlate, IPoint.Create(offset_plate[0], offset_plate[1]))
#         section.SubComponents.Add(subComponent)

#     return section

# def utilisation_strength(solution, Fx, Myy, Mzz):
#     '''
#     Obtain the utilisation ratio for a given Fx, Myy, Mzz
#     Note that the notation is consistent with Oasys GSA. 
#     Fx, Myy, Mzz must be iterable. 
#     '''

#     # Initialise np array
#     arr_utilisation = np.zeros(len(Fx))

#     for i in range(len(Fx)):
#         # Prepare the data
#         axialForce = Force(float(Fx[i]), ForceUnit.Kilonewton)
#         majorAxisBending = Moment(float(Myy[i]), MomentUnit.KilonewtonMeter)
#         minorAxisBending = Moment(float(Mzz[i]), MomentUnit.KilonewtonMeter)
#         load = ILoad.Create(axialForce, majorAxisBending, minorAxisBending)
#         strengthResult = solution.Strength.Check(load)

#         # Extract the utilisation
#         arr_utilisation[i] = strengthResult.LoadUtilisation.Percent

#     return arr_utilisation

# def curve_convertToDataframe(curve):
#     '''
#     Convert IList<ILoadCurve> to pandas.dataframe
#     '''

#     # Determine the number of points
#     n_points = curve.get_Item(0).Points.Count

#     # Initialise an empty numpy array
#     arr_curve = np.zeros([n_points,3])

#     # Get each point and input to array
#     for i in range(n_points):
#         arr_curve[i,0] = curve.get_Item(0).Points.get_Item(i).X.Kilonewtons
#         arr_curve[i,1] = curve.get_Item(0).Points.get_Item(i).YY.KilonewtonMeters
#         arr_curve[i,2] = curve.get_Item(0).Points.get_Item(i).ZZ.KilonewtonMeters

#     # Convert to dataframe
#     df_curve = pl.DataFrame(arr_curve, columns=['Fx','Myy','Mzz'])

#     return df_curve

# def curve_ForceMomentDataFrame(solution, angle):
#     '''
#     Get the force moment interaction curve in dataframe. 
#     Angle is in degree from -180 to 180
#     '''

#     curve = solution.Strength.GetForceMomentInteractionCurve(Angle(float(angle), AngleUnit.Degree))
#     df_curve = curve_convertToDataframe(curve)

#     return df_curve

# def curve_DataFrameToPolygon(df_curve, bool_Fx, bool_Myy, bool_Mzz):
#     '''
#     Convert df curve to list of coordinates. 
#     '''

#     # Number of points
#     n_points = len(df_curve['Fx'])

#     # Check inputs
#     if (bool_Fx) and (bool_Myy) and (bool_Mzz):
#         print('Error: Curve is only for 2D graph.')
#         return 0

#     # Initialise curve_polygon as empty list
#     curve_polygon = []

#     if (bool_Fx) and (bool_Myy):
#         # x is Myy and y is Fx
#         for i in range(n_points):
#             curve_polygon.append((df_curve.loc[i,'Myy'],df_curve.loc[i,'Fx']))
#         curve_polygon.append((df_curve.loc[0,'Myy'],df_curve.loc[0,'Fx']))
#     elif (bool_Fx) and (bool_Mzz):
#         # x is Mzz and y is Fx
#         for i in range(n_points):
#             curve_polygon.append((df_curve.loc[i,'Mzz'],df_curve.loc[i,'Fx']))
#         curve_polygon.append((df_curve.loc[0,'Mzz'],df_curve.loc[0,'Fx']))
#     elif (bool_Myy) and (bool_Mzz):
#         # x is Myy and y is Mzz
#         for i in range(n_points):
#             curve_polygon.append((df_curve.loc[i,'Myy'],df_curve.loc[i,'Mzz']))
#         curve_polygon.append((df_curve.loc[0,'Myy'],df_curve.loc[0,'Mzz']))

#     return curve_polygon

# def curve_FxToMyyrange(solution, Fx):
#     '''
#     Obtain the moment capacity (+ve and -ve) for a given axial load. 
#     Fx in kN unit. 
#     '''

#     # Get the force moment interaction curve
#     df_curve = curve_ForceMomentDataFrame(solution, 0)
    
#     if not utilisation_strength(solution, [Fx, ], [0, ], [0, ])[0]>=100:
#         # Get the maximum and minimum M
#         max_Myy = max(df_curve['Myy'])
#         min_Myy = min(df_curve['Myy'])
#         line = sl.geometry.LineString([(min_Myy-10,Fx), (max_Myy+10,Fx)])
#         polygon = sl.geometry.Polygon(curve_DataFrameToPolygon(df_curve, True, True, False))

#         # Get the intersection
#         intersection = polygon.exterior.intersection(line)
#         Myyrange = (intersection[0].coords[0][0], intersection[1].coords[0][0])
#     else:
#         Myyrange = (0, 0)

#     return Myyrange

# def curve_FxrangeToMyyrange(solution, Fxrange):
#     '''
#     Obtain the maximum moment capacity (+ve and -ve) for a range of axial load. 
#     '''

#     # Get the force moment interaction curve
#     df_curve = curve_ForceMomentDataFrame(solution, 0)

#     # Filter the list of Myy for Fx within the given range
#     upper_Fx = max(Fxrange)
#     lower_Fx = min(Fxrange)
#     key = (df_curve.loc[:, 'Fx']>=lower_Fx) & (df_curve.loc[:, 'Fx']<=upper_Fx) 
#     sr_Myy = df_curve.loc[key, 'Myy']
#     if sr_Myy.empty:
#         lower_Myyrange = curve_FxToMyyrange(solution, lower_Fx)
#         upper_Myyrange = curve_FxToMyyrange(solution, upper_Fx)
#         Myyrange = (min(min(lower_Myyrange), min(upper_Myyrange)), max(max(lower_Myyrange), max(upper_Myyrange)))
#     else:
#         Myyrange = (min(sr_Myy), max(sr_Myy))

#     return Myyrange