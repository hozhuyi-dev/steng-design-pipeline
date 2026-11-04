"""Etabs: wrapper over the ETABS API, through pythonnet: materials, sections, piers and objects, and table exports
"""

import sys

import clr
import polars as pl
import System  # type: ignore

from steng.connectors.csi import CsiApi
from steng.geometry.section_line import parse_rectangle
from steng.utils import check_folder

assemblies = [a.GetName().Name for a in System.AppDomain.CurrentDomain.GetAssemblies()]
if "ETABSv1" not in assemblies:
    clr.AddReference(R'C:\Program Files\Computers and Structures\ETABS 21\ETABSv1.dll')
import ETABSv1  # type: ignore

COLUMN_PIER_NAME = "_columns"
COUPLING_BEAM_GROUP_NAME = "_coupling_beams"
FLOOR_GROUP_PREFIX = "_floors_"
PILE_GROUP_NAME = "_piles"
POINT_GROUP_PREFIX = "_points"
WALL_GROUP_PREFIX = "_walls_"

class Etabs(CsiApi):
    def __init__(self, is_attach=True, pid=None, folder=R".\etabs", file="test.edb", remote_name=None, etabs_path=None):
        """Initialise object with attached instance
        """

        check_folder(folder)

        #create API helper object
        helper = ETABSv1.cHelper(ETABSv1.Helper())


        if is_attach:
            if not pid:
                #attach to a running instance of ETABS
                try:
                    #get the active ETABS object        
                    if remote_name is not None:
                        self.model = ETABSv1.cOAPI(helper.GetObjectHost(remote_name, "CSI.ETABS.API.ETABSObject"))
                    else:
                        self.model = ETABSv1.cOAPI(helper.GetObject("CSI.ETABS.API.ETABSObject"))
                except:
                    print("No running instance of the program found or failed to attach.")
                    self.model = None
                    sys.exit(-1)
            else:
                # attach to a running instance of ETABS given the process id
                try:
                    self.model = ETABSv1.cOAPI(helper.GetObjectProcess("CSI.ETABS.API.ETABSObject", pid))
                except:
                    print(
                        "No running instance of the program found with process id " 
                        + str(pid) + " or failed to attach."
                    )
                    self.model = None
                    sys.exit(-1)

        else:
            if etabs_path is not None:
                try:
                    #'create an instance of the ETABS object from the specified path
                    if remote_name is not None:
                        self.model = ETABSv1.cOAPI(helper.CreateObjectHost(remote_name, etabs_path))
                    else:
                        self.model = ETABSv1.cOAPI(helper.CreateObject(etabs_path))
                except :
                    print("Cannot start a new instance of the program from " + etabs_path)
                    self.model = None
                    sys.exit(-1)
            else:
                try: 
                    #create an instance of the ETABS object from the latest installed ETABS
                    if remote_name is not None:
                        self.model = ETABSv1.cOAPI(helper.CreateObjectProgIDHost(remote_name, "CSI.ETABS.API.ETABSObject"))
                    else:
                        self.model = ETABSv1.cOAPI(helper.CreateObjectProgID("CSI.ETABS.API.ETABSObject"))

                except:
                    print("Cannot start a new instance of the program.")
                    self.model = None
                    sys.exit(-1)

            self.model.ApplicationStart()
        
        CsiApi.__init__(self)
    
    # ----- database tables -----

    def write_tables(self, keys, format="parquet", GroupName="All", folder=R".\etabs", purpose="", model="etabs"):
        return super().write_tables(keys, format, GroupName, folder, purpose, model)
    
    # ----- pier label -----

    def get_pier_label_name_list(self):
        """Retrieve the names of all defined Pier Labels 
        """

        NumberNames = 0
        MyName = []

        ret, NumberNames, MyName = (
            self.model.SapModel
            .PierLabel.GetNameList(NumberNames, MyName)
        )
        assert ret == 0

        return list(MyName)
    
    def set_pier(self, Name):
        """Adds a new Pier Label
        """

        ret = self.model.SapModel.PierLabel.SetPier(Name)
        assert ret == 0

        return self
    
    # ----- sap model -----

    def set_units_kn_m_c(self, units=ETABSv1.eUnits(6)) -> "Etabs":
        """Set the units to kN, m, and C
        """
        
        self = CsiApi.set_units_kn_m_c(self, units)

        return self
    
    # ----- session -----

    def add_material(self, grade):
        """Add a new material if not already in model
        """

        MatType, Region, Standard, Grade = _match_grade(grade)

        self, grades = self._get_table_column_list("material_properties_-_general", "grade")
        if Grade in grades:
            return self

        return self._add_material(MatType, Region, Standard, Grade)
    
    def add_prop_beam(self, section_line: str, grade: int):
        """Add a new prop frame if not already in model
        """

        _, _, _, Grade = _match_grade(grade)

        if section_line.startswith("R(m)"):
            T3, T2 = parse_rectangle(section_line)
            self = (
                self
                ._set_prop_frame_rectangle(f"BM{section_line} {str(grade)}", Grade, T3, T2)
                ._set_prop_frame_rebar_beam(f"BM{section_line} {str(grade)}")
            )

        return self
    
    def add_prop_brace(self, section_line: str, grade: int):
        """Add a new prop frame if not already in model
        """

        _, _, _, Grade = _match_grade(grade)

        if section_line.startswith("R(m)"):
            T3, T2 = parse_rectangle(section_line)
            self = (
                self
                ._set_prop_frame_rectangle(f"BR{section_line} {str(grade)}", Grade, T3, T2)
                ._set_prop_frame_rebar_beam(f"BR{section_line} {str(grade)}")
            )

        return self
    
    def add_prop_column(self, section_line: str, grade: int):
        """Add a new prop frame if not already in model
        """

        _, _, _, Grade = _match_grade(grade)

        if section_line.startswith("R(m)"):
            T3, T2 = parse_rectangle(section_line)
            self = (
                self
                ._set_prop_frame_rectangle(f"CL{section_line} {str(grade)}", Grade, T3, T2)
                # TODO: Set column section
            )

        return self
    
    def add_prop_slab(self, thickness: float, grade: int):
        """Add a new prop slab if not already in model
        """

        _, _, _, Grade = _match_grade(grade)

        return self._set_prop_area_slab(
            f"S(m) {str(thickness)} {str(grade)}",
            ETABSv1.eSlabType.Slab,
            ETABSv1.eShellType.ShellThin,
            Grade,
            thickness
        )
    
    def add_prop_wall(self, thickness: float, grade: int):
        """Add a new prop slab if not already in model
        """

        _, _, _, Grade = _match_grade(grade)

        return self._set_prop_area_wall(
            f"W(m) {str(thickness)} {str(grade)}",
            ETABSv1.eWallPropType.Specified,
            ETABSv1.eShellType.ShellThin,
            Grade,
            thickness
        )
    
    # ----- utility -----

    def assign_pier_columns(self):
        """Assign all columns as pier
        """

        if COLUMN_PIER_NAME not in self.get_pier_label_name_list():
            self = self.set_pier(COLUMN_PIER_NAME)
        
        self, uniquenames = self._get_table_column_list("column_object_connectivity", "uniquename")
        df_pier = (
            pl.DataFrame([pl.Series("uniquename", uniquenames, dtype=pl.String)])
            .with_columns(pierlabel = pl.lit(COLUMN_PIER_NAME))
        )

        self = self._get_table_for_editing("frame assignment pier label")
        is_change = (
            self.data_interactive["data"]
            .filter(
                pl.col("uniquename").is_in(uniquenames),
                pl.col("pierlabel") == COLUMN_PIER_NAME,
            )
            .shape[0]
        ) != len(uniquenames)
        if is_change:
            self.data_interactive["data"] = (
                self.data_interactive["data"]
                .filter(pl.col("uniquename").is_in(uniquenames).not_())
                .vstack(df_pier)
            )
            self = (
                self
                ._set_table_for_editing()
                ._apply_edited_tables()
            )

        return self
    
    def assign_group_piles(self):
        """Assign all piles as group
        """
        self = self._get_group_name_list()
        if PILE_GROUP_NAME in self.cache["last_groups"]:
            self = self._delete_group(PILE_GROUP_NAME)
        self = self._set_group(PILE_GROUP_NAME)

        self, names = self._get_table_column_list("joint_assignments_-_springs", "uniquename")
        for name in names:
            self = self._set_point_obj_group_assign(name, PILE_GROUP_NAME)

        return self
    
    def assign_group_floors_story(self, story):
        """Assign walls at a story as group
        """

        self = self._get_group_name_list()
        if (FLOOR_GROUP_PREFIX + story) in self.cache["last_groups"]:
            self = self._delete_group(FLOOR_GROUP_PREFIX + story)
        self = self._set_group(FLOOR_GROUP_PREFIX + story)

        self = self.get_tables("floor_object_connectivity")
        names = (
            self.data["floor_object_connectivity"]
            .select("uniquename", "story")
            .filter(pl.col("story") == story)["uniquename"]
            .to_list()
        )

        for name in names:
            self = self._set_area_obj_group_assign(name, FLOOR_GROUP_PREFIX + story)

        return self
    
    def assign_group_walls_story(self, story):
        """Assign walls at a story as group
        """

        self = self._get_group_name_list()
        if (WALL_GROUP_PREFIX + story) in self.cache["last_groups"]:
            self = self._delete_group(WALL_GROUP_PREFIX + story)
        self = self._set_group(WALL_GROUP_PREFIX + story)

        self = self.get_tables("wall_object_connectivity")
        names = (
            self.data["wall_object_connectivity"]
            .select("uniquename", "story")
            .filter(pl.col("story") == story)["uniquename"]
            .to_list()
        )

        for name in names:
            self = self._set_area_obj_group_assign(name, WALL_GROUP_PREFIX + story)

        return self
    
    def assign_group_coupling_beams(self):
        """Assign walls with property modifers on f11 as group of coupling beams
        """

        self = self._get_group_name_list()
        if (COUPLING_BEAM_GROUP_NAME) in self.cache["last_groups"]:
            self = self._delete_group(COUPLING_BEAM_GROUP_NAME)
        self = self._set_group(COUPLING_BEAM_GROUP_NAME)

        self = self.get_tables("area_assignments_-_stiffness_modifiers")
        names = (
            self.data["area_assignments_-_stiffness_modifiers"]
            .select("uniquename", "label")
            .filter(
                pl.col("label").str.contains("^W"),
            )["uniquename"]
            .to_list()
        )

        for name in names:
            self = self._set_area_obj_group_assign(name, COUPLING_BEAM_GROUP_NAME)

        return self
    
def _match_grade(grade: int):
    """Return input to add material

    TODO add support for rebar
    """

    Region = "Europe"

    if grade <= 90:
        MatType = ETABSv1.eMatType.Concrete
        Standard = "EN 1992-1-1 per EN 206-1"
    elif grade <= 450:
        MatType = ETABSv1.eMatType.Steel
        Standard = "EN 1993-1-1 per EN 10025-2"
    else:
        MatType = None
        Standard = None

    if grade < 16: Grade = "C12/15"
    elif grade < 20: Grade = "C16/20"
    elif grade < 25: Grade = "C20/25"
    elif grade < 30: Grade = "C25/30"
    elif grade < 35: Grade = "C30/37"
    elif grade < 40: Grade = "C35/45"
    elif grade < 45: Grade = "C40/50"
    elif grade < 50: Grade = "C45/55"
    elif grade < 55: Grade = "C50/60"
    elif grade < 60: Grade = "C55/67"
    elif grade < 70: Grade = "C60/75"
    elif grade < 80: Grade = "C70/85"
    elif grade < 90: Grade = "C80/95"
    elif grade == 90: Grade = "C90/105"
    elif grade < 275: Grade = "S235"
    elif grade < 355: Grade = "S275"
    elif grade < 450: Grade = "S355"
    elif grade == 450: Grade = "S450"

    return MatType, Region, Standard, Grade