"""A library that wraps the SAFE API.

Classes:
SAFE -- A class attach to an instance with useful methods

Functions:
"""

import sys

import clr
import polars as pl
import System  # type: ignore

from steng.connectors.csi import CsiApi
from steng.geometry.section_line import parse_rectangle
from steng.utils import check_folder

assemblies = [a.GetName().Name for a in System.AppDomain.CurrentDomain.GetAssemblies()]
if "SAFEv1" not in assemblies:
    clr.AddReference(R'C:\Program Files\Computers and Structures\SAFE 21\SAFEv1.dll')
import SAFEv1  # type: ignore

ABS_TOL = 0.001

class Safe(CsiApi):
    def __init__(self, is_attach=True, pid=None, folder=R".\safe", file="test.fdb", remote_name=None, safe_path=None):
        """Initialise object with attached instance
        """

        check_folder(folder)

        #create API helper object
        helper = SAFEv1.cHelper(SAFEv1.Helper())
        
        if is_attach:
            if not pid:
                #attach to a running instance of SAFE
                try:
                    #get the active SAFE object        
                    if remote_name is not None:
                        self.model = SAFEv1.cOAPI(helper.GetObjectHost(remote_name, "CSI.SAFE.API.ETABSObject"))
                    else:
                        self.model = SAFEv1.cOAPI(helper.GetObject("CSI.SAFE.API.ETABSObject"))
                except:
                    print("No running instance of the program found or failed to attach.")
                    self.model = None
                    sys.exit(-1)
            else:
                # attach to a running instance of SAFE given the process id
                try:
                    self.model = SAFEv1.cOAPI(helper.GetObjectProcess("CSI.SAFE.API.ETABSObject", pid))
                except:
                    print(
                        "No running instance of the program found with process id " 
                        + str(pid) + " or failed to attach."
                    )
                    self.model = None
                    sys.exit(-1)

        else:
            if safe_path is not None:
                try:
                    #'create an instance of the object from the specified path
                    if remote_name is not None:
                        self.model = SAFEv1.cOAPI(helper.CreateObjectHost(remote_name, safe_path))
                    else:
                        self.model = SAFEv1.cOAPI(helper.CreateObject(safe_path))
                except :
                    print("Cannot start a new instance of the program from " + safe_path)
                    self.model = None
                    sys.exit(-1)
            else:
                try: 
                    #create an instance of the object from the latest installed SAFE
                    #'SAFE uses API infrastructure from ETABS; as a consequence, the main connection object is called the ETABSObject       
                    if remote_name is not None:
                        self.model = SAFEv1.cOAPI(helper.CreateObjectProgIDHost(remote_name, "CSI.SAFE.API.ETABSObject"))
                    else:
                        self.model = SAFEv1.cOAPI(helper.CreateObjectProgID("CSI.SAFE.API.ETABSObject"))

                except:
                    print("Cannot start a new instance of the program.")
                    self.model = None
                    sys.exit(-1)

            #start SAFE application
            self.model.ApplicationStart()
        
        CsiApi.__init__(self)
    
    # ----- database tables -----

    def write_tables(self, keys, format="parquet", GroupName="All", folder=r".\safe", purpose="", model="safe"):
        return super().write_tables(keys, format, GroupName, folder, purpose, model)

    # ----- sap model -----

    def set_units_kn_m_c(self, units=SAFEv1.eUnits(6)) -> "Safe":
        """Set the units to kN, m, and C
        """
        
        self = CsiApi.set_units_kn_m_c(self, units)

        return self
    
    # ----- session -----

    def update_story(self, DatumName, StryNmAbove, StryNmBelow, ModelDatum, StryHtAbove, StryHtBelow):
        """Update the story names, datum and heights
        """

        self = self._get_table_for_editing("Grid Definitions - General")

        self.data_interactive["data"] = (
            self.data_interactive["data"]
            .with_columns(
                datumname = pl.lit(str(DatumName)),
                strynmabove = pl.lit(str(StryNmAbove)),
                strynmbelow = pl.lit(str(StryNmBelow)),
                modeldatum = pl.lit(str(ModelDatum)),
                stryhtabove = pl.lit(str(StryHtAbove)),
                stryhtbelow = pl.lit(str(StryHtBelow)),
            )
        )

        return (
            self
            ._set_table_for_editing()
            ._apply_edited_tables()
        )
    
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
            SAFEv1.eSlabType.Slab,
            SAFEv1.eShellType.ShellThin,
            Grade,
            thickness
        )
    
    def add_prop_wall(self, thickness: float, grade: int):
        """Add a new prop slab if not already in model
        """

        _, _, _, Grade = _match_grade(grade)

        return self._set_prop_area_wall(
            f"W(m) {str(thickness)} {str(grade)}",
            SAFEv1.eWallPropType.Specified,
            SAFEv1.eShellType.ShellThin,
            Grade,
            thickness
        )

def _match_grade(grade: int):
    """Return input to add material

    TODO add support for rebar
    """

    Region = "Europe"

    if grade <= 90:
        MatType = SAFEv1.eMatType.Concrete
        Standard = "EN 1992-1-1 per EN 206-1"
    elif grade <= 450:
        MatType = SAFEv1.eMatType.Steel
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