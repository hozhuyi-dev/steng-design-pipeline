"""An interface for interacting with CSi software when the API calls are the same

Interfaces:
CsiApi -- A interface for CSi software

Functions:
"""

import datetime
import os

import numpy as np
import polars as pl
import System  # type: ignore

from steng.utils import check_folder, convert_to_list, match_options


class CsiApi:
    def __init__(self):
        """Initialise object with attached instance
        """

        self.path, self.name = self._get_filepath_filename()
        self.datetime = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
        self.data = {}
        self.data_interactive = {}
        self.cache = {}

        self.set_units_kn_m_c()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        """

        self.model = None
    
    # ----- area obj -----

    def _add_area_obj_by_coord(self, X, Y, Z, PropName="Default", UserName=None):
        """Add an area object by coordinates
        """

        assert len(Y) == len(X)
        assert len(Z) == len(X)

        NumberPoints = len(X)

        Name = ""

        ret, X, Y, Z, Name = self.model.SapModel.AreaObj.AddByCoord(
            NumberPoints,
            System.Array[System.Double](X), System.Array[System.Double](Y), System.Array[System.Double](Z), 
            Name, PropName=PropName, UserName=UserName
        )
        # print(ret)
        assert ret == 0

        self.cache["last_added_area"] = Name

        return self
    
    def _set_area_obj_group_assign(self, Name, GroupName, Remove=False):
        """Adds or removes area objects from a specified group. 
        """

        ret = self.model.SapModel.AreaObj.SetGroupAssign(Name, GroupName, Remove=Remove)
        assert ret == 0

        return self

    # ----- database tables -----

    def _get_available_tables_for_display(self):
        """Get a list of available table keys for display
        """

        NumberTables = 0
        TableKey     = []
        TableName    = []
        ImportType   = []

        ret, NumberTables, TableKey, TableName, ImportType = (
            self.model.SapModel
            .DatabaseTables
            .GetAvailableTables(NumberTables, TableKey, TableName, ImportType)
        )
        assert ret == 0

        return list(TableKey)
    
    def _get_table_column_list(self, TableKey, column_name, contain="", not_contain="!"):
        """Return self and a list of selected column from a table
        """

        self = self.get_tables(TableKey)
        column = (
            self.data[TableKey]
            .select(column_name)
            .filter(
                pl.col(column_name).str.contains(contain),
                ~pl.col(column_name).str.contains(not_contain),
            )
            .to_series().to_list()
        )

        return self, column

    def set_load_pattern_selected(self, contain="", not_contain="!"):
        """Set the load patterns selected for display
        """

        self, LoadPatternList = (
            self
            ._get_table_column_list("load_pattern_definitions", "name", contain=contain, not_contain=not_contain)
        )

        ret, LoadPatternList = (
            self.model
            .SapModel
            .DatabaseTables
            .SetLoadPatternsSelectedForDisplay(LoadPatternList)
        )
        assert ret == 0

        return self
    
    def set_load_case_selected(self, contain="", not_contain="!"):
        """Set the load patterns selected for display
        """

        self, LoadCaseList = (
            self
            ._get_table_column_list("load_case_definitions_-_summary", "name", contain=contain, not_contain=not_contain)
        )

        ret, LoadCaseList = (
            self.model
            .SapModel
            .DatabaseTables
            .SetLoadCasesSelectedForDisplay(LoadCaseList)
        )
        assert ret == 0

        return self

    def set_load_combination_selected(self, contain="", not_contain="!"):
        """Set the load patterns selected for display
        """

        self, LoadCombinationList = (
            self
            ._get_table_column_list("load_combination_definitions", "name", contain=contain, not_contain=not_contain)
        )

        ret, LoadCombinationList = (
            self.model
            .SapModel
            .DatabaseTables
            .SetLoadCombinationsSelectedForDisplay(LoadCombinationList)
        )
        assert ret == 0

        return self

    def set_output_options(
            self, IsUserBaseReactionLocation=False, UserBaseReactionX=0.0, 
            UserBaseReactionY=0.0, UserBaseReactionZ=0.0, IsAllModes=True, 
            StartMode=1, EndMode=3, IsAllBucklingModes=True, StartBucklingMode=1, 
            EndBucklingMode=3, MultistepStatic=2, NonlinearStatic=2, 
            ModalHistory=2, DirectHistory=2, Combo=2
        ):
        """Set the output options for display
        
        MultistepStatic: Indicates how multistep static load case results are displayed
            1=Envelopes, 
            2=Step-by-step, 
            3=Last Step.
        NonlinearStatic: Indicates how nonlinear static load case results are displayed 
            1=Envelopes, 
            2=Step-by-step, 
            3=Last Step.
        ModalHistory: Indicates how multistep modal time history load case results are displayed
            1=Envelopes, 
            2=Step-by-step, 
            3=Last Step.
        DirectHistory: Indicates how direct integration time load case results are displayed
            1=Envelopes, 
            2=Step-by-step, 
            3=Last Step.
        Combo: Indicates how load combination results are displayed
            1=Envelopes, 
            2=Multiple Values If Possible.
        """
        
        ret = (
            self.model.SapModel
            .DatabaseTables
            .SetOutputOptionsForDisplay(
                IsUserBaseReactionLocation, UserBaseReactionX, UserBaseReactionY, 
                UserBaseReactionZ, IsAllModes, StartMode, EndMode, 
                IsAllBucklingModes, StartBucklingMode, EndBucklingMode, 
                MultistepStatic, NonlinearStatic, ModalHistory, DirectHistory, Combo
            )
        )
        assert ret == 0
        
        return self

    def get_tables(self, keys, GroupName="All"):
        """The function extracts the requested table as a dataframe object
        """

        keys = convert_to_list(keys)

        available_tables = self._get_available_tables_for_display()
        for TableKey in keys:
            TableKey = match_options(TableKey, available_tables)
            if not TableKey:
                continue

            FieldKeyList = []
            TableVersion = 0
            FieldsKeysIncluded = []
            NumberRecords = 0
            TableData = []
            
            ret, GroupName, TableVersion, FieldsKeysIncluded, NumberRecords, TableData = (
                self.model.SapModel
                .DatabaseTables.GetTableForDisplayArray(TableKey, FieldKeyList, GroupName, TableVersion, FieldsKeysIncluded, NumberRecords, TableData)
            )
            assert ret == 0, f"Error when getting {TableKey}"
            data = np.array(TableData).astype("str")
            
            if len(data) != 0:
                data = np.reshape(data, (NumberRecords, -1))
                df = pl.DataFrame(data, schema={key: pl.String for key in FieldsKeysIncluded})
            else:
                df = pl.DataFrame(schema={key: pl.String for key in FieldsKeysIncluded})
            
            key = "_".join(TableKey.lower().split())
            self.data[key] = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})

        return self
    
    def write_tables(self, keys, format="parquet", GroupName="All", folder=R".\data", purpose="", model="csi"):
        """The function saves the requested table as a .csv file"""
        
        keys = convert_to_list(keys)

        folder = folder + os.sep + self.datetime + "_" + self.name + "_" + purpose + "_" + model
        check_folder(folder)

        available_tables = self._get_available_tables_for_display()
        for TableKey in keys:
            self = self.get_tables([TableKey,], GroupName=GroupName)
            
            TableKey = match_options(TableKey, available_tables)
            if not TableKey:
                continue
            
            key = "_".join(TableKey.lower().split())
            path = folder + os.sep + key + "." + format
            
            if format == "parquet":
                self.data[key].write_parquet(path)
            elif format == "csv":
                self.data[key].write_csv(path, separator=",")
        
        return self
    
    def _get_available_tables_for_editing(self):
        """Get a list of available table keys for editing
        """

        NumberTables = 0
        TableKey     = []
        TableName    = []
        ImportType   = []
        IsEmpty      = []

        ret, NumberTables, TableKey, TableName, ImportType, IsEmpty = (
            self.model.SapModel
            .DatabaseTables.GetAllTables(NumberTables, TableKey, TableName, ImportType, IsEmpty)
        )
        assert ret == 0

        df = pl.DataFrame([
            pl.Series("key", TableKey, dtype=pl.String),
            pl.Series("name", TableName, dtype=pl.String),
            pl.Series("import_type", ImportType, dtype=pl.UInt8),
            pl.Series("is_empty", IsEmpty, dtype=pl.Boolean),
        ])

        is_locked = self.model.SapModel.GetModelIsLocked()
        import_types = [3] if is_locked else [2, 3]

        return (
            df
            .filter(pl.col("import_type").is_in(import_types))
            ["key"].to_list()
        )
    
    def _get_table_for_editing(self, TableKey, GroupName="All"):
        """Get the table for editing in polars dataframe
        """

        available_tables = self._get_available_tables_for_editing()
        TableKey = match_options(TableKey, available_tables)
        if not TableKey:
            return self
        
        TableVersion = 0
        FieldsKeysIncluded = []
        NumberRecords = 0
        TableData =[]
        
        ret, TableVersion, FieldsKeysIncluded, NumberRecords, TableData = (
            self.model.SapModel
            .DatabaseTables.GetTableForEditingArray(TableKey, GroupName, TableVersion, FieldsKeysIncluded, NumberRecords, TableData)
        )
        assert ret == 0
        data = np.array(TableData).astype("str")

        if len(data) != 0:
            data = np.reshape(data, (NumberRecords, -1))
            df = pl.DataFrame(data, schema={key: pl.String for key in FieldsKeysIncluded})
        else:
            df = pl.DataFrame(schema={key: pl.String for key in FieldsKeysIncluded})
        
        df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})
        self.data_interactive["key"] = TableKey
        self.data_interactive["header"] = list(FieldsKeysIncluded)
        self.data_interactive["data"] = df
        self.cache[TableKey] = {
            "header": FieldsKeysIncluded,
            "data": df.clone(),
        }

        return self
    
    def _set_table_for_editing(self):
        """Set the interactive table to be applied into ETABS, make sure data type is string
        """

        TableVersion = 0
        NumberRecords = self.data_interactive["data"].shape[0]
        TableData = self.data_interactive["data"].to_numpy().flatten().tolist()

        ret, TableVersion, FieldsKeysIncluded, TableData = (
            self.model.SapModel
            .DatabaseTables.SetTableForEditingArray(
                self.data_interactive["key"],
                TableVersion,
                System.Array[System.String](self.data_interactive["header"]),
                NumberRecords,
                System.Array[System.String](TableData),
            )
        )
        assert ret == 0

        return self

    def _apply_edited_tables(self, FillImportLog=False):
        """Apply all tables set for application
        """

        NumFatalErrors = 0
        NumErrorMsgs = 0
        NumWarnMsgs = 0
        NumInfoMsgs = 0
        ImportLog = ""
        
        ret, NumFatalErrors, NumErrorMsgs, NumWarnMsgs, NumInfoMsgs, ImportLog = (
            self.model.SapModel.DatabaseTables.ApplyEditedTables(FillImportLog, NumFatalErrors, NumErrorMsgs, NumWarnMsgs, NumInfoMsgs, ImportLog)
        )
        assert ret == 0

        return self

    def _cancel_editing_table(self):
        """Cancel editing table and clear all tables set for editing
        """

        ret = self.model.SapModel.DatabaseTables.CancelTableEditing()
        assert ret == 0

        return self
    
    def _undo_edited_tables(self, keys):
        """Undo edited table
        """

        keys = convert_to_list(keys)

        for key in keys:
            if key not in list(self.cache.keys()):
                continue
            
            self.data_interactive["key"] = key
            self.data_interactive["header"] = self.cache[key]["header"]
            self.data_interactive["data"] = self.cache[key]["data"]
            self = self._set_table_for_editing()

        return (
            self
            ._apply_edited_tables()
        )
    
    # ----- frame obj -----

    def _add_frame_obj_by_coord(self, XI, YI, ZI, XJ, YJ, ZJ, PropName="Default", UserName=None):
        """Add a frame object by coordinates
        """

        Name = ""

        ret, Name = self.model.SapModel.FrameObj.AddByCoord(XI, YI, ZI, XJ, YJ, ZJ, Name, PropName=PropName, UserName=UserName)
        # print(ret)
        assert ret == 0

        self.cache["last_added_frame"] = Name

        return self
    
    def _add_frame_obj_by_coord_null(self, XI, YI, ZI, XJ, YJ, ZJ, UserName=None):
        """Add a frame object by coordinates
        """

        Name = ""

        ret, Name = self.model.SapModel.FrameObj.AddByCoord(XI, YI, ZI, XJ, YJ, ZJ, Name, PropName="", UserName=UserName)
        # assert ret == 0 # will not be zero if adding null

        return self
    
    # ----- group -----

    def _delete_group(self, Name):
        """Deletes the specified group. 
        """

        ret = self.model.SapModel.GroupDef.Delete(Name)
        assert ret == 0

        return self
    
    def _get_group_name_list(self):
        """Get all of the group names
        """

        NumberNames = 0
        MyName = []

        ret, NumberNames, MyName = self.model.SapModel.GroupDef.GetNameList(NumberNames, MyName)
        # print(ret)
        assert ret == 0

        if NumberNames != 0:
            self.cache["last_groups"] = list(MyName)

        return self

    def _set_group(self, Name):
        """If this is the name of an existing group, that group is modified, otherwise a new group is added.
        """

        ret = self.model.SapModel.GroupDef.SetGroup(Name)
        assert ret == 0

        return self
    
    # ----- point obj -----

    def _add_point_obj_by_coord(self, X, Y, Z, UserName=None, MergeOff=False):
        """Add a point object by coordinates

        MergeOff -- False for merging, True for not merging
        """

        Name = ""

        ret, Name = self.model.SapModel.PointObj.AddCartesian(X, Y, Z, Name, UserName=UserName, MergeOff=MergeOff)
        assert ret == 0

        self.cache["last_added_point"] = Name

        return self
    
    def _set_point_obj_group_assign(self, Name, GroupName, Remove=False):
        """Adds or removes point objects from a specified group. 
        """

        ret = self.model.SapModel.PointObj.SetGroupAssign(Name, GroupName, Remove=Remove)
        assert ret == 0

        return self
    
    def _set_point_obj_restraint(self, Name, Value):
        """Add restraint to point
        """

        ret, Value = self.model.SapModel.PointObj.SetRestraint(Name, System.Array[System.Boolean](Value))
        # print(ret)
        assert ret == 0

        return self
    
    def _set_point_obj_spring_assignment(self, Name, SpringProp):
        """Assign spring to point
        """

        ret = self.model.SapModel.PointObj.SetSpringAssignment(Name, SpringProp)
        # print(ret)
        assert ret == 0

        return self
    
    # ----- prop area -----

    def _set_prop_area_slab(self, Name, SlabType, ShellType, MatProp, Thickness):
        """Initialises a slab property
        """

        ret = self.model.SapModel.PropArea.SetSlab(Name, SlabType, ShellType, MatProp, Thickness, color=-1, notes="", GUID="")
        assert ret == 0

        return self
    
    def _set_prop_area_wall(self, Name, WallPropType, ShellType, MatProp, Thickness):
        """Initialises a wall property
        """

        ret = self.model.SapModel.PropArea.SetWall(Name, WallPropType, ShellType, MatProp, Thickness, color=-1, notes="", GUID="")
        assert ret == 0

        return self

    # ----- prop area spring -----
    
    # ----- prop frame -----

    def _set_prop_frame_rectangle(self, Name, MatProp, T3, T2):
        """Initialises a solid rectangular frame section property

        If this property already exists, all items for the section are reset to their default value
        """

        ret = self.model.SapModel.PropFrame.SetRectangle(Name, MatProp, T3, T2, Color=-1, Notes="", GUID="")
        assert ret == 0

        return self
    
    def _set_prop_frame_rebar_beam(
            self, Name, MatPropLong="Rebar", MatPropConfine="Rebar", 
            CoverTop=0.075, CoverBot=0.075, 
            TopLeftArea=(0.032**2)/4*np.pi, TopRightArea=(0.032**2)/4*np.pi, 
            BotLeftArea=(0.032**2)/4*np.pi, BotRightArea=(0.032**2)/4*np.pi,
        ):
        """Assigns beam rebar data to frame sections.
        """

        (
            self.model.SapModel
            .PropFrame.SetRebarBeam(
                Name, MatPropLong, MatPropConfine, 
                CoverTop, CoverBot, TopLeftArea, TopRightArea, BotLeftArea, BotRightArea
            )
        )
        # assert ret == 0

        return self

    # ----- prop link -----

    # ----- prop material -----

    def _add_material(self, MatType, Region, Standard, Grade):
        """Add material
        """

        Name = ""

        ret, Name = (
            self.model.SapModel.PropMaterial
            .AddMaterial(Name, MatType, Region, Standard, Grade, UserName="")
        )
        assert ret == 0

        return self

    # ----- prop point spring -----

    def _set_prop_point_spring(self, Name, k):
        """Creates a new named point spring property, or modifies an existing named point spring property 
        """

        SpringOption = 1 # User defined / link properties

        ret, k = (
            self.model.SapModel.PropPointSpring
            .SetPointSpringProp(Name, SpringOption, System.Array[System.Double](k))
        )
        assert ret == 0

        return self

    
    # ----- sap model -----

    def _get_filepath_filename(self):
        """Get the filename
        """

        filepath = self.model.SapModel.GetModelFilepath()
        filename = self.model.SapModel.GetModelFilename()

        return (
            filepath,
            filename[len(filepath):(len(filename) - 4)],
        )
    
    def set_units_kn_m_c(self, units):
        """Set the units to kN, m, and C
        """
        
        ret = self.model.SapModel.SetPresentUnits(units)
        assert ret == 0

        return self
    
    def set_units_kn_m_c_2(self, forceUnits=4, lengthUnits=6, temperatureUnits=2):
        """Set the units to kN, m, and C
        """
        
        ret = (
            self.model.SapModel.SetPresentUnits_2(
                forceUnits=forceUnits, 
                lengthUnits=lengthUnits, 
                temperatureUnits=temperatureUnits
            )
        )
        assert ret == 0

        return self
    
    # ----- session -----
    
    def add_beam_obj_by_coord(
            self, 
            XI: float, YI: float, ZI: float, XJ: float, YJ: float, ZJ: float, 
            section_line: str, grade: int
        ):
        """Add a frame object by coord, does not check if it already exists
        """

        return self._add_frame_obj_by_coord(
            XI, YI, ZI, XJ, YJ, ZJ, PropName=f"BM{section_line} {str(grade)}"
        )
    
    def add_brace_obj_by_coord(
            self, 
            XI: float, YI: float, ZI: float, XJ: float, YJ: float, ZJ: float, 
            section_line: str, grade: int
        ):
        """Add a frame object by coord, does not check if it already exists
        """

        return self._add_frame_obj_by_coord(
            XI, YI, ZI, XJ, YJ, ZJ, PropName=f"BR{section_line} {str(grade)}"
        )
    
    def add_column_obj_by_coord(
            self, 
            XI: float, YI: float, ZI: float, XJ: float, YJ: float, ZJ: float, 
            section_line: str, grade: int
        ):
        """Add a frame object by coord, does not check if it already exists
        """

        return self._add_frame_obj_by_coord(
            XI, YI, ZI, XJ, YJ, ZJ, PropName=f"CL{section_line} {str(grade)}"
        )
    
    def add_frame_obj_by_coord_null(
            self, 
            XI: float, YI: float, ZI: float, XJ: float, YJ: float, ZJ: float
        ):
        """Add a null frame object by coord, does not check if it already exists
        """

        return self._add_frame_obj_by_coord_null(
            XI, YI, ZI, XJ, YJ, ZJ
        )
    
    def add_slab_obj_by_coord(self, X, Y, Z, thickness: float, grade: int):
        """Add a slab object by coord, does not check if it already exists
        """

        return self._add_area_obj_by_coord(X, Y, Z, f"S(m) {str(thickness)} {str(grade)}")
    
    def add_wall_obj_by_coord(self, X, Y, Z, thickness: float, grade: int):
        """Add a wall object by coord, does not check if it already exists
        """

        return self._add_area_obj_by_coord(X, Y, Z, f"W(m) {str(thickness)} {str(grade)}")
    
    def add_restraint(self, X, Y, Z, restraint_ux, restraint_uy, restraint_uz, restraint_rxx, restraint_ryy, restraint_rzz):
        """Add restraint to point, if point does not exist add point
        """

        self = self._add_point_obj_by_coord(X, Y, Z)

        return self._set_point_obj_restraint(self.cache["last_added_point"], [restraint_ux, restraint_uy, restraint_uz, restraint_rxx, restraint_ryy, restraint_rzz])
    
    def add_prop_point_spring(self, section_line, fz, uz):
        """Add point spring property
        """

        return self._set_prop_point_spring(f"{section_line} {fz:.0f} {uz:.3f}", [float(fz)/uz/10, float(fz)/uz/10, float(fz)/uz, 0.0, 0.0, 0.0])
    
    def add_point_spring(self, X, Y, Z, section_line, fz, uz):
        """Add point spring to point, if point does not exist add point
        """

        self = self._add_point_obj_by_coord(X, Y, Z)

        return self._set_point_obj_spring_assignment(self.cache["last_added_point"], f"{section_line} {fz} {uz}")
    
    # ----- view -----

    def refresh_view(self, Zoom=True):
        """Refresh the view
        """

        Window = 0

        ret = self.model.SapModel.View.RefreshView(Window, Zoom)
        assert ret == 0

        return self