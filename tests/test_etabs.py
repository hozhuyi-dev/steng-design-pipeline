"""Reading ETABS result tables: the same results exported with no step, a step type, or a numbered step

ETABS adds StepType for envelopes ("Max", "Min") and StepNumber for stepped cases. Every result table
reads to the same standard model whichever columns it has, with the step appended to the case name.
"""

from collections import defaultdict

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from steng.model.etabs import EtabsMixin
from steng.session import SCHEMA, _cast_df

FORCES = {"p": ["-1200.5", "-1180.0"], "v2": ["35.2", "35.2"], "v3": ["-4.1", "-4.1"],
          "t": ["0.3", "0.3"], "m2": ["12.0", "-8.5"], "m3": ["140.2", "-60.7"]}
TABLES = {
    # table: (expander, standard model attribute, one case's rows as ETABS exports them)
    "base_reactions": ("_expand_base_reaction", "base_reaction", {
        "fx": ["1.5"], "fy": ["-2.5"], "fz": ["8000"], "mx": ["10"], "my": ["-20"], "mz": ["3"],
    }),
    "story_forces": ("_expand_story_force", "story_force", {
        "story": ["L1", "L1"], "location": ["Top", "Bottom"],
        "p": ["-8000", "-8000"], "vx": ["-120", "-120"], "vy": ["0", "0"], "t": ["0", "0"], "mx": ["0", "0"], "my": ["-300", "-840"],
    }),
    "story_max_over_avg_displacements": ("_expand_story_displacement", "story_displacement", {
        "story": ["L1", "L1"], "direction": ["X", "Y"],
        "maximum": ["0.011", "0.002"], "average": ["0.010", "0.0018"], "ratio": ["1.1", "1.1"],
    }),
    "story_max_over_avg_drifts": ("_expand_story_drift", "story_drift", {
        "story": ["L1", "L1"], "direction": ["X", "Y"],
        "max_drift": ["0.0011", "0.0002"], "avg_drift": ["0.0010", "0.00018"], "ratio": ["1.1", "1.1"],
    }),
    "joint_displacements": ("_expand_joint_displacement", "joint_displacement", {
        "uniquename": ["17", "18"], "story": ["Base", "Base"],
        "ux": ["0", "0"], "uy": ["0", "0"], "uz": ["-0.004", "-0.005"], "rx": ["0", "0"], "ry": ["0", "0"], "rz": ["0", "0"],
    }),
    "joint_reactions": ("_expand_joint_reaction", "joint_reaction", {
        "uniquename": ["17", "18"], "story": ["Base", "Base"],
        "fx": ["10", "11"], "fy": ["0", "0"], "fz": ["4000", "4100"], "mx": ["0", "0"], "my": ["5", "6"], "mz": ["0", "0"],
    }),
    "element_forces_-_columns": ("_expand_force_column", "force_column", {
        "uniquename": ["40", "40"], "story": ["L1", "L1"], "station": ["0", "4.5"], **FORCES,
    }),
    "pier_forces": ("_expand_force_pier", "force_pier", {
        "pier": ["P1", "P1"], "story": ["L1", "L1"], "location": ["Top", "Bottom"], **FORCES,
    }),
    "spandrel_forces": ("_expand_force_spandrel", "force_spandrel", {
        "spandrel": ["S1", "S1"], "story": ["L2", "L2"], "location": ["Left", "Right"], **FORCES,
    }),
}
STEPS = {
    # export: (columns ETABS adds, the case name read)
    "no step": ({}, "ULSENV"),
    "step type": ({"steptype": "Max"}, "ULSENV_max"),
    "numbered step": ({"steptype": "Step By Step", "stepnumber": "3"}, "ULSENV_stepbystep3"),
}


class Tables(EtabsMixin):
    """ETABS tables by name, as Session.read_etabs holds them, None for those not exported"""

    def __init__(self, tables):
        self.etabs = defaultdict(lambda: None, tables)


def read(table, step):
    """Read one table exported with the given step columns, as Session.read_etabs does"""
    expander, attribute, rows = TABLES[table]
    n = len(next(iter(rows.values())))
    columns, _ = STEPS[step]
    df = pl.DataFrame({
        **rows, "outputcase": ["ULSENV"] * n, "casetype": ["Combination"] * n,
        **{column: [value] * n for column, value in columns.items()},
    })
    return _cast_df(getattr(Tables({table: df}), expander)(), SCHEMA[attribute])


@pytest.mark.parametrize("table", list(TABLES))
@pytest.mark.parametrize("step", list(STEPS))
def test_case_name_and_type(table, step):
    df = read(table, step)
    assert df["outputcase"].unique().to_list() == [STEPS[step][1]]
    assert df["outputcasetype"].unique().to_list() == ["combination"]


@pytest.mark.parametrize("table", list(TABLES))
@pytest.mark.parametrize("step", ["step type", "numbered step"])
def test_results_do_not_depend_on_the_step_columns(table, step):
    assert_frame_equal(read(table, step).drop("outputcase"), read(table, "no step").drop("outputcase"))


def test_pier_forces_with_design_forces():
    pier = pl.DataFrame({
        "pier": ["P1"], "story": ["L1"], "outputcase": ["ULS1"], "casetype": ["Combination"], "location": ["Bottom"],
        **{k: v[:1] for k, v in FORCES.items()},
    })
    design = pier.rename({"outputcase": "combo"}).drop("casetype").with_columns(combo=pl.lit("ULS13"))
    df = Tables({"pier_forces": pier, "design_forces_-_piers": design})._expand_force_pier()
    assert df["outputcase"].to_list() == ["ULS1", "ULS13"]
    assert df["outputcasetype"].to_list() == ["combination", "combination"]
    assert df["fx"].to_list() == [-1200.5, -1200.5]
