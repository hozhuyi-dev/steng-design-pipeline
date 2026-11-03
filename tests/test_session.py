"""Reading the example tower into the standard model: from .$et text files and from ETABS tables"""

from pathlib import Path

import polars as pl
import pytest

from steng.session import Session

EXAMPLES = Path(__file__).parents[1] / "examples"
ET_FILES = ["tower_rev01.$et", "tower_rev02.$et", "tower_rev03.$et"]
TABLES = [
    "20250506-101500_tower_rev01_log_etabs",
    "20250603-143000_tower_rev02_log_etabs",
    "20250701-091000_tower_rev03_log_etabs",
]


@pytest.fixture(scope="module")
def from_et():
    return Session().read_models(ET_FILES, abs_path=str(EXAMPLES))


@pytest.fixture(scope="module")
def from_tables():
    return Session().read_models(TABLES, abs_path=str(EXAMPLES)).create_latest_model()


def test_et_stories_and_members(from_et):
    model = from_et.sessions["tower_rev01"]
    assert model.story["story"].to_list() == ["L5", "L4", "L3", "L2", "L1", "Base"]
    assert model.story["elev"].to_list() == pytest.approx([18.9, 15.3, 11.7, 8.1, 4.5, 0.0])
    # Per story: 12 columns, 20 beams, 4 core walls, 8 slabs; 16 grid points on 6 levels
    assert (model.column.height, model.beam.height, model.wall.height, model.floor.height) == (60, 100, 20, 40)
    assert model.point.height == 96
    assert model.point.filter(pl.col("restraint_uz")).height == 16


def test_et_wall_properties(from_et):
    model = from_et.sessions["tower_rev01"]
    wall = model.wall.filter(pl.col("wall") == "W1", pl.col("story") == "L1").row(0, named=True)
    assert (wall["pier"], wall["thickness"], wall["grade"]) == ("P1", 0.4, 40)
    assert wall["area"] == pytest.approx(8.0 * 4.5)
    assert sorted(set(wall["zs"])) == [0.0, 4.5]


def test_et_revisions_side_by_side(from_et):
    walls = from_et.get_df("wall").group_by("model").agg(pl.len(), pl.col("thickness").max()).sort("model")
    assert walls.rows() == [("tower_rev01", 20, 0.4), ("tower_rev02", 20, 0.5), ("tower_rev03", 24, 0.5)]


def test_tables_latest_model_and_history(from_tables):
    assert from_tables.name == "tower_rev03"
    history = from_tables.get_df("modal").filter(pl.col("outputcase") == "Modal_1").sort("datetime")
    assert history["period"].to_list() == [0.62, 0.54, 0.66]


def test_tables_story_results(from_tables):
    model = from_tables.sessions["tower_rev01"]
    base = model.base_reaction.filter(pl.col("outputcase") == "WX").row(0, named=True)
    shear = model.story_force.filter(pl.col("outputcase") == "WX", pl.col("story") == "L1", pl.col("station") == 0.0)
    assert shear["fy"].item() == pytest.approx(base["fx"])
    assert model.story_drift.filter(pl.col("outputcase") == "WX")["ux_max"].sum() == pytest.approx(0.0120 * 1.1)
