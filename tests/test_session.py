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
    assert model.story["story"].to_list() == [f"L{k}" for k in range(18, 0, -1)] + ["Base"]
    # L1 is 4.5 m high, the 17 storeys above 3.6 m
    assert model.story["elev"].to_list() == pytest.approx([4.5 + 3.6 * k for k in range(17, -1, -1)] + [0.0])
    # Per story: 12 columns, 20 beams, 6 core wall piers, 8 slabs; 16 grid points and 4 door jambs on 19 levels
    assert (model.column.height, model.beam.height, model.wall.height, model.floor.height) == (216, 360, 108, 144)
    assert model.point.height == 380
    assert model.point.filter(pl.col("restraint_uz")).height == 20


def test_et_wall_properties(from_et):
    model = from_et.sessions["tower_rev01"]
    wall = model.wall.filter(pl.col("wall") == "W1", pl.col("story") == "L1").row(0, named=True)
    assert (wall["pier"], wall["thickness"], wall["grade"]) == ("P1", 0.3, 40)
    assert wall["area"] == pytest.approx(2.5 * 4.5)  # W1 runs to the door
    assert sorted(set(wall["zs"])) == [0.0, 4.5]


def test_et_revisions_side_by_side(from_et):
    walls = from_et.get_df("wall").group_by("model").agg(pl.len(), pl.col("thickness").max()).sort("model")
    assert walls.rows() == [("tower_rev01", 108, 0.3), ("tower_rev02", 108, 0.35), ("tower_rev03", 114, 0.35)]


def test_tables_latest_model_and_history(from_tables):
    assert from_tables.name == "tower_rev03"
    history = from_tables.get_df("modal").filter(pl.col("outputcase") == "Modal_1").sort("datetime")
    assert history["period"].to_list() == [1.92, 1.64, 1.75]


def test_tables_story_results(from_tables):
    model = from_tables.sessions["tower_rev01"]
    base = model.base_reaction.filter(pl.col("outputcase") == "WX").row(0, named=True)
    shear = model.story_force.filter(pl.col("outputcase") == "WX", pl.col("story") == "L1", pl.col("station") == 0.0)
    assert shear["fy"].item() == pytest.approx(base["fx"])
    assert model.story_drift.filter(pl.col("outputcase") == "WX")["ux_max"].sum() == pytest.approx(0.085 * 1.1)


def test_tables_session_named_for_its_model(from_tables):
    model = from_tables.sessions["tower_rev01"]
    assert (model.name, model.datetime) == ("tower_rev01", "20250506-101500")


def test_write_and_read_model(from_tables, tmp_path):
    from_tables.write_model(str(tmp_path))
    folder = tmp_path / "20250701-091000_tower_rev03_model"
    model = Session().read_model(str(folder))
    assert (model.name, model.datetime) == ("tower_rev03", "20250701-091000")
    for attribute in ["story", "story_force", "story_drift", "modal"]:
        assert model.__dict__[attribute].equals(from_tables.__dict__[attribute])


def test_et_revisions_dated_when_saved(from_et):
    # The "$ File ... saved" line, as ETABS writes it: month first, 12-hour clock
    assert [from_et.sessions[name.split(".")[0]].datetime for name in ET_FILES] == [
        "20250506-101500", "20250603-143000", "20250701-091000",
    ]
    assert Session().read_models(ET_FILES, abs_path=str(EXAMPLES)).create_latest_model().name == "tower_rev03"
