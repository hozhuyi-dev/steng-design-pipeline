"""The example tower as 3D solids, against its volumes by hand"""

import struct
from pathlib import Path

import polars as pl
import pytest

from steng.report.model3d import member_solids, volume, write_stl
from steng.session import Session

EXAMPLES = Path(__file__).parents[1] / "examples"
HEIGHT = 4.5 + 18 * 3.6  # rev03: 19 stories, 69.3 m


@pytest.fixture(scope="module")
def solids():
    tower = Session().read_models(["tower_rev03.$et"], abs_path=str(EXAMPLES)).sessions["tower_rev03"]
    return member_solids(tower).with_columns(volume=pl.col("faces").map_elements(volume, return_dtype=pl.Float64))


def test_volumes(solids):
    by_kind = dict(solids.group_by("kind").agg(pl.col("volume").sum()).iter_rows())
    assert by_kind["wall"] == pytest.approx(30.0 * 0.35 * HEIGHT)          # 30 m of piers, 728 m3
    assert by_kind["column"] == pytest.approx(12 * 0.8 * 0.8 * HEIGHT)     # 12 columns, 532 m3
    assert by_kind["floor"] == pytest.approx(8 * 8.0 * 8.0 * 0.25 * 19)    # 8 bays on 19 floors, 2432 m3
    assert by_kind["beam"] == pytest.approx(20 * 8.0 * 0.5 * 0.8 * 19)     # 20 beams on 19 floors, 1216 m3


def test_walls_keep_their_piers(solids):
    walls = solids.filter(pl.col("kind") == "wall")
    assert sorted(walls["pier"].unique()) == ["P1", "P2", "P3", "P4", "P5", "P6"]


def test_stl(solids, tmp_path):
    write_stl(solids, tmp_path / "tower.stl")
    data = (tmp_path / "tower.stl").read_bytes()
    triangles = struct.unpack("<I", data[80:84])[0]
    assert triangles == solids.height * 12  # 6 faces of 2 triangles for each 4-sided prism
    assert len(data) == 84 + 50 * triangles
