"""Wall and column designs over trial bar layouts, pinned to their outputs

The golden files in tests/golden were written by this module's write_golden() from the design
before its numbered copies (calculate_pm_wall_3, _5, _6 and the rest) were collapsed into single
functions over any number of layers, and rewritten for each intended change since. The trial
ratios reach 12%, so columns get up to three layers of bars, and the 350 mm walls the two that fit.
Rewrite them only for an intended change in the results:

    uv run python tests/test_layers_pinned.py
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from steng.design_session import DesignSession

EXAMPLES = Path(__file__).parents[1] / "examples"
GOLDEN = Path(__file__).parent / "golden"
PERC_VERT = np.array([0.4, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
STORIES = ["L1", "L2", "L10", "L19"]


def design_walls():
    return (
        DesignSession()
        .read_models(["20250701-091000_tower_rev03_rc-wall_etabs"], abs_path=str(EXAMPLES))
        .create_rc_wall()
        .design_rc_wall(contain_story="^L(1|2|10|19)$", contain_outputcase="^ULS", perc_vert=PERC_VERT, use_adsec=False)
    )


def design_columns():
    return (
        DesignSession()
        .read_models(["20250701-091000_tower_rev03_rc-column_etabs"], abs_path=str(EXAMPLES))
        .create_rc_column()
        .design_rc_column(
            contain_column="^C(1|2)$", contain_story="^L(1|2|10|19)$", contain_outputcase="^ULS",
            perc_vert=PERC_VERT, use_adsec=False,
        )
    )


def tables(walls, columns):
    """The tables to pin, sorted so that row order does not matter"""
    return {
        "wall_pier": walls.design_rc_wall_pier.sort("pier", "story", "key"),
        "wall_force_pmm": walls.design_rc_wall_force_pmm.sort("pier", "story", "key", "outputcase", "station"),
        "wall_force_v2": walls.design_rc_wall_force_v2.sort("pier", "story", "key", "outputcase", "station"),
        "wall_force_v3": walls.design_rc_wall_force_v3.sort("pier", "story", "key", "outputcase", "station"),
        "column": columns.design_rc_column_column.sort("column", "story", "key"),
        "column_force_pmm": columns.design_rc_column_force_pmm.sort("column", "story", "key", "outputcase", "station"),
    }


@pytest.fixture(scope="module")
def designed():
    return tables(design_walls(), design_columns())


@pytest.mark.parametrize("name", [
    "wall_pier", "wall_force_pmm", "wall_force_v2", "wall_force_v3", "column", "column_force_pmm",
])
def test_matches_golden(designed, name):
    golden = pl.read_parquet(GOLDEN / f"{name}.parquet")
    assert_frame_equal(designed[name], golden, check_column_order=False, rel_tol=1e-9, abs_tol=1e-9)


def test_golden_covers_every_layer_count():
    walls = pl.read_parquet(GOLDEN / "wall_pier.parquet")
    columns = pl.read_parquet(GOLDEN / "column.parquet")
    wall_layers = walls.select(sum((pl.col(f"dia_vert{k}") > 0).cast(pl.Int32) for k in range(6))).to_series().unique()
    column_layers = columns.select(sum((pl.col(f"dia_vert{k}") > 0).cast(pl.Int32) for k in range(3))).to_series().unique()
    assert sorted(wall_layers) == [1, 2]
    assert sorted(column_layers) == [1, 2, 3]


def write_golden():
    GOLDEN.mkdir(exist_ok=True)
    for name, df in tables(design_walls(), design_columns()).items():
        df.write_parquet(GOLDEN / f"{name}.parquet")
        print("wrote", GOLDEN / f"{name}.parquet", df.shape)


if __name__ == "__main__":
    write_golden()
