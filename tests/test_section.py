"""Trial reinforcement layouts for wall sections"""

import numpy as np
import polars as pl
import pytest

from steng.design import section


@pytest.mark.parametrize("n_layers", [1, 3, 6])
def test_wall_trials_for_every_thickness(n_layers):
    # 0.35 m does not survive a round trip through mm (0.35 * 1000 / 1000 = 0.35000000000000003),
    # so the trials must join back on the thickness as given
    df = pl.DataFrame({"pier": ["P1", "P2", "P3"], "thickness_min": [0.25, 0.30, 0.35], "length": [6.0, 6.0, 6.0]})
    trials = section.populate_reo_wall_vert(df, np.array([1.0, 2.0]), np.array([12, 16, 20, 25, 32]), np.array([200]), n_layers)
    # One trial per ratio for each pier, and the thickness back in mm
    assert trials.group_by("pier").len().sort("pier")["len"].to_list() == [2, 2, 2]
    assert trials.sort("pier")["thickness_min"].to_list() == pytest.approx([250, 250, 300, 300, 350, 350])


def test_column_links_at_least_a_quarter_of_the_largest_bar():
    # 9.5.3(1): links at least max(6 mm, phi_max / 4). H40 and H25 bars need 10 mm links, even where
    # 6 mm links would carry the shear
    df = pl.DataFrame(
        {"A_sw_s_req2": [0.05], "dia_horz0": [10], "n2_horz0": [2], "dia_vert0": [40], "dia_vert1": [25], "dia_vert2": [0]},
        schema_overrides={c: pl.Int32 for c in ["dia_horz0", "n2_horz0", "dia_vert0", "dia_vert1", "dia_vert2"]},
    )
    links = section.generate_reo_column_horz(df, "A_sw_s_req2", "2", [6, 8, 10, 12], [200], 0.9).row(0, named=True)
    assert links["dia_horz02"] == 10


def test_wall_bar_layers_fit_the_thickness():
    # Layers 25 mm or a bar apart from each face (8.2(2)) must leave as much clear between the two
    # innermost layers. In a 350 mm wall with 30 mm cover and 16 mm horizontal bars, two layers of H32
    # reach 126 mm from each face, leaving 66 mm; a third would cross the middle
    df = pl.DataFrame(
        {"pier": ["P1"], "thickness_min": [0.35], "length": [6.0], "s_vert0": [None], **{f"dia_vert{k}": [None] for k in range(6)}},
        schema_overrides={"s_vert0": pl.Int32, **{f"dia_vert{k}": pl.Int32 for k in range(6)}},
    )
    trials = section.populate_reo_wall_vert(df, np.array([2.0, 6.0, 12.0]), np.array([12, 16, 20, 25, 32]), np.array([200]), 6)
    for row in trials.iter_rows(named=True):
        layers = [row[f"dia_vert{k}"] for k in range(6) if row[f"dia_vert{k}"] > 0]
        x = 30 + 16 + layers[0] / 2 + sum(o / 2 + max(o, 25) + i / 2 for o, i in zip(layers, layers[1:]))
        assert 350 - 2 * x - layers[-1] >= max(layers[-1], 25)
    assert trials.select(sum((pl.col(f"dia_vert{k}") > 0).cast(pl.Int32) for k in range(6))).max().item() == 2
