"""Track the example tower across its revisions, design its core walls, and save the charts and 3D model used in
the README

    python examples/make_example_tower.py      # .$et model files
    python examples/make_example_results.py    # ETABS result tables
    python examples/render_charts.py
"""

from pathlib import Path

import matplotlib
import numpy as np

from steng.design_session import DesignSession
from steng.report.model3d import member_solids, write_stl
from steng.report.plot import plot_wall_design, plot_wall_overview
from steng.session import Session

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = Path(__file__).parent
IMG = HERE.parent / "img"
UR_LIMIT = 0.9
# Trial vertical reinforcement ratios [%], with the default bar sizes at 200 mm
PERC_VERT = np.array([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
AZIMUTH = -62  # deg, the 3D view


def design_core_walls():
    """Design the latest revision's core walls with the Eurocode 2 simplified method, without AdSec"""
    model = sorted(p.name for p in (HERE / "etabs").glob("*_rc-wall_etabs"))[-1]
    return (
        DesignSession()
        .read_models([model], abs_path=str(HERE))
        .create_rc_wall()
        .design_rc_wall(contain_outputcase="^ULS", ur_limit_pmm=UR_LIMIT, perc_vert=PERC_VERT, use_adsec=False)
    )


def render_tower(walls, path_png, path_stl):
    """Draw the latest revision in 3D, its core walls coloured by the vertical reinforcement chosen,
    and write it as an STL file, which GitHub shows as a 3D model

    The drawing cuts away the frame in the quarter of the plan nearest the viewer, to show the core.
    """
    tower = Session().read_models(["tower_rev03.$et"], abs_path=str(HERE)).sessions["tower_rev03"]
    solids = (
        member_solids(tower)
        .join(walls.select("pier", "story", "perc_vert"), on=["pier", "story"], how="left")
    )
    write_stl(solids, path_stl)

    # The blue ramp of the reinforcement chart, light to dark; the rest of the frame in recessive greys
    reinforcement = LinearSegmentedColormap.from_list("reinforcement", ["#cde2fb", "#0d366b"])
    scale = Normalize(0, 4)
    colours = {"column": (0.80, 0.80, 0.78, 1.0), "beam": (0.86, 0.86, 0.84, 1.0), "floor": (0.90, 0.90, 0.88, 0.18)}
    light = np.array([-0.4, -0.7, 0.6]) / np.linalg.norm([-0.4, -0.7, 0.6])
    # Painted back to front in three groups: the frame behind the core, the core, the frame in front.
    # Depth along the view: the camera looks from azimuth AZIMUTH, nearer is larger
    view = np.array([np.cos(np.radians(AZIMUTH)), np.sin(np.radians(AZIMUTH))])
    core_depth = view @ [12.0, 12.0]
    groups = {"behind": ([], []), "core": ([], []), "front": ([], [])}
    for kind, perc, member_faces in solids.select("kind", "perc_vert", "faces").iter_rows():
        centre = np.mean([corner[:2] for face in member_faces for corner in face], axis=0)
        if kind != "wall" and centre[0] > 12 and centre[1] < 12:
            continue
        group = "core" if kind == "wall" else "front" if view @ centre > core_depth else "behind"
        groups[group][0].extend(member_faces)
        if kind == "wall":
            groups[group][1].extend([reinforcement(scale(perc))] * len(member_faces))
            continue
        # The frame lit gently from above and in front, so its faces read; the walls keep the scale's colours
        for face in member_faces:
            normal = np.cross(np.subtract(face[1], face[0]), np.subtract(face[2], face[0]))
            brightness = 0.85 + 0.15 * max(0.0, light @ normal / np.linalg.norm(normal))
            r, g, b, a = colours[kind]
            groups[group][1].append((r * brightness, g * brightness, b * brightness, a))

    fig = plt.figure(figsize=(8, 9), dpi=110)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    for zorder, (faces, face_colours) in enumerate(groups.values()):
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=face_colours, edgecolors=(1, 1, 1, 0.35), linewidths=0.3, zorder=zorder,
        ))
    ax.set(xlim=(0, 24), ylim=(0, 24), zlim=(0, 70))
    ax.set_box_aspect((24, 24, 70))
    ax.view_init(elev=16, azim=AZIMUTH)
    ax.set_axis_off()
    colourbar = fig.colorbar(ScalarMappable(scale, reinforcement), ax=ax, shrink=0.4, pad=0.0)
    colourbar.set_label("Core walls: vertical reinforcement [%]", color="#52514e")
    colourbar.outline.set_visible(False)
    fig.savefig(path_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    # Each folder is one ETABS export, named <datetime>_<model>_<purpose>_etabs
    models = sorted(p.name for p in (HERE / "etabs").glob("*_log_etabs"))
    session = Session().read_models(models, abs_path=str(HERE)).create_latest_model()
    walls = design_core_walls().get_rc_wall(UR_LIMIT)

    charts = {
        "modal-periods": session.log_modal_period_vs_time(),
        "base-reactions": session.log_base_reaction_vs_time(),
        "story-drift": session.log_story_drift_ux_max(["WX"], drift_limit=500),
        "rc-wall-overview": plot_wall_overview(walls, UR_LIMIT, "Core wall piers - lightest reinforcement that passes"),
        "rc-wall": plot_wall_design(walls, "P5", UR_LIMIT, "Core wall P5, 8 m long - reinforcement by storey"),
    }
    IMG.mkdir(exist_ok=True)
    for name, chart in charts.items():
        chart.save(IMG / f"{name}.png", scale_factor=1)
        print("saved", IMG / f"{name}.png")

    render_tower(walls, IMG / "tower-3d.png", HERE / "tower_rev03.stl")
    print("saved", IMG / "tower-3d.png", "and", HERE / "tower_rev03.stl")
