"""Track the example tower across its revisions and save the charts used in the README

    python examples/make_example_tower.py      # .$et model files
    python examples/make_example_results.py    # ETABS result tables
    python examples/render_charts.py
"""

from pathlib import Path

from steng.session import Session

HERE = Path(__file__).parent
IMG = HERE.parent / "img"

if __name__ == "__main__":
    # Each folder is one ETABS export, named <datetime>_<model>_<purpose>_etabs
    models = sorted(p.name for p in (HERE / "etabs").glob("*_etabs"))
    session = Session().read_models(models, abs_path=str(HERE)).create_latest_model()

    charts = {
        "modal-periods": session.log_modal_period_vs_time(),
        "base-reactions": session.log_base_reaction_vs_time(),
        "story-drift": session.log_story_drift_ux_max(["WX"], drift_limit=500),
    }
    IMG.mkdir(exist_ok=True)
    for name, chart in charts.items():
        chart.save(IMG / f"{name}.png", scale_factor=1)
        print("saved", IMG / f"{name}.png")
