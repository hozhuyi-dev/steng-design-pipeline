"""DesignSession: a Session with the reinforced concrete design methods
"""

from steng.design.rc_design import RcDesignMixin
from steng.session import Session


class DesignSession(Session, RcDesignMixin):
    """A Session that also designs reinforced concrete columns, walls, coupling beams and piles

    The design methods come from RcDesignMixin. AdSec, which confirms the simplified checks, is loaded
    only when a design calls it:

        walls = (
            DesignSession()
            .read_models(["20250701-091000_tower_rev03_rc-wall_etabs"], abs_path="examples")
            .create_rc_wall()
            .design_rc_wall(contain_outputcase="^ULS", use_adsec=False)
        )
        walls.get_rc_wall(0.9)
    """

    def __init__(self, name: str | None = None) -> None:
        """Start an empty session, as Session does"""

        Session.__init__(self, name=name)
        RcDesignMixin.__init__(self)
