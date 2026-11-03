"""A class with additional methods for working with standardised models

Classes:
DesignSession -- A class for working standardised model with oasys.adsec that is
    incompatible with rhino

Functions:
"""

from steng.design.rc_design import RcDesignMixin
from steng.session import Session


class DesignSession(Session, RcDesignMixin):
    def __init__(self, name=None):
        """Initialise an instance of model
        """

        Session.__init__(self, name=name)
        RcDesignMixin.__init__(self)