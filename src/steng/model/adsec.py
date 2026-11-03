"""An interface to interact with Adsec

Classes:
AdSecMixin -- An interface to interact with Adsec

Functions:
"""

import polars as pl


def _adsec():
    """Start AdSec, which loads its .NET assemblies on import"""
    from steng.connectors.adsec import AdSec

    return AdSec()


class AdSecMixin:
    def __init__(self):
        """Initialise instance
        """
        pass

    def adsec_calculate_pm(self, angle, section_type, df):
        """Generate the pm curve
        """

        self.curve = (
            _adsec()
            .get_force_moment_curve(angle, section_type, df)
            .filter(pl.col("moment") >= 0)
        )

        return self

    def adsec_calculate_pmm(self, section_type, df):
        """Generate the pmm surface
        """

        self.surface = (
            _adsec()
            .get_failure_surface(section_type, df)
            .filter(
                ((pl.col("yy1") >= 0) & (pl.col("zz1") >= 0)) | ((pl.col("yy2") >= 0) & (pl.col("zz2") >= 0)) | ((pl.col("yy3") >= 0) & (pl.col("zz3") >= 0))
            )
        )

        return self