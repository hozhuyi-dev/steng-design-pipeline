"""Wrapper over the GSA API, through pythonnet
"""

import datetime
import os

import clr

GSA_INSTALL_PATH = r"C:/Program Files/Oasys/GSA 10.2"
clr.AddReference(os.path.join(GSA_INSTALL_PATH, "GsaAPI.dll"))
import GsaAPI as gsa

from steng.utils import _check_file, _check_folder


class Gsa:
    def __init__(self, folder=r"./gsa", file="test.gwb", saveAs=None):
        """Initialise gsa model
        """

        if not saveAs:
            saveAs = file

        _check_folder(folder)

        self.path = folder + os.sep + file
        self.path_new = folder + os.sep + saveAs
        self.datetime = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
        self.model = gsa.Model()
        self.data = {}

    def __enter__(self):
        """Open the model
        """
        is_file_exist = _check_file(self.path)

        if is_file_exist:
            self.model.Open(self.path)


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save and close the model
        """

        self.model.SaveAs(self.path_new)
        self.model.Close()