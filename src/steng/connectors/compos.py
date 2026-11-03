"""A library that wraps the COMPOS API.

Classes:
COMPOS -- A class 

Functions:
"""

import datetime
import os

import comtypes.client

from steng.utils import _check_file, _check_folder


class Compos:
    def __init__(self, folder=r"./compos", file="test.cob", saveAs=None):
        """Initialise compos model
        """

        if not saveAs:
            saveAs = file

        _check_folder(folder)

        self.path = folder + os.sep + file
        self.path_new = folder + os.sep + saveAs
        self.datetime = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
        self.model = comtypes.client.CreateObject("Compos.Automation")
        self.data = {}

    def __enter__(self):
        """Open the model
        """
        is_file_exist = _check_file(self.path)

        if is_file_exist:
            self.model.Open(self.path)
        else:
            self.model.New()


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save and close the model
        """

        self.model.SaveAs(self.path_new)
        self.model.Close()