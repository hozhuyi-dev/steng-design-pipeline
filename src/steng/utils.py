"""Small helpers for files, folders and lists
"""

import difflib
import os


def check_file(filepath):
    """Check if a file exists
    """

    return os.path.isfile(filepath)

def check_folder(folder):
    """Create a folder if it does not exist
    """

    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass

def convert_to_list(data):
    """Convert Python built-in data types to a list.
    """

    if (data == None) or (type(data) == str) or (type(data) == int) or (type(data) == float) or (type(data) == bool):
        return [data,]
    elif type(data) == list:
        return data
    
    return list(data)

def match_options(word: str, options: list):
    """Match the given string to the closest match of existing list."""

    word = word.lower()
    options_lower = [s.lower() for s in options]
    match = difflib.get_close_matches(word, options_lower, n=1, cutoff=0.8)
    if match:
        match = match[0]
        index = options_lower.index(match)
        return str(options[index])
    else:
        return None

def generate_full_path(model):
    """Return the full path relative to the project folder
    """

    if model.endswith("_etabs"):
        folder = r"./etabs/" + model + r"/"
    else:
        folder = model

    return folder