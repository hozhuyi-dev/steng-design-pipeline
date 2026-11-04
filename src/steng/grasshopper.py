"""Helpers for Grasshopper Python: nested lists to and from Grasshopper data trees

Grasshopper's data trees are converted with ghpythonlib.treehelpers, which ships with Rhino.
"""


def list_to_tree(input, none_and_holes=True, source=[0]):
    """Transforms nestings of lists or tuples to a Grasshopper DataTree"""
    from ghpythonlib.treehelpers import list_to_tree  # type: ignore

    return list_to_tree(input, none_and_holes, source)


def tree_to_list(input, retrieve_base=lambda x: x[0]):
    """Returns a list representation of a Grasshopper DataTree"""
    from ghpythonlib.treehelpers import tree_to_list  # type: ignore

    return tree_to_list(input, retrieve_base)


def input_to_list(*inputs):
    """Data should always be list of lists

    First index is for each object, second index is for the data
    """

    outputs = []

    for input in inputs:
        if type(input) == list:
            if type(input[0]) == list:
                outputs.append(input)
            else:
                outputs.append([input, ])
        elif type(input) in [str, int, float, bool, type(None)]:
            outputs.append([[input,]])
        else:
            output = tree_to_list(input, retrieve_base=(lambda x: x))

            while type(output[0][0]) == list:
                output = output[0]

            outputs.append(output)

    return outputs
