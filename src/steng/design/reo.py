"""Functions for generating the reinforcement in slabs, beams, columns and walls

Functions:
"""

import numpy as np
import polars as pl

from steng.design._checks import _check_inputs, _drop_outputs


def generate_reo_beam_flexure(df, ur_limit, diameter):
    """Generate the beam reinforcement
    """

    # Input
    missing_column = _check_inputs(df, ["b", "A_top_req", "A_bot_req"])
    if missing_column: return df
    b = np.array(df.select("b").to_series())                    #mm
    A_top_req = np.array(df.select("A_top_req").to_series())    #mm2
    A_bot_req = np.array(df.select("A_bot_req").to_series())    #mm2
    
    # Constant
    n_rows = df.shape[0]

    # Calculations
    widths = np.unique(b)
    dia_top00 = np.full(n_rows, 0)
    dia_top01 = np.full(n_rows, 0)
    dia_top02 = np.full(n_rows, 0)
    dia_top03 = np.full(n_rows, 0)
    dia_top04 = np.full(n_rows, 0)
    dia_top05 = np.full(n_rows, 0)
    dia_top06 = np.full(n_rows, 0)
    dia_top07 = np.full(n_rows, 0)
    dia_top08 = np.full(n_rows, 0)
    dia_top09 = np.full(n_rows, 0)
    dia_bot00 = np.full(n_rows, 0)
    dia_bot01 = np.full(n_rows, 0)
    dia_bot02 = np.full(n_rows, 0)
    dia_bot03 = np.full(n_rows, 0)
    dia_bot04 = np.full(n_rows, 0)
    dia_bot05 = np.full(n_rows, 0)
    dia_bot06 = np.full(n_rows, 0)
    dia_bot07 = np.full(n_rows, 0)
    dia_bot08 = np.full(n_rows, 0)
    dia_bot09 = np.full(n_rows, 0)
    n_top00 = np.full(n_rows, 0)
    n_top01 = np.full(n_rows, 0)
    n_top02 = np.full(n_rows, 0)
    n_top03 = np.full(n_rows, 0)
    n_top04 = np.full(n_rows, 0)
    n_top05 = np.full(n_rows, 0)
    n_top06 = np.full(n_rows, 0)
    n_top07 = np.full(n_rows, 0)
    n_top08 = np.full(n_rows, 0)
    n_top09 = np.full(n_rows, 0)
    n_bot00 = np.full(n_rows, 0)
    n_bot01 = np.full(n_rows, 0)
    n_bot02 = np.full(n_rows, 0)
    n_bot03 = np.full(n_rows, 0)
    n_bot04 = np.full(n_rows, 0)
    n_bot05 = np.full(n_rows, 0)
    n_bot06 = np.full(n_rows, 0)
    n_bot07 = np.full(n_rows, 0)
    n_bot08 = np.full(n_rows, 0)
    n_bot09 = np.full(n_rows, 0)
    A_top_prov = np.full(n_rows, 0.0)
    A_bot_prov = np.full(n_rows, 0.0)
    for width in widths:
        is_current_width = b == width
        n_min = 2
        n_max = max(int(width // 100), 2)
        n = np.arange(n_min, n_max + 1, 1)

        (
            dia_00, dia_01, dia_02, dia_03, dia_04, dia_05, dia_06, dia_07, dia_08, dia_09, 
            n_00, n_01, n_02, n_03, n_04, n_05, n_06, n_07, n_08, n_09
        ) = _generate_reo_beam_flexure(diameter, n)
        A = (
            n_00 * dia_00 ** 2 * np.pi / 4
            + n_01 * dia_01 ** 2 * np.pi / 4
            + n_02 * dia_02 ** 2 * np.pi / 4
            + n_03 * dia_03 ** 2 * np.pi / 4
            + n_04 * dia_04 ** 2 * np.pi / 4
            + n_05 * dia_05 ** 2 * np.pi / 4
            + n_06 * dia_06 ** 2 * np.pi / 4
            + n_07 * dia_07 ** 2 * np.pi / 4
            + n_08 * dia_08 ** 2 * np.pi / 4
            + n_09 * dia_09 ** 2 * np.pi / 4
        )
        indices = np.argsort(A)
        dia_00 = dia_00[indices]
        dia_01 = dia_01[indices]
        dia_02 = dia_02[indices]
        dia_03 = dia_03[indices]
        dia_04 = dia_04[indices]
        dia_05 = dia_05[indices]
        dia_06 = dia_06[indices]
        dia_07 = dia_07[indices]
        dia_08 = dia_08[indices]
        dia_09 = dia_09[indices]
        n_00 = n_00[indices]
        n_01 = n_01[indices]
        n_02 = n_02[indices]
        n_03 = n_03[indices]
        n_04 = n_04[indices]
        n_05 = n_05[indices]
        n_06 = n_06[indices]
        n_07 = n_07[indices]
        n_08 = n_08[indices]
        n_09 = n_09[indices]
        A = A[indices]
        
        indices = np.searchsorted(A, A_top_req[is_current_width] / ur_limit, side='right')
        indices[indices == len(A)] -= 1
        dia_top00[is_current_width] = dia_00[indices]
        dia_top01[is_current_width] = dia_01[indices]
        dia_top02[is_current_width] = dia_02[indices]
        dia_top03[is_current_width] = dia_03[indices]
        dia_top04[is_current_width] = dia_04[indices]
        dia_top05[is_current_width] = dia_05[indices]
        dia_top06[is_current_width] = dia_06[indices]
        dia_top07[is_current_width] = dia_07[indices]
        dia_top08[is_current_width] = dia_08[indices]
        dia_top09[is_current_width] = dia_09[indices]
        n_top00[is_current_width] = n_00[indices]
        n_top01[is_current_width] = n_01[indices]
        n_top02[is_current_width] = n_02[indices]
        n_top03[is_current_width] = n_03[indices]
        n_top04[is_current_width] = n_04[indices]
        n_top05[is_current_width] = n_05[indices]
        n_top06[is_current_width] = n_06[indices]
        n_top07[is_current_width] = n_07[indices]
        n_top08[is_current_width] = n_08[indices]
        n_top09[is_current_width] = n_09[indices]
        A_top_prov[is_current_width] = A[indices]
        indices = np.searchsorted(A, A_bot_req[is_current_width] / ur_limit, side='right')
        indices[indices == len(A)] -= 1
        dia_bot00[is_current_width] = dia_00[indices]
        dia_bot01[is_current_width] = dia_01[indices]
        dia_bot02[is_current_width] = dia_02[indices]
        dia_bot03[is_current_width] = dia_03[indices]
        dia_bot04[is_current_width] = dia_04[indices]
        dia_bot05[is_current_width] = dia_05[indices]
        dia_bot06[is_current_width] = dia_06[indices]
        dia_bot07[is_current_width] = dia_07[indices]
        dia_bot08[is_current_width] = dia_08[indices]
        dia_bot09[is_current_width] = dia_09[indices]
        n_bot00[is_current_width] = n_00[indices]
        n_bot01[is_current_width] = n_01[indices]
        n_bot02[is_current_width] = n_02[indices]
        n_bot03[is_current_width] = n_03[indices]
        n_bot04[is_current_width] = n_04[indices]
        n_bot05[is_current_width] = n_05[indices]
        n_bot06[is_current_width] = n_06[indices]
        n_bot07[is_current_width] = n_07[indices]
        n_bot08[is_current_width] = n_08[indices]
        n_bot09[is_current_width] = n_09[indices]
        A_bot_prov[is_current_width] = A[indices]

    ur_top = A_top_req / A_top_prov
    ur_bot = A_bot_req / A_bot_prov

    # Output
    df = _drop_outputs(
        df, 
        [
            "dia_top00", "dia_bot00", "n_top00", "n_bot00",
            "dia_top01", "dia_bot01", "n_top01", "n_bot01",
            "dia_top02", "dia_bot02", "n_top02", "n_bot02",
            "dia_top03", "dia_bot03", "n_top03", "n_bot03",
            "dia_top04", "dia_bot04", "n_top04", "n_bot04",
            "dia_top05", "dia_bot05", "n_top05", "n_bot05",
            "dia_top06", "dia_bot06", "n_top06", "n_bot06",
            "dia_top07", "dia_bot07", "n_top07", "n_bot07",
            "dia_top08", "dia_bot08", "n_top08", "n_bot08",
            "dia_top09", "dia_bot09", "n_top09", "n_bot09",
            "A_top_prov", "A_bot_prov", "ur_top", "ur_bot",
        ]
    )
    df = df.hstack(
        pl.DataFrame({
            "dia_top00": dia_top00, "dia_bot00": dia_bot00, "n_top00": n_top00, "n_bot00": n_bot00,
            "dia_top01": dia_top01, "dia_bot01": dia_bot01, "n_top01": n_top01, "n_bot01": n_bot01,
            "dia_top02": dia_top02, "dia_bot02": dia_bot02, "n_top02": n_top02, "n_bot02": n_bot02,
            "dia_top03": dia_top03, "dia_bot03": dia_bot03, "n_top03": n_top03, "n_bot03": n_bot03,
            "dia_top04": dia_top04, "dia_bot04": dia_bot04, "n_top04": n_top04, "n_bot04": n_bot04,
            "dia_top05": dia_top05, "dia_bot05": dia_bot05, "n_top05": n_top05, "n_bot05": n_bot05,
            "dia_top06": dia_top06, "dia_bot06": dia_bot06, "n_top06": n_top06, "n_bot06": n_bot06,
            "dia_top07": dia_top07, "dia_bot07": dia_bot07, "n_top07": n_top07, "n_bot07": n_bot07,
            "dia_top08": dia_top08, "dia_bot08": dia_bot08, "n_top08": n_top08, "n_bot08": n_bot08,
            "dia_top09": dia_top09, "dia_bot09": dia_bot09, "n_top09": n_top09, "n_bot09": n_bot09,
            "A_top_prov": A_top_prov, "A_bot_prov": A_bot_prov, "ur_top": ur_top, "ur_bot": ur_bot,
        })
    )

    return df

def _generate_reo_beam_flexure(diameters, ns):
    """Generate the possible arrange of rebars
    """

    N_LAYER = 10
    n_max = ns[-1]
    dia_list = []
    n_list = []

    for layer in range(N_LAYER):
        for n in ns:
            if (layer == 0) and (n != n_max):
                continue
            if (int(n_max // 2) == 0) and (layer > 0) and (int(n // 2) == 1):
                continue
            for diameter in diameters:
                dia_current = np.full(N_LAYER, 0)
                n_current = np.full(N_LAYER, 0)
                dia_current[:layer] = diameters[-1]
                n_current[:layer] = ns[-1]
                dia_current[layer] = diameter
                n_current[layer] = n
                dia_list.append(list(dia_current))
                n_list.append(list(n_current))
    
    dia_arr = np.array(dia_list)  # Shape: (N, 10)
    n_arr = np.array(n_list)      # Shape: (N, 10)
    dia_00, dia_01, dia_02, dia_03, dia_04, dia_05, dia_06, dia_07, dia_08, dia_09 = dia_arr.T
    n_00, n_01, n_02, n_03, n_04, n_05, n_06, n_07, n_08, n_09 = n_arr.T

    return (
        dia_00, dia_01, dia_02, dia_03, dia_04, dia_05, dia_06, dia_07, dia_08, dia_09, 
        n_00, n_01, n_02, n_03, n_04, n_05, n_06, n_07, n_08, n_09,
    )

def generate_reo_beam_shear(df, ur_limit, diameter, spacing):
    """Generate the beam reo for shear
    """

    # Input
    missing_column = _check_inputs(df, ["b", "A_sw_s_req"])
    if missing_column: return df
    b = np.array(df.select("b").to_series())                    #mm
    A_sw_s_req = np.array(df.select("A_sw_s_req").to_series())    #mm2/mm
    
    # Constant
    n_rows = df.shape[0]

    # Calculations
    widths = np.unique(b)
    diameter = np.array(diameter)
    spacing = np.array(spacing)
    dia_link = np.full(n_rows, 0)
    n_link = np.full(n_rows, 0)
    s_link = np.full(n_rows, 0)
    A_sw_s_prov = np.full(n_rows, 0.0)
    for width in widths:
        is_current_width = b == width
        n_min = 2
        n_max = max(int(width // 100), 2)
        n = np.arange(n_min, n_max + 1, 2)

        diameter_current, spacing_current, n_current = np.meshgrid(diameter, spacing, n, indexing='ij')
        diameter_current = diameter_current.ravel()
        spacing_current = spacing_current.ravel()
        n_current = n_current.ravel()
        A_sw_s = n_current * (diameter_current ** 2) / 4 * np.pi / spacing_current

        indices = np.argsort(A_sw_s)
        diameter_current = diameter_current[indices]
        spacing_current = spacing_current[indices]
        n_current = n_current[indices]
        A_sw_s = A_sw_s[indices]

        indices = np.searchsorted(A_sw_s, A_sw_s_req[is_current_width] / ur_limit, side='right')
        indices[indices == len(A_sw_s)] -= 1

        n_link[is_current_width] = n_current[indices]
        s_link[is_current_width] = spacing_current[indices]
        dia_link[is_current_width] = diameter_current[indices]
        A_sw_s_prov[is_current_width] = A_sw_s[indices]

    ur_link = A_sw_s_req / A_sw_s_prov

    # Output
    df = _drop_outputs(df, ["A_sw_s_prov", "n_link", "s_link", "dia_link", "ur_link"])
    df = df.hstack(
        pl.DataFrame({
            "A_sw_s_prov": A_sw_s_prov,
            "n_link": n_link,
            "s_link": s_link,
            "dia_link": dia_link,
            "ur_link": ur_link,
        })
    )

    return df