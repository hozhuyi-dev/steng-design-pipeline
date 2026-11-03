"""Functions built on numpy and sympy for reinforced concrete calculations

Functions:
"""

import numpy as np

EFFECTIVE_DEPTH_FACTOR = 0.9

def A(
        dia00, n00, 
        dia01, n01, 
        dia02, n02, 
        dia03, n03, 
        dia04, n04, 
        dia05, n05, 
        dia06, n06, 
        dia07, n07, 
        dia08, n08, 
        dia09, n09, 
    ):
    A00 = dia00 ** 2 * np.pi / 4 * n00
    A01 = dia01 ** 2 * np.pi / 4 * n01
    A02 = dia02 ** 2 * np.pi / 4 * n02
    A03 = dia03 ** 2 * np.pi / 4 * n03
    A04 = dia04 ** 2 * np.pi / 4 * n04
    A05 = dia05 ** 2 * np.pi / 4 * n05
    A06 = dia06 ** 2 * np.pi / 4 * n06
    A07 = dia07 ** 2 * np.pi / 4 * n07
    A08 = dia08 ** 2 * np.pi / 4 * n08
    A09 = dia09 ** 2 * np.pi / 4 * n09
    return A00 + A01 + A02 + A03 + A04 + A05 + A06 + A07 + A08 + A09

def centroid(
        cover, dia_link, 
        dia00, n00, 
        dia01, n01, 
        dia02, n02, 
        dia03, n03, 
        dia04, n04, 
        dia05, n05, 
        dia06, n06, 
        dia07, n07, 
        dia08, n08, 
        dia09, n09, 
    ):
    A00 = dia00 ** 2 * np.pi / 4 * n00
    A01 = dia01 ** 2 * np.pi / 4 * n01
    A02 = dia02 ** 2 * np.pi / 4 * n02
    A03 = dia03 ** 2 * np.pi / 4 * n03
    A04 = dia04 ** 2 * np.pi / 4 * n04
    A05 = dia05 ** 2 * np.pi / 4 * n05
    A06 = dia06 ** 2 * np.pi / 4 * n06
    A07 = dia07 ** 2 * np.pi / 4 * n07
    A08 = dia08 ** 2 * np.pi / 4 * n08
    A09 = dia09 ** 2 * np.pi / 4 * n09
    c00 = cover + dia_link + dia00 / 2
    c01 = c00 + dia00 / 2 + np.maximum(dia00, 25) + dia01 / 2
    c02 = c01 + dia01 / 2 + np.maximum(dia01, 25) + dia02 / 2
    c03 = c02 + dia02 / 2 + np.maximum(dia02, 25) + dia03 / 2
    c04 = c03 + dia03 / 2 + np.maximum(dia03, 25) + dia04 / 2
    c05 = c04 + dia04 / 2 + np.maximum(dia04, 25) + dia05 / 2
    c06 = c05 + dia05 / 2 + np.maximum(dia05, 25) + dia06 / 2
    c07 = c06 + dia06 / 2 + np.maximum(dia06, 25) + dia07 / 2
    c08 = c07 + dia07 / 2 + np.maximum(dia07, 25) + dia08 / 2
    c09 = c08 + dia08 / 2 + np.maximum(dia08, 25) + dia09 / 2
    A_total = A00 + A01 + A02 + A03 + A04 + A05 + A06 + A07 + A08 + A09
    cA_total = A00 * c00 + A01 * c01 + A02 * c02 + A03 * c03 + A04 * c04 + A05 * c05 + A06 * c06 + A07 * c07 + A08 * c08 + A09 * c09
    return cA_total / A_total

def d(h, centroid):
    return h - centroid

def d2(h, cover, dia_horz, dia_vert):
    return (h - cover - dia_horz - dia_vert / 2) * EFFECTIVE_DEPTH_FACTOR

def d3(h, cover, dia_horz, dia_vert):
    return h - cover - dia_horz - dia_vert / 2

def d3_6(h, cover, dia_horz, dia_vert0, dia_vert1, dia_vert2, dia_vert3, dia_vert4, dia_vert5):
    d3 = h - cover - dia_horz - dia_vert0 / 2
    d3[dia_vert1 != 0] -= dia_vert0[dia_vert1 != 0] / 2 + np.maximum(dia_vert0[dia_vert1 != 0], 25) + dia_vert1[dia_vert1 != 0] / 2
    d3[dia_vert2 != 0] -= dia_vert1[dia_vert2 != 0] / 2 + np.maximum(dia_vert1[dia_vert2 != 0], 25) + dia_vert2[dia_vert2 != 0] / 2
    d3[dia_vert3 != 0] -= dia_vert2[dia_vert3 != 0] / 2 + np.maximum(dia_vert2[dia_vert3 != 0], 25) + dia_vert3[dia_vert3 != 0] / 2
    d3[dia_vert4 != 0] -= dia_vert3[dia_vert4 != 0] / 2 + np.maximum(dia_vert3[dia_vert4 != 0], 25) + dia_vert4[dia_vert4 != 0] / 2
    d3[dia_vert5 != 0] -= dia_vert4[dia_vert5 != 0] / 2 + np.maximum(dia_vert4[dia_vert5 != 0], 25) + dia_vert5[dia_vert5 != 0] / 2
    return d3

def z5(d):
    return 0.5 * d

def z9(d):
    return 0.9 * d

def A_c(h, b_w):
    return h * b_w

def A_s_wall_3(s_vert, dia_vert0, dia_vert1, dia_vert2, h):
    A_vert0 = (dia_vert0 ** 2) / 4 * np.pi
    A_vert1 = (dia_vert1 ** 2) / 4 * np.pi
    A_vert2 = (dia_vert2 ** 2) / 4 * np.pi
    return 2 * (A_vert0 + A_vert1 + A_vert2) * h / s_vert

def A_s_wall(s_vert, dia_vert, h):
    return 2 * (dia_vert ** 2) / 4 * np.pi * h / s_vert

def l_0(l, k):
    return l * k

def l_02(l, k, beta):
    return l * k * beta