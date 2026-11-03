"""Ray-triangle intersection for checking design forces against a failure surface

A design force (P, M2, M3) is a ray from the origin. Scaling it by t reaches the failure surface,
so the utilisation ratio is 1 / t. Each ray is paired with one candidate triangle of the surface.

Functions:
rays_triangles_intersection -- Intersect each ray from the origin with its triangle
"""

import numpy as np

EPS = 1e-9


def rays_triangles_intersection(directions, triangles):
    """Intersect ray i, from the origin along directions[i], with triangle triangles[i]

    Moller and Trumbore (1997), "Fast, minimum storage ray-triangle intersection". The ray
    O + tD meets the triangle V0 + u(V1 - V0) + v(V2 - V0) where u, v >= 0 and u + v <= 1;
    Cramer's rule gives t, u and v from scalar triple products. Vectorised over all pairs.

    Parameters
    ----------
    directions : np.ndarray (n, 3)
        Ray directions
    triangles : np.ndarray (n, 3, 3)
        Vertices of each triangle

    Returns
    -------
    intersected : np.ndarray (n,) of bool
    t : np.ndarray (n,)
        Distance along each ray in multiples of its direction, NaN where it misses
    """

    directions = np.asarray(directions, dtype=float)
    triangles = np.asarray(triangles, dtype=float)

    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0

    # det = e1 . (d x e2) is zero when the ray is parallel to the triangle
    p = np.cross(directions, e2)
    det = np.einsum("ij,ij->i", e1, p)
    is_parallel = np.abs(det) < EPS
    inv_det = np.divide(1.0, det, out=np.zeros_like(det), where=~is_parallel)

    # s = origin - v0
    s = -v0
    q = np.cross(s, e1)
    u = np.einsum("ij,ij->i", s, p) * inv_det
    v = np.einsum("ij,ij->i", directions, q) * inv_det
    t = np.einsum("ij,ij->i", e2, q) * inv_det

    intersected = ~is_parallel & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > EPS)

    return intersected, np.where(intersected, t, np.nan)
