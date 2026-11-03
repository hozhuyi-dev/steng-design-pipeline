import numpy as np
import pytest

from steng.geometry.rays import rays_triangles_intersection

# A triangle in the plane x = 2, around the x axis
TRIANGLE = [[2.0, -1.0, -1.0], [2.0, 1.0, -1.0], [2.0, 0.0, 1.0]]


@pytest.mark.parametrize(
    "direction, hit, t",
    [
        ([1.0, 0.0, 0.0], True, 2.0),  # straight through: the force is half the capacity
        ([0.5, 0.0, 0.0], True, 4.0),  # t is in multiples of the direction
        ([-1.0, 0.0, 0.0], False, None),  # pointing away
        ([0.0, 1.0, 0.0], False, None),  # parallel to the plane
        ([1.0, 2.0, 0.0], False, None),  # passes beside the triangle
    ],
)
def test_single_rays(direction, hit, t):
    intersected, t_out = rays_triangles_intersection(np.array([direction]), np.array([TRIANGLE]))
    assert intersected[0] == hit
    if hit:
        assert t_out[0] == pytest.approx(t)
    else:
        assert np.isnan(t_out[0])


def _reference(direction, triangle):
    """Intersect the ray with the triangle's plane, then test the point against each edge"""
    v0, v1, v2 = triangle
    normal = np.cross(v1 - v0, v2 - v0)
    denom = normal @ direction
    if abs(denom) < 1e-12:
        return False, np.nan
    t = (normal @ v0) / denom
    if t <= 0:
        return False, np.nan
    point = t * direction
    for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
        if np.cross(b - a, point - a) @ normal < 0:
            return False, np.nan
    return True, t


def test_matches_reference_on_random_triangles():
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(2000, 3))
    triangles = directions[:, None, :] * 2 + rng.normal(scale=0.8, size=(2000, 3, 3))

    intersected, t = rays_triangles_intersection(directions, triangles)

    expected = [_reference(d, tri) for d, tri in zip(directions, triangles)]
    assert intersected.tolist() == [hit for hit, _ in expected]
    np.testing.assert_allclose(t[intersected], [t for hit, t in expected if hit], rtol=1e-9)
    assert 0.1 < intersected.mean() < 0.9
