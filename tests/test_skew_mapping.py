import numpy as np

from longrangefy.matched import build_correction


def test_skew_nearest_image_maps_to_correct_target_cell():
    cell = np.array([[1., 0, 0], [1., .1, 0], [0, 0, 1.]])
    correction = build_correction(
        cell, np.zeros((1, 3)), np.zeros((1, 3, 3)), np.eye(3),
        (1, 1, 1), np.zeros((1, 1, 3, 1, 3)), np.zeros((1, 3), int),
        (3, 1, 1), nreal=0, nrecip=0,
    )
    positions = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
    positions[0] += [-.02, .049, 0]
    mapping = correction.map_sites(np.diag([3, 1, 1]) @ cell, positions, tol=.053)
    np.testing.assert_array_equal(mapping.canonical_of_target, [0, 1, 2])
