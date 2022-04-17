import numpy as np


def get_h_coord(points):
    insertion = np.ones(shape=(points.shape[1],))
    points_h = np.insert(points, points.shape[0], values=insertion, axis=0)

    return points_h


'''function name:lowercase?'''


def h_for_p_svd(objps, imgps):
    assert objps.shape == imgps.shape, "Number of points don't match"
    assert objps.shape[1] >= 4, "Number of points isn't enough"

    # objps_h = np.vstack([objps, np.array([1]*objps.shape[1])])
    # imgps_h = np.vstack([imgps, np.array([1]*imgps.shape[1])])

    '''points normalization'''
    m1 = np.mean(objps[:2], axis=1)
    max_std1 = max(np.std(objps[:2], axis=1)) + 1e-9
    norm_mat1 = np.diag([1 / max_std1, 1 / max_std1, 1])
    norm_mat1[0][2], norm_mat1[1][2] = -m1[0] / max_std1, -m1[1] / max_std1

    norm_objps = np.dot(norm_mat1, objps)

    m2 = np.mean(imgps[:2], axis=1)
    max_std2 = max(np.std(imgps[:2], axis=1))
    norm_mat2 = np.diag([1 / max_std2, 1 / max_std2, 1])
    norm_mat2[0][2], norm_mat2[1][2] = -m2[0] / max_std2, -m2[1] / max_std2

    norm_imgps = np.dot(norm_mat2, imgps)

    corrs = objps.shape[1]

    A = np.zeros(shape=(2 * corrs, 9))
    for i in range(corrs):
        A[2 * i] = np.array([norm_objps[0][i], norm_objps[1][i], 1, 0, 0, 0,
                             -norm_objps[0][i] * norm_imgps[0][i], -norm_objps[1][i] * norm_imgps[0][i],
                             -norm_imgps[0][i]])
        A[2 * i + 1] = np.array([0, 0, 0, norm_objps[0][i], norm_objps[1][i], 1,
                                 -norm_objps[0][i] * norm_imgps[1][i], -norm_objps[1][i] * norm_imgps[1][i],
                                 -norm_imgps[1][i]])

    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    H = np.dot(np.linalg.inv(norm_mat2), np.dot(H, norm_mat1))

    H /= H[2][2]

    return H


def h_for_p_ransac(objps, imgps):
    assert objps.shape == imgps.shape, "Number of points don't match"
    assert objps.shape[1] >= 4, "Number of points isn't enough"

    # objps_h = np.vstack([objps, np.array([1]*objps.shape[1])])
    # imgps_h = np.vstack([imgps, np.array([1]*imgps.shape[1])])

    '''points normalization'''
    m1 = np.mean(objps[:2], axis=1)
    max_std1 = max(np.std(objps[:2], axis=1)) + 1e-9
    norm_mat1 = np.diag([1 / max_std1, 1 / max_std1, 1])
    norm_mat1[0][2] = -m1[0] / max_std1
    norm_mat1[1][2] = -m1[1] / max_std1

    norm_objps = np.dot(norm_mat1, objps)

    m2 = np.mean(imgps[:2], axis=1)
    max_std2 = max(np.std(imgps[:2], axis=1))
    norm_mat2 = np.diag([1 / max_std2, 1 / max_std2, 1])
    norm_mat2[0][2] = -m2[0] / max_std2
    norm_mat2[1][2] = -m2[1] / max_std2

    norm_imgps = np.dot(norm_mat2, imgps)

    corrs = objps.shape[1]

    A = np.zeros(shape=(2 * corrs, 8))
    for i in range(corrs):
        A[2 * i] = np.array([norm_objps[0][i], norm_objps[1][i], 1, 0, 0, 0,
                             -norm_objps[0][i] * norm_imgps[0][i], -norm_objps[1][i] * norm_imgps[0][i],
                             -norm_imgps[0][i]])
        A[2 * i + 1] = np.array([0, 0, 0, norm_objps[0][i], norm_objps[1][i], 1,
                                 -norm_objps[0][i] * norm_imgps[1][i], -norm_objps[1][i] * norm_imgps[1][i],
                                 -norm_imgps[1][i]])

    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    H = np.dot(np.linalg.inv(norm_mat2), np.dot(H, norm_mat1))

    H /= H[2][2]

    return H


def h_for_bite_ca(objps, imgps):
    '''
    :param objps: row the same coordinate, collum the same point
    :param imgps: row the same coordinate, collum the same point
    :return:
    '''
    assert objps.shape == imgps.shape, "number of points don't match"
    assert objps.shape[1] >= 3, "number of points isn't enough"

    '''points normalization'''
    m1 = np.mean(objps[:2], axis=1)
    max_std = max(np.std(objps[:2], axis=1)) + 1e-9
    norm_mat1 = np.diag([1 / max_std, 1 / max_std, 1])
    norm_mat1[0][2], norm_mat1[1][2] = -m1[0] / max_std, -m1[1] / max_std

    norm_objps = np.dot(norm_mat1, objps)

    m2 = np.mean(imgps[:2], axis=1)
    norm_mat2 = np.diag([1 / max_std, 1 / max_std, 1])
    norm_mat2[0][2], norm_mat2[1][2] = -m2[0] / max_std, -m2[1] / max_std

    norm_imgps = np.dot(norm_mat2, imgps)

    A = np.concatenate((norm_objps[:2], norm_imgps[:2]), axis=0)
    U, S, V = np.linalg.svd(A.T)

    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:]

    H = np.vstack([np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros(shape=(2, 1))), axis=1), np.array([0, 0, 1])])

    H = np.dot(np.linalg.inv(norm_mat2), np.dot(H, norm_mat1))

    return H
