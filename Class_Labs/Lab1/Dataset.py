import numpy as np

def dataset(N):
    # class 0
    xy_c1 = np.random.randn(N, 2)  # 利用标准正态分布创建N行2列的矩阵
    x1 = xy_c1[:, 0].T
    y1 = xy_c1[:, 1].T
    data_0 = [(0, x, y) for x, y in zip(x1, y1)]

    # class 1
    xy_c2 = np.random.randn(2 * N, 2)  # 利用标准正态分布创建2N行2列的矩阵
    x2 = 2 * xy_c2[:, 0].T
    y2 = 2 * xy_c2[:, 1].T
    data_1 = [(1, x, y) for x, y in zip(x2, y2) if x ** 2 + y ** 2 > 4 and x + y < 0]

    data = np.concatenate((data_0, data_1), 0)
    np.random.shuffle(data)

    return data
