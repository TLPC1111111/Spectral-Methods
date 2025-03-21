# import numpy as np
# def gauss_legendre_2d_integral(f, x_min, x_max, y_min, y_max, n, t, random_coff, p=None, q=None):
#     '''
#     gauss数值积分！
#     :param f: 要求的函数
#     :param x_min: x轴积分下限
#     :param x_max: x轴积分上限
#     :param y_min: y轴积分下线
#     :param y_max: y轴积分上限
#     :param n: 节点个数
#     :return: 积分值
#     '''
#     # 高斯-勒让德节点和权重
#     x, wx = np.polynomial.legendre.leggauss(n)
#     y, wy = np.polynomial.legendre.leggauss(n)
#     # 将节点从区间[-1, 1]映射到对应积分区域
#     x_mapped = 0.5 * (x_max - x_min) * x + 0.5 * (x_max + x_min)
#     y_mapped = 0.5 * (y_max - y_min) * y + 0.5 * (y_max + y_min)
#     xv, yv = np.meshgrid(x_mapped, y_mapped, indexing='ij')
#     # 计算积分
#     integral_values = f(t, xv, yv, p, q, random_coff)
#     integral = np.sum(wx[:, None] * wy[None, :] * integral_values) * 0.25 * (x_max - x_min) * (y_max - y_min)
#     return integral
#
# def func(t , x , y , p , q , random_coeff):
#     return t * (x + y)
#
# result = gauss_legendre_2d_integral(func , 0 , 1 , 0 , 1 , 20 , 1 , 3)
#
# print(result)
from multiprocessing import Pool

import numpy as np
x_max = 1
y_max = 1
x_min = 0
y_min = 0
n_gauss = 20
x_gauss, wx_gauss = np.polynomial.legendre.leggauss(n_gauss)
y_gauss, wy_gauss = np.polynomial.legendre.leggauss(n_gauss)
x_mapped = 0.5 * (x_max - x_min) * x_gauss + 0.5 * (x_max + x_min)
y_mapped = 0.5 * (y_max - y_min) * y_gauss + 0.5 * (y_max + y_min)
xv, yv = np.meshgrid(x_mapped, y_mapped, indexing='ij')


def compute_chunk(chunk, x_mapped, y_mapped, wx_gauss, wy_gauss, f, t, p, q, random_coff):
    integral_chunk = 0
    for i in chunk:
        x = x_mapped[i]
        for j in range(len(y_mapped)):
            y = y_mapped[j]
            integral_chunk += wx_gauss[i] * wy_gauss[j] * f(t, x, y, p, q, random_coff)
    return integral_chunk
def gauss_legendre_2d_integral(f, x_min, x_max, y_min, y_max, n, t, random_coff, p=None, q=None):
    # 将积分点分配到多个进程
    # 将积分点分成若干组
    chunk_size = n_gauss // 4  # 每组的大小
    chunks = [range(i, min(i + chunk_size, n_gauss)) for i in range(0, n_gauss, chunk_size)]

    # 使用多进程计算
    with Pool(processes=4) as pool:
        results = pool.map(compute_chunk, chunks)

    # 合并结果
    integral = sum(results) * 0.25 * (x_max - x_min) * (y_max - y_min)
    return integral

def func(t , x , y , p , q , random_coeff):
    return t * (x + y)

result = gauss_legendre_2d_integral(func , x_min ,x_max , y_min , y_max , 20 , 0 , 0)
print(result)