import numpy as np
import tqdm
import csv
from numba import jit
import sys
import os
class SpectralMethods:
    def __init__(self, x_min, x_max, y_min, y_max, N, time_len, dt):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.N = N
        self.time_len = time_len
        self.dt = dt
        self.t_steps = int(self.time_len / dt)  # 总时间步长
        self.random_coff = np.array(np.loadtxt("random_coef.txt"))
        self.mean_random_coff = np.mean(self.random_coff)
        self.mean_coeff = 3
        self.u_coff = np.zeros((len(self.random_coff), self.N, self.N))
        # 初始化Gauss节点和权重：
        self.n_gauss = 20
        self.x_gauss, self.wx_gauss = np.polynomial.legendre.leggauss(self.n_gauss)
        self.y_gauss, self.wy_gauss = np.polynomial.legendre.leggauss(self.n_gauss)
        self.x_mapped = 0.5 * (x_max - x_min) * self.x_gauss + 0.5 * (x_max + x_min)
        self.y_mapped = 0.5 * (y_max - y_min) * self.y_gauss + 0.5 * (y_max + y_min)
        self.xv, self.yv = np.meshgrid(self.x_mapped, self.y_mapped, indexing='ij')

    @staticmethod
    @jit(nopython=True)
    def true_func(t , x , y , p , q , random_coff):  # 修 设置新初值！
        return (1 + random_coff) * (1 - x)**3 * x**3 * (1 - y)**3 * y**3 * np.exp(2 * np.pi * t)

    @staticmethod
    @jit(nopython=True)
    def source_term_func(t , x , y , random_coff): # 修
        '''
        源项
        :param t: 时间
        :param x: x
        :param y: y
        :param random_coff: 随机系数
        '''
        return ((1 + random_coff) * (1 - x)**3 * x**3 * (1 - y)**3 * y**3 * np.exp(2 * np.pi *t) * 2 * np.pi + \
                random_coff * (1 + random_coff) * np.exp(2 * np.pi *t) * ((1 - y)**3 * y**3 * (-360 * x**2 + 360 * x - 72) +
                                                                           (1 - x)**3 * x**3 * (-360 * y**2 + 360 * y - 72) +
                                                                           2 * (6 * x - 36 * x**2 + 60 * x**3 - 30 * x**4) *
                                                                           (6 * y - 36 * y**2 + 60 * y**3 - 30 * y**4)))

    @staticmethod
    @jit(nopython=True)
    def basis_func(p , q , x , y):
        '''
        基函数
        :param index_x: x轴Fourier模态索引
        :param index_y: y轴Fourier模态索引
        :param x: x
        :param y: y
        '''
        return np.sin(p * np.pi * x) * np.sin(q * np.pi * y)

    def gauss_legendre_2d_integral(self, f , x_min, x_max, y_min, y_max, n , t , random_coff , p = None , q = None):
        '''
        gauss数值积分！
        :param f: 要求的函数
        :param x_min: x轴积分下限
        :param x_max: x轴积分上限
        :param y_min: y轴积分下线
        :param y_max: y轴积分上限
        :param n: 节点个数
        :return: 积分值
        '''
        # 计算积分
        integral_values = f(t, self.xv, self.yv, p, q, random_coff)
        integral = np.sum(self.wx_gauss[:, None] * self.wy_gauss[None, :] * integral_values) * 0.25 * (x_max - x_min) * (y_max - y_min)
        return integral

    def u_init_times_basis_func(self , t , x , y , p , q , random_coff):
        return (1 + random_coff) * (1 - x)**3 * x**3 * (1 - y)**3 * y**3 * self.basis_func(p , q , x, y)

    def f_times_basis_func(self , t , x , y , p , q , random_coff):
        return self.source_term_func(t , x , y , random_coff) * self.basis_func(p , q , x , y)

    def init_time_coff(self):
        for j in range(len(self.random_coff)):
            for p in range(1, self.N + 1):
                for q in range(1, self.N + 1):
                    self.u_coff[j][p - 1][q - 1] = 4 * self.gauss_legendre_2d_integral(self.u_init_times_basis_func,
                                                                                  self.x_min, self.x_max, self.y_min,
                                                                                  self.y_max, 20, t = 0,
                                                                                  random_coff=self.random_coff[j],
                                                                                  p=p, q=q)
        for j in tqdm.tqdm(range(len(self.random_coff)) , desc = "Processing"):
            for t in range(1 , self.t_steps + 1):
                for p in range(1 , self.N + 1):
                    for q in range(1 , self.N + 1):
                        self.u_coff[j][p - 1][q - 1] = (1 + self.dt * (self.mean_random_coff - self.random_coff[j]) *
                                                   ((p * np.pi) ** 4 + 2 * (p * np.pi) ** 2 * (q * np.pi) ** 2 + (
                                                               q * np.pi) ** 4)) / \
                                                  (1 + self.dt * self.mean_random_coff * (
                                                              (p * np.pi) ** 4 + 2 * (p * np.pi) ** 2 * (
                                                                  q * np.pi) ** 2 + (q * np.pi) ** 4)) \
                                                  * self.u_coff[j][p - 1][q - 1] + 4 * self.dt * \
                                                  1 / (1 + self.dt * self.mean_random_coff * (
                                                (p * np.pi) ** 4 + 2 * (p * np.pi) ** 2 * (q * np.pi) ** 2 + (q * np.pi) ** 4)) * \
                                                  self.gauss_legendre_2d_integral(self.f_times_basis_func, self.x_min,
                                                                                  self.x_max, self.y_min, self.y_max,
                                                                                  20, t * self.dt, self.random_coff[j] , p, q)

    def ensamble_u(self , t_1 , x , y , p , q):
        u = []
        for j in range(len(self.random_coff)):
            u_j = 0
            for p in range(1 , self.N + 1):
                for q in range(1 , self.N + 1):
                    u_j += self.u_coff[j][p-1][q-1] * self.basis_func(p , q , x , y)
            u.append(u_j)
        return np.array(u)

    def error_func(self , t , x , y , p , q , random_coff):
        true_val = self.true_func(t, x, y, p, q, self.mean_coeff)
        ensemble_u = np.mean(self.ensamble_u(t , x , y , p , q) , axis = 0)
        return (true_val - ensemble_u)**2

    def L2_error_func(self):
        return np.sqrt(self.gauss_legendre_2d_integral(self.error_func , self.x_min , self.x_max , self.y_min , self.y_max ,
                                                                            20 , self.time_len , self.random_coff))

    def save_results(self):
        '''
        存！
        :param u_true: 真解
        :param u: 数值解
        '''
        with open("results.csv", mode = 'w' , newline = '') as wp:
            write = csv.writer(wp)
            write.writerow(["dt" , "Error" , "N"])
            write.writerow([self.dt, self.L2_error_func() , self.N])

    def main_run(self):
        self.init_time_coff()
        # L2误差分析：
        error = self.L2_error_func()
        print(error)
        # self.save_results()

        # 读取上一次运行的时间步长：
        # with open('results.csv' , mode = 'r' , newline = '') as file:
        #     reader = csv.reader(file)
        #     next(reader)
        #     dt_1 = next(reader)[0]
        #     dt_1 = float(dt_1)
        # if(dt_1 == self.dt):
        #     print("时间步长与上次时间步长相同！请您选择不同的时间步长！")
        #     sys.exit()

        last_errors = []

        # with open('results.csv' , mode = 'r' , newline = '') as file3:
        #     reader = csv.reader(file3)
        #     next(reader)
        #     N_1 = next(reader)[-1]
        #     N_1 = int(N_1)
        # if(N_1 == self.N):
        #     print("N与上次相同！！")
        #     sys.exit()

        # with open('results.csv'  , mode = 'r' , newline = '') as file2:
        #     reader = csv.reader(file2)
        #     next(reader)
        #     for row in reader:
        #         error_list = row[-2]
        #         error_list = error_list.strip("[]")
        #         float_error_list = np.fromstring(error_list , sep = " ")
        #         last_errors.append(float_error_list)
        #
        # print(f"时间收敛阶：{np.log(np.abs(error) / np.abs(np.array(last_errors))) / np.log(self.dt / dt_1)}")
        # # print(f"空间收敛阶：{np.log(np.abs(error) / np.abs(np.array(last_errors))) / np.log(self.N / N_1)}")
        # self.save_results()

        file_name = 'saved_results.csv'
        file_exists = os.path.exists(file_name)

        with open(file_name, mode='a', newline='') as wp:
            writer = csv.writer(wp)
            if not file_exists:  # 只有文件不存在时才写入表头
                writer.writerow(["dt", "Error"])
            writer.writerow([error])  # 追加数据


