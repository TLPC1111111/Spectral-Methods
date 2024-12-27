import numpy as np
import tqdm
import csv
import sys
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
        self.random_coff = np.array([0.2107 , 0.1841 , 0.2691])  # 随机系数
        self.mean_random_coff = np.mean(self.random_coff)

    def save_results(self , u_true , u):
        '''
        存！
        :param u_true: 真解
        :param u: 数值解
        '''
        with open("results.csv", mode = 'w' , newline = '') as wp:
            write = csv.writer(wp)
            write.writerow(["dt , random_coeff", "true value", "Numerical Value", "Error"])
            for j in range(len(self.random_coff)):
                write.writerow([self.dt, self.random_coff[j], u_true[j], u[j], np.abs(np.array(u_true[j]) - np.array(u[j]))])

    def true_func(self , t , x , y , random_coff):
        return (1 + random_coff) * np.sin(2 * np.pi * x) * np.sin(2* np.pi * y) * np.sin(t)

    def basis_func(self , p , q , x , y):
        '''
        基函数
        :param index_x: x轴Fourier模态索引
        :param index_y: y轴Fourier模态索引
        :param x: x
        :param y: y
        '''
        return np.sin(p * np.pi * x) * np.sin(q * np.pi * y)

    def source_term_func(self , t , x , y , random_coff):
        '''
        源项在这！！！
        :param t: 时间
        :param x: x
        :param y: y
        :param random_coff: 随机系数
        '''
        return (1 + random_coff) * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.cos(t) \
                + 4 * random_coff * (1 + random_coff) * 16 * np.pi**4 * np.sin(t) * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def gauss_legendre_2d_integral(self, f , x_min, x_max, y_min, y_max, n , t , random_coff , p , q):
        '''
        数学王子数值积分！
        :param f: 要求的函数
        :param x_min: x轴积分下限
        :param x_max: x轴积分上限
        :param y_min: y轴积分下线
        :param y_max: y轴积分上限
        :param n: 节点个数
        :return: 积分值
        '''
        # 高斯-勒让德节点和权重
        x, wx = np.polynomial.legendre.leggauss(n)
        y, wy = np.polynomial.legendre.leggauss(n)
        # 将节点从区间[-1, 1]映射到对应积分区域
        x_mapped = 0.5 * (x_max - x_min) * x + 0.5 * (x_max + x_min)
        y_mapped = 0.5 * (y_max - y_min) * y + 0.5 * (y_max + y_min)
        # 计算积分
        integral = 0
        for i in range(n):
            for j in range(n):
                integral += wx[i] * wy[j] * f(t , x_mapped[i], y_mapped[j] , p , q , random_coff)
        integral *= 0.25 * (x_max - x_min) * (y_max - y_min)
        return integral

    def u_init_times_basis_func(self , t , x , y , p , q , random_coff):
        return 0

    def f_times_basis_func(self , t , x , y , p , q , random_coff):
        return self.source_term_func(t , x , y , random_coff) * self.basis_func(p , q , x , y)

    def main_run(self):
        u_coff = np.zeros((3 , self.N , self.N))
        for j in range(len(self.random_coff)):
            for p in range(1 , self.N + 1):
                for q in range(1 , self.N + 1):
                    u_coff[j][p-1][q-1] = 4 * self.gauss_legendre_2d_integral(self.u_init_times_basis_func , self.x_min , self.x_max ,
                                                                           self.y_min , self.y_max , 10 , t = 0 , random_coff = self.random_coff[j] ,
                                                                           p = p , q = q)
        u = np.zeros(len(self.random_coff)) # 初始化u
        for j in range(len(self.random_coff)):
            for t in tqdm.tqdm(range(1 , self.t_steps + 1) , desc = "Processing"):
                for p in range(1 , self.N + 1):
                    for q in range(1 , self.N + 1):
                        u_coff[j][p-1][q-1] = (1 + self.dt * (self.mean_random_coff - self.random_coff[j]) *
                                            ((p * np.pi)**4 + 2 * (p * np.pi)**2 * (q * np.pi)**2 + (q * np.pi)**4)) / \
                                           (1 + self.dt * self.mean_random_coff * ((p * np.pi)**4 + 2 * (p * np.pi)**2 * (q * np.pi)**2 + (q * np.pi)**4)) \
                                            * u_coff[j][p-1][q-1] + 4 * self.dt * \
                                           1 / (1 + self.dt * self.mean_random_coff * ((p * np.pi)**4 + 2 * (p * np.pi)**2 * (q * np.pi)**2 + (q * np.pi)**4)) * \
                                            self.gauss_legendre_2d_integral(self.f_times_basis_func , self.x_min , self.x_max , self.y_min , self.y_max ,
                                                                            10 , t * self.dt , self.random_coff[j] , p , q)
        for j in range(len(self.random_coff)):
            for p in range(1 , self.N + 1):
                for q in range(1 , self.N + 1):
                    u[j] += u_coff[j][p-1][q-1] * self.basis_func(p , q , 0.222 , 0.555)
        # 输出真解：
        u_true = []
        for j in range(len(self.random_coff)):
            u_true.append(self.true_func(1 , 0.222 , 0.555 , self.random_coff[j]))
        # 读取上一次运行的时间步长：
        with open('results.csv' , mode = 'r' , newline = '') as file:
            reader = csv.reader(file)
            next(reader)
            dt_1 = next(reader)[0]
            dt_1 = float(dt_1)
        if(dt_1 == self.dt):
            print("时间步长与上次时间步长相同！请您选择不同的时间步长！")
            sys.exit()
        last_errors = []
        with open('results.csv'  , mode = 'r' , newline = '') as file2:
            reader = csv.reader(file2)
            next(reader)
            for row in reader:
                last_errors.append(float(row[-1]))
        print(f"真解：{u_true}")
        print(f"数值解：{u}")
        print(f"误差：{np.array(u) - np.array(u_true)}")
        print(f"收敛阶：{np.log(np.abs(np.array(u) - np.array(u_true)) / np.abs(np.array(last_errors))) / np.log(self.dt / dt_1)}")
        self.save_results(u_true = u_true , u = u)





