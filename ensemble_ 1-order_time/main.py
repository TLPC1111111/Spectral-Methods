import argparse
import numpy as np
from spectral_methods import SpectralMethods
import time
def Default_arguments():   # 跑到32即可，包含32
    '''
    input parameters that can be provided directly.
    '''
    parser = argparse.ArgumentParser(description="Solving the fourth-order parabolic equation with random coefficients using spectral methods")
    parser.add_argument("--min_x" , type = float , default = 0 , help = "The minimum value of the region D along the x-axis")
    parser.add_argument("--max_x" , type = float , default = 1 , help = "The maximum value of the region D along the x-axis")
    parser.add_argument("--min_y" , type = float , default = 0 , help = "The minimum value of the region D along the y-axis")
    parser.add_argument("--max_y" , type = float , default = 1 , help = "The maximum value of the region D along the y-axis")
    parser.add_argument("--N" , type = int , default = 8 , help = "The number of terms of a Fourier series")
    parser.add_argument("--time_len" , type = float , default = 1 , help = "Total time length")
    parser.add_argument("-dt" , type = float , default = 1/32 , help = "Time step")
    return  parser

def main():  # 重新设置初值条件！
    ed = time.time()
    args = Default_arguments().parse_args()
    method = SpectralMethods(x_min=args.min_x,
                             x_max=args.max_x,
                             y_min=args.min_y,
                             y_max=args.max_y,
                             N=args.N,
                             time_len=args.time_len,
                             dt=args.dt)
    method.main_run()
    cost= time.time() - ed
    print(f"cost:{cost}")

if __name__ == '__main__':
    main()