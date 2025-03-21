import numpy as np

mean_value = 3.0
deviation = 0.3

while True:
    numbers = np.random.uniform(mean_value - deviation, mean_value + deviation, 10000)
    mean_calculated = np.mean(numbers)
    max_deviation = np.max(np.abs(numbers - mean_calculated))
    min_value = np.min(numbers)
    if max_deviation < min_value:
        break

filename = "random_coef.txt"
np.savetxt(filename, numbers, fmt="%.6f")
print(f"数据已存入 {filename}")

