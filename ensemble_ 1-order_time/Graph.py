import numpy as np
import matplotlib.pyplot as plt

# 时间步长 (Δt)
dt_values = np.array([1/8, 1/16, 1/32, 1/64, 1/128])

# 对应的 L2 误差 (E1, E2, E3)
epsilon_E1 = np.array([2.2952e-3, 1.0956e-3, 5.3446e-4, 2.6385e-4, 1.3107e-4])
epsilon_E2 = np.array([9.1622e-3, 4.3275e-3, 2.0974e-3, 1.0318e-3, 5.1156e-4])
epsilon_E3 = np.array([8.0752e-3, 3.9189e-3, 1.9297e-3, 9.5742e-4, 4.7685e-4])

# 计算 log(Δt) 和 log(误差)
log_dt = np.log(dt_values)
log_epsilon_E1 = np.log(epsilon_E1)
log_epsilon_E2 = np.log(epsilon_E2)
log_epsilon_E3 = np.log(epsilon_E3)

# 创建一个3张图的子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 绘制第一个误差图：ε_L2^E,1
axes[0].plot(log_dt, log_epsilon_E1, 'o-', label='ε_L2^E,1', color='blue')
axes[0].plot(log_dt, log_dt, '--', label='Reference Line: slope = 1', color='red')
axes[0].set_xlabel('log(Δt)')
axes[0].set_ylabel('log(ε_L2^E,1)')
axes[0].set_title('Error vs. Time Step for ε_L2^E,1')
axes[0].grid(True)
axes[0].legend()

# 绘制第二个误差图：ε_L2^E,2
axes[1].plot(log_dt, log_epsilon_E2, 's-', label='ε_L2^E,2', color='green')
axes[1].plot(log_dt, log_dt, '--', label='Reference Line: slope = 1', color='red')
axes[1].set_xlabel('log(Δt)')
axes[1].set_ylabel('log(ε_L2^E,2)')
axes[1].set_title('Error vs. Time Step for ε_L2^E,2')
axes[1].grid(True)
axes[1].legend()

# 绘制第三个误差图：ε_L2^E,3
axes[2].plot(log_dt, log_epsilon_E3, 'd-', label='ε_L2^E,3', color='orange')
axes[2].plot(log_dt, log_dt, '--', label='Reference Line: slope = 1', color='red')
axes[2].set_xlabel('log(Δt)')
axes[2].set_ylabel('log(ε_L2^E,3)')
axes[2].set_title('Error vs. Time Step for ε_L2^E,3')
axes[2].grid(True)
axes[2].legend()

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()
