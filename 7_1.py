import numpy as np
import matplotlib.pyplot as plt
f0 = 1e6
c_true = 1500
alpha = np.radians(60)
t = np.linspace(0, 3600, 5000)
dt = t[1] - t[0]
v_true = np.zeros_like(t)

mask1 = t <= 600 # 1 участок
v_true[mask1] = (1.2 / 600) * t[mask1]
mask2 = (t > 600) & (t <= 2400) # 2 участок
v_true[mask2] = 1.2
mask3 = t > 2400 # 3 участок
v_true[mask3] = 1.2 * (1 - (t[mask3] - 2400) / 1200)
x_true = np.cumsum(v_true) * dt

np.random.seed(42) #Блок погрешностей
gamma_deg = 2.0
cos_gamma = np.cos(np.radians(gamma_deg))
delta_f_noise = np.random.normal(0, 3, len(t))
fd_izmer = (4 * v_true * cos_gamma * f0 * np.sin(alpha)) / c_true + delta_f_noise
T_water = 3.0
c_hat = 1500 + 3.0 * (T_water - 15)  # 1464 м/с
v_izmer = (fd_izmer * c_hat) / (4 * f0 * np.sin(alpha))
S_d = 0.001
b_d = 0.02
v_izmer = (1 + S_d) * v_izmer + b_d
v_izmer = np.maximum(v_izmer, 0)
x_izmer = np.cumsum(v_izmer) * dt
x_error = x_izmer - x_true
error_v = v_izmer - v_true

fig, axes = plt.subplots(1, 1, figsize=(15, 5))
axes.plot(t / 60, v_true, 'g-', linewidth=2, label='Истинная скорость', alpha=0.9)
axes.plot(t / 60, v_izmer, 'r-', linewidth=1.5, label='Измеренная скорость', alpha=0.8)
axes.fill_between(t / 60, v_true, v_izmer, alpha=0.2, color='red')
axes.set_xlabel('Время, минуты', fontsize=10)
axes.set_ylabel('Скорость, м/с', fontsize=10)
axes.set_title('Уход показаний скорости', fontsize=11)
axes.legend(fontsize=9)
axes.grid(True, alpha=0.3)
axes.set_xlim(0, 60)
axes.set_ylim(0, 1.4)
fig, axes = plt.subplots(1, 1, figsize=(15, 5))
v_range = np.linspace(0, 1.5, 100)
fd_range = (4 * v_range * f0 * np.sin(alpha)) / c_true
axes.plot(v_range, fd_range, 'b-', linewidth=2.5)
axes.set_xlabel('Скорость БПА, м/с', fontsize=10)
axes.set_ylabel('Разность частот f_d, Гц', fontsize=10)
axes.set_title('Зависимость скорости от разности частот', fontsize=11)
axes.grid(True, alpha=0.3)
fig, axes = plt.subplots(1, 1, figsize=(15, 5))
axes.plot(t / 60, x_error, 'r-', linewidth=1.8, label='Ошибка позиционирования')
axes.axhline(y=0, color='k', linestyle='--', linewidth=1)
axes.fill_between(t / 60, 0, x_error, where=(x_error > 0), alpha=0.3, color='red', label='Завышение')
axes.fill_between(t / 60, 0, x_error, where=(x_error < 0), alpha=0.3, color='blue', label='Занижение')
axes.set_xlabel('Время, минуты', fontsize=10)
axes.set_ylabel('Ошибка координаты, м', fontsize=10)
axes.set_title('Накопление ошибки позиционирования\n(интеграл ошибки скорости)', fontsize=11)
axes.legend(fontsize=8)
axes.grid(True, alpha=0.3)
axes.set_xlim(0, 60)
plt.tight_layout()
plt.show()

print("Результаты моделирования")
print(f"\nИстинная скорость:")
print(f"  Средняя ошибка скорости: {np.mean(error_v):.4f} м/с")
print(f"  СКО ошибки скорости: {np.std(error_v):.4f} м/с")
print(f"  Максимальная ошибка скорости: {np.max(np.abs(error_v)):.4f} м/с")
print(f"  Истинное пройденное расстояние: {x_true[-1]/1000:.2f} км")
print(f"  Измеренное расстояние: {x_izmer[-1]/1000:.2f} км")
print(f"  Абсолютная ошибка позиции: {x_error[-1]:.1f} м")
print(f"  Относительная ошибка: {100 * abs(x_error[-1]/x_true[-1]):.2f}%")