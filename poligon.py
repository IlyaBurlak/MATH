import matplotlib.pyplot as plt
import numpy as np

# Данные
t = [1, 3, 4, 6, 7]
z = [0.54, 0.927, 1.148, 1.518, 1.652]

# Функция для аппроксимации
def approx_func(t):
    return t/(1.41 + 0.48*t)

# Создание значений для графика функции
t_values = np.linspace(0, 10, 100)
z_values = approx_func(t_values)

# Построение графика
plt.figure(figsize=(8, 6))
plt.plot(t, z, 'ro', label='Исходные данные')
plt.plot(t_values, z_values, label='Аппроксимация')

plt.xlabel('t')
plt.ylabel('z')
plt.title('График аппроксимации функции и данных')
plt.legend()
plt.grid(True)
plt.show()