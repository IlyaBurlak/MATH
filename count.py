import matplotlib.pyplot as plt
import numpy as np

# Данные
x = [1, 2, 3, 4, 5, 6, 7]
y = [62.1, 87.2,	109.3,	127.3,	134.7,	136.2,	126.9]

# Функция для аппроксимации
def approx_func(x):
    return x/(0.009 + 0.005 * x)

# Создание значений для графика функции
t_values = np.linspace(0, 10, 100)
z_values = approx_func(t_values)

# Построение графика
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'ro', label='Исходные данные')
plt.plot(t_values, z_values, label='Аппроксимация')

plt.xlabel('t')
plt.ylabel('z')
plt.title('График аппроксимации функции и данных')
plt.legend()
plt.grid(True)
plt.show()