from collections import Counter
from math import erf

import matplotlib.pyplot as plt
import numpy as np


class StatisticsCalculator:

    def __init__(self, data, interval_width=0.02):
        self.data = data
        self.interval_width = interval_width
        self.data_min = min(self.data)
        self.data_max = max(self.data)
        self.num_intervals = int((self.data_max - self.data_min) / self.interval_width) + 1
        self.interval_dict = {i: 0 for i in range(self.num_intervals)}
        self.cumulative_frequency = {}
        self.midpoints = {i: round(self.data_min + (i + 0.5) * self.interval_width, 2) for i in
                          range(self.num_intervals)}
        self.relative_frequency = {}

    def calculate_interval_data(self):
        for num in self.data:
            interval = int((num - self.data_min) // self.interval_width)
            self.interval_dict[interval] += 1

    def calculate_cumulative_freq(self):
        cumulative_freq = 0
        for i, freq in self.interval_dict.items():
            cumulative_freq += freq
            self.cumulative_frequency[i] = cumulative_freq

    def calculate_relative_freq(self):
        total_data_points = len(self.data)
        self.relative_frequency = {i: freq / total_data_points for i, freq in self.interval_dict.items()}

    def print_variational_series(self):
        variational_series = dict(Counter(self.data))
        sorted_variational_series = dict(sorted(variational_series.items()))
        print("\nВариационный ряд:")
        for key, value in sorted_variational_series.items():
            print(f"{key}: {value}")

    def print_interval_data(self):
        print("\nИнтервальные данные:")
        for i, freq in self.interval_dict.items():
            interval_start = round(self.data_min + i * self.interval_width, 2)
            interval_end = round(self.data_min + (i + 1) * self.interval_width, 2)
            print(f"{interval_start} - {interval_end}: {freq}")

    def print_midpoints(self):
        print("\nСредние точки интервалов:")
        for i, midpoint in self.midpoints.items():
            print(f"{midpoint}: {self.interval_dict[i]}")

    def print_cumulative_freq(self):
        print("\nНакопленная частота:")
        for i, cum_freq in self.cumulative_frequency.items():
            interval_start = round(self.data_min + i * self.interval_width, 2)
            interval_end = round(self.data_min + (i + 1) * self.interval_width, 2)
            print(f"{interval_start} - {interval_end}: {cum_freq}")

    def print_relative_freq(self):
        print("\nОтносительная частота:")
        for i, rel_freq in self.relative_frequency.items():
            interval_start = round(self.data_min + i * self.interval_width, 2)
            interval_end = round(self.data_min + (i + 1) * self.interval_width, 2)
            print(f"{interval_start} - {interval_end}: {rel_freq}")

    def calculate_relative_freq_density(self):
        bin_counts = list(self.interval_dict.values())
        bin_width = self.interval_width
        total_data_points = len(self.data)
        self.relative_freq_density = {i: freq / (total_data_points * bin_width) for i, freq in enumerate(bin_counts)}

    def print_relative_freq_density(self):
        print("\nОтносительная плотность частот:")
        for i, freq_density in self.relative_freq_density.items():
            interval_start = round(self.data_min + i * self.interval_width, 2)
            interval_end = round(self.data_min + (i + 1) * self.interval_width, 2)
            print(f"{interval_start} - {interval_end}: {freq_density}")

    def plot_relative_freq_histogram(self):
        plt.bar(list(self.midpoints.values()), list(self.relative_frequency.values()), width=self.interval_width,
                linewidth=2)
        plt.xlabel('Средние значения интервалов')
        plt.ylabel('Относительная частота')
        plt.title('Гистограмма относительной частоты')
        plt.grid(True)
        plt.show()

    def plot_relative_freq_density_histogram(self):
        plt.bar(list(self.midpoints.values()), list(self.relative_freq_density.values()), width=self.interval_width,
                linewidth=2)
        plt.xlabel('Средние значения интервалов')
        plt.ylabel('Плотность относительной частоты')
        plt.title('Гистограмма плотности относительной частоты')
        plt.grid(True)
        plt.show()

    def calculate_moments(self, order):
        moments = {}
        for k in range(1, order + 1):
            raw_moment = sum([num ** k for num in self.data]) / len(self.data)
            moments[f'Момент порядка {k}'] = raw_moment

            mean = sum(self.data) / len(self.data)
            central_moment = sum([(num - mean) ** k for num in self.data]) / len(self.data)
            moments[f'Центральный момент {k}'] = central_moment

        print("\nМоменты:")
        for key, value in moments.items():
            print(f"{key}: {value}")

        return moments

    # Расчет асимметрии
    def calculate_skewness(self):
        mean = np.mean(self.data)
        variance = np.var(self.data)
        std_dev = np.sqrt(variance)

        moment_3 = np.sum(((self.data - mean) / std_dev) ** 3) / len(self.data)
        skewness = moment_3

        print(f"\nАсимметрия: {skewness}")

        return skewness

    # Расчет эксцесса
    def calculate_kurtosis(self):
        mean = np.mean(self.data)
        variance = np.var(self.data)
        std_dev = np.sqrt(variance)

        moment_4 = np.sum(((self.data - mean) / std_dev) ** 4) / len(self.data)
        kurtosis = moment_4 - 3

        print(f"\nЭксцесс: {kurtosis}")

        return kurtosis

    def calculate_within_3_sigma(self):
        mean = np.mean(self.data)
        std_dev = np.std(self.data)

        lower_bound = mean - 3 * std_dev
        upper_bound = mean + 3 * std_dev

        within_3_sigma = [num for num in self.data if lower_bound <= num <= upper_bound]
        percentage_within_3_sigma = len(within_3_sigma) / len(self.data) * 100

        print(f"\nПроцент значений в пределах 3 стандартных отклонений: {percentage_within_3_sigma:.2f}%")

    def calculate_empirical_distribution_function(self):
        sorted_data = sorted(self.data)
        n = len(self.data)
        x_unique = sorted(set(sorted_data))
        edf = []
        cumulative_sum = 0
        for val in x_unique:
            count_val = sorted_data.count(val)
            cumulative_sum += count_val
            edf_val = cumulative_sum / n
            edf.append(edf_val)

        print("\nЭмпирическая функция распределения:")
        for i in range(len(x_unique)):
            print(f"{x_unique[i]}: {edf[i]}")

        return x_unique, edf

    def plot_empirical_distribution_function(self):
        x_unique, edf = self.calculate_empirical_distribution_function()

        plt.step(x_unique, edf, where='post', linewidth=2)
        plt.xlabel('Значения данных')
        plt.ylabel('Эмпирическая функция распределения')
        plt.title('График эмпирической функции распределения (Дискретная)')
        plt.grid(True)
        plt.show()

    def plot_relative_freq_histogram_and_polygon(self):
        plt.bar(list(self.midpoints.values()), list(self.relative_frequency.values()), width=self.interval_width,
                linewidth=2, alpha=1, label='Гистограмма относительной частоты')

        plt.plot(list(self.midpoints.values()), list(self.relative_frequency.values()), marker='o',
                 color='r', linestyle='-', linewidth=2, label='Полигон относительной частоты')

        plt.xlabel('Средние значения интервалов')
        plt.ylabel('Относительная частота')
        plt.title('Гистограмма и полигон относительной частоты')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_relative_freq_density_histogram_with_probabilities(self):
        plt.figure(figsize=(12, 6))

        # Отобразить относительную плотность частот в качестве основной гистограммы
        plt.bar(list(self.midpoints.values()), list(self.relative_freq_density.values()), width=self.interval_width,
                linewidth=2, label='Относительная плотность частот')

        # Рассчитать вероятности для каждого интервала
        probabilities = [freq * self.interval_width for freq in self.relative_freq_density.values()]

        # Постройте вторую гистограмму с вероятностями
        plt.bar(list(self.midpoints.values()), probabilities, width=self.interval_width,
                alpha=0.7, label='Вероятность попадания в интервал')

        print("\n")
        for idx, prob in enumerate(probabilities):
            plt.text(list(self.midpoints.values())[idx], list(self.relative_freq_density.values())[idx] + prob / 2,
                     f' {prob:.3f}', ha='center', va='center')

        for idx, prob in enumerate(probabilities):
            print(f'Интервал {list(self.midpoints.keys())[idx]}: Вероятность = {prob}')

        # Рассчитать среднее и стандартное отклонение для нормальной кривой
        mean = np.mean(self.data)
        std_dev = np.std(self.data)

        # Постройте нормальную кривую на том же графике
        x = np.linspace(self.data_min, self.data_max, 100)
        y = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        plt.plot(x, y, label='Нормальное распределение', color='r', linestyle='--')

        plt.xlabel('Средние значения интервалов')
        plt.ylabel('Плотность относительной частоты / Вероятность')
        plt.title('Гистограмма плотности относительной частоты и вероятности попадания в интервалы')
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_empirical_and_theoretical_distribution_functions(self):
        x_unique, edf = self.calculate_empirical_distribution_function()
        data, tdf = self.calculate_theoretical_distribution_function()

        plt.step(x_unique, edf, where='post', linewidth=2, label='Эмпирическая функция распределения (EFР)')
        plt.plot(data, tdf, label='Теоретическая функция распределения (TFP)', color='r', linestyle='--')

        plt.xlabel('Значения данных')
        plt.ylabel('Вероятность')
        plt.title('Эмпирическая vs Теоретическая функции распределения')
        plt.legend()
        plt.grid(True)
        plt.show()


    def calculate_theoretical_distribution_function(self, distribution_type='normal'):
        if distribution_type == 'normal':
            mean = np.mean(self.data)
            std_dev = np.std(self.data)

            x_values = np.linspace(self.data_min, self.data_max, self.num_intervals * 10)  # генерируем больше точек для более плавной кривой
            if self.num_intervals == 1:
                # Для данных с только одним уникальным значением нельзя создать кривую распределения
                return None, None

            tdf = []
            for x in x_values:
                if std_dev != 0:
                    tdf_val = 0.5 * (1 + erf((x - mean) / (std_dev * np.sqrt(2))))
                else:
                    tdf_val = 0.0
                tdf.append(tdf_val)

            return x_values, tdf
        else:
            return None, None

# Usage
data = [6.75, 6.77, 6.77, 6.73, 6.76, 6.74, 6.7, 6.75, 6.71, 6.72,
        6.77, 6.79, 6.71, 6.78, 6.73, 6.7, 6.73, 6.77, 6.75, 6.74,
        6.71, 6.7, 6.78, 6.76, 6.81, 6.69, 6.8, 6.8, 6.77, 6.68,
        6.74, 6.7, 6.7, 6.74, 6.77, 6.83, 6.76, 6.76, 6.82, 6.77,
        6.71, 6.74, 6.77, 6.75, 6.74, 6.75, 6.77, 6.72, 6.74, 6.8,
        6.75, 6.8, 6.72, 6.78, 6.7, 6.75, 6.78, 6.78, 6.78, 6.77,
        6.74, 6.74, 6.77, 6.73, 6.74, 6.77, 6.73, 6.74, 6.75, 6.74,
        6.76, 6.76, 6.74, 6.74, 6.74, 6.74, 6.76, 6.74, 6.72, 6.8,
        6.76, 6.78, 6.73, 6.7, 6.76, 6.76, 6.77, 6.75, 6.78, 6.72,
        6.76, 6.78, 6.68, 6.75, 6.73, 6.82, 6.73, 6.8, 6.81, 6.71,
        6.82, 6.77, 6.8, 6.8, 6.7, 6.7, 6.82, 6.72, 6.69, 6.73,
        6.76, 6.74, 6.77, 6.72, 6.76, 6.78, 6.78, 6.73, 6.76, 6.8,
        6.76, 6.72, 6.76, 6.76, 6.74, 6.73, 6.75, 6.77, 6.77, 6.7,
        6.81, 6.74, 6.73, 6.77, 6.74, 6.78, 6.69, 6.74, 6.71, 6.76,
        6.76, 6.77, 6.81, 6.74, 6.74, 6.77, 6.75, 6.8, 6.74, 6.76,
        6.77, 6.77, 6.81, 6.75, 6.78, 6.73, 6.76, 6.76, 6.76, 6.77,
        6.76, 6.8, 6.77, 6.74, 6.77, 6.72, 6.75, 6.76, 6.77, 6.81,
        6.76, 6.76, 6.76, 6.8, 6.74, 6.8, 6.74, 6.73, 6.75, 6.77,
        6.74, 6.76, 6.77, 6.77, 6.75, 6.76, 6.74, 6.82, 6.76, 6.73,
        6.74, 6.75, 6.76, 6.72, 6.78, 6.72, 6.76, 6.77, 6.75, 6.78]

# Usage
calc = StatisticsCalculator(data)
calc.calculate_interval_data()
calc.calculate_cumulative_freq()
calc.calculate_relative_freq()
calc.calculate_relative_freq_density()
calc.calculate_moments(4)
calc.calculate_skewness()
calc.calculate_kurtosis()

# Printing results
calc.print_variational_series()
calc.print_interval_data()
calc.print_midpoints()
calc.calculate_empirical_distribution_function()
calc.print_cumulative_freq()
calc.print_relative_freq()
calc.print_relative_freq_density()
calc.calculate_within_3_sigma()

# Plottin
calc.plot_empirical_distribution_function()
calc.plot_relative_freq_histogram_and_polygon()
calc.plot_relative_freq_density_histogram_with_probabilities()
calc.plot_empirical_and_theoretical_distribution_functions()
