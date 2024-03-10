import numpy as np
from scipy.stats import mannwhitneyu, norm
from sympy import symbols, diff, log

# 1. Генерация параметров для нормального распределения
mu = np.random.uniform(0, 10)  # диапазон для mu
sigma2 = np.random.uniform(1, 5)  # случайное значение sigma^2 в интервале [1, 5]
sigma = np.sqrt(sigma2)

# 2. Генерация Yn точек из нормального распределения
n = 1000
Yn = np.random.normal(mu, sigma, n)

# 3. Определение функции g и её производной
x = symbols('x')
g = (x**2 + 3*x)/(x + 2) + sin(x-1)/log(x + 5)
g_prime = diff(g, x)

# 4. Вычисление g(mu) и g'(mu)
g_mu = g.subs(x, mu)
g_prime_mu = g_prime.subs(x, mu)
var_g = (g_prime_mu**2) * sigma2

# 5. Вычисление значений g(Yn) для всех точек Yn
g_Yn = [float(g.subs(x, y)) for y in Yn]  # Преобразование к float сразу при вычислении

# 6. Генерация выборки из нормального распределения
normal_sample = np.random.normal(float(g_mu), np.sqrt(float(var_g)), n)

# 7. Применение U-теста
g_Yn_array = np.array(g_Yn)
u_stat, p_val_u = mannwhitneyu(g_Yn_array, normal_sample, alternative='two-sided')

print("Статистика U:", u_stat)
print("P-value:", p_val_u)




