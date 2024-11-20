import numpy as np

x_values = np.array([0, 3, 6, 9, 12])  # Współrzędne x
y_values = np.array([4, 5, 4, 1, 2])   # Współrzędne y
m = 3  # Stopień wielomianu

# Tworzenie macierzy Vandermonde'a (do aproksymacji)
A = np.vander(x_values, m + 1, increasing=True)

# Rozwiązywanie układu równań A * a = y
coefficients = np.linalg.lstsq(A, y_values, rcond=None)[0]

# Wypisanie wyników
print("Współczynniki wielomianu aproksymującego (od a_0 do a_3):")
for i, coeff in enumerate(coefficients):
    print(f"a_{i} = {coeff:.4f}")

# Wyświetlenie funkcji aproksymującej
print("\nFunkcja aproksymująca:")
print("P(x) =", " + ".join(f"{coeff:.4f}*x^{i}" for i, coeff in enumerate(coefficients)))





# import sympy as sp

# # Wartości węzłów (x) oraz wartości funkcji (y)
# x_values = [0, 3, 6, 9, 12]
# y_values = [4, 5, 4, 1, 2]

# # Stopień wielomianu
# m = 3

# # Zmienna
# x = sp.symbols('x')

# # Tworzymy macierz Vandermonde'a (do aproksymacji)
# A = [[x_val**i for i in range(m + 1)] for x_val in x_values]

# # Tworzymy wektor y
# y = sp.Matrix(y_values)

# # Tworzymy macierz A w formacie sympy
# A_matrix = sp.Matrix(A)

# # Rozwiązujemy układ równań A * a = y (A_matrix * coeffs = y)
# coeffs = A_matrix.LUsolve(y)

# # Wypisanie współczynników
# print("Współczynniki wielomianu aproksymującego (od a_0 do a_3):")
# for i, coeff in enumerate(coeffs):
#     print(f"a_{i} = {coeff:.4f}")

# # Budujemy funkcję aproksymującą P(x)
# P = sum(coeff * x**i for i, coeff in enumerate(coeffs))

# # Wyświetlenie funkcji aproksymującej
# print("\nFunkcja aproksymująca:")
# print("P(x) =", P)

