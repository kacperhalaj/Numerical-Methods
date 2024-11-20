# def lagrange_interpolation(x_values, y_values, x):
#     n = len(x_values)
#     Px = 0  # Wielomian interpolacyjny

#     for i in range(n):
#         # Obliczanie wielomianu bazowego L_i(x)
#         L_i = 1
#         for j in range(n):
#             if j != i:
#                 L_i *= (x - x_values[j]) / (x_values[i] - x_values[j])
        
#         # Dodawanie do sumy
#         Px += y_values[i] * L_i

#     return Px

# # Węzły interpolacyjne
# x_values = [1, 2, 3]
# y_values = [5, 7, 6]
# x = 2.5
# result = lagrange_interpolation(x_values, y_values, x)

# print(f"Wartość funkcji interpolującej w punkcie x = {x}: {result}")




import sympy as sp

x = sp.symbols('x')

x_vals = [1, 2, 3]
y_vals = [5, 7, 6]

# Funkcje Lagrange'a
def lagrange_basis(x_vals, i, x):
    result = 1
    for j in range(len(x_vals)):
        if j != i:
            result *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
    return result

# Wielomian interpolacyjny Lagrange'a
L = 0
for i in range(len(x_vals)):
    L += y_vals[i] * lagrange_basis(x_vals, i, x)

L = sp.simplify(L)
print(f"Wielomian interpolacyjny Lagrange'a: {L}")
