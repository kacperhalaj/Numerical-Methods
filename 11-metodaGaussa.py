import numpy as np

# Dane
A = np.array([[3, 0, 6], [1, 2, 8], [4, 5, -2]], dtype=float)
b = np.array([-12, -12, 39], dtype=float)

# Eliminacja Gaussa
def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        # Pivoting
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        
        # Eliminate
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

# Rozkład LU
def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()
    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            U[j, i:] -= factor * U[i, i:]
            L[j, i] = factor
    return L, U

# Obliczenia
x = gauss_elimination(A.copy(), b.copy())
L, U = lu_decomposition(A)

print("Rozwiązanie metodą Gaussa:", x)
print("Macierz L:\n", L)
print("Macierz U:\n", U)
