import math

def f(x):
    return math.sin(x) * math.exp(-3 * x) + x**3

def metodaParaboli(a, b, n):
    h = (b - a) / n
    wynik = f(a) + f(b)
    for i in range(1, n, 2):
        wynik += 4 * f(a + i * h)
    for i in range(2, n - 1, 2):
        wynik += 2 * f(a + i * h)
    wynik *= h / 3
    return wynik

a = -3
b = 1
n = 100 
wynik = metodaParaboli(a, b, n)
print(f"Wartość całki metodą Simpsona: {wynik}")

# Błąd można oszacować jako |E| ≈ K * (b - a) * h^4 / 180
# gdzie K to maksimum wartości czwartej pochodnej funkcji na przedziale [a, b].

def czwartapochodna(x):
    # Przybliżenie czwartej pochodnej funkcji f(x) = sin(x) * e^(-3x) + x^3
    return 81 * math.sin(x) * math.exp(-3 * x) - 54 * math.cos(x) * math.exp(-3 * x)

# Szacujemy maksymalną wartość czwartej pochodnej na przedziale
max_K = max(abs(czwartapochodna(a)), abs(czwartapochodna(b)))
blad = max_K * (b - a) * (h ** 4) / 180
print(f"Oszacowany błąd maksymalny: {blad}")
