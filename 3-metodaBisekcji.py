def f(x):
    return x**3 + x - 1

def metodaB(a, b, epsilon):

    if f(a) * f(b) >= 0:
        print("Funkcja nie zmienia znaku w przedziale [a, b].")
        return None
    
    while (b - a) / 2 > epsilon:
        c = (a + b) / 2  # Środek przedziału
        if f(c) == 0:
            return c  # Znaleźliśmy dokładny pierwiastek
        elif f(a) * f(c) < 0:
            b = c  # Pierwiastek jest w przedziale [a, c]
        else:
            a = c  # Pierwiastek jest w przedziale [c, b]
    
    return (a + b) / 2  # Zwracamy przybliżony pierwiastek

a = 0  # Początek przedziału
b = 1  # Koniec przedziału
epsilon = 0.01  # Dokładność

wynik = metodaB(a, b, epsilon)
if wynik is not None:
    print(f"Przybliżona wartość pierwiastka: {wynik}")
