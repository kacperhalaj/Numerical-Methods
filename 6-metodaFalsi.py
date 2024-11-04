import math

def f(x):
    """
    Funkcja, której pierwiastek chcemy znaleźć.
    f(x) = 3x - cos(x) - 1
    """
    return 3 * x - math.cos(x) - 1

def false_position_method(a, b, epsilon, max_iterations=1000):
    """
    Funkcja implementująca metodę regulacji falsi.
    
    Parametry:
    a, b: przedział, w którym szukamy pierwiastka
    epsilon: dokładność obliczeń
    max_iterations: maksymalna liczba iteracji
    
    Zwraca:
    Przybliżoną wartość pierwiastka lub informację o braku zbieżności
    """
    if f(a) * f(b) >= 0:
        print("Funkcja nie zmienia znaku w przedziale [a, b].")
        return None
    
    for _ in range(max_iterations):
        # Obliczamy nowe przybliżenie
        x_new = b - (f(b) * (b - a)) / (f(b) - f(a))
        
        # Sprawdzamy warunek zakończenia
        if abs(f(x_new)) < epsilon:
            return x_new
        
        # Wybieramy nowy przedział, w którym funkcja zmienia znak
        if f(a) * f(x_new) < 0:
            b = x_new
        else:
            a = x_new
    
    print("Przekroczono maksymalną liczbę iteracji. Brak zbieżności.")
    return None

# Przykład użycia:
a = 0.25  # Początek przedziału
b = 0.75  # Koniec przedziału
epsilon = 0.00001  # Dokładność

root = false_position_method(a, b, epsilon)
if root is not None:
    print(f"Przybliżona wartość pierwiastka: {root}")
