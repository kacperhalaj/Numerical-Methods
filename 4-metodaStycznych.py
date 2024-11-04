import math

def f(x):
    """
    Funkcja, której pierwiastek chcemy znaleźć.
    f(x) = sin(x) - (1/2)x
    """
    return math.sin(x) - 0.5 * x

def df(x):
    """
    Pochodna funkcji f(x).
    df(x) = cos(x) - 1/2
    """
    return math.cos(x) - 0.5

def newton_method(x0, epsilon, max_iterations=1000):
    """
    Funkcja implementująca metodę Newtona-Raphsona.
    
    Parametry:
    x0: początkowe przybliżenie pierwiastka
    epsilon: dokładność obliczeń
    max_iterations: maksymalna liczba iteracji
    
    Zwraca:
    Przybliżoną wartość pierwiastka lub informację o braku zbieżności
    """
    for _ in range(max_iterations):
        fx = f(x0)
        dfx = df(x0)
        
        if dfx == 0:
            print("Pochodna równa zeru. Metoda Newtona-Raphsona nie może być zastosowana.")
            return None
        
        # Obliczamy nowe przybliżenie
        x1 = x0 - fx / dfx
        
        # Sprawdzamy warunek zakończenia
        if abs(x1 - x0) < epsilon:
            return x1
        
        x0 = x1
    
    print("Przekroczono maksymalną liczbę iteracji. Brak zbieżności.")
    return None

# Przykład użycia:
x0 = math.pi / 2  # Początkowe przybliżenie w przedziale [pi/2, pi]
epsilon = 0.01  # Dokładność

root = newton_method(x0, epsilon)
if root is not None:
    print(f"Przybliżona wartość pierwiastka: {root}")








