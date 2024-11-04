def f(x):
    """
    Funkcja, której pierwiastek chcemy znaleźć.
    f(x) = x^3 + x^2 - 3x - 3
    """
    return x**3 + x**2 - 3*x - 3

def secant_method(x0, x1, epsilon, max_iterations=1000):
    """
    Funkcja implementująca metodę siecznych.
    
    Parametry:
    x0, x1: dwa początkowe przybliżenia pierwiastka
    epsilon: dokładność obliczeń
    max_iterations: maksymalna liczba iteracji
    
    Zwraca:
    Przybliżoną wartość pierwiastka lub informację o braku zbieżności
    """
    for _ in range(max_iterations):
        f_x0 = f(x0)
        f_x1 = f(x1)
        
        if f_x1 - f_x0 == 0:
            print("Dzielenie przez zero. Metoda siecznych nie może być zastosowana.")
            return None
        
        # Obliczamy nowe przybliżenie
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        
        # Sprawdzamy warunek zakończenia
        if abs(x2 - x1) < epsilon:
            return x2
        
        # Aktualizujemy wartości do następnej iteracji
        x0, x1 = x1, x2
    
    print("Przekroczono maksymalną liczbę iteracji. Brak zbieżności.")
    return None

# Przykład użycia:
x0 = 1  # Początkowe przybliżenie
x1 = 2  # Drugie początkowe przybliżenie
epsilon = 0.0001  # Dokładność

root = secant_method(x0, x1, epsilon)
if root is not None:
    print(f"Przybliżona wartość pierwiastka: {root}")
