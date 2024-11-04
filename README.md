# Numerical-Methods

# MetodyNumeryczne

Implementacjom algorytmów matematycznych w języku Python.

### Tematy:

#### [Lista zadań](https://github.com/kacperhalaj/Numerical-Methods/blob/main/ZADANIA.md)

#### ZADANIA
- [horner wyliczanie wartosci w danym punkcie](https://github.com/kacperhalaj/Numerical-Methods/blob/main/1-schematHornera.py)
    - kod:

```python
def horner(lista, x):
    wynik = lista[0]  # Zaczynamy od najwyższego współczynnika
    # Iterujemy po pozostałych współczynnikach
    for element in lista[1:]:
        wynik = wynik * x + element

    return wynik

lista = [2, -6, 2, -1]  # 2x^3 - 6x^2 + 2x - 1
x = 3  # Punkt
wartosc = horner(lista, x)
print(f"Wartość wielomianu w punkcie {x}: {wartosc}")
```
- [horner dzielenie wielomianu](https://github.com/kacperhalaj/Numerical-Methods/blob/main/2-dzielenieWielomianu.py)
  - kod:

```python
def horner_dzielenie(lista, c):
    n = len(lista) - 1  # Stopień wielomianu
    iloraz = [0] * n  # Lista na współczynniki ilorazu, o jeden stopień mniejsza niż wielomian
    iloraz[0] = lista[0]  # Pierwszy współczynnik ilorazu to najwyższy współczynnik wielomianu

    # Schemat Hornera dla ilorazu
    for i in range(1, n):
        iloraz[i] = iloraz[i - 1] * c + lista[i]

    # Reszta: ostatni krok Hornera daje resztę
    reszta = iloraz[n - 1] * c + lista[-1]
    
    return iloraz, reszta


lista = [2, -6, 2, -1]  # 2x^3 - 6x^2 + 2x - 1
c = 3  # Dzielimy przez (x - 3)
iloraz, reszta = horner_dzielenie(lista, c)

print(f"Współczynniki ilorazu: {iloraz}")
print(f"Reszta: {reszta}")
```

 - [bisekcja](https://github.com/kacperhalaj/Numerical-Methods/blob/main/3-metodaBisekcji.py)
    - kod:

```python
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

```

 - [Metoda stycznych](https://github.com/kacperhalaj/Numerical-Methods/blob/main/4-metodaStycznych.py)
    - kod:
 
 ```python
import math

def f(x):
    return math.sin(x) - 0.5 * x

def df(x):
    return math.cos(x) - 0.5

def metodaStycznych(x0, epsilon, max_iterations=1000):
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

wynik = metodaStycznych(x0, epsilon)
if wynik is not None:
    print(f"Przybliżona wartość pierwiastka: {wynik}")
```
- [Metoda siecznych](https://github.com/kacperhalaj/Numerical-Methods/blob/main/5-metodaSiecznych.py)
   - cod:

```python
def f(x):
    return x**3 + x**2 - 3*x - 3

def metodaSiecznych(x0, x1, epsilon, max_iterations=1000):
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

wynik = metodaSiecznych(x0, x1, epsilon)
if wynik is not None:
    print(f"Przybliżona wartość pierwiastka: {wynik}")

```
- [Metoda falsi](https://github.com/kacperhalaj/Numerical-Methods/blob/main/6-metodaFalsi.py)
  - kod:

```python
import math

def f(x):
    return 3 * x - math.cos(x) - 1

def metodaFalsi(a, b, epsilon, max_iterations=1000):
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

wynik = metodaFalsi(a, b, epsilon)
if wynik is not None:
    print(f"Przybliżona wartość pierwiastka: {wynik}")

```

