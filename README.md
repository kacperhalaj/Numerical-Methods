# Numerical-Methods

# MetodyNumeryczne

Implementacjom algorytmów matematycznych w języku Python.

### Tematy:

#### [Lista zadań](https://github.com/kacperhalaj/Numerical-Methods/blob/main/ZADANIA.md)

#### ZADANIA
- [horner wyliczanie wartosci w danym punkcie](https://github.com/Prawy126/MetodyNumeryczne/tree/main/horner)
    - kod:

```python
def horner(tablica, x):
    wynik = tablica[0]
    for i in range(1, len(tablica)):
        wynik = wynik * x + tablica[i]
    return wynik

stopien_wielomianu = int(input("Podaj stopień wielomianu: "))
tablica = []

print("Podaj współczynniki wielomianu, zaczynając od współczynnika przy najwyższej potędze:")
for i in range(stopien_wielomianu + 1):
    y = int(input())
    tablica.append(y)

x = int(input("Podaj wartość x: "))

print("Oto twoja tablica współczynników:", tablica)
wynik = horner(tablica, x)
print("Wynik wielomianu dla x =", x, "wynosi", wynik)
```
- [horner dzielenie wielomianu](https://github.com/Prawy126/MetodyNumeryczne/tree/main/horner2)
  - kod:

```python
def hornerDzielenie(tablica, x, rzad):
    wynik = []
    for i in range(rzad):
        wartosc = tablica[i] + x * (wynik[-1] if wynik else 0)
        wynik.append(wartosc)
    return wynik
rzad = int(input("Proszę podać rząd wielomianu: "))
tablica = []
print("Podaj współczynniki wielomianu, zaczynając od współczynnika przy najwyższej potędze:")
for i in range(rzad + 1):
    y = int(input())
    tablica.append(y)

x = int(input("Proszę podać liczbę przez, którą będziemy dzielić: "))

wynik = hornerDzielenie(tablica, x, rzad)
```

 - [bisekcja](https://github.com/Prawy126/MetodyNumeryczne/tree/main/bisekcja)
    - kod:

```python
def bisekcja(func, a, b, error_accept):
    def f(x):
        return func(x)

    if f(a) * f(b) >= 0:
        raise ValueError("Założenia bisekcji nie zostały spełnione - f(a) i f(b) muszą mieć różne znaki.")

    while abs(b - a) > error_accept:
        c = (a + b) / 2
        if f(c) == 0:
            break  # Znaleziono dokładny pierwiastek
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2

```

 - [Metoda stycznych](https://github.com/Prawy126/MetodyNumeryczne/tree/main/metoda_stycznych)
    - cod:
 
 ```python

import math
from sympy import symbols, diff

# Definiujemy zmienną symboliczną
x = symbols('x')


def wyborX0(rownanie, x_val):
    f = rownanie(x_val)
    fdiff = diff(rownanie(x), x).subs(x, x_val)
    return f * fdiff


def styczne(rownanie, dokladnosc, dolnyPrzedzial, gornyPrzedzial):
    a = rownanie(dolnyPrzedzial)
    b = rownanie(gornyPrzedzial)

    assert a * b < 0, "Funkcja nie zmienia znaku na przedziale"

    if wyborX0(rownanie, dolnyPrzedzial) > 0:
        x0 = dolnyPrzedzial
    elif wyborX0(rownanie, gornyPrzedzial) > 0:
        x0 = gornyPrzedzial
    else:
        raise ValueError("Nie można wybrać x0")

    while True:
        fx0 = rownanie(x0)
        fprime_x0 = float(diff(rownanie(x), x).subs(x, x0))
        x1 = x0 - fx0 / fprime_x0

        m = min(abs(float(diff(rownanie(x), x, 2).subs(x, dolnyPrzedzial))),
                abs(float(diff(rownanie(x), x, 2).subs(x, gornyPrzedzial))))
        if abs(rownanie(x1)) / m <= dokladnosc:
            return x1
        else:
            x0 = x1


rownanie = lambda x: x ** 3 - 2 * x ** 2 - 3 * x - 5
print(styczne(rownanie, 0.0001, 3, 4))


```
- [Metoda siecznych](https://github.com/Prawy126/MetodyNumeryczne/tree/main/metoda_siecznych)
   - cod:

```python
# Definicja funkcji f(x) = x^3 + x^2 - 3x - 3
def f(x):
    return x**3 + x**2 - 3*x - 3

# Implementacja metody siecznych
def metoda_siecznych(x0, x1, E):
    """
    x0, x1: punkty startowe
    E: dokładność
    """
    while abs(x1 - x0) >= E:
        f_x0 = f(x0)
        f_x1 = f(x1)
        if f_x0 == f_x1:
            break
        x2 = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
        x0, x1 = x1, x2
    return x1

# Początkowe punkty i dokładność
x0 = 1
x1 = 2
E = 0.0001

# Wywołanie metody siecznych
rozwiazanie = metoda_siecznych(x0, x1, E)
print(rozwiazanie)

```
- [Metoda falsi](https://github.com/Prawy126/MetodyNumeryczne/tree/main/metoda_stycznych)
  - cod:

```python
# Definicja funkcji f(x) = 3x - cos(x) - 1
import math
def f(x):
    return 3 * x - math.cos(x) - 1


# Implementacja metody regula falsi
def metoda_regula_falsi(x0, x1, e):
    """
    x0, x1: punkty startowe
    e: dokładność
    """
    f_x0 = f(x0)
    f_x1 = f(x1)

    if f_x0 * f_x1 > 0:
        return None  # Funkcja musi mieć różne znaki na końcach przedziału

    while abs(f_x1 - f_x0) > e:
        x2 = (x0 * f_x1 - x1 * f_x0) / (f_x1 - f_x0)  # Obliczenie nowego przybliżenia
        f_x2 = f(x2)

        if abs(f_x2) < e:
            return x2

        # Aktualizacja punktów x0 i x1
        if f_x2 * f_x0 < 0:
            x1 = x2
            f_x1 = f_x2
        else:
            x0 = x2
            f_x0 = f_x2

    return (x0 + x1) / 2


# Początkowe punkty i dokładność
x0 = 0.25
x1 = 0.75
e = 0.00001

# Wywołanie metody regula falsi
rozwiazanie = metoda_regula_falsi(x0, x1, e)
print(rozwiazanie)

```

