import math

def f(x):
    return 0.06*x**2+2

def metodaProstokatow(a, b, n):
    h = (b - a) / n  # szerokość podprzedziału
    wynik = 0
    for i in range(n):
        wynik += f(a + i * h)  # używamy lewej wartości funkcji w każdym podprzedziale
    wynik *= h
    return wynik

a = 1 
b = 4 
n = 1000  # liczba podprzedziałów im wiecej tym dokladniej

wynik = metodaProstokatow(a, b, n)
print(f"Wartość całki metodą prostokątów: {wynik}")