import math

def f(x):
    return math.sqrt(1 + x)

def metodaTrapezow(a, b, h):
    n = int((b - a) / h) #liczba przedzialow
    wynik = f(a) + f(b) #wyliczenie wartosci na krancach przedzialow
    for i in range(1, n):   #petla do sumowania srodkowych przedzialow
        wynik += 2 * f(a + i * h) #wyliczmay wartosci i podwajamy srodkowe
    wynik *= h / 2
    return wynik


a = 0
b = 1
h = 1/3

wynik = metodaTrapezow(a, b, h)
print(f"Wartość całki metodą trapezów: {wynik}")
