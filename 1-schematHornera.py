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
