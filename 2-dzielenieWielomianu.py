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
