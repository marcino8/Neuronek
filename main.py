# welcome to the endgame

import pandas as pd
import numpy as np
import math

heart_df = pd.read_csv("dane.csv", sep=',')
print(heart_df.head())
#print(heart_df)

print("\nIlość danych:")
print(heart_df.shape)

# sprawdzanie braków:

print(heart_df.isna().sum())

# pierwotnie wystąpiło 201 braków w "bmi", całość obserwacji wynosi 5110, dlatego postanawiamy usunąć wiersze z brakami:

dane = heart_df.dropna(axis=0)

print(dane.shape)
print(dane.isna().sum())

# sprawdzamy typy występujące w danych:
print("Typy danych:")
print(dane.dtypes)

# Tworzymy osobne kolumny dla każdej kategorii pracy – jeśli dana występuje, w kolumnie jest 1, jeśli nie, 0
# children
# Private
# Self-employed
# Govt_job
# Never_worked

dane = dane.copy()

nazwy = ["children", "Private", "Self-employed", "Govt_job", "Never_worked"]
pom = []

for i in nazwy:
        for n in dane.loc[:, 'work_type']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom  # dodanie wektora jako nową kolumnę do danych
        pom = []

nazwy = ["Male", "Female", "Other"]
for i in nazwy:
        for n in dane.loc[:, 'gender']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

nazwy = ["Yes", "No"]
for i in nazwy:
        for n in dane.loc[:, 'ever_married']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

nazwy = ["Rural", "Urban"]
for i in nazwy:
        for n in dane.loc[:, 'Residence_type']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

nazwy = ['formerly smoked', 'never smoked', 'smokes']

for i in nazwy:
        for n in dane.loc[:, 'smoking_status']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

dane.drop('gender', axis='columns', inplace=True)
dane.drop('id', axis='columns', inplace=True)
dane.drop('work_type', axis='columns', inplace=True)
dane.drop('smoking_status', axis='columns', inplace=True)
dane.drop('Residence_type', axis='columns', inplace=True)
dane.drop('ever_married', axis='columns', inplace=True)






















# ogarniając o co chodzi, czytałam linki, które powrzucałam do notatek na timsach, bardzo polecam je poczytać razem z czytaniem kodu,
# dużo jaśniej się w głowie robi, seeerio :D

# ta klasa trzyma informacje o jakimś konkretnym połączeniu neuronka (np. pierwszego neuronka z drugiej warstwy ukrytej z trzecim neuronkiem pierwszej warstwy ukrytej):
class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron  # wskazuje, o którego neuronka z aktualnej (nie poprzedniej!) warstwy nam chodzi
        self.weight = np.random.normal()  # losuje wagę dla połączenia z tym neuronkiem
        self.dWeight = 0.0  # delta wag, to przyda nam się później, będzie przechowywać informację o tym, jak bardzo trzeba zmienić weight (inny po każdej iteracji)


class Neuron:  # któryś jeden neuronek z aktualnej warstwy, nad którą pracujemy
    eta = 0.001  # współczynnik, który wpływa na wielkość skoku (wielkość zmiany wagi połączenia), Learning Rate
    alpha = 0.01  # współczynnik, który zwany jest momentum, współczynnik wnoszący bezwładność. Przyspiesza i stabilizuje uczenie.
    # Używane są obydwa w propagacji wstecznej.

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0  # wyliczany jako error*pochodnaFunkcjiAktywacyjnej(output), gdzie output obliczany jest w forward, a gradient w back propagation
        self.output = 0.0  # tę zmienną wypluwa nasz pojedynczy neuronek, gdy go nakarmimy informacjami i wagami poprzednich neuronków
        if layer is None:  # layer to po prostu poprzednia warstwa (np. dla danych wejściowych nie ma żadnej warstwy, więc będzie None,
            # ale dla pierwszej warstwy ukrytej już mamy jedną warstwę – dane wejściowe)
            pass
        else:
            for neuron in layer:  # dla każdego pojedynczego neuronka w podanej poprzedniej warstwie
                con = Connection(neuron)  # utwórz połączenie (tutaj ustalamy jego losową wagę)
                self.dendrons.append(con)  # i dodaj je do listy połączeń naszego aktualnego neuronka w aktualnej warstwie, którą się zajmujemy

    # czyli generalnie __init__ robi tyle, że dla każdego konkretnego neuronka w podanej aktualnej warstwie layer tworzy listę, na której ma napisane,
    # z iloma neuronkami z poprzedniej warstwy się ten nasz neuronek łączy oraz ustala dla każdego tego połączenia losową wagę.

    def addError(self, err):  # używane w propagacji wstecznej; zbiera nam informacje o błędach z każdego neuronka z poprzedniej warstwy
        self.error = self.error + err

    def sigmoid(self, x):  # to jest funkcja aktywacyjna, tutaj sigmoidalna. Leci ze wzoru 1/(1+e^(-z)),
        # gdzie z = suma wag*output neuronków poprzednich + bias
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):  # pochodna funkcji aktywacyjnej, używana do propagacji wstecznej.
        return x * (1.0 - x)  # dla funkcji simgoidalnej pochodna wynosi sigmoid*(1-sigmoid).
        # Nasza output to już sigmoid obliczony, dlatego tutaj mamy same x
        # (output liczymy w feedForword zanim użyjemy backPropagate)

    def setError(self, err):  # ustala wartość błędu
        self.error = err

    def setOutput(self, output):  # ustala wartość output neuronka
        # (użyte to będzie w forward propagation po tym, jak już nakarmimy neuronka informacjami z neuronków poprzedniej warstwy)
        self.output = output

    def getOutput(self):  # wiadomo, zwraca wartość output (to, co wypluwa neuronek)
        return self.output

    def feedForword(self):  # dokładnie to samo, co robiliśmy w excelu, taka jakby jednokierunkowa sieć
        sumOutput = 0
        if len(self.dendrons) == 0:  # jeżeli nie ma żadnego połączenia dla wybranego neuronka, zamyka działanie funkcji
            # (tak będzie dla warstwy wejściowej, która nie ma żadnych poprzednich warstw)
            return
        for dendron in self.dendrons:  # dla każdego połączenia (z kolejnymi neuronkami z poprzedniej warstwy) to nam zlicza waga * wyjście z poprzednich neuronków,
            sumOutput = sumOutput + float(dendron.connectedNeuron.getOutput()) * dendron.weight  # suma tego daje nam wartość,
        self.output = self.sigmoid(sumOutput)  #  wrzucamy wartość do funkcji aktywacyjnej i mamy output naszego neuronka (jak w excelu)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:  # to dobrze opisuje ten link: http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
            dendron.dWeight = Neuron.eta * self.gradient * dendron.connectedNeuron.output + self.alpha * dendron.dWeight  # tutaj alfa określa, jaka część poprzedniej delty wag
            # powinna zostać włączona do aktualnego liczenia (za pierwszym podejściem dWeight była ustalona na 0), bo zmiana wagi powinna być proporcjonalna do gradientu. Chodzi
            # o to, żeby zminimalizować ryzyko wpadnięcia w ekstrema lokalne; na jednym z linków był przykład z wolnymi sankami i taką łódką czy coś xD
            # Jak dzięki alfie nadamy "szybkości" naszej zmianie wagi, to ona przeleci sobie po ekstremum lokalnym i będzie dalej szukać ekstremum globalnego.
            # Output aktualnego neuronu decyduje o tym, jak duża będzie zmiana wagi, by zminimalizować błąd, a gradient zdecyduje, w którym
            # kierunku zmienić wagę, by zminimalizować błąd.
            dendron.weight = dendron.weight + dendron.dWeight  # poprawiona waga
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)  # wysyłamy błąd połączenia neuronka z poprzedniej warstwy do wszystkich z aktualnej warstwy,
            # z którymi jest połączony. Używamy
            # funkcji addError zamiast setError, żeby nie nadpisać informacji wysłanej przez te pozostałe neuronki z poprzedniej warstwy. W skrócie ta linijka
            # sumuje błędy wyliczone z każdego neuronka poprzedniej warstwy, które trafiają do każdego neuronka z warstwy aktualnej. Na przykład
            # jeśli pierwsza warstwa ukryta ma 10 neuronków, a druga ma 5 neuronków, to każdy z tych 4 neuronków (cztery, bo na jednym z nich pracujemy wsteczną propagacją
            # i usuniemy całkowicie jego błąd) dostanie swoją własną sumę błędów popełnionych przy połączeniach z tymi 10 neuronkami z poprzedniej warstwy.
        self.error = 0  # tutaj zerujemy błąd tego konkretnego aktualnego neuronka na którym pracowaliśmy, bo dzięki wstecznej propagacji usunęliśmy ten błąd :)


class Network:
    def __init__(self, topology):  # topology to LISTA, która dostarcza nam informacje o ilości warstw oraz ilości neuronków w każdej z nich.
        # Na przykład lista [5, 8, 1] mówi, że mamy 3 warstwy.
        # Pierwsza (zawsze to DANE!) ma 5 neuronków,
        # druga (warstwa ukryta) 8 neuronków,
        # trzecia (zawsze to WYNIK, ostatni neuron!!) 1 neuronka (tutaj w naszym projekcie będzie 1, ale oni mają 2, bo na dole inputs outputs to listy dwuelementowe)

        # teraz, zgodnie z tym ile chcemy warstw i ilości neuronków w każdej, tworzymy sobie listę warstw i neuronków w środku:
        self.layers = []

        for numNeuron in topology:  # dla każdej kolejnej warstwy w liście topology…
            layer = []
            for i in range(numNeuron):  # …przejdź po podanej ilości neuronków (np. warstwa z liczbą "5" utworzy sobie pierwszego, drugiego, …, piątego neuronka)
                if len(self.layers) == 0:  # sprawdza, czy już utworzyliśmy jakąś warstwę (ilość elementów w "layers" to ilość utworzonych warstw), jeśli nie…
                    layer.append(Neuron(None))  # dodaje do aktualnie tworzonej warstwy neuronka, który nie ma żadnych poprzednich warstw (neuronki z danymi wejściowymi będą tak tworzony)
                else:  # a jeżeli już mamy dodaną jakąś warstwę (np. warstwę neuronków trzymających informacje o danych)
                    layer.append(Neuron(self.layers[-1]))  # to do aktualnie tworzonej warstwy "layer" dodaje neuronka, którego poprzednią warstwą jest ostatni element z "layers",
                    # jeżeli np. utworzyliśmy warstwę neuronków z danymi wejściowymi [1], to teraz tworzymy np. pierwszą warstwę ukrytą, a jej poprzednią warstwą jest [1]
            layer.append(Neuron(None))  # Tutaj tworzymy bias dla tej naszej tworzonej warstwy. Dajemy None, bo bias to ustalona z góry stała liczba, on nie ma żadnych poprzednich warstw :)  (jak w excelu)
            layer[-1].setOutput(1)  # Ustalili wartość biasa (ostatniego dodanego neuronka) na 1, tutaj można zobaczyć, co on robi: https://stackoverflow.com/questions/2480650/what-is-the-role-of-the-bias-in-neural-networks
            self.layers.append(layer)  # dodajemy świeżo utworzoną warstwę do naszej kolekcji warstw "layers"

    def setInput(self, inputs):  # ta metoda pozwala nam dodać do pierwszej warstwy nasze dane (bo pierwsza warstwa to warstwa danych, wiadomo :D)
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])  # "0", bo w listach numerowanie jest od 0. "i" przechodzi po kolejnych wierszach

    def getError(self, target):  # liczy całościowy błąd popełniony przez calutką sieć (wszystkie warstwy) dla konkretnej pary input output
        err = 0
        # oni tutaj mają pętlę, bo ich ostatnia warstwa wyników składa się z dwóch neuronków
        for i in range(len(target)):  # target to to, co chcemy, żeby wypluł nasz ostatni neuronek po tym, jak zje wszystkie poprzednie (lista wyników, na dole mamy to nazwane "outputs")
            e = (target[i] - self.layers[-1][i].getOutput())  # oblicza różnicę dla podanego input i output pomiędzy target a tym, co wypluł neuronek
            err = err + e ** 2  # err to będzie suma kwadratów powyższego działania dla każdej kolejnej pary input output
        err = err / len(target)  # tutaj wyliczamy średnią
        err = math.sqrt(err)  # liczymy pierwiastek; wzór ten podany jest bezpośrednio na stronie: https://thecodacus.com/2017/08/14/neural-network-scratch-python-no-libraries/
        return err

    def feedForword(self):  # w tej funkcji wywołujemy nauczanie neuronków
        for layer in self.layers[1:]:  # dla każdej kolejnej warstwy idąc od pierwszej ukrytej (indeks 0 to nauronki danych!!)
            for neuron in layer:  # i dla każdego neuronka w danej warstwie
                neuron.feedForword()  # naucz go

    def backPropagate(self, target):  # wsteczna propagacja
        for i in range(len(target)):  # dla każdej pary input output (to są dwie wartości, dlatego tu jest pętla! Bo mają wyniki jako listy dwuelementowe, u nas nie będzie pętli)
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())  # ustawia błąd dla konkretnego neuronka wynikowego (ta funkcja użyta będzie PO feed forward),
            # później kolejne "naprawione" już neuronki będą przesyłać obliczone błędy dla następnych poprzednich warstw (bo idziemy od końca :))
        for layer in self.layers[::-1]:  # idziemy sobie od ostatniej warstwy: https://stackoverflow.com/questions/41430791/python-list-error-1-step-on-1-slice
            for neuron in layer:
                neuron.backPropagate()

    def getResults(self):  # to tutaj zwraca nam czysty wynik, dosłownie to, co wypluwa neuron…
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()  # removing the bias neuron
        return output

    def getThResults(self):  # …a to tutaj zwraca nam już zinterpretowany wynik (dla funkcji sigmoidalnej >0.5 ustawia wynik na 1, inaczej na 0)
        output = []
        for neuron in self.layers[-1]:  # dla każdego neuronka w ostatniej warstwie
            o = neuron.getOutput()
            if o > 0.5:
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()  # usuwanie biasa… szczerze mówiąc mnie mocno zastanawia co robi bias w ostatniej warstwie,
        # można by go nie dodawać w ogóle przy tworzeniu warstwy tej bo jest zupełnie niepotrzebny xD
        return output


topology = []
# ustalanie ilości neuronków w każdej kolejnej warstwie:
topology.append(2)
topology.append(3)
topology.append(2)  # dwie, bo potrzebują wyniku jako dwuelementowej listy
net = Network(topology)  # tworzenie sieci z zadanymi parametrami
# sztywne ustalanie współczynników, ja bym to poustawiała tymi metodami z neta, bo to się dostosowuje do danych jakoś:
Neuron.eta = 0.09
Neuron.alpha = 0.015
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]

while True:
    err = 0
    for i in range(len(inputs)):
        # wg tego co czytałam, dla każdej iteracji powinnyśmy mieszać dane, tu tego nie ma:
        net.setInput(inputs[i])
        net.feedForword()
        net.backPropagate(outputs[i])
        err = err + net.getError(outputs[i])
    print("error: ", err)
    if err < 0.91:  # TUTAJ BYŁO 0.01, ALE ZMIENIŁAM, ŻEBY DAŁO SIĘ SZYBCIEJ SPRAWDZAĆ, JAK TO DZIAŁA :)
        break

while True:
    a = input("type 1st input :")
    b = input("type 2nd input :")
    net.setInput([a, b])
    net.feedForword()
    print(net.getThResults())


# podsumowując, czego zauważyłam, że na pewno brakuje:
# Ustalania współczynników eta i alfa konkretnym wzorem, nie na sztywno.
# Nie podzielili zbioru input na zbiory uczący, walidacyjny i testowy, my musimy to zrobić.
# Nie sprawdzili na samym końcu jak sieć zachowuje się na zbiorze walidacyjnym i testowym,
# my możemy przy okazji robienia tego wypluć jakieś wykresy do raportu.
# Zmiana struktury sieci będzie z podobnym kodem banalna, więc jeśli napiszemy sobie wykresy do tego
# i różne "testowe" rzeczy, to później tylko kopiuj wklej wyników do raportu i jakieś super wnioski bazujące
# najlepiej na tym linku, bo przesuper zrobili:
# https://www.cri.agh.edu.pl/uczelnia/tad/inteligencja_obliczeniowa/08%20-%20Uczenie%20-%20poglądowe.pdf
# (pozachwycałam się tym na timsach w notatkach xD Swoją drogą moim zdaniem najlepsze opracowania są z AGH... xD)
# No i musimy sobie dostosować ten kod pod siebie, bo jest robiona pod dane, których wynikiem ma być dwuelementowa lista,
# ale to nie będzie za skomplikowane. Myślałam też, żebyśmy jakoś spróbowały schemat czy projekt "rysowankę" sieci
# zrobić przed pisaniem tego, przyda się do raportu plus nie oskarży nas o skopiowane kodu z neta :D





'''
mapper = {'children': 0, 'Private': 1, 'Self-employed': 2, 'Govt_job': 3, 'Never_worked': 4}
dane.loc[:, 'work_type'] = dane.loc[:, 'work_type'].replace(mapper)

mapper2 = {'Male': 0, 'Female': 1, 'Other': 2}
dane.loc[:, 'gender'] = dane.loc[:, 'gender'].replace(mapper2)

mapper3 = {'No': 0, 'Yes': 1}
dane.loc[:, 'ever_married'] = dane.loc[:, 'ever_married'].replace(mapper3)

mapper4 = {'Rural': 0, 'Urban': 1}
dane.loc[:, 'Residence_type'] = dane.loc[:, 'Residence_type'].replace(mapper4)

# ZAPYTAĆ O TO !!!
mapper5 = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}
dane.loc[:, 'smoking_status'] = dane.loc[:, 'smoking_status'].replace(mapper5)

mask = dane['work_type'].str.startswith('c')
dane.loc[mask, 'work_type'] = 0

mask = dane['work_type'].str.startswith('P')
dane.loc[mask, 'work_type'] = 1

mask = dane['work_type'].str.startswith('S')
dane.loc[mask, 'work_type'] = 0
'''




