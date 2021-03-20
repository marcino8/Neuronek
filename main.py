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
for i in nazwy:
        pom = []
        for n in dane.loc[:, 'work_type']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom  # dodanie wektora jako nową kolumnę do danych
        pom = []

nazwy = ["Male", "Female", "Other"]
for i in nazwy:
        pom = []
        for n in dane.loc[:, 'gender']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

nazwy = ["Yes", "No"]
for i in nazwy:
        pom = []
        for n in dane.loc[:, 'ever_married']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

nazwy = ["Rural", "Urban"]
for i in nazwy:
        pom = []
        for n in dane.loc[:, 'Residence_type']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

nazwy = ['formerly smoked', 'never smoked', 'smokes']

for i in nazwy:
        pom = []
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


# przetrzymuje informacje o neuronku:
class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + float(dendron.connectedNeuron.getOutput()) * dendron.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight
            dendron.weight = dendron.weight + dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0


class Network:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if len(self.layers) == 0:
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))  # bias neuron
            layer[-1].setOutput(1)  # setting output of bias neuron as 1
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword()

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:  # reverse the order
            for neuron in layer:
                neuron.backPropagate()

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()  # removing the bias neuron
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if o > 0.5:
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()  # removing the bias neuron
        return output


topology = []
topology.append(2)
topology.append(3)
topology.append(2)
net = Network(topology)
Neuron.eta = 0.09
Neuron.alpha = 0.015
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
while True:
    err = 0
    for i in range(len(inputs)):
        net.setInput(inputs[i])
        net.feedForword()
        net.backPropagate(outputs[i])
        err = err + net.getError(outputs[i])
    print("error: ", err)
    if err < 0.91:
        break

while True:
    a = input("type 1st input :")
    b = input("type 2nd input :")
    net.setInput([a, b])
    net.feedForword()
    print(net.getThResults())






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
'''

'''
mask = dane['work_type'].str.startswith('c')
dane.loc[mask, 'work_type'] = 0

mask = dane['work_type'].str.startswith('P')
dane.loc[mask, 'work_type'] = 1

mask = dane['work_type'].str.startswith('S')
dane.loc[mask, 'work_type'] = 0
'''




