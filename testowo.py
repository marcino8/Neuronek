import pandas as pd
import random
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




# randomowe listy:
# https://stackoverflow.com/questions/9755538/how-do-i-create-a-list-of-random-numbers-without-duplicates

daneWejsciowe = dane
daneWejsciowe.drop('stroke', axis='columns', inplace=True)
losowane = []

for i in range(3):  # np. dla 3 losowań
    losowane = daneWejsciowe.sample(frac=1).reset_index(drop=True)
    print(losowane)








