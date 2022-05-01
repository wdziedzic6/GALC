from sklearn import tree
from csv import DictReader  # Import modułu do przetwarzania plików CSV
import pandas as pd
import utils

class DecisionTreeClassifier:
    def __init__(self):
        pass

    def classify(self, training_data, test_data, percentage_range):
        print("Start classification with DecisionTreeClassifier")

        objects_with_headers = utils.get_objects(training_data)
        objects_without_headers = objects_with_headers[1:]

        decisions_array = []  # tablica do przechowywania wyznaczonych obiektom testowym atrybutów decyzyjnych

        for i in range(len(test_data)):
            # training_set_for_object = self.prepare_the_most_similar_data()
            # Tu musi zostać zaimplementowane wybranie najbardziej podobnych obiektów do testowego (nowy zbiór treningowy)

            training_set_for_object_without_decision = utils.get_objects_without_decision(objects_without_headers)
            decisions_set_for_object = utils.get_decisions_set(objects_without_headers)

            conditional_attributes = training_set_for_object_without_decision
            decision_attributes = decisions_set_for_object

            clf = tree.DecisionTreeClassifier()
            clf.fit(conditional_attributes, decision_attributes)

            clf.predict(test_data[i])

        accuracy_of_classification = 0  # Tu wyznaczyć dokładność klasyfikacji
        returned_object = [decisions_array, test_data, percentage_range, accuracy_of_classification]

        return returned_object

    def prepare_the_most_similar_data(self):
        return []


# Struktura obiektu zwracanego w wyniku klasyfikacji:
# [tablica_przypisanych etykiet, tablica_obiektow_z_rzeczywistymi_etykietami, procent_zakresu_treningowego,
# wyznaczona_dokladnosc]