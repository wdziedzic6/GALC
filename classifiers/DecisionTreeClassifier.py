import pandas as pd
import numpy as np
import utils
from sklearn import tree
from csv import DictReader  # Import modułu do przetwarzania plików CSV
from sklearn.model_selection import train_test_split

class DecisionTreeClassifier:
    def __init__(self):
        pass

    def classify(self, training_data, test_data, percentage_range):
        print("Start classification with DecisionTreeClassifier")

        # objects_with_headers = utils.get_objects(training_data)
        # objects_without_headers = objects_with_headers[1:]

        test_dataset = utils.get_objects(test_data)
        test_dataset_pd = pd.read_csv(test_data)

        my_criterion = "gini"  # Kryterium podziału węzła drzewa podczas budowy drzewa: 'gini', 'entropy'
        my_max_depth = 5
        my_min_samples_split = 10
        my_min_samples_leaf = 10
        my_max_leaf_nodes = 30
        my_min_impurity_decrease = 0.02
        # Utworzenie obiektu przykładowego modelu klasyfikatora (k-NN)
        model = tree.DecisionTreeClassifier(criterion=my_criterion,
                                       max_depth=my_max_depth,
                                       min_samples_split=my_min_samples_split,
                                       min_samples_leaf=my_min_samples_leaf,
                                       max_leaf_nodes=my_max_leaf_nodes,
                                       min_impurity_decrease=my_min_impurity_decrease)

        decisions_array = []  # tablica do przechowywania wyznaczonych obiektom testowym atrybutów decyzyjnych

        for i in range(len(test_dataset)):
            # training_set_for_object = self.prepare_the_most_similar_data()
            # Tu musi zostać zaimplementowane wybranie najbardziej podobnych obiektów do testowego (nowy zbiór treningowy)

            print("Analizowany obiekt #", i)
            print("Atrybuty test_dataset[i]:", test_dataset[i])

            tr_dataset = pd.read_csv(training_data)
            # tu trzeba zrobić zwrócenie zbioru treningowego najbardizej podobnych obiektów do obiektu testowego
            # i musi być rozgraniczenie na zmienną z samymi tylko atrybutami warunkowymi i na zmienną
            # z samymi tylko etykietami
            # te zmienne potem posłużą w metodzie fit do zbudowania klasyfikatora

            no_column = tr_dataset.shape[1]  # Ustalenie liczby kolumn w danych
            features = tr_dataset.iloc[:, :no_column - 1]  # Wyodrębnienie częśći warunkowej danych
            labels = tr_dataset.iloc[:, [no_column - 1]]  # Wyodrębnienie kolumny decyzyjnej

            datasets = train_test_split(features, labels, test_size=0.6, random_state=1234)
            features_train = datasets[0]
            features_test = datasets[1]
            labels_train = datasets[2]
            labels_test = datasets[3]

            # training_set_for_object_without_decision = utils.get_objects_without_decision(objects_without_headers)
            # decisions_set_for_object = utils.get_decisions_set(objects_without_headers)
            # conditional_attributes = training_set_for_object_without_decision
            # decision_attributes = decisions_set_for_object

            # model.fit(training_set_for_object_without_decision, np.ravel(decisions_set_for_object))
            model.fit(features_train, np.ravel(labels_train))

            print("test test_object_data_frame: ", test_object_data_frame)
            test_object_data_frame = test_dataset_pd.iloc[[i], :]
            print("test_object_data_frame: ", test_object_data_frame)
            print("type(test_object_data_frame): ", type(test_object_data_frame))

            labels_predicted = model.predict(test_object_data_frame) # Generowanie decyzji dla obiektu testowego
            # labels_predicted = model.predict(features_test) # Generowanie decyzji dla obiektu testowego
            print("labels_predicted=", labels_predicted)

        accuracy_of_classification = 0  # Tu wyznaczyć dokładność klasyfikacji
        returned_object = [decisions_array, test_data, percentage_range, accuracy_of_classification]

        return returned_object

    def prepare_the_most_similar_data(self):
        return []


# Struktura obiektu zwracanego w wyniku klasyfikacji:
# [tablica_przypisanych etykiet, tablica_obiektow_z_rzeczywistymi_etykietami, procent_zakresu_treningowego,
# wyznaczona_dokladnosc]