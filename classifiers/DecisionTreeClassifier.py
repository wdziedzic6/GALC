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
        print("")
        test_dataset_dataframe = pd.read_csv(test_data)
        test_objects_with_headers = utils.get_objects(test_data)
        test_objects_without_headers = test_objects_with_headers[1:]

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
        print("len(test_objects_without_headers)=", len(test_objects_without_headers))
        for i in range(len(test_objects_without_headers)):
            # training_set_for_object = self.prepare_the_most_similar_data()
            # Tu musi zostać zaimplementowane wybranie najbardziej podobnych obiektów do testowego (nowy zbiór treningowy)

            print("Analizowany obiekt #=", i)
            print("test_objects_without_headers[i]=", test_objects_without_headers[i])

            tr_dataset_dataframe = pd.read_csv(training_data)
            no_column = tr_dataset_dataframe.shape[1]  # Ustalenie liczby kolumn w danych
            train_features = tr_dataset_dataframe.iloc[:, :no_column - 1]  # Wyodrębnienie częśći warunkowej danych
            train_labels = tr_dataset_dataframe.iloc[:, [no_column - 1]]  # Wyodrębnienie kolumny decyzyjnej

            model.fit(train_features, np.ravel(train_labels))

            current_test_object = test_dataset_dataframe.iloc[[i]]
            no_column = current_test_object.shape[1]
            current_test_object_features = current_test_object.iloc[:, :no_column - 1]

            labels_predicted = model.predict(current_test_object_features) # Generowanie decyzji dla obiektu testowego
            print("label_predicted =", labels_predicted)

        accuracy_of_classification = 0  # Tu wyznaczyć dokładność klasyfikacji
        returned_object = [decisions_array, test_data, percentage_range, accuracy_of_classification]

        return returned_object

    def prepare_the_most_similar_data(self):
        return []


# Struktura obiektu zwracanego w wyniku klasyfikacji:
# [tablica_przypisanych etykiet, tablica_obiektow_z_rzeczywistymi_etykietami, procent_zakresu_treningowego,
# wyznaczona_dokladnosc]