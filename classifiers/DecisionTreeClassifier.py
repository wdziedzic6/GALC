import pandas as pd
import numpy as np
import utils
from sklearn import tree


class DecisionTreeClassifier:
    def __init__(self):
        pass

    def classify(self, training_data, test_data, percentage_range, metrics):
        print("Start classification with DecisionTreeClassifier")
        print("")
        test_dataset_dataframe = pd.read_csv(test_data)
        test_objects_with_headers = utils.get_objects(test_data)
        test_objects_without_headers = test_objects_with_headers[1:]
        real_labels = utils.get_labels(test_objects_without_headers)

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
        for i in range(len(test_objects_without_headers)):
            utils.prepare_the_most_similar_data(metrics, percentage_range, training_data, test_objects_without_headers[i])

            print("Analizowany obiekt #=", i)
            print("Wartosci atrybutow:", test_objects_without_headers[i])

            tr_dataset_dataframe = pd.read_csv("data/the_most_similar_objects.csv")
            no_column = tr_dataset_dataframe.shape[1]  # Ustalenie liczby kolumn w danych
            train_features = tr_dataset_dataframe.iloc[:, :no_column - 1]  # Wyodrębnienie częśći warunkowej danych
            train_labels = tr_dataset_dataframe.iloc[:, [no_column - 1]]  # Wyodrębnienie kolumny decyzyjnej

            model.fit(train_features, np.ravel(train_labels))

            current_test_object = test_dataset_dataframe.iloc[[i]]
            no_column = current_test_object.shape[1]
            current_test_object_features = current_test_object.iloc[:, :no_column - 1]

            label_predicted = model.predict(current_test_object_features)  # Generowanie decyzji dla obiektu testowego
            # print("label_predicted =", label_predicted[0])
            decisions_array.append(label_predicted[0])

        number_of_correct_labels = utils.get_number_of_correct_labels(decisions_array, real_labels)
        accuracy_of_classification = number_of_correct_labels/len(test_objects_without_headers)

        print("number_of_correct_labels:", number_of_correct_labels)
        print("")

        # Zwracana jest dokładność klasyfikacji
        return accuracy_of_classification
