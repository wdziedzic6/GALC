from classifiers.KNeighboursClassifier import KNeighboursClassifier
from classifiers.NaiveBayesianClassifier import NaiveBayesianClassifier
from classifiers.DecisionTreeClassifier import DecisionTreeClassifier
import math
import csv


class TrainAndTestManager:
    def __init__(self):
        pass

    def execute_a_series_of_classifications(self, data_set, classifier_name, percentage_range, metrics):

        classifier = None
        # Wyznaczenie rodzaju klasyfikatora na podstawie parametru classifier_name
        if classifier_name == "KNeighboursClassifier":
            classifier = KNeighboursClassifier()
        elif classifier_name == "NaiveBayesianClassifier":
            classifier = NaiveBayesianClassifier()
        elif classifier_name == "DecisionTreeClassifier":
            classifier = DecisionTreeClassifier()

        # tutaj jednego zbioru danych powinna powstać tablica zawierająca 10 obiektów,
        # po jednym dla każdej serii klasyfikacji.
        # Do każdej serii klasyfikacji trafia obiekt, w którym pod indeksem 0 znajduje się zbiór treningowy,
        # a pod indeksem 1 zbiór testowy
        # train and test ze zbioru wejsciowego tworzy treningowy i testowy na podstawie losowego przydziału
        # według okreslonych proporcji np. 50-50, 60-40 lub 70-30

        # lista zawiertająca obiekty do testów (prób)
        # struktura obiektu: [ścieżka do zbioru treningowego, ścieżka do zbioru testowego]
        test_objects = self.get_train_and_test_objects(data_set)  # tu do zmiennej trzeba przypisać rezultat
        # metody przygotowującej tablice obiektow z zbiorami danych dla poszczegolnych 10-ciu prób
        test_results = []

        for i in range(10):
            training_data = test_objects[i][0]
            test_data = test_objects[i][1]
            print("Klasyfikacja dla procentu lokalności " + percentage_range + ", seria nr " + i+1)
            classification_result = classifier.classify(training_data, test_data, percentage_range, metrics)

        # Wyznaczenie średniej dokładności klasyfikacji
        sum = 0
        for i in test_results:
            sum = sum + i
        average_accuracy = sum / len(test_results)

        # Wyznaczenie odchylenia standardowego
        sum_of_squared_difference = 0
        for i in test_results:
            sum_of_squared_difference = sum_of_squared_difference + ((i - average_accuracy)**2)

        standard_deviation = math.sqrt((sum_of_squared_difference / len(test_results)))

        # Utworzenie i zwrócenie obiektu klasyfikacji dla danego zakresu lokalności
        result_for_the_scope_of_locality = [percentage_range, average_accuracy, standard_deviation]
        return result_for_the_scope_of_locality

    def get_train_and_test_objects(self, data_set):  # implementacja tej metody jest do przerobienia gdyz jest kopią z crossvalidation
        # Utworzenie tablicy z obiektami (plus na początku header)
        objects = []
        with open(data_set, "r") as a_file:
            for line in a_file:
                line = line[:-1]  # Pozbycie się znaku nowej lini
                obj = line.split(
                    ",")  # Utworzenie obiektu składającego się z atrybutów na podstawie zadanego separatora
                objects.append(obj)

        # Podział na 10 grup
        groups = []
        objects_without_header = objects[1:]
        separator = int((len(objects)-1) / 10)
        start = 0
        end = int(separator)

        for i in range(10):
            groups.append(objects_without_header[start:end])
            start = start + separator
            end = end + separator

        # Zapis do plików odpowiednich danych i zwrócenie ścieżek z plikami w formie tablicy obiektów
        dataset_paths = []
        for i in range(10):
            paths = []

            test_path = 'data_cases/test' + str(i) + '.csv'
            with open(test_path, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(objects[0])
                write.writerows(groups[i])

            train_path = 'data_cases/train' + str(i) + '.csv'
            with open(train_path, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(objects[0])
                for j in range(10):
                    if j != i:
                        write.writerows(groups[j])

            paths.append(train_path)
            paths.append(test_path)
            dataset_paths.append(paths)
        return dataset_paths
