from classifiers.KNeighboursClassifier import KNeighboursClassifier
from classifiers.NaiveBayesianClassifier import NaiveBayesianClassifier
from classifiers.DecisionTreeClassifier import DecisionTreeClassifier


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

        for i in range(10):
            print("Klasyfikacja dla procentu lokalności " + percentage_range + ", seria nr " + i+1)
            # classification_result = classifier.classify(training_data, test_data, percentage_range, metrics)