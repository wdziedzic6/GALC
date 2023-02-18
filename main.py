# GALC - Global and Local Classifier
from TrainAndTestManager import TrainAndTestManager
from CrossValidationManager import CrossValidationManager
import pandas as pd


class GALC:
    def __init__(self):
        pass

    def classify(self, data_set, classifier_name, local_ranges, metrics):

        # ETAP KLASYFIKACJI

        # tablica przechowujaca wyniki poszczegolnych klasyfikacji
        results_set = []
        number_of_objects = len(pd.read_csv(data_set)) - 1
        classification_manager = None
        if number_of_objects <= 1000:
            classification_manager = CrossValidationManager()
        else:
            classification_manager = TrainAndTestManager()

        # Dokonanie serii klasyfikacji dla kazdego ze wskazanych zakresow lokalnych
        for percentage_range in local_ranges:
            classification_result = classification_manager.execute_a_series_of_classifications(data_set,
                                                                                               classifier_name,
                                                                                               percentage_range,
                                                                                               metrics)
            # Dodanie rezultatu klasyfikacji dla obslugiwanego zakrezu do tablicy agregujacej wyniki
            results_set.append(classification_result)

        # Dokonanie klasyfikacji globalnej (brane pod uwage 100% obiektow treningowych)
        classification_result = classification_manager.execute_a_series_of_classifications(data_set, classifier_name,
                                                                                           100, metrics)
        # Dodanie rezultatu klasyfikacji globalnej do tablicy agregujacej wyniki
        results_set.append(classification_result)

        # ETAP PODSUMOWANIA WYNIKOW

        # Struktura obiektu zwracanego w wyniku klasyfikacji w danym zakresie lokalności:
        # [procent_zakresu_treningowego, średnia dokładność, odchylenie standardowe]

        for i in range(len(results_set)):
            print("")
            print(i+1, "Klasyfikacja z przeszukiwaniem obiektow podobnych o procentowym zakresie =", results_set[i][0])
            print("- Średnia dokladność klasyfikacji:", results_set[i][1])
            print("- Odchylenie standardowe klasyfikacji:", results_set[i][2])
            print("")


# Glowna metoda, gdzie wprowadzane zostaja parametry klasyfikacji:
# zbior danych z decyzją (ścieżka), nazwa klasyfikatora, tablica zakresow klasyfikacji lokalnych oraz metryka
def main():
    classifier = GALC()

    # data_set = "data\winequality-red_train.csv"
    # data_set = "data\winequality-red_decision.csv"
    # data_set = "data\winequality-red_decision_v2.csv"
    data_set = "data\winequality-red_decision_v2_bigger_version.csv"

    # classifier.classify(data_set, "KNeighboursClassifier", [20, 40, 60, 80], "METRYKA_EUKLIDESOWA")
    # classifier.classify(data_set, "NaiveBayesianClassifier", [20, 40, 60, 80], "METRYKA_EUKLIDESOWA")
    classifier.classify(data_set, "DecisionTreeClassifier", [20, 40, 60, 80], "METRYKA_EUKLIDESOWA")


# Uruchomienie skryptu
if __name__ == "__main__":
    main()

