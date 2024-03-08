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
        classification_result = classification_manager.execute_a_series_of_classifications(data_set,
                                                                                           classifier_name,
                                                                                           100,
                                                                                           metrics)
        # Dodanie rezultatu klasyfikacji globalnej do tablicy agregujacej wyniki
        results_set.append(classification_result)

        return results_set
        # ETAP PODSUMOWANIA WYNIKOW

        # Struktura obiektu zwracanego w wyniku klasyfikacji w danym zakresie lokalności:
        # [procent_zakresu_treningowego, średnia dokładność, odchylenie standardowe]

# Glowna metoda, gdzie wprowadzane zostaja parametry klasyfikacji:
# zbior danych z decyzją (ścieżka), nazwa klasyfikatora, tablica zakresow klasyfikacji lokalnych oraz metryka
def main():
    classifier = GALC()
    results = {}

    # Dane do klasyfikacji
    datasets = ["iodegradation", "gender_voice", "ivehicle", "german_credit_data",
                "pima-indians-diabetes", "wine", "winequality-red_train", "data-balance", "apple_quality", "cancer-data"]

    classifier_name = "LogisticRegressionClassifier"
    scopes = [20, 40, 60, 80]
    metric = "METRYKA_EUKLIDESOWA"  # Możliwe wartości: "METRYKA_EUKLIDESOWA", "METRYKA_MANHATTAN", "METRYKA_KOSINUSOWA"

    # Przechodzenie przez zbiory danych i klasyfikację
    for data_set in datasets:
        print(f'{data_set}')
        dataset_file = f'data/{data_set}.csv'
        results_set = classifier.classify(dataset_file, classifier_name, scopes, metric)
        results[data_set] = results_set

    # Wyświetlanie wyników
    print(f"Wyniki dla klasyfikatora: {classifier_name} i metryki: {metric}")
    print("")
    for result in results:
        print(f"Wyniki dla zbioru danych: {result}")
        for i, res in enumerate(results[result]):
            print(f"{i+1}. Klasyfikacja z przeszukiwaniem obiektów podobnych o procentowym zakresie = {res[0]}")
            print(f"   - Średnia dokładność klasyfikacji: {res[1]}")
            print(f"   - Odchylenie standardowe klasyfikacji: {res[2]}")
        print("----------------------------------------------------------------------------------------------")


# Uruchomienie skryptu
if __name__ == "__main__":
    main()

