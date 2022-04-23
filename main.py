# GALC - Global and Local Classifier
from KNeighboursClassifier import KNeighboursClassifier
from NaiveBayesianClassifier import NaiveBayesianClassifier
from DecisionTreeClassifier import DecisionTreeClassifier


class GALC:
    def __init__(self):
        pass

    def classify(self, training_data, test_data, classifier_name, local_ranges):

        # ETAP KLASYFIKACJI

        # tablica przechowujaca wyniki poszczegolnych klasyfikacji
        results_set = []
        classifier = None
        # Wyznaczenie rodzaju klasyfikatora na podstawie parametru classifier_name
        if classifier_name == "KNeighboursClassifier":
            print("")
            classifier = KNeighboursClassifier()
        elif classifier_name == "NaiveBayesianClassifier":
            classifier = NaiveBayesianClassifier()
        elif classifier_name == "DecisionTreeClassifier":
            classifier = DecisionTreeClassifier()

        # Dokonanie klasyfikacji dla kazdego ze wskazanych zakresow lokalnych
        for range in local_ranges:
            classification_result = classifier.classify(training_data, test_data, range)
            # Dodanie rezultatu klasyfikacji dla obslugiwanego zakrezu do tablicy agregujacej wyniki
            results_set.append(classification_result)

        # Dokonanie klasyfikacji globalnej (brane pod uwage 100% obiektow treningowych)
        classification_result = classifier.classify(training_data, test_data, 100)
        # Dodanie rezultatu klasyfikacji globalnej do tablicy agregujacej wyniki
        results_set.append(classification_result)

        # ETAP PODSUMOWANIA WYNIKOW

        for i in range(len(results_set)):
            print(i+1, "Klasyfikacja z przeszukiwaniem obiektow podobnych o procentowym zakresie =", results_set[i][2])
            print("- Dokladnosc klasyfikacji:", results_set[i][3])



# Glowna metoda, gdzie wprowadzane zostaja parametry klasyfikacji:
# zbior treningowy, zbior testowy, nazwa klasyfikatora, tablica zakresow klasyfikacji lokalnych
def main():
    classifier = GALC()
    # classifier.classify("winequality-red_train", "winequality-red_test", "KNeighboursClassifier", [20, 40, 60, 80])
    classifier.classify("winequality-red_train", "winequality-red_test", "NaiveBayesianClassifier", [20, 40, 60, 80])
    # classifier.classify("winequality-red_train", "winequality-red_test", "DecisionTreeClassifier", [20, 40, 60, 80])


# Uruchomienie skryptu
if __name__ == "__main__":
    main()


# Struktura obiektu zwracanego w wyniku klasyfikacji:
# [tablica_przypisanych etykiet, tablica_obiektow_z_rzeczywistymi_etykietami, procent_zakresu_treningowego, wyznaczona_dokladnosc]