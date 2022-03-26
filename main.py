# GALC - Global and Local Classifier
# from KNeighboursClassifier import KNeighboursClassifier
from NaiveBayesianClassifier import NaiveBayesianClassifier
from DecisionTreeClassifier import DecisionTreeClassifier


class GALC:
    def __init__(self):
        pass

    def classify(self, training_data, test_data, classifier_name):
        if classifier_name == "KNeighboursClassifier":
            print("")
            # classifier = KNeighboursClassifier()
            # classifier.classify(training_data, test_data)
        elif classifier_name == "NaiveBayesianClassifier":
            classifier = NaiveBayesianClassifier()
            classifier.classify(training_data, test_data)
        elif classifier_name == "DecisionTreeClassifier":
            classifier = DecisionTreeClassifier()
            classifier.classify(training_data, test_data)


def main():
    classifier = GALC()
    # classifier.classify("winequality-red_train", "winequality-red_test", "KNeighboursClassifier")
    classifier.classify("winequality-red_train", "winequality-red_test", "NaiveBayesianClassifier")
    # classifier.classify("winequality-red_train", "winequality-red_test", "DecisionTreeClassifier")


if __name__ == "__main__":
    main()
