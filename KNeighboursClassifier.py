# GALC - Global and Local Classifier
#import numpy as np
#import pandas as pd
#from sklearn.neighbors import KNeighborsClassifier


class KNeighboursClassifier:
    def __init__(self):
        pass

    def classify(self, training_data, test_data, range):
        print("Start classification")


def myformat(number):
    return "{0:.4f}".format(float(number))


#datasetTrain = pd.read_csv('./winequality-red _train.csv')
#datasetTest = pd.read_csv('./winequality-red _test.csv')

trainColumn = [0,1,2,3,4,5,6,7,8,9,10,11]
#datasetTrain = datasetTrain.iloc[:,trainColumn]
testColumn = [0,1,2,3,4,5,6,7,8,9,10]
#datasetTest = datasetTest.iloc[:,testColumn]

#noColumnTrain = datasetTrain.shape[1]
#noColumnTest = datasetTest.shape[1]

#features_train = datasetTrain.iloc[:,:noColumnTrain-1]
#labels_train = datasetTrain.iloc[:,[noColumnTrain-1]]

#features_test = datasetTest.iloc[:,:noColumnTrain-1]


#model = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
#model.fit(features_train, np.ravel(labels_train))


#labels_predicted = model.predict(features_test) #Testowanie tablicy testowej
#print("Decyzje wygenerowane:",labels_predicted)

#labels_predicted_proba = model.predict_proba(features_test) #Wyliczenie prawdopodonieństwa poszczególnych decyzji
print("Prawdopodobieństwo decyzji:")
#for i in labels_predicted_proba:
#    print(i)

print("---------")
#print(datasetTest)
#datasetTest.insert((datasetTest.shape[1]),"quality",labels_predicted,True)
print("\n\n\n")
#print(datasetTest)
#datasetTest.to_csv('winequality-red_decision',index=False, header=True, encoding='utf-8')