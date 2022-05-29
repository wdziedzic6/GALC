from csv import DictReader  # Import modułu do przetwarzania plików CSV
import math
import csv


def get_data_without_headers(data_set_with_headers):
    with open(data_set_with_headers, "r") as f:
        data_set_without_headers = f.readlines()[1:]
        return data_set_without_headers


def get_headers(data_set_with_headers):
    file = open(data_set_with_headers, "r")  # Otwarcie pliku z danymi o nazwach atrybutów
    first_line = file.readline()  # Odczyt pierwszej lini (nagłówek zawierający atrybuty)
    first_line = first_line[:-1]  # Pozbycie się znaku nowej lini
    attributes = first_line.split(",")  # Utworzenie listy atrybutów na podstawie zadanego separatora
    file.close()
    return attributes


def get_objects(data_set_with_headers):
    objects = []
    with open(data_set_with_headers, "r") as a_file:
        for line in a_file:
            line = line[:-1]  # Pozbycie się znaku nowej lini
            obj = line.split(",")  # Utworzenie obiektu składającego się z atrybutów na podstawie zadanego separatora
            objects.append(obj)
    return objects


def get_objects_without_decision(data_set_without_headers):
    # print("Do zaimplementowania")
    return []


def get_decisions_set(data_set_without_headers):
    # print("Do zaimplementowania")
    return []


def prepare_the_most_similar_data(metrics, percentage_of_objects, full_train_set, current_obj):
    train_objects_with_headers = get_objects(full_train_set)
    train_objects_without_headers = train_objects_with_headers[1:]

    indexes_and_similarities = []

    for i in range(len(train_objects_without_headers)):
        similarity_index = calculate_the_similarity(metrics, current_obj, train_objects_without_headers[i])
        indexes_and_similarities.append({'index':i, 'similarity':similarity_index})

    indexes_and_similarities.sort(key=sort_by_similarity)
    number_of_all_objects = len(train_objects_without_headers)
    object_boundary = int((percentage_of_objects * number_of_all_objects)/100)
    returned_list = []

    for i in range(object_boundary):
        index = indexes_and_similarities[i]['index']
        returned_list.append(train_objects_without_headers[index])

    with open('data/the_most_similar_objects.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(train_objects_with_headers[0])
        write.writerows(returned_list)


def sort_by_similarity(e):
    return e['similarity']


def calculate_the_similarity(metrics, current_obj, train_objects_without_headers):
    if metrics == "METRYKA_EUKLIDESOWA":
        return calculate_the_similarity_using_euclidean_metric(current_obj, train_objects_without_headers)
    elif metrics == "METRYKA_MANHATTAN":
        return calculate_the_similarity_using_manhattan_metric(current_obj, train_objects_without_headers)
    elif metrics == "METRYKA_KOSINUSOWA":
        print("Oblicznie z metryki kosinusowej")


def calculate_the_similarity_using_euclidean_metric(current_obj, train_object):
    distance = 0
    for i in range(len(current_obj)):
        distance = distance + ((float(current_obj[i]) - float(train_object[i])) ** 2)

    distance = math.sqrt(distance)
    return distance


def calculate_the_similarity_using_manhattan_metric(current_obj, train_object):
    distance = 0
    for i in range(len(current_obj)):
        distance = distance + abs(float(current_obj[i]) - float(train_object[i]))

    return distance


def calculate_the_similarity_using_cosine_metric(current_obj, train_object):
    distance = 0
    # for i in range(len(current_obj)):
    #     distance = distance + abs(float(current_obj[i]) - float(train_object[i]))
    # ta funkcja do zrobienia
    return distance


def get_number_of_correct_labels(predicted_labels, real_labels):
    number_of_correct_labels = 0
    for i in range(len(predicted_labels)):
        if str(predicted_labels[i]) == str(real_labels[i]):
            number_of_correct_labels = number_of_correct_labels + 1

    return number_of_correct_labels


def get_labels(test_objects_without_headers):
    labels = []
    for i in range(len(test_objects_without_headers)):
        labels.append(test_objects_without_headers[i][-1])

    return labels
