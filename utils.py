from csv import DictReader  # Import modułu do przetwarzania plików CSV


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
