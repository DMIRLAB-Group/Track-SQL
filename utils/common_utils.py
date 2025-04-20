import json
import pickle

def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
    return data

def save_json_file(path,data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
def load_pickle_file(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data
def save_pickle_file(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)


def is_number(value):
    try:
        _ = float(value)
        return True
    except:
        return False

def get_key_from_value(value, dictionary):
    return next((key for key, val in dictionary.items() if val == value), None)