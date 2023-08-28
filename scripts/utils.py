# save(data, filename)
import json

def load(filename):
    with open(filename, "r") as f:
        # Load the lines in the file into a list
        read_file = f.read()
    return read_file

def load_json(filename):
    with open(filename, "r") as f:
        # Load the lines in the file into a list
        read_file = json.load(f)
    return read_file

# file_exists(path)