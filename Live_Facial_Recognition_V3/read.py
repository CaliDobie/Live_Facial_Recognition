def count_from_file(file_path):
    with open(file_path, 'r') as file:
        count = int(file.read())
    return count
