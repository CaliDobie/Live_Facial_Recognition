def count_to_file(file_path, count):
    with open(file_path, 'w') as file:
        file.write(str(count))
