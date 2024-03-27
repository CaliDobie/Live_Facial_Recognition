import os


def create(path, folder_name):
    full_path = os.path.join(path, folder_name)

    try:
        os.makedirs(full_path)
        print(f"The folder for '{folder_name}' was created successfully")
        print("\n")

        # Loads count.txt path
        count_file_path = full_path + r"\count.txt"

        # Create count.txt in the new folder
        with open(count_file_path, 'x') as file:
            file.write("0")  # Write 0 in count.txt

    except FileExistsError:
        if folder_name == "":
            print(f"The folder for '{folder_name}' can not be created")
            print("\n")

        else:
            print(f"The folder for '{folder_name}' already exists")
            print("\n")
