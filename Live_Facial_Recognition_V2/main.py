import os
import cv2
import load
import read
import write
import folder
import keyboard
import recognize

known_faces_folder = r"\your\folder\path\here"

frames = 60  # Frame rate value(fps)

# Open a connection to the camera (0 is the default camera)
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW opens camera faster
video_capture.set(cv2.CAP_PROP_FPS, frames)  # Sets the frame rate

# Load known faces and their encodings
known_faces, known_names = load.saved_known_faces(known_faces_folder)

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True

while True:
    # Capture each frame from the camera
    ret, frame = video_capture.read()

    # Check for f key press
    if keyboard.is_pressed("f"):  # new personal folder in known faces folder
        # Allow the user to input a name for the folder
        folder_name = input("Enter the name for the folder: ")

        # Create the folder
        folder.create(known_faces_folder, folder_name)

    # Check for p key press
    if keyboard.is_pressed("p"):  # New picture in personal folder
        folder_name = input("Enter the person's name: ")

        # Loads personal folder path
        personal_folder_path = known_faces_folder + "\\" + folder_name
        # Loads count.txt path
        count_file_path = known_faces_folder + "\\" + folder_name + r"\count.txt"

        # Check if the path exists
        if os.path.exists(personal_folder_path and count_file_path):
            # Reads count from count.txt
            count = read.count_from_file(count_file_path)

            # Increment count
            count += 1

            # Specify the path where the image will be saved
            image_path = os.path.join(personal_folder_path, f"{count}.jpg")

            # Save the captured frame as an image
            cv2.imwrite(image_path, frame)

            print("Picture saved.")
            print("\n")

            # Updates count in count.txt
            write.count_to_file(count_file_path, count)

            # Load known faces and their encodings
            known_faces, known_names = load.saved_known_faces(known_faces_folder)

        else:
            print(f"The folder for '{folder_name}' does not exist.")
            print("\n")

    # Check for d key press
    if keyboard.is_pressed("d"):  # Delete last picture in personal folder
        folder_name = input("Enter the person's name: ")

        # Loads personal folder path
        personal_folder_path = known_faces_folder + "\\" + folder_name
        # Loads count.txt path
        count_file_path = known_faces_folder + "\\" + folder_name + r"\count.txt"

        # Check if the path exists
        if os.path.exists(personal_folder_path and count_file_path):
            # Reads count from count.txt
            count = read.count_from_file(count_file_path)

            # Checks to make sure count isn't 0
            if count > 0:
                # Specify the path where the image will be deleted from
                image_path = os.path.join(personal_folder_path, f"{count}.jpg")

                # Delete the image at the path
                os.remove(image_path)

                # Decrement count
                count -= 1

                print("Picture deleted.")
                print("\n")

                # Updates count in count.txt
                write.count_to_file(count_file_path, count)

                # Load known faces and their encodings
                known_faces, known_names = load.saved_known_faces(known_faces_folder)

            # if count is 0
            else:
                print("There are no pictures in the folder.")
                print("\n")

        else:
            print(f"The folder for '{folder_name}' does not exist.")
            print("\n")

    # Conduct facial recognition on camera feed
    unknown_face_locations, recognized_names = recognize.face(frame, known_faces, known_names)

    # Display the recognized names on the frame
    for (top, right, bottom, left), name in zip(unknown_face_locations, recognized_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame in a preview window
    cv2.imshow("F = New Folder, P = New Picture in Folder, D = Delete Last Picture, Esc = Close", frame)

    # Check for Esc key press
    if cv2.waitKey(1) & 0xFF == 27:  # exits
        break

# Release the camera and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
