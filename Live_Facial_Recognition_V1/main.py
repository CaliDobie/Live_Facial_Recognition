import os
import cv2
import keyboard
import face_recognition

# Function to load all images and their corresponding encodings from a folder
def load_known_faces(folder_path):
    known_faces = []
    known_encodings = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            known_image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(known_image)[0]
            known_faces.append(filename.split(".")[0])  # Use the file name (excluding extension) as the person's identifier
            known_encodings.append(encoding)

    return known_faces, known_encodings

# Specify the path to the folder containing known faces
folder_path = r"\your\folder\path\here"

# Load known faces and their encodings
known_faces, known_encodings = load_known_faces(folder_path)

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)  # 0 represents the default camera
video_capture.set(cv2.CAP_PROP_FPS, 60)  # sets the frame rate

while True:
    # Capture each frame from the camera
    _, frame = video_capture.read()

    # Resize the frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Check for tab key press
    if keyboard.is_pressed("tab"):
        # Allow the user to input a name for the image
        image_name = input("Enter a name for the new Known Face: ") + ".jpg"

        # Specify the path where the image will be saved
        image_path = os.path.join(folder_path, image_name)

        # Save the captured frame as an image
        cv2.imwrite(image_path, frame)

        print(f"New Known Face '{image_name.split(".")[0]}' saved in '{folder_path}'. \n")

        # Load known faces and their encodings
        known_faces, known_encodings = load_known_faces(folder_path)

    if process_this_frame:
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Check if any face matches the known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, use the name of the matching person
            if True in matches:
                first_match_index = matches.index(True)
                name = known_faces[first_match_index]

    # You can perform further actions based on the recognized face
    process_this_frame = not process_this_frame

    # Draw rectangles around the faces
    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Facial Recognition - Press 'Tab' to capture a new Known Face", frame)

    # Check for 'Esc' key press to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object
video_capture.release()
cv2.destroyAllWindows()
