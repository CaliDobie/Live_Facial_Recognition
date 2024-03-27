import os
import face_recognition


def saved_known_faces(known_faces_folder):
    known_faces = []
    known_names = []

    for person_folder in os.listdir(known_faces_folder):
        person_path = os.path.join(known_faces_folder, person_folder)

        if os.path.isdir(person_path):
            face_encodings = []

            for filename in os.listdir(person_path):
                if filename.endswith(('.jpg', '.png')):
                    face_image = face_recognition.load_image_file(os.path.join(person_path, filename))
                    face_landmarks = face_recognition.face_landmarks(face_image)

                    if len(face_landmarks) > 0:
                        face_encoding = face_recognition.face_encodings(face_image)[0]
                        face_encodings.append((face_encoding, face_landmarks[0]))

            if face_encodings:
                average_face_encoding = sum([encoding for encoding, landmarks in face_encodings]) / len(face_encodings)
                known_faces.append(average_face_encoding)
                known_names.append(person_folder)

    return known_faces, known_names
