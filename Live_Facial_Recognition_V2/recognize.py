import face_recognition


def face(frame, known_faces, known_names):
    # Find all face locations and face encodings in the current frame
    unknown_face_locations = face_recognition.face_locations(frame)
    unknown_face_encodings = face_recognition.face_encodings(frame, unknown_face_locations)

    recognized_names = []

    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        recognized_names.append(name)

    return unknown_face_locations, recognized_names
