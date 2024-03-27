import face_recognition


def face(frame, known_faces, known_names):
    # Find all face locations and face encodings in the current frame
    unknown_face_locations = face_recognition.face_locations(frame)
    unknown_face_encodings = face_recognition.face_encodings(frame, unknown_face_locations)

    recognized_names = []

    for face_encoding in unknown_face_encodings:
        matches = []

        for known_face, known_name in zip(known_faces, known_names):
            result = face_recognition.compare_faces([known_face], face_encoding)

            if result[0]:
                matches.append((known_name, result))

        if len(matches) > 0:
            matches.sort(key=lambda x: x[1][0], reverse=True)
            name = matches[0][0]

        else:
            name = "Unknown"

        recognized_names.append(name)

    return unknown_face_locations, recognized_names
