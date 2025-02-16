import face_recognition
import os
import cv2
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.unknown_face_encodings = []
        self.similarity_threshold = 0.6  # Threshold for face similarity

    def load_encoding_images(self, images_path):
        """Load images from the specified directory, encode them, and store with names."""
        print("[INFO] Loading known faces...")
        for person_folder in os.listdir(images_path):
            person_path = os.path.join(images_path, person_folder)
            if not os.path.isdir(person_path):
                continue
            
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(person_folder)
        print(f"[INFO] Loaded {len(self.known_face_names)} known faces.")

    def detect_known_faces(self, frame):
        """Detect faces and recognize known faces."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)
        return face_locations, face_encodings, face_names

    def is_similar_to_existing_unknown(self, new_encoding):
        """Check if the new face encoding is similar to any stored unknown encodings."""
        if not self.unknown_face_encodings:
            return False

        matches = face_recognition.compare_faces(self.unknown_face_encodings, new_encoding)
        face_distances = face_recognition.face_distance(self.unknown_face_encodings, new_encoding)

        # Check if any matches are within the similarity threshold
        return any(matches) and min(face_distances) < self.similarity_threshold

    def add_unknown_face(self, encoding):
        """Add a new unknown face encoding to the list."""
        self.unknown_face_encodings.append(encoding)
