import cv2
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import os
from PIL import Image, ImageTk
import numpy as np
import face_recognition


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images_from_subfolders(self, image_folder):
        """Manually load face encodings from images in subfolders inside a specified folder."""
        for person_name in os.listdir(image_folder):

            person_folder = os.path.join(image_folder, person_name)
            if os.path.isdir(person_folder):  # If it's a directory
                for img_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_name)
                    img = cv2.imread(img_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_encoding = face_recognition.face_encodings(rgb_img)
                    if face_encoding:
                        self.known_face_encodings.append(face_encoding[0])
                        self.known_face_names.append(person_name)

    def detect_known_faces(self, frame):
        """Detect known faces in a given frame."""
        face_locations = []
        face_encodings = []
        face_names = []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations_in_frame = face_recognition.face_locations(rgb_frame)
        face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations_in_frame)

        for encoding, location in zip(face_encodings_in_frame, face_locations_in_frame):
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_locations.append(location)
            face_encodings.append(encoding)
            face_names.append(name)

        return face_locations, face_encodings, face_names


class FaceRecognitionAttendance:
    def __init__(self):
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images_from_subfolders("images/")  # Load face encodings from subfolders inside images/
        print("Loaded face encodings from 'images/' subfolders")  # Debug: Check if encodings are loaded
        self.attendance = []
        self.unknown_counter = 1
        self.unknown_encodings = []  # To track previously seen unknown faces
        self.cap = None
        self.running = False

        os.makedirs("captured", exist_ok=True)

        # Set up the GUI
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance")

        self.start_button = tk.Button(self.root, text="Start Attendance", command=self.start_attendance)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Attendance", command=self.stop_attendance)
        self.stop_button.pack(pady=10)

        self.tree = ttk.Treeview(self.root, columns=("Name", "Time"), show='headings', height=15)
        self.tree.heading("Name", text="Name")
        self.tree.heading("Time", text="Time")
        self.tree.pack(pady=10)

        self.register_button = tk.Button(self.root, text="Register Person", command=self.register_person)
        self.register_button.pack(pady=10)

    def mark_attendance(self, name, frame):
        """Mark attendance of a person and save the corresponding image."""
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        self.attendance.append({"Name": name, "Time": date_time})
        self.tree.insert("", tk.END, values=(name, date_time))

        # Save the frame with person's name and time as filename
        image_path = f"captured/{name}_{date_time.replace(' ', '_').replace(':', '-')}.jpg"
        cv2.imwrite(image_path, frame)

    def capture_images(self, name, frame, count):
        """Save multiple images in a separate folder for the registered person."""
        person_folder = f"images/{name}"
        os.makedirs(person_folder, exist_ok=True)
        image_path = f"{person_folder}/{name}_{count}.jpg"
        cv2.imwrite(image_path, frame)

    def register_person(self):
        """Open a window for registering a new person with multiple images."""
        name = simpledialog.askstring("Input", "Enter the person's name:")
        if name:
            self.open_camera_for_registration(name)

    def open_camera_for_registration(self, name):
        """Open a camera feed window to capture 5 images for registration."""
        register_window = tk.Toplevel(self.root)
        register_window.title(f"Registering {name}")

        camera_label = tk.Label(register_window)
        camera_label.pack()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not found!")
            register_window.destroy()
            return

        capture_count = 0

        def capture_image_callback():
            nonlocal capture_count
            ret, frame = self.cap.read()
            if ret:
                self.capture_images(name, frame, capture_count + 1)
                capture_count += 1
                messagebox.showinfo("Success", f"Captured image {capture_count} of 5")

                if capture_count >= 5:
                    self.cap.release()
                    register_window.destroy()

                    # Reload encodings after registration
                    self.sfr.load_encoding_images_from_subfolders("images/")  # Load the newly added person's encoding
                    messagebox.showinfo("Completed", f"Registration for {name} is complete! Now you can start attendance.")

        capture_button = tk.Button(register_window, text="Capture Image", command=capture_image_callback)
        capture_button.pack()

        def update_camera_feed():
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(image=img)
                    camera_label.config(image=img_tk)
                    camera_label.image = img_tk

                camera_label.after(10, update_camera_feed)

        update_camera_feed()

    def start_attendance(self):
        """Start the face recognition attendance system."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not found!")
            return

        self.running = True
        self.run_face_recognition()

    def run_face_recognition(self):
        """Run face recognition and mark attendance."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect faces in the current frame
            face_locations, face_encodings, face_names = self.sfr.detect_known_faces(frame)

            print(f"Detected faces: {len(face_locations)}")  # Debug: Check how many faces were detected

            for face_loc, encoding, name in zip(face_locations, face_encodings, face_names):
                y1, x2, y2, x1 = face_loc

                # If the face is known (not unknown)
                if name != "Unknown":
                    print(f"Recognized known face: {name}")  # Debug: Show recognized name
                    if name not in [entry["Name"] for entry in self.attendance]:
                        self.mark_attendance(name, frame)
                else:
                    print("Unknown face detected.")  # Debug: If an unknown face is detected

                    # Only show the prompt once for the unknown person (based on unique encoding)
                    if not self.is_known_unknown(encoding):
                        self.show_unknown_person_prompt(frame, face_loc)
                        self.unknown_encodings.append(encoding)  # Add encoding to known unknowns

                # Draw bounding box and label on the face
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.imshow("Face Recognition Attendance", frame)
            key = cv2.waitKey(1)
            if key == 27:  # Escape key to stop
                break

        self.save_attendance()
        self.cap.release()
        cv2.destroyAllWindows()

    def is_known_unknown(self, encoding):
        """Check if an unknown person's encoding is already in the list of known unknowns."""
        for known_encoding in self.unknown_encodings:
            matches = face_recognition.compare_faces([known_encoding], encoding)
            if True in matches:
                return True  # This encoding has already been seen

        return False

    def show_unknown_person_prompt(self, frame, face_loc):
        """Prompt when an unknown face is detected."""
        messagebox.showinfo("Unknown Person", "Unknown person detected, attendance marked as Unknown.")
        y1, x2, y2, x1 = face_loc
        unknown_image_path = f"captured/Unknown_{self.unknown_counter}.jpg"
        cv2.imwrite(unknown_image_path, frame)
        self.unknown_counter += 1

    def stop_attendance(self):
        """Stop the attendance recording."""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def save_attendance(self):
        """Save the attendance data to a CSV file."""
        df = pd.DataFrame(self.attendance)
        df.to_csv("attendance.csv", index=False)
        messagebox.showinfo("Attendance Saved", "Attendance has been saved to 'attendance.csv'.")

    def run(self):
        """Start the GUI."""
        self.root.mainloop()


if __name__ == "__main__":
    attendance_system = FaceRecognitionAttendance()
    attendance_system.run()
