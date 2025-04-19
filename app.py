import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import av
from PIL import Image

# Ensure the known_faces directory exists
if not os.path.exists("known_faces"):
    os.makedirs("known_faces")

# Utility: Load known faces
def load_known_faces():
    encodings = []
    names = []
    for file in os.listdir("known_faces"):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = face_recognition.load_image_file(f"known_faces/{file}")
            encoding = face_recognition.face_encodings(img)
            if encoding:
                encodings.append(encoding[0])
                names.append(os.path.splitext(file)[0])
    return encodings, names

# Utility: Mark attendance
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    new_row = {"Name": name, "Date": date, "Time": time}

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        if not ((df['Name'] == name) & (df['Date'] == date)).any():
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv("attendance.csv", index=False)
    else:
        pd.DataFrame([new_row]).to_csv("attendance.csv", index=False)

# Video processor class for face recognition
class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.known_encodings, self.known_names = load_known_faces()
        self.attended = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = self.known_names[match_index]

                if name not in self.attended:
                    mark_attendance(name)
                    self.attended.add(name)

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.set_page_config(page_title="Smart Attendance", layout="centered")
st.title("📸 Smart Face Attendance System")

menu = st.sidebar.radio("Choose an option", ["📷 Webcam Attendance", "🧍 Register New Face", "📊 View Attendance Log", "🧩 Add More Faces"])

# 1. Webcam Attendance Mode
if menu == "📷 Webcam Attendance":
    st.info("Show your face to the webcam. If you are registered, your attendance will be logged.")
    webrtc_streamer(key="face-attendance", video_processor_factory=FaceRecognitionProcessor)

# 2. Register New Face
elif menu == "🧍 Register New Face":
    st.subheader("Register a New Person")
    name = st.text_input("Enter Full Name")
    uploaded_image = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"])

    if name and uploaded_image:
        image = Image.open(uploaded_image)
        save_path = os.path.join("known_faces", f"{name.lower().replace(' ', '_')}.jpg")
        image.save(save_path)
        st.success(f"{name} registered successfully! You can now use webcam attendance.")
    elif uploaded_image and not name:
        st.warning("Please enter a name before uploading.")

# 3. Add More Faces
elif menu == "🧩 Add More Faces":
    st.subheader("Add New Face")
    new_name = st.text_input("Enter Name (e.g., Alice)", key="new_name")
    new_face_img = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"], key="new_face_img")

    if new_name and new_face_img:
        image = Image.open(new_face_img)
        save_path = os.path.join("known_faces", f"{new_name.lower().replace(' ', '_')}.jpg")
        # Save the uploaded image to the known_faces folder
        image.save(save_path)
        st.success(f"Face of {new_name} has been successfully registered!")
    else:
        st.warning("Please provide both a name and a face image.")

# 4. View Attendance Log
elif menu == "📊 View Attendance Log":
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)
    else:
        st.warning("No attendance data yet.")
