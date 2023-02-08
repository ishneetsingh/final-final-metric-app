import streamlit as st
from insightface.app import FaceAnalysis
import cv2
import time

@st.cache(allow_output_mutation=True)
def load_insightface():
    app = FaceAnalysis(allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

INSIGHTFACE = load_insightface()

def blur_faces_insightface(frame):
    face_box_time_start = time.time()
    faces = INSIGHTFACE.get(frame)
    face_box_time_end = time.time()

    blurring_start = time.time()
    height, width = frame.shape[:2]
    kernel_size = int(0.0000076853 * height * width + 10.0335)
    if kernel_size % 2 == 0:
        kernel_size += 1

    for face in faces:
            # Blurring
            x1  = int(face['bbox'][0])
            y1  = int(face['bbox'][1])
            x2 = int(face['bbox'][2])
            y2 = int(face['bbox'][3])

            roi = frame[y1:y2, x1:x2]
            roi = cv2.GaussianBlur(roi, (23, 23), 30)
            frame[y1:y2, x1:x2] = roi 
    blurring_end = time.time()

    face_box_time = face_box_time_end - face_box_time_start
    blurring_time = blurring_end - blurring_start
    return face_box_time, blurring_time