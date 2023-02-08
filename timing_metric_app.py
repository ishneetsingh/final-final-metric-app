import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from MoveNet_Processing_Utils import movenet_processing
import av
import time


st.title('timing test')

def callback(frame):
    img = frame.to_ndarray(format="rgb24")

    global rendering_time_arr
    global action_recognition_arr
    global face_detection_arr
    global overall_time_arr
    global noFrames
    global startTime
    
    out_image = img.copy()
    out_image, rendering_time, classifying_time, face_detection_time = movenet_processing(out_image, max_people=max_people, blur_faces=blurring, insightface=insightface)
    noFrames += 1

    if rendering_time != -1:
        rendering_time_arr.append(rendering_time)
        action_recognition_arr.append(classifying_time)
        face_detection_arr.append(face_detection_time)

    mean_rendering_time = sum(rendering_time_arr) / len(rendering_time_arr)
    mean_action_recognition_time = sum(action_recognition_arr) / len(action_recognition_arr)
    mean_face_detection_time = sum(face_detection_arr) / len(face_detection_arr)
    
    currTime = time.time()
    fps = noFrames / (currTime - startTime)

    print('-'*20)
    print(f'ACTION RECOGNITION TIME - {mean_action_recognition_time * 1000: 03f}ms')

    if blurring:
        if not insightface:
            print(f'FACE DETECTION TIME [Proposed] - {mean_face_detection_time * 1000: 03f}ms')
        else:
            print(f'FACE DETECTION TIME [InsightFace] - {mean_face_detection_time * 1000: 03f}ms')

    print(f'RENDERING TIME - {mean_rendering_time * 1000: 03f}ms')
    print(f'FPS - {fps: 01f}')
    print('-'*20)

    return av.VideoFrame.from_ndarray(out_image, format="rgb24")

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Note: inter is for interpolating the image (to shrink it)
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        ratio = width/float(w)
        dim = (int(w * ratio), height)
    else:
        ratio = width/float(w)
        dim = (width, int(h * ratio))

    # Resize image
    return cv2.resize(image, dim, interpolation=inter)

render_metric, ar_metric, face_det_metric, fps_metric = st.columns(4)

max_people = st.number_input('Maximum Number of People', value = 1, min_value=1, max_value=6)
blurring = st.checkbox("Face Blurring", value=False)
insightface = st.checkbox("InsightFace", value=False)

ctx = webrtc_streamer(
    key="real-time",
    video_frame_callback=callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    # For Deploying
    rtc_configuration={
            "iceServers": [
        {
            "urls": "stun:openrelay.metered.ca:80",
        },
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443?transport=tcp",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        ]
    }
)

if ctx.state.playing:
    rendering_time_arr = []
    action_recognition_arr = []
    face_detection_arr = []
    overall_time_arr = []
    noFrames = 0
    startTime = time.time()