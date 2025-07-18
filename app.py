import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

class SLineOverlay(VideoTransformerBase):
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = img.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                cv2.putText(img, "👑 S-Line Queen!", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        return img

st.set_page_config(page_title="S-Line Live Cam", layout="centered")
st.title("💃 S-Line Queen LIVE Detector")
st.write("📸 Arahkan kamera HP-mu dan lihat siapa yang jadi S-Line Queen secara langsung! 👑")

webrtc_streamer(key="sline", video_processor_factory=SLineOverlay)
