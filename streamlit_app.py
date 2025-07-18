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

                # Hitung posisi bounding box wajah
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # Hitung titik tengah atas kepala (kira-kira posisi mahkota)
                center_x = int((x1 + x2) / 2)
                top_y = y1

                # Gambar garis vertikal ke atas
                line_length = 150  # panjang garis ke atas
                line_thickness = 3
                line_color = (0, 0, 255)  # merah

                cv2.line(
                    img,
                    (center_x, top_y),
                    (center_x, max(0, top_y - line_length)),
                    line_color,
                    line_thickness
                )

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

st.set_page_config(page_title="S-Line Live Cam", layout="centered")
st.title("💃 S-Line Queen LIVE Detector")
st.write("📸 Arahkan kamera HP-mu dan lihat siapa yang jadi S-Line Queen secara langsung! 👑")

webrtc_streamer(
    key="sline",
    video_processor_factory=SLineOverlay,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    }
)
