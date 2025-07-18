import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np

mp_face_detection = mp.solutions.face_detection

def detect_face_and_add_sline(image):
    image_np = np.array(image)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if not results.detections:
            return image, "Wajah tidak terdeteksi 😢"

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", size=30)  # pastikan font tersedia

        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            h, w = image.size[1], image.size[0]
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            draw.text((x, y - 40), "👑 S-Line Queen!", fill="magenta", font=font)

        return image, f"{len(results.detections)} wajah terdeteksi 👀"
