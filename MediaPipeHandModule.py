import mediapipe as mp
import cv2
import numpy as np
from time import time

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


class MediaPipeHandModule:
    def __init__(self, model_path='hand_landmarker.task', num_hands=1, min_confidence=0.5):
        self.results = None
        self.results_ts = None
        self.free = True
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                num_hands=num_hands,
                min_hand_detection_confidence=min_confidence,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self.callback
            )
        )

    def callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.results = result
        self.results_ts = timestamp_ms
        self.free = True

    def detect(self, frame, start_time):
        if self.free:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.landmarker.detect_async(mp_image, int(time() * 1000) - start_time)
            self.free = False

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = cv2.cvtColor(np.copy(rgb_image), cv2.COLOR_BGR2RGB)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks_list[idx]
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )
        return annotated_image
