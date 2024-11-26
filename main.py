import mediapipe as mp
import cv2
import numpy as np
from time import time

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import requests
from os.path import exists

from MediaPipeHandModule import *
from Turtle import *


def landmarks_to_nparray(landmarks):
    return np.array([[keypoint.x, keypoint.y] for keypoint in landmarks])


def draw_info_on_field(field, turtle):
    d, u = turtle.internals.get('prev_thumb_down', False), turtle.internals.get('prev_thumb_up', False)
    field = cv2.putText(field, f'Speed:', [5, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, [90, 90, 90], 2, cv2.LINE_AA)
    field = cv2.putText(field, f'<', [125, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, [90, 90, 90+60*d], 2+2*d, cv2.LINE_AA)
    field = cv2.putText(field, f'{turtle.speed:03d}', [160, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, [90, 90, 90], 2, cv2.LINE_AA)
    field = cv2.putText(field, f'>', [228, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, [90, 90, 90+60*u], 2+2*u, cv2.LINE_AA)
    return field


def generate_output(frame, turtle: Turtle):
    output = np.zeros([max(frame.shape[0], turtle.field_size[0]), frame.shape[1] + turtle.field_size[1], 3], dtype='uint8')
    field = (np.ones([turtle.field_size[0], turtle.field_size[1], 3], dtype='uint8') * 255).astype('uint8')

    start = max(turtle.field_size[0], frame.shape[0]) // 2 - frame.shape[0] // 2
    output[start:start + frame.shape[0], 0:frame.shape[1], :] = frame

    turtle_rotated = turtle.get_rotated_image()
    turtle_image_pos = turtle.get_image_corner()

    turtle_aplha = np.concatenate((turtle_rotated[:, :, [3]], turtle_rotated[:, :, [3]], turtle_rotated[:, :, [3]]), axis=2) / 255.

    # caclulations for upper-left and lower-right corners of visible part of field and turtle image to draw
    a, b = max(turtle_image_pos[0], 0), min(turtle_image_pos[0] + turtle.image_shape[0], turtle.field_size[0])
    c, d = max(turtle_image_pos[1], 0), min(turtle_image_pos[1] + turtle.image_shape[1], turtle.field_size[1])
    e, f = max(-turtle_image_pos[0], 0), min(turtle.field_size[0] - turtle_image_pos[0], turtle.image_shape[0])
    g, h = max(-turtle_image_pos[1], 0), min(turtle.field_size[1] - turtle_image_pos[1], turtle.image_shape[1])
    # drawing visible part of turtle on field
    field[a:b, c:d] = turtle_rotated[e:f, g:h, :3] * turtle_aplha[e:f, g:h, :] + \
                        field[a:b, c:d] * (1 - turtle_aplha[e:f, g:h, :])
    
    field = draw_info_on_field(field, turtle)

    start = max(turtle.field_size[0], frame.shape[0]) // 2 - turtle.field_size[0] // 2
    output[start:start+turtle.field_size[0], frame.shape[1]:, :] = field
    return output


class MainApp:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.hand_module = MediaPipeHandModule()
        self.turtle = Turtle()
        self.start_time = int(time() * 1000)

    def main(self):
        prev_frame_time = time()
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                continue

            frame = self._prepare_frame(frame)
            self.hand_module.detect(frame, self.start_time)

            if self.hand_module.results:
                frame = self.hand_module.draw_landmarks_on_image(frame, self.hand_module.results)
                if self.hand_module.results.hand_landmarks:
                    self.turtle.analyse_gesture(landmarks_to_nparray(self.hand_module.results.hand_landmarks[0]))
                    if self.turtle.is_running:
                        self.turtle.move(time() - prev_frame_time)
            prev_frame_time = time()

            output_frame = generate_output(frame, self.turtle)

            cv2.imshow('MainApp', output_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.video.release()
        cv2.destroyAllWindows()

    def _prepare_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, y, _ = frame.shape
        if(x > y):
            frame = frame[(x-y)//2:(x+y)//2, y:0:-1, :]
        else:
            frame = frame[0:x, (y+x)//2:(y-x)//2:-1, :]
        return frame


if __name__ == '__main__':
    if not exists('hand_landmarker.task'):
        response = requests.get('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
        with open('hand_landmarker.task', 'wb') as file:
            file.write(response.content)

    app = MainApp()
    app.main()