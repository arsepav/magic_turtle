import cv2
import numpy as np
from time import time

def angle_of_vector(p1, p2):
    v = p2 - p1
    return np.arctan2(v[1], v[0])


def angle_between_vectors(p1, p2, p3, p4):
    v1 = p2 - p1
    v2 = p4 - p3
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))/np.pi*180

def direction_between_vectors(p1, p2, p3, p4):
    v1 = p2 - p1
    v2 = p4 - p3
    return np.cross(v1, v2) > 0


class Turtle:
    def __init__(self, speed=100, field_size=(800, 800), turtle_image_path='turtle.png'):
        self.pos = np.array([200, 200])
        self.angle = 0
        self.is_running = False
        self.field_size = np.array(field_size)
        self.speed = speed
        self.image = cv2.resize(cv2.imread(turtle_image_path, cv2.IMREAD_UNCHANGED), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
        self.image_shape = np.array(self.image.shape)[:2]
        self.internals = {}

    def move(self, dt):
        self.pos = self.pos + np.array([np.cos(self.angle/180*np.pi), np.sin(self.angle/180*np.pi)]) * self.speed * dt
        self.pos[np.where(self.pos < 0)] = 0
        self.pos = np.minimum(self.pos, self.field_size)

    def analyse_gesture(self, pos):
        palm_center = pos[[0, 5, 9, 13, 17]].mean(axis=0)
        palm_size = np.linalg.norm(pos[[0, 5, 9, 13, 17]] - palm_center, axis=1).mean() * 2
        closed = [
            angle_between_vectors(*pos[[0, 17, 19, 20]]) > 90,
            angle_between_vectors(*pos[[0, 13, 15, 16]]) > 90,
            angle_between_vectors(*pos[[0, 9, 11, 12]]) > 90,
            angle_between_vectors(*pos[[0, 5, 7, 8]]) > 90,
            not ((direction_between_vectors(*pos[[0, 5, 3, 4]]) == direction_between_vectors(*pos[[0, 17, 0, 5]])) and
                angle_between_vectors(*pos[[0, 5, 3, 4]]) > 25)
        ]

        if closed == [True, True, True, False, True]:
            self.is_running = True
            self.angle = angle_of_vector(*(pos[[7, 8]][:, ::-1])) / np.pi * 180
        else:
            self.is_running = False

        if closed == [True, True, True, True, False]:
            direction = angle_of_vector(*(pos[[1, 4]][:, ::-1]))/np.pi*180
            self.update_speed(direction)
        else:
            self.internals['prev_thumb_up'] = False
            self.internals['prev_thumb_down'] = False

    def update_speed(self, direction):
        if direction < -145 or direction > 125:
            self._handle_thumb_up()
        elif direction > -45 and direction < 45:
            self._handle_thumb_down()

    def _handle_thumb_up(self):
        if self.internals.get('prev_thumb_up', False):
            if time() - self.internals.get('prev_thumb_up_trig', 0) > 1:
                self.internals['prev_thumb_up_trig'] = time()
                self.speed = min(self.speed + 10, 500)
                print('TU')
        else:
            self.internals['prev_thumb_up'] = True
            self.internals['prev_thumb_up_trig'] = time()


    def _handle_thumb_down(self):
        if self.internals.get('prev_thumb_down', False):
            if time() - self.internals.get('prev_thumb_down_trig', 0) > 1:
                self.internals['prev_thumb_down_trig'] = time()
                self.speed = max(self.speed - 10, 10)
                print('TD')
        else:
            self.internals['prev_thumb_down'] = True
            self.internals['prev_thumb_down_trig'] = time()

    def get_image_corner(self):
        return self.pos.astype('int') - self.image_shape // 2

    def get_rotated_image(self):
        rot_mat = cv2.getRotationMatrix2D(self.image_shape / 2, self.angle, 1.0)
        return cv2.warpAffine(self.image, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_LINEAR)

