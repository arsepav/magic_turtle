import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from time import time, sleep


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

turtle = cv2.resize(cv2.imread('turtle.png', cv2.IMREAD_UNCHANGED), None, fx=0.3, fy = 0.3, interpolation=cv2.INTER_CUBIC)
turtle_shape = np.array(turtle.shape)[:2]


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


class Mediapipe_HandModule():
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.mp_hands = solutions.hands
        self.results = None
        self.results_ts = None
        self.turtle_pos = np.array([200, 200])
        self.turtle_angle = 0
        self.turtle_running = False
        self.field_size = np.array([800, 800])
        self.analysis_fps = 15
        self.free = True
        self.internals = {}
        self.speed = 100

    def draw_landmarks_on_image(self, rgb_image, detection_result, ts, cur_t):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        #Hand_landmarks_list = detection_result
        annotated_image = cv2.cvtColor(np.copy(rgb_image), cv2.COLOR_BGR2RGB)

        # Loop through the detected Hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            # Draw the Hand landmarks.
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

    # Create a Hand landmarker instance with the live stream mode:
    def callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.free = True
        self.results = result
        self.results_ts = timestamp_ms

    def turtle_move(self, dt):
        self.turtle_pos = self.turtle_pos + np.array([np.cos(self.turtle_angle/180*np.pi), np.sin(self.turtle_angle/180*np.pi)]) * self.speed * dt
        self.turtle_pos[np.where(self.turtle_pos < 0)] = 0
        if self.turtle_pos[0] > self.field_size[0]:
            self.turtle_pos[0] = self.field_size[0]
        if self.turtle_pos[1] > self.field_size[1]:
            self.turtle_pos[1] = self.field_size[1]


    def analyse_gesture(self, pos):
        #print(pos[[0, 1, 5, 9, 15, 17]])
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
            self.turtle_running = True
            self.turtle_angle = angle_of_vector(*(pos[[7, 8]][:, ::-1]))/np.pi*180
        else:
            self.turtle_running = False

        if closed == [True, True, True, True, False]:
            direction = angle_of_vector(*(pos[[1, 4]][:, ::-1]))/np.pi*180
            if direction < -145 or direction > 125: # thumb up
                if self.internals.get('prev_thumb_up', False):
                    if time() - self.internals.get('prev_thumb_up_seen', 0) > 2:
                        self.internals['prev_thumb_up_trig'] = time()

                    if time() - self.internals.get('prev_thumb_up_trig', 0) > 1:
                        self.internals['prev_thumb_up_trig'] = time()
                        self.speed = min(self.speed+10, 500)
                        print('TU')
                else:
                    self.internals['prev_thumb_up'] = True
                    self.internals['prev_thumb_up_trig'] = time()
                        
                self.internals['prev_thumb_up_seen'] = time()
            else:
                self.internals['prev_thumb_up'] = False

            if direction > -45 and direction < 45: # thumb down
                if self.internals.get('prev_thumb_down', False):
                    if time() - self.internals.get('prev_thumb_down_seen', 0) > 2:
                        self.internals['prev_thumb_down_trig'] = time()

                    if time() - self.internals.get('prev_thumb_down_trig', 0) > 1:
                        self.internals['prev_thumb_down_trig'] = time()
                        self.speed = max(self.speed-10, 10)
                        print('TD')
                else:
                    self.internals['prev_thumb_down'] = True
                    self.internals['prev_thumb_down_trig'] = time()

                self.internals['prev_thumb_down_seen'] = time()
            else:
                self.internals['prev_thumb_down'] = False
        else:
            self.internals['prev_thumb_up'] = False
            self.internals['prev_thumb_down'] = False




    def output(self, frame, field_size):
        output = np.zeros([max(frame.shape[0], field_size[0]), frame.shape[1] + field_size[1], 3], dtype='uint8')
        field = (np.ones([field_size[0], field_size[1], 3], dtype='uint8') * 255).astype('uint8')
        #print(output.dtype)
        start = max(field_size[0], frame.shape[0]) // 2 - frame.shape[0] // 2
        output[start:start+frame.shape[0], 0:frame.shape[1], :] = frame
        turtle_pos_cur = self.turtle_pos.astype('int') - turtle_shape // 2

        #print(turtle_shape / 2)
        rot_mat = cv2.getRotationMatrix2D(turtle_shape / 2, self.turtle_angle, 1.0)
        turtle_rotated = cv2.warpAffine(turtle, rot_mat, turtle.shape[1::-1], flags=cv2.INTER_LINEAR)
        turtle_aplha = np.concatenate((turtle_rotated[:, :, [3]],turtle_rotated[:, :, [3]],turtle_rotated[:, :, [3]]), axis=2) / 255.

        a = max(turtle_pos_cur[0], 0)
        b = min(turtle_pos_cur[0]+turtle_shape[0], field_size[0])
        c = max(turtle_pos_cur[1], 0)
        d = min(turtle_pos_cur[1]+turtle_shape[1], field_size[1])
        e = max(-turtle_pos_cur[0], 0)
        f = min(field_size[0]-turtle_pos_cur[0], turtle_shape[0])
        g = max(-turtle_pos_cur[1], 0)
        h = min(field_size[1]-turtle_pos_cur[1], turtle_shape[1])
        field[a:b, c:d] = \
            turtle_rotated[e:f, g:h, :3] * turtle_aplha[e:f, g:h, :] + \
            field[a:b, c:d] * (1-turtle_aplha[e:f, g:h, :])
        
        d, u = self.internals.get('prev_thumb_down', False), self.internals.get('prev_thumb_up', False)
        field = cv2.putText(field, f'Speed:', [5, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, [90, 90, 90], 2, cv2.LINE_AA)
        field = cv2.putText(field, f'<', [125, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, [90, 90, 90+60*d], 2+2*d, cv2.LINE_AA)
        field = cv2.putText(field, f'{self.speed:03d}', [160, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, [90, 90, 90], 2, cv2.LINE_AA)
        field = cv2.putText(field, f'>', [228, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, [90, 90, 90+60*u], 2+2*u, cv2.LINE_AA)

        start = max(field_size[0], frame.shape[0]) // 2 - field_size[0] // 2
        output[start:start+field_size[0], frame.shape[1]:, :] = field
        return output

    
    def main(self):
        self.video = cv2.VideoCapture(0)
        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=self.callback
            )
        )
        self.start_time = int(time()*1000)
        self.new_analysis_time = time()

        while self.video.isOpened():
            # Capture frame-by-frame
            ret, frame = self.video.read()

            if not ret:
                continue

            x, y, _ = frame.shape
            if(x > y):
                frame = frame[(x-y)//2:(x+y)//2, y:0:-1, :]
            else:
                frame = frame[0:x, (y+x)//2:(y-x)//2:-1, :]

            #if self.new_analysis_time < time():
            if self.free:
                self.free = False
                #self.new_analysis_time += 1/self.analysis_fps

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.landmarker.detect_async(mp_image, int(time()*1000) - self.start_time)
            
            if(not (self.results is None)):
                frame = self.draw_landmarks_on_image(mp_image.numpy_view(), self.results, self.results_ts, int(time()*1000) - self.start_time)
                if self.results.hand_landmarks:
                    self.analyse_gesture(np.array([[i for i in [keypoint.x, keypoint.y]] for keypoint in self.results.hand_landmarks[0]]))
                else:
                    self.turtle_running = False

            if self.turtle_running:
                print('moving')
                self.turtle_move(time() - self.internals.get('prev_frame_time', time()))
            self.internals['prev_frame_time'] = time()
            out_image = self.output(frame, self.field_size)
            cv2.imshow('Show', out_image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.video.release()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    body_module = Mediapipe_HandModule()
    body_module.main()
