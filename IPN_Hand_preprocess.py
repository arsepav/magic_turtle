import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from time import sleep, time
from os import listdir
from os.path import isdir, exists


class Data:
    def recreate(self, data_size):
        self.data = [None for i in range(data_size)]
        self.in_process = 0


    def callback(self, result, output_image, idx):
        if result.hand_landmarks:
            tmp = np.array([i for keypoint in result.hand_landmarks[0] for i in [keypoint.x, keypoint.y]])
        else:
            tmp = np.array([0 for _ in range(21 * 2)])
        self.data[idx] = tmp
        self.in_process -= 1


start_time = time()
_print = print
def print(*args, **kwargs):
    dt = int(time() - start_time)
    _print(f'[{dt//60:02d}:{dt%60:02d}]', *args, **kwargs)


data = Data()

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=data.callback)

inputs_path = 'IPN_Hand/frames/'
folders = sorted([i for i in listdir(inputs_path) if isdir(inputs_path+i)])
for foldername in folders:
    res_path = inputs_path + foldername
    if exists(f'{res_path}.csv'):
        print(f'File {res_path}.csv already exists')
        continue
    images = sorted([i for i in listdir(res_path) if i[-3:] == 'jpg'])#[:200]
    ll = len(images)
    data.recreate(ll)
    detector = vision.HandLandmarker.create_from_options(options)
    
    for idx, imgname in enumerate(images):
        data.in_process += 1
        while data.in_process > 400:
            #print(f'cur in process {data.in_process} with last {idx-1} waiting to add')
            sleep(0.5)
        if idx%300 == 0:
            print(f'{foldername}: {idx}/{ll}')
        detector.detect_async(
           mp.Image.create_from_file(f'{res_path}/{imgname}'),
           idx
        )
        sleep(0.01)

    print('Waiting to finish')
    while data.in_process > 0:
        #print(f'cur in process {data.in_process} waiting to finish')
        sleep(1)
    np.savetxt(f'{res_path}.csv', data.data, fmt='%.08f', delimiter=',')
    print(f'{res_path}.csv finished')
    #break