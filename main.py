"""
+============================================================+
- Tác Giả: Hoàng Thành
- Viện Toán Ứng dụng và Tin học(SAMI - HUST)
- Email: thanh.hoangvan051199@gmail.com
- Github: https://github.com/thanhhoangvan
+============================================================+
"""

import os
import csv
import time

import cv2
from cv2 import waitKey
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Queue

# from tensorflow.keras.models import load_model
from make_model import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

(height, width, channel) = (50, 100, 3)
class_names = ['car', 'non_car']
n_classes = len(class_names)

border_color_red = (255, 0, 0)
border_color_green = (0, 255, 0)

weights_path = 'my_h5_model.h5'
positions_csv = 'positions.csv'
video_sample = 'carPark.mp4'

def CSV2Positions(csv_path):
    Positions = {}

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        position_count = 0
        for row in csv_reader:
            if position_count != 0:
                raw_position = [int(i) for i in row]
                Positions[raw_position[0]] = raw_position[1:]
            position_count += 1
    return Positions


class ExtractParkingSlot:
    def __init__(self, mp_running, positions, input_queue=None, output_queue=None):
        self.running = mp_running

        self.input_queue = input_queue
        self.output_queue = output_queue

        self.positions = positions
        self.processExtracting = None

    def start(self):
        self.processExtracting = Process(target=self.extracting, args=(self.running, self.positions, self.input_queue, self.output_queue))
        self.processExtracting.start()

    def extracting(self, running, positions, input_queue, output_queue):
        ids_position = positions.keys()

        while running.value == 1:
            if not input_queue.empty():
                image = input_queue.get()
                for spot_id in ids_position:
                    (x0, y0, x1, y1) = positions[spot_id]
                    img_spot = image[y0:y1, x0:x1].copy()
                    img_spot = cv2.cvtColor(img_spot, cv2.COLOR_BGR2RGB)
                    img_spot = cv2.resize(img_spot, (100, 50))

                    if not output_queue.full():
                        output_queue.put((spot_id, img_spot))

    def stop(self):

        while True:
            if not self.input_queue.empty:
                _ = self.input_queue.get()
            elif not self.output_queue.empty:
                _ = self.output_queue.get()
            else:
                break
        
        self.processExtracting.join()


class ClassifyParkingSpot:
    def __init__(self, mp_running, input_queue=None, output_queue=None):
        self.running = mp_running
        self.processTracking = None
        self.input_queue = input_queue
        self.output_queue = output_queue
    
    def start(self):
        self.processTracking = Process(target=self.predictModel, args=(self.running, self.input_queue, self.output_queue))
        self.processTracking.start()
    
    def predictModel(self, running_flag, input_queue, out_queue):

        model = make_model(n_classes, (height, width, channel))
        model.load_weights(weights_path)

        while running_flag.value == 1:
            if not input_queue.empty():
                (parking_spot_id, img_parking_spot) = input_queue.get()
                img_parking_spot = np.expand_dims(img_parking_spot, axis=0)
                
                result = model.predict(img_parking_spot)

                if not out_queue.full():
                    out_queue.put((parking_spot_id, class_names[np.argmax(result[0])]))

    def stop(self):

        while True:
            if not self.input_queue.empty:
                _ = self.input_queue.get()
            elif not self.output_queue.empty:
                _ = self.output_queue.get()
            else:
                break
        
        self.processTracking.join()


class ParkingSpotTracking:
    def __init__(self):
        self.mp_running = Value("I", 0)
        self.positions = CSV2Positions(positions_csv)
        self.video = cv2.VideoCapture(video_sample)
        self.parking_status = {}

        self.frame_queue = Queue(maxsize=100)
        self.slot_queue = Queue(maxsize=100)
        self.predict_queue = Queue(maxsize=100)

        self.Extract = ExtractParkingSlot(self.mp_running, self.positions, self.frame_queue, self.slot_queue)
        self.Predict = ClassifyParkingSpot(self.mp_running, self.slot_queue, self.predict_queue)

    def start(self):
        self.mp_running.value = 1
        time.sleep(0.5)
        
        self.initStatus()

        self.Extract.start()
        self.Predict.start()

        self.Tracking()

    def Visualize(self, image) -> np.array:
        for ids in self.positions.keys():
            (x0, y0, x1, y1) = self.positions[ids]
            status = self.parking_status[ids]
            
            if status[0]=='non_car':
                cv2.rectangle(image, (x0, y0), (x1, y1), border_color_green, thickness=2)
            elif status[0]=='car':
                cv2.rectangle(image, (x0, y0), (x1, y1), border_color_red, thickness=2)

        return image
 
    def initStatus(self):
        idSpot = self.positions.keys()
        for ids in idSpot:
            self.parking_status[ids] = ['non_car', time.time()]

    def Tracking(self):
        time.sleep(3)
        while True:
            ret, frame = self.video.read()
            if frame is None:
                continue
            if ret:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    
                    if not self.predict_queue.empty():
                        (ids, status_label) = self.predict_queue.get()
                        self.parking_status[ids] = [status_label, time.time()]
                
                    image = self.Visualize(frame.copy())
                    cv2.imshow('Tracking', image)

                    if waitKey(25)==ord('q'):
                        break
            
            else:
                break

        cv2.destroyAllWindows()
        self.stop()

    def stop(self):
        self.mp_running.value = 0
        time.sleep(2)
        self.Extract.stop()
        self.Predict.stop()


if __name__=='__main__':
    PST = ParkingSpotTracking()
    PST.start()
