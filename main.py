"""
+============================================================+
- Tác Giả: Hoàng Thành
- Viện Toán Ứng dụng và Tin học(SAMI - HUST)
- Email: thanh.hoangvan051199@gmail.com
- Github: https://github.com/thanhhoangvan
+============================================================+
"""

import os

from cv2 import VideoCapture

from ParkTracking import CSV2Positions
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from make_model import *


class ParkingSpotTracking:
    def __init__(self, model, position_csv, class_names) -> None:
        self.Tracking = {}
        self.positions = self.CSV2Positions(position_csv)
        self.model = model
        self.class_names = class_names

    def drawResult(self, image, Result={}):
        border_color_red = (255, 0, 0)
        border_color_green = (0, 255, 0)

        slotIDs = Result.keys()
        for ids in slotIDs:
            type = Result[ids]
            (x0, y0, x1, y1) = self.positions[ids]
            if type=='non_car':
                cv2.rectangle(image, (x0, y0), (x1, y1), border_color_green, thickness=2)
            elif type=='car':
                cv2.rectangle(image, (x0, y0), (x1, y1), border_color_red, thickness=2)

        return image

    def CSV2Positions(self, csv_path):
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

    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        result = self.model.predict(image)
        return self.class_names[np.argmax(result[0])]

    def tracking(self, image):
        results = {}
        parking_slots = self.extractParkinSlot(image)
        slot_ids = parking_slots.keys()
        for slotID in slot_ids:
            slot_type = self.predict(parking_slots[slotID])
            results[slotID] = slot_type
        return results

    def extractParkinSlot(self, image):
        parking_slots = {}
        position_ids = self.positions.keys()
        
        for id in position_ids:
            (x0, y0, x1, y1) = self.positions[id]
            image_slot = image[y0:y1, x0:x1].copy()
            try:
                image_slot = cv2.resize(image_slot, (100, 50))
                parking_slots[id] = image_slot
            except:
                pass
        
        return parking_slots

    def main(self):
        video = cv2.VideoCapture('carPark.mp4')
        
        while True:
            ret, frame = video.read()
            if ret:
                result_tracking = self.tracking(frame)
                img_result = self.drawResult(frame, result_tracking)
                
                cv2.imshow('Park Tracking - Thanh HoangVan', img_result)
		        
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

if __name__=='__main__':
    n_classes = 2
    image_shape = (50, 100, 3)
    class_names = ['car', 'non_car']

    model = make_model(n_classes, image_shape)
    model.load_weights('my_h5_model.h5')
    
    position_csv = './positions.csv'
    
    PST = ParkingSpotTracking(model, position_csv, class_names)
    PST.main()
