"""
+============================================================+
- Tác Giả: Hoàng Thành
- Viện Toán Ứng dụng và Tin học(SAMI - HUST)
- Email: thanh.hoangvan051199@gmail.com
- Github: https://github.com/thanhhoangvan
+============================================================+
"""

import os
import cv2
import csv

position_csv = './positions.csv'
video_url = 'carPark.mp4'

data_dir   = './data/'
car_dir    = data_dir + 'car/'
noncar_dir = data_dir + 'non_car/'

(height, width) = (50, 100)

def CSV2Positions(csv_file_link):
	Positions = {}

	with open(csv_file_link, 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		position_count = 0
		for row in csv_reader:
			if position_count != 0:
				raw_position = [int(i) for i in row]
				Positions[raw_position[0]] = raw_position[1:]
			position_count += 1
	return Positions

def cropParkingImages(image=None, positions={}):
    count = 0
    position_ids = positions.keys()
    for id in position_ids:
        (x0, y0, x1, y1) = positions[id]
        parking_slot = image[y0:y1, x0:x1].copy()

        try:
            parking_slot = cv2.resize(parking_slot, (width, height))
            cv2.imwrite(data_dir + str(count)+'.jpg', parking_slot)
        except:
            pass

        count += 1

    cv2.destroyAllWindows()

if __name__=='__main__':
    positions = CSV2Positions(position_csv)

    video = cv2.VideoCapture(video_url)
    while True:
        ret, frame = video.read()
        
        if ret:
            cropParkingImages(frame, positions)
        else:
            break
    video.release()
    print('crop successful')