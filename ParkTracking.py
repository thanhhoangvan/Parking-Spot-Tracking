import time
import csv
import cv2
import numpy as np

def show(video_link, positions):
	video = cv2.VideoCapture(video_link)

	while True:
		ret, frame = video.read()
		if not ret:
			break

		for p in positions:
			(x0, y0, x1, y1) = positions[p]
			cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)

		cv2.imshow('Park Tracking - Thanh HoangVan', frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()

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

if __name__=='__main__':
	video_link = './data/carPark.mp4'
	csv_link   = './positions.csv'

	positions = CSV2Positions(csv_link)
	show(video_link, positions)