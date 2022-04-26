"""
+============================================================+
- Tác Giả: Hoàng Thành
- Viện Toán Ứng dụng và Tin học(SAMI - HUST)
- Email: thanh.hoangvan051199@gmail.com
- Github: https://github.com/thanhhoangvan
+============================================================+
"""


import csv
import cv2
import numpy as np


def marked(img, x0, y0, x1, y1):
	shapes = np.zeros_like(img, np.uint8)
	cv2.rectangle(shapes, (x0, y0), (x1, y1), (118, 185, 0), cv2.FILLED)
	cv2.rectangle(shapes, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
	img_marked = img.copy()
	alpha = 0.2
	mask = shapes.astype(bool)
	img_marked[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]
	return img_marked

def markingParkingArea(event, x, y, flags, param):
	global ix, iy, drawing, img, img_temp, count, csv_writer

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix = x
		iy = y
		img_temp = img.copy()

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			img = img_temp.copy()
			img = marked(img, ix, iy, x, y)

	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		img = img_temp.copy()
		img = marked(img, ix, iy, x, y)

		count += 1
		POSITION[count] =  [ix, iy, x, y]
		print('add new parking area: {:4} {:4} {:4} {:4}'.format(ix, iy, x, y))
		csv_writer.writerow({'ID': count, 'x0': ix, 'y0': iy, 'x1': x, 'y1': y})


if __name__ == '__main__':

	# Variables
	ix = -1
	iy = -1
	drawing = False
	POSITION = {}
	count = 0

	# Input video
	video = cv2.VideoCapture('./data/carPark.mp4')
	_, img = video.read()
	img_temp = img.copy()
	video.release()

	# Config CSV file output
	fieldnames = ['ID', 'x0', 'y0', 'x1', 'y1']
	csv_file = open('./positions.csv','w')
	csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
	csv_writer.writeheader()

	# Mouse Event config
	windowsName = 'Marking Parking Area! - Thanh HoangVan'
	cv2.namedWindow(windowsName)
	cv2.setMouseCallback(windowsName, markingParkingArea)  

	# Show results
	while True:
		cv2.imshow(windowsName, img)
		key = cv2.waitKey(1)
		if key==ord('q'):
			break
	cv2.destroyAllWindows()

	print('='*40)
	print('You have marked {} Parking Areas'.format(count))
	for i in range(count):
		(x0,y0, x1, y1) = POSITION[i+1]
		print('ID: {:2} - ({:4}, {:4})->({:4}, {:4})'.format(i+1,x0, y0, x1, y1))

	csv_file.close()
	print('='*40)
	print('Saved all positions to csv file!')
