import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join, isdir, exists
from distutils.dir_util import copy_tree

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t')
parser.add_argument('-i')

args = parser.parse_args()

classifier = cv2.CascadeClassifier('result/cascade.xml')


def find(img) :
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	items = classifier.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in items:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	return img

def detect() :
	global args
	if int(args.t) == 0 :
		img = cv2.imread(args.i)
		if img is None: return

		cv2.imshow('img',find(img))
		
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	elif int(args.t) == 1:
		cap = cv2.VideoCapture(0)

		while(True):
		    ret, frame = cap.read()
		    cv2.imshow('frame',find(frame))
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		        break

		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()
	else:
		files = [f for f in listdir('images') if isfile(join('images', f))]
		os.system('mkdir detect_result')
		for f in files:
			cv2.imwrite('detect_result/' + f, find(cv2.imread('images/' + f)))


detect()