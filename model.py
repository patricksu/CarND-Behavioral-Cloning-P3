import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import cv2
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
import math
from random import shuffle

driving_log_files = []
driving_log_files.append("./data/Peng_data")   #_track1_center
driving_log_files.append("./data/Peng_data_track1_recovery")
driving_log_files.append("./data/Peng_data_track1_reverse")

samples = []
steering = []

for f in driving_log_files:
	log_file_path = f + "/driving_log.csv"
	with open(log_file_path) as driving_log:
		reader = csv.reader(driving_log)
		for line in reader:
			samples.append(line)

train_samples, valid_samples = train_test_split(samples, test_size = 0.2)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset : offset + batch_size]
			images = []
			angles = []
			for record in batch_samples:
				file = line[0].split('/')
				path = 'data' + '/' + file[-3] + '/' + file[-2] + '/' + file[-1]
				center_image = cv2.imread(path)
				center_angle = float(line[3])
				images.append(center_image)
				angles.append(center_angle)
			X = np.array(images)
			y = np.array(angles)
			yield sklearn.utils.shuffle(X, y)
batch_size = 32
train_generator = generator(train_samples, batch_size = batch_size)
valid_generator = generator(valid_samples, batch_size = batch_size)


model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0-0.5))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

print(math.ceil(len(train_samples)/batch_size))

model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, steps_per_epoch = math.ceil(len(train_samples)/batch_size), epochs = 3,
	verbose = 1, validation_data = valid_generator, validation_steps = math.ceil(len(valid_samples)/batch_size))
model.save('model.h5')
