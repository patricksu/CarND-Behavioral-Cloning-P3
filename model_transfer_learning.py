# Only use the images with angle > 0.05. Reducing from 8K to 3.5K
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import cv2
import csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras import optimizers
import math
from random import shuffle
import random
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 10, 'the number of epochs')
flags.DEFINE_integer('batch_size', 32, 'the batch size')
flags.DEFINE_integer('cols', 320, 'number of pixels horizontal')
flags.DEFINE_integer('rows', 160, 'number of pixels vertical')


driving_log_udacity = "./data"
driving_log_dir = []
driving_log_dir.append("./afterBridge3")

udacity_samples = []
samples = []

log_file_path = driving_log_udacity + "/driving_log.csv"
with open(log_file_path) as driving_log:
	reader = csv.reader(driving_log)
	for line in reader:
		udacity_samples.append(line)

for f in driving_log_dir:
	log_file_path = f + "/driving_log.csv"
	with open(log_file_path) as driving_log:
		reader = csv.reader(driving_log)
		for line in reader:
			samples.append(line)

# Augmentation1: adust brightness
def augment_brightness(img1):
	img = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
	img = np.array(img, dtype = np.float64)
	light = np.random.uniform() + 0.5
	img[:,:,2] = img[:,:,2] * light
	img[:,:,2][img[:,:,2] > 255] = 255
	img = np.array(img, dtype = np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
	return img
# Augmentation2: use left and right camera images
# This is implemented in the generator

# Augmentation3: Horizontal and vertical shifts
def trans_image(image,steer,trans_range):
    # Translation
	tr_x = trans_range*np.random.uniform()-trans_range/2
	steer_ang = steer + tr_x * 0.004
	tr_y = 40*np.random.uniform()-40/2
	#tr_y = 0
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
	image_tr = cv2.warpAffine(image,Trans_M,(FLAGS.cols,FLAGS.rows))
	return image_tr,steer_ang

# Augmentation4: shadow augmentation

def add_random_shadow(image):
	top_y = 320*np.random.uniform()
	top_x = 0
	bot_x = 160
	bot_y = 320*np.random.uniform()
	image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
	shadow_mask = 0*image_hls[:,:,1]
	X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
	shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
	#random_bright = .25+.7*np.random.uniform()
	if np.random.randint(2)==1:
		random_bright = .5
		cond1 = shadow_mask==1
		cond0 = shadow_mask==0
		if np.random.randint(2)==1:
			image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
		else:
			image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
	image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
	return image

# Augmentation5: flipping
# Implemented in the generator

def preprocess_image_file_train(line_data):
	i_lrc = np.random.randint(3)
	file = ''
	if (i_lrc == 0):
		file = line_data[1].strip()
		shift_ang = .25
	if (i_lrc == 1):
		file = line_data[0].strip()
		shift_ang = 0.
	if (i_lrc == 2):
		file = line_data[2].strip()
		shift_ang = -.25
	eles = file.split('/')
	path = eles[-3] + '/' + eles[-2] + '/' + eles[-1]
	y_steer = float(line_data[3]) + shift_ang
	image = cv2.imread(path)
	image = augment_brightness(image)
	# image,y_steer = trans_image(image,y_steer,20)
	image = add_random_shadow(image)
	image = np.array(image)
	ind_flip = np.random.randint(2)
	if ind_flip==0:
		image = cv2.flip(image,1)
		y_steer = -y_steer
	return image,y_steer

def generator_train(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		X = []
		y = []
		for i in range(batch_size):
			i_line = np.random.randint(num_samples)
			line_data = samples[i_line]
			img, y_steer = preprocess_image_file_train(line_data)
			X.append(img)
			y.append(y_steer)
		X = np.array(X)
		y = np.array(y)
		yield X, y


def generator_valid(samples, batch_size=32):
	num_samples = len(samples)
	while True:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset : offset + batch_size]
			images = []
			angles = []
			for record in batch_samples:
				eles = record[0].split('/')
				path = eles[-3] + '/' + eles[-2] + '/' + eles[-1]
				center_image = cv2.imread(path)
				center_angle = float(record[3])
				images.append(center_image)
				angles.append(center_angle)
			X = np.array(images)	
			y = np.array(angles)
			yield X, y

_, valid_samples = train_test_split(udacity_samples, test_size = 0.1)

train_generator = generator_train(samples, batch_size = FLAGS.batch_size)
valid_generator = generator_valid(valid_samples, batch_size = FLAGS.batch_size)

model = load_model('model_afterBridge_2.h5')
adam = optimizers.adam(lr = 0.0005)
model.compile(optimizer=adam,loss='mse')
model.fit_generator(train_generator, steps_per_epoch = math.ceil(len(samples)/FLAGS.batch_size), 
	epochs = FLAGS.epochs, verbose = 1, validation_data = valid_generator, 
	validation_steps = math.ceil(len(valid_samples)/FLAGS.batch_size))
model.save('model_afterBridge_3.h5')
