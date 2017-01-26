# import necessary modules
import os
import numpy as np
import pandas as pd
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import BaseLogger
from sklearn.utils import shuffle
import cv2
from pathlib import Path

print('Modules loaded.')

# ====================================================
# ====================================================
# ====================================================
# set some paramters needed for the training of the model
batch = 10
epoch = 1
validation = 0.2
use_model_version = 8

# ====================================================
# ====================================================
# ====================================================


# change working directory to current directory
#os.chdir("/home/alex/Documents/Udacity-SDC-T1-P3/")

# remove fiel of numpy array if needed
data_array_file_name = "data-array.npy"

my_file = Path(data_array_file_name)
if my_file.is_file():
    print('file exists... we remove it')
    os.remove(data_array_file_name)
    

# Load the csv data
df = pd.read_csv('/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/data2/driving_log.csv', header=0)
 
print('csv loaded.')  

# we do not need the entire dataframe - we only need two columns, so we 
# cut them out
df = df[['center', 'steering']]

# print some info about the dataframe
print("rows {} columns {}".format(df.shape[0], df.shape[1]))


# function to normalize a greyscale image so that is has the range of -0.5 and 0.5
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

# function to scale the iamge by 50% 
def cut_and_scale(img):
    img = img[60:160, 0:320]
    img = cv2.resize(img,(160, 50), interpolation = cv2.INTER_CUBIC)
    return img
    
    
# create array where we store the combination of dataframe and image data
arr = list()
x = len(df['steering'])

# for each row of the csv we load the center camera image and add the image data to array
for i in range(0, x):
    path = df['center'][i]
    img = cv2.imread(path,0)    
    arr.append(cut_and_scale(img))

# convert list to array
myarray = np.asarray(arr) 
myarray = np.expand_dims(myarray, 3)    
myarray = normalize_grayscale(myarray)    
print(myarray.shape)

print("--------------- SAVE -----------------") 
np.save("data-array.npy", myarray, allow_pickle=True, fix_imports=True)
print("--------------- data-array.npy SAVED  -----------------") 


print("--------------- LOAD data-array.npy -----------------")
myarray = np.load('data-array.npy')
print("--------------- LOADED -----------------")



X_train = myarray

df = pd.read_csv('/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/data2/driving_log.csv', header=0)

print('csv loaded.')  

# for training we need the Y values as list
y_train = df['steering'].values.tolist()

# we shuffle data to prevent some side effects
X_train, y_train = shuffle(X_train, y_train) 

print("shape of X_train")
print(X_train.shape)

print("--------------- BUILDING THE MODEL  -----------------")

# building the model - see readme.md for details
model = None
model = Sequential()

# add two convolutional layers
model.add(Convolution2D(64, 3, 3, input_shape=(50, 160, 1)))
print(model.output_shape)
model.add(MaxPooling2D((2, 2)))
print(model.output_shape)
# dropout to prevent overfitting
model.add(Dropout(0.3))
# another convolutional layer
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(MaxPooling2D((2, 2)))
print(model.output_shape)
# dropout to prevent overfitting
model.add(Dropout(0.3))
# convert the array to a one dimensional array
model.add(Flatten())
print(model.output_shape)
# recuce number of nodes from several thousand to 1
# and add several Relu as activation function
model.add(Dense(1000))
model.add(Activation('relu'))
print(model.output_shape)
model.add(Dense(50))
model.add(Activation('relu'))
print(model.output_shape)
model.add(Dense(1))
# final node/layer and activation fucnction tanh to have range from -1 to 1
model.add(Activation('tanh'))
print(model.output_shape)  
    
print("--------------- Save model architecture  -----------------")
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()

# save the model
print("--------------- TRAINING THE MODEL  -----------------")

# set optimization algorithm and how we evaluate the outcome - we use 
# mse = mean squared error
model.compile(loss='mse', optimizer='adam')

# Train on 10027 samples, validate on 2507 samples
model.fit(X_train, y_train, batch_size=batch, nb_epoch=epoch, verbose=1, callbacks=[BaseLogger()], validation_split=validation)

print("--------------- Save weights  -----------------")
# save the results 
model.save_weights('model.h5')
print("--------------- ALL DONE  -----------------")