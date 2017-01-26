import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from decimal import *
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import os
import sys

import math

#os.chdir("F:\\CODE\\Udacity-Self-Driving-Car\\Term-1\\Project-3")
#os.chdir("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/")
#os.chdir("/home/alex/Documents/Udacity-SDC-T1-P3/")

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


# normalize greyscale like in training data
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

# scale the image and cut something away like in training data
def cut_and_scale(img):
    img = img[60:160, 0:320]
    img = cv2.resize(img,(160, 50), interpolation = cv2.INTER_CUBIC)
    return img
    
# ensure speed is not getting more than 10 
def speed_control(speed):
    #print("speed:")
    #print(speed)
    throttle = 0.2
    if Decimal(speed) > Decimal("10"):
        throttle = 0.0
    return throttle

@sio.on('telemetry')
def telemetry(sid, data):

    # throttle data from car
    throttle = data["throttle"]
    # speed data fromcar
    speed = data["speed"]
    # image data from car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    # convert image to numpy array
    image_array = np.asarray(image)
    
    cv2.imwrite('test-camera-raw.jpg',image_array)    
    img = cv2.imread('test-camera-raw.jpg', 0)
    

    # cut the image and scale it down to right size
    image_array = cut_and_scale(img)
    
    # normalze grey scale as we did with training data
    image_array = normalize_grayscale(image_array)  
    
    # expand dimensions of the array
    image_array = np.expand_dims(image_array, 3) 

    # ensure array has right shape for model
    transformed_image_array = image_array[None, :, :, :]
    
    # predict steering angle using the model 
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
   
    # in case steerin angle is bigger than 1 etc. then limit it
    if steering_angle > 1:
        steering_angle = 0.99
    if steering_angle < -1:
        steering_angle = -0.99
        
    print(steering_angle)
    
    # we limit the speed and ensure speed is constant
    throttle = speed_control(speed)
    
    # send the throttle and steering angle to car
    send_control(steering_angle, throttle)
    

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    #print("in send_control")
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Remote Driving')
    #parser.add_argument('model', type=str,
    #help='Path to model definition json. Model weights should be on the same path.')
    #args = parser.parse_args()
    
    file_name = 'model.json'
    with open(file_name, 'r') as jfile:

        print("reading model")
        model = model_from_json(jfile.read())


    print("compile model")
    model.compile("adam", "mse")
    weights_file = file_name.replace('json', 'h5')
    print("loading weights")
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)