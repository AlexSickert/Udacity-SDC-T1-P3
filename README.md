# Introduction
This file describes the approach and architecture of project 3. In hindsight, the project did not involve that much coding. The main challenge was to find out what kind of training data to feed into the model. Several model architectures were tried out but the differences in outcome were minimal. Therefore, the focus was later on providing better training data. 

# Approach
I am not an experet in CNN and deep learning. Therefore my apporach was to start with things I know and then adapt them to the needs ot the project. This means, I started with the archtiecture of traffic sign classifier and subsequent examples using Keras provided by Udacity. Of course project 3 was not a classification problem. Instead, a coninuous number was the desired outcome (steering angle). Initially I thought I could still treat it as a classfication problem, but was not successful. The next step was then to modify the last layer of the model in such a way that we have a continuous variable and not a classifier. 

# How training data was created

An important idea was that the upper part of the camera image is not carrying useful infromation. only the street up to the horizon is relevant. Therefore the top 60 pixel could be cut away. 

I knew from the traffic sign classifier project that the images necessary to take decisions are actually very small. So I tried to scale down the images by cutting them in half. The result are images in size of 160 x 50 pixel. I looked at them and asked myself if I could decide where to with these images and said yes - it should be sufficient information. 

Subesquently I thought that I should create a lot of trianing data. This is what i read in the forum. So I recorded a lot of images by driving nicely around the track. I recorded 35k images of the central camera. However, no matter which model I used, how many epochs used for training - the result was disappointing. The algorithm regulalry did not decisively enough modify the steering angle and so the car got off track. This led to massive frustration as a lot of time was invested into this project. 

At some point I decided to start from scratch after having this thought: What we need in terms of traingn data is more explicit driver action and we need to record the situations where the car is about to go off-track. In other words - the "nice driving" is useless. I then recorded not continuous driving, but only very short sequences of a few seconds. i recorded only situations when the car was on the edge of getting off track. I compiled a small dataset of 2k images and the result was much better. Then I optimized this training set by recording more situations at the locations where the car tends to go off-track. 


# Architecture

I tried out many different models - approximately 10. But I could not observer that any of the models perofrmed significantly better than others. Also differetn activation fucntions, optimizers, number of epochs did not show significant impact. Ultimately I used this architecture: 

1. convolution 
2. Max Pooling
3. Dropout 30%
4. Convolution
5. Max Pooling
6. Dropout 30%
7. Flatten
8. Fully conntected layer with 1000 neurons
9. Relu
10. Fully connected layer with 50 neurons
11. Relu 
12. Fully connected layer with 1 neuron
13. Tanh activation function to obtain a numeric range of -1 to 1



# Result

# Insight

