# Introduction
This file describes the approach and architecture of project 3. In hindsight, the project did not involve that much coding. The main challenge was to find out what kind of training data to feed into the model. Several model architectures were tried out but the differences in outcome were minimal. Therefore, the focus was later on providing better training data. 

# Approach
I am not an expert in CNN and deep learning. Therefore my approach was to start with things I know and then adapt them to the needs of the project. This means, I started with the architecture of traffic sign classifier and subsequent examples using Keras provided by Udacity. Of course project 3 was not a classification problem. Instead, a continuous number was the desired outcome (steering angle). Initially I thought I could still treat it as a classification problem, but was not successful. The next step was then to modify the last layer of the model in such a way that we have a continuous variable and not a classifier. 

# How training data was created

An important idea was that the upper part of the camera image is not carrying useful information. Only the street up to the horizon is relevant. Therefore the top 60 pixel could be cut away. 

example image: 

![alt tag](https://github.com/AlexSickert/Udacity-SDC-T1-P3/blob/master/example-raw-image.jpg?raw=true)

I knew from the traffic sign classifier project that the images necessary to take decisions are actually very small. So I tried to scale down the images by cutting them in half. The result are images in size of 160 x 50 pixel. I looked at them and asked myself if I could decide where to with these images and said yes - it should be sufficient information. 

example image:

![alt tag](https://github.com/AlexSickert/Udacity-SDC-T1-P3/blob/master/example-scaled-image.jpg?raw=true)

Subsequently I thought that I should create a lot of training data. This is what i read in the forum. So I recorded a lot of images by driving nicely around the track. I recorded 35k images of the central camera. However, no matter which model I used, how many epochs used for training - the result was disappointing. The algorithm regularly did not decisively enough modify the steering angle and so the car got off track. This led to massive frustration as a lot of time was invested into this project. 

At some point I decided to start from scratch after having this thought: What we need in terms of training data is more explicit driver action and we need to record the situations where the car is about to go off-track. In other words - the "nice driving" is useless. I then recorded not continuous driving, but only very short sequences of a few seconds. i recorded only situations when the car was on the edge of getting off track. I compiled a small dataset of 2k images and the result was much better. Then I optimized this training set by recording more situations at the locations where the car tends to go off-track. 


# Architecture

I tried out many different models - approximately 10. But I could not observe that any of the models performed significantly better than others. Also different activation functions, optimizers, number of epochs did not show significant impact. Ultimately I used this architecture: 

1. convolution 
2. Max Pooling
3. Dropout 30%
4. Convolution
5. Max Pooling
6. Dropout 30%
7. Flatten
8. Fully connected layer with 1000 neurons
9. Relu
10. Fully connected layer with 50 neurons
11. Relu 
12. Fully connected layer with 1 neuron
13. Tanh activation function to obtain a numeric range of -1 to 1

In addition to that several other approaches were tried out in terms of architecture: 

- from project 1 I tried out to do draw the lines of the border using CV2 (huelines). But the result was not ideal.
- I tried to increase the contrast by using a function of CV2 (v2.equalizeHist(img))  but I could not observe any positive  impact

# Training and evaluation

I played with various combinations of epoch, batch sizes, validation ratio. What i realized is that after the first epoch the gain in improvement is minimal. Big batch sizes lead to bad results, so i reduced batch size to 10. The validation ratio is 0.2. 
What I observed is that the loss calculated by Keras (mean squared error) says absolutely nothing about how the model will perform when used in the car. I had one model using a big training set and many epochs and at the then the mse loss was 0.02, but the car got off track constantly. Once I started from scratch with different training data the mse/loss is 0.1 but the car stays on track and drvies nicely. This is the reason why I did not include test data. I explicitly excluded test data as it would make the whole process even more complicated without further insight. by testing on the car i could see that performance of the model depends on the speed of driving. Therefore I created a speed control that locks-in the speed at approx 10 miles per hour. This was my test setup. 

# Result
The ultimate dataset created a Numpy array of 800MB and on my Linux laptop training took approx 5 minutes. I considered both acceptable and did not try other improvements to make training more efficient. 

I tested the final result by letting the car go three rounds without going off-track once. 



