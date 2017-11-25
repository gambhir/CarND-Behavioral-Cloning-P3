**Behavioral Cloning Project**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/dataset-preview.png "Dataset Preview"
[image3]: ./images/dataset-angle.png "Dataset Angles"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 123-136)

The model includes RELU layers to introduce nonlinearity (code line 126-130), and the data is normalized in the model using a Keras lambda layer (code line 124).

#### 2. Attempts to reduce overfitting in the model

I tried using a drop out layer, but found it didn't do anything substantial.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 131). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.  The easiest method to reduce overfitting is just watching the loss for training and validation and making sure they converge and stopping at that point.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 138).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I tried various methods of data collection and used all images collected(center, left, and right).  I attempted to use track 1 and 2 and also different input methods.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to essentially use models that were already created.  I tried LeNet, Nvidia's, and also variations of the two.

My first step was to just get something working to give me the confidence to understand how the data was responding on the actual track.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. More often then not things were equal in terms of mean squared error.  The starting point with LeNet provided a decent base and moving to Nvidia's model worked even better enough.

I found my main problem was at the first turn after the bridge where no lane lines were present and you could run off into the dirt.  This problem wasn't solved by changing the model though.  This problem was solved prior to the creation of the model in preprocessing of the data.

#### 2. Final Model Architecture

The final model architecture (model.py lines 123-136) consisted of a convolution neural network with the following layers and layer sizes.

| Layer (type)                     | Output Shape        | Param #  | Connected to          |
| -------------------------------- |:-------------------:| --------:| --------------------: |
| lambda_1 (Lambda)                | (None, 160, 320, 3) | 0        | lambda_input_1[0][0]  |
| cropping2d_1 (Cropping2D)        | (None, 65, 320, 3)  | 0        | lambda_1[0][0]        |
| convolution2d_1 (Convolution2D)  | (None, 31, 158, 24) | 1824     | cropping2d_1[0][0]    |
| convolution2d_2 (Convolution2D)  | (None, 14, 77, 36)  | 21636    | convolution2d_1[0][0] |
| convolution2d_3 (Convolution2D)  | (None, 5, 37, 48)   | 43248    | convolution2d_2[0][0] |
| convolution2d_4 (Convolution2D)  | (None, 3, 35, 64)   | 27712    | convolution2d_3[0][0] |
| convolution2d_5 (Convolution2D)  | (None, 1, 33, 64)   | 36928    | convolution2d_4[0][0] |
| flatten_1 (Flatten)              | (None, 2112)        | 0        | convolution2d_5[0][0] |
| dense_1 (Dense)                  | (None, 100)         | 211300   | flatten_1[0][0]       |
| dense_2 (Dense)                  | (None, 50)          | 5050     | dense_1[0][0]         |
| dense_3 (Dense)                  | (None, 10)          | 510      | dense_2[0][0]         |
| dense_4 (Dense)                  | (None, 1)           | 11       | dense_3[0][0]         |

**Total params**: 348,219

**Trainable params**: 348,219

**Non-trainable params**: 0


Here is a visualization of the architecture.

![alt text][image1]
Image taken from - http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I started by driving 2 laps clockwise and then drove counter-clockwise.  Also would some times run into issues on the bridge so I collected extra data back and forth on that section.  I collected data in several different ways.  

Here are a few random images from the training set left, center, and then right.
![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to move back to center.

To augment the data set, I also flipped images and angles thinking that this would first increase the number of data samples I had, but also strengthen my distributions of left, right, and straight angles. For example here is a rouge break down of angles with left being -.15 and right being .15 less than or greater than.

![alt text][image3]

Total Samples :  79734

