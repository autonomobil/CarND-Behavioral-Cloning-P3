[//]: # (Image References)

[image1]: ./images/Data_Distribution.png "Data Distribution"
[image2]: ./images/example1.jpg "example from dataset"
[image3]: ./images/example2.jpg "example from dataset"
[image4]: ./images/example3.jpg "example from dataset"
---
# SDCND Project 3: Behavioral Cloning
## Succeeded in track 1 & 2 with one model!
---

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## 1. Overview
###1.1 Goal:
* Develop an end-to-end approach for a self-driving car in a simulated environment
* end-to-end means: The model gets camera images as input and controls the vehicle with the steering angle as output. No manual extraction of features or route planning needed. [Similar to NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) or [comma.ai](https://arxiv.org/abs/1608.01230) approach.
* Model must drive the car at least 1 round in track 1
* As challenge: Let the model drive the car in track 2

###1.2 Approach:
* Literature research:
    * [NVIDIAs paper on their approach ](https://arxiv.org/abs/1604.07316)
    * [Sentdex' Python Plays GTA V project](https://psyber.io/).
    * [Siraj Raval' Video on this topic](https://www.youtube.com/watch?v=EaY5QiZwSP4&t=4s)
    * Slack/Forum of Udacity
* Use the provided Udacity simulator to collect data of good driving behavior as training data
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with the training and validation set
* Test the models autonomous functionality in track 1 in the simulation and check if it behaves correctly
* Change network architecture and data pipeline if necessary
* Train and test the model in track 2
* Repeat until *one* model can drive the car in track 1 and track 2

###1.3 Content of this project

My project includes the following files:
* model.py containing the script to create and train the model
* helper.py containing various helper functions used by model.py
* data_augmentation.py is an universal image augmentation functions, gets called
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* [link for the acquired and used training data](https://mega.nz/#F!REtxTTwZ)


###1.4 How to run the model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
Python, Keras and all dependencies must be installed.

###1.5 What's inside model.py?

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
##2. Dataset exploration, acquisition, cleaning and augmentation
###2.1 Exploration
Udacity provided a dataset with 24108 images (24108 / 3 perspectives = 8036 scene samples) for track 1, which was very helpful. They are 320 pixels width * 160 pixels height and taken from 3 perspectives by virtual cameras (center, left, right).

Exploration of the dataset lead to the some observations:
* Simple and similar graphics, easy to learn but danger of overfitting.
* Track 1 is clockwise round track with almost all curves being left - unbalanced, left-heavy dataset.
* Track 2 is very different looking, with lots of sharp bends and vertical up and downs. Developing a model to generalize and drive both tracks will be challenging


### 2.2 Acquisition
For track 1 17259 images were acquired by using a joystick as input. Here I came across the first problem: Unity doesn't take the (Windows-)calibrated signals from a joystick, solution see [here(forum)](https://discussions.udacity.com/t/using-a-joystick-in-the-simulator-wrong-calibrated/423354/5). The acquired data contains 2 laps of track 1, also 2 in reverse and different critical scene, like the bridge, the curve after the bridge,. etc. Folders are: me_data, me_data2, me_data3, me_data4

For track 2 21933 images were acquired, this are 4 complete laps driving as near to the middle line as possible.
Folders are: me_track2_data, me_track2_data

Example of a scene sample(center, left, right):

![image2] ![image3] ![image4]
### 2.3 Cleaning
Most of the samples driving straight which leads to a *big* bias for going straight. A function was implemented, which deletes as much sample as there over the maximum in one bin.

The histogram shows:
* blue: distribution before cleaning,
* orange : distribution after cleaning.


![image1]
Note: y axis is logarithmic

### 2.4 Data augmentation
To provide more data (more data is *always* better) and reduce shortcomings in the dataset, several data augmentation techniques were used. This enables the extraction of more information which helps prevent overfitting and getting generalization. A general image augmentation function data_augmentation.py was implemented and used.

Used techniques (every technique outputs a new image):
* Use the left and right camera image, correct the steering angle with a factor of 0.175
* Flip (np.fliplr) the image and mupltiply steering angle with -1
* Use create_augmen_img_bytransform inside data_augmentation.py, which with the settings I used apply:  
    * a random color/brightness-shift
    * a small zoom of -/+ 5 %

These techniques are called inside the generator for the fit_generator, will be described below.

##3.Network Architecture and Training Strategy
###3.1 Network Architecture

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
