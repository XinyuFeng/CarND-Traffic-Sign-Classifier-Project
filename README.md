# CarND-Traffic-Sign-Classifier-Project

#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./pictures/1.jpg "Traffic Sign 1"
[image5]: ./pictures/2.jpg "Traffic Sign 2"
[image6]: ./pictures/3.jpg "Traffic Sign 3"
[image7]: ./pictures/4.jpg "Traffic Sign 4"
[image8]: ./pictures/5.jpg "Traffic Sign 5"
[image9]: ./pictures/6.jpg "Traffic Sign 6"
[image10]: ./pictures/7.jpg "Traffic Sign 7"
[image11]: ./pictures/8.jpg "Traffic Sign 8"
[image12]: ./pictures/9.jpg "Traffic Sign 9"
[image13]: ./pictures/10.jpg "Traffic Sign 10"
[image14]: ./pictures/11.jpg "Traffic Sign 11"
[image15]: ./pictures/12.jpg "Traffic Sign 12"
[image16]: ./examples/classify.png "classify"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/XinyuFeng/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It first shows 43 representative images for each classes, then a bar chart showing how the data distribution in training set

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I fill some extra images into some classes which have fewer images than average by rotate some digree of random images in training dataset. I did this since less data in one class can lead to misclassify for the CNN of that class

Then I grayscale image to make them into 32x32x1 dimentions and normalize them in order to make it easier for CNN to training and these ways can increase the final accuracy.

Here is an example of a traffic sign image before and after grayscaling.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| flatten					|	outputs 400						|
| Fully connected		| Input 400, output 120						|
| RELU					|												|
| dropout					|												|
| Fully connected		| Input 120, output 84						|
| RELU					|												|
| dropout					|												|
| Fully connected		| Input 84, output 43						|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the folloing hyperparameter:
Epochs: 30
Batch size: 150
Learning rate: 0.00097 (I found this is better than 0.0001 on validation accuracy)
dropout probability: 0.7
Finally, I use Adamoptimizer to train the model


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were: 
* validation set accuracy of 0.951 
* test set accuracy of 0.933

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I first use the classical LeNet without dropout layer since it really do a good job on digit recognition, and for traffic sign classify, when it turn into grayscale, will have the same image format as hand-written digit image. Thus i believe this net can also do a good job on traffic sign classigication.


* What were some problems with the initial architecture?

I found out that without dropout, when validation accuracy is exceed arround 4%-5% compared to test accuracy, which leads me believe that its the result of overfitting.


* How was the architecture adjusted and why was it adjusted? 

I added dropout layers after fully connected layers


* Which parameters were tuned? How were they adjusted and why?

learning rate, epochs and drop prob. A low learning rate can give me a better final result when trained enough epochs. After my try and trial, I found 30 epochs is enough to make the validation accuracy to reach a stable number. Finally, I choose drop prob as 0.7, which can give me the best result. 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I choose 12 signs and here are 12 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]

![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]

![alt text][image12] ![alt text][image13] ![alt text][image14] ![alt text][image15]

The images with number might be difficult to classify due to more complex feature shapes plus the low resolution of images
The left turn sign should also be difficult.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or left      		| Go straight or left   									| 
| Keep right     			| Keep right 										|
| No entry					| No entry											|
| No vehicles	      		| No vehicles					 				|
| Priority road			| Priority road      							|
| yield			| yield     							|
| Speed limit (20km/h)			| Roundabout mandatory      							|
| Ahead only 			| Ahead only      							|
| Dangerous curve to the left			| Road work      							|
| Speed limit(60km/h)			| End of speed limit(80km/h)      							|
| Speed limit(80km/h)			| Speed limit(80km/h)     							|
| Stop		| Stop    							|


The model was able to correctly guess 9 of the 12 traffic signs, which gives an accuracy of 75%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

I choose some typical images to describe:

For the 1th image, the model is sure that this is a Go straight or left sign (probability of 1.0), which is correct. 


For the 2nd image, as the same as first, very sure of the result that it's a keep right sign. as well as 5th, 6th image.


For the 3rd image, which a 0.35 probability of turn right and 0.63 as No entry, which is correct. This reult might be caused by the white bar in the image that can be misclassified as a turn right sign.


For the 4th image, with a 0.88 probalitify as No vehicles and very small probabilities for speed limit since the image has a circle shape, and might look same as a speed limit sign, but the final result is correct


For the 9th image due to the blur of the image, cause it wrong classify.

etc.
The original clasify image is here:

![alt text][image16]

