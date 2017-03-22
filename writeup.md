#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/galery_dataset.jpg "Visualization Images"
[image2]: ./writeup_images/barchart_dataset.jpg "Sign Type Distribution"
[image3]: ./writeup_images/image_greyscale.jpg "Example Image Greyscale"
[image4]: ./writeup_images/barchart_augmented_dataset.png "Sign Type Distribution After Data Augmentation"
[image5]: ./traffic-signs-for-classification/IMG_20170308_120539.png "Traffic Sign 1"
[image6]: ./traffic-signs-for-classification/IMG_20170308_120554.png "Traffic Sign 2"
[image7]: ./traffic-signs-for-classification/IMG_20170308_120903.png "Traffic Sign 3"
[image8]: ./traffic-signs-for-classification/IMG_20170308_121151.png "Traffic Sign 4"
[image9]: ./traffic-signs-for-classification/IMG_20170308_121403.png "Traffic Sign 5"
[image10]: ./traffic-signs-for-classification/IMG_20170308_122420.png "Traffic Sign 6"
[image11]: ./writeup_images/bar_chart_my_signs.png "Classification Result with Softmax"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used python and numpy libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth and sixth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set and a bar chart showing the distribution of sign types

![alt text][image1]![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 20th code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because this would speed up training time without decreasing prediction quality

Here is an example of a traffic sign image after grayscaling.

![alt text][image3]

As a last step, I normalized the image.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the training, validation and test data as it was provided by udacity

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630

to increase the number of training data, to balance out the vastly uneven distribution of sign types in the training data I generated additional data from the existent images.

I generated new data (especially for the signs with low occurance) by
* rotation by -1, -2, 1, and 2 degrees around the center of the images
* zoom in by 5% and 10%

The data augmentation is realized in the 11th code cell of the IPython notebook. 

This increased the size of the training set from 34799 to 302864 images and balanced out the distribution of different sign type quite good

![alt text][image3]

The images in the validation set and test set remained unchanged.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 24th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   					| 
| Conv1 3x3          	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Conv2 3x3     	    | 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Dropout				| 0.7											|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 		    		|
| Conv3 3x3     	    | 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Dropout				| 0.7											|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 		    		|
| Fully connected		| 1024       									|
| RELU          		|            									|
| Dropout				| 0.7											|
| Fully connected		| 512       									|
| RELU          		|            									|
| Dropout				| 0.7											|
| Fully connected		| 43       		    							|
| Softmax				|             									|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 27th cell of the ipython notebook. 

To train the model, I used:

* Adam Optimizer
* Learning Rate of 0.0001
* batch size of 128
* 50 Epochs

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 27th and 28th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.958
* test set accuracy of 0.941
* accuracy on my traffic signs 1.0

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I first tried the architecture like it is given in the lecture notes.

* What were some problems with the initial architecture?
validation accuracy maxed at 0.91 and test accuracy at 0.83.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
To increase the over all accuracy I introduced an additional convolutional layer and an additional fully connected layer.
The gap between validation accuracy and test accuracy indicates overfitting. To reduce this gap I introduced dropouts (0.7) after conv2, con3, fc1 and fc2

* Which parameters were tuned? How were they adjusted and why?
I generated as much new training data as there were no memory errors while training
learning rate of 0.001 and 0.00001 didn't work. validation accuracy was flat at 0.05.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The most important design choice was to include dropouts.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
the test accuray (0.941) is very near the validation accuracy (0.958). That proves that the model works well, generalizes well and does not overfit.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10] 

I took these pictures im Munich on my way to the office.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			            |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Yield      		        | Yield   									    | 
| No entry     			    | No entry 										|
| Turn right ahead		    | Turn right ahead								|
| Road narrows on the right | Road narrows on the right					 	|
| Yield			            | Yield      							        |
| Speed limit (60km/h)		| Speed limit (60km/h)      					|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1%

####3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 33th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a yield sign (probability of 1.0), and the image does contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | Yield   							    		| 
| 0.0                	| Speed limit (20km/h) 							|
| 0.0           		| Speed limit (30km/h)							|
| 0.0            	    | Speed limit (50km/h)					 		|
| 0.0           		| Speed limit (60km/h)      					|


For the second image, the model is absolutely sure that this is a no entry sign (probability of 1.0), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | No entry 							    		| 
| 1.58849149e-28     	| Stop               							|
| 1.12626075e-32		| Roundabout mandatory							|
| 6.28699791e-34	    | Go straight or left					 		|
| 6.83392467e-35		| Keep right                  					|

For the third image, the model is absolutely sure that this is a Turn right ahead sign (probability of 1.0), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Turn right ahead					    		| 
| 8.36154435e-09     	| Ahead only               						|
| 4.27053564e-17		| Right of way at the next intersection			|
| 7.30437704e-19	    | Road work					 		            |
| 4.13292444e-19		| Yield                  				    	|

For the fourth image, the model is pretty sure that this is a Road narrows on the right sign (probability of 0.99), and the image does contain a Road narrows on the right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99971390e-01        | Road narrows on the right					    | 
| 2.84816797e-05     	| Children crossing               				|
| 6.20672722e-08		| Pedestrians			                        |
| 7.05641166e-13	    | Right of way at the next intersection			|
| 1.22637252e-13		| Traffic signals                  				|

For the fifth image, the model is absolutely sure that this is a Yield sign (probability of 1.0), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | Yield   							    		| 
| 0.0                	| Speed limit (20km/h) 							|
| 0.0           		| Speed limit (30km/h)							|
| 0.0            	    | Speed limit (50km/h)					 		|
| 0.0           		| Speed limit (60km/h)      					|

For the sixth image, the model is absolutely sure that this is a Speed limit (60km/h) sign (probability of 1.0), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | Speed limit (60km/h)				    		| 
| 0.0                	| Speed limit (80km/h) 							|
| 0.0           		| Speed limit (50km/h)							|
| 0.0            	    | Turn left ahead   					 		|
| 0.0           		| Ahead only                  					|

Visualization with a bar chart:
![alt text][image11]

