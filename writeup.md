# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./writeup_images/galery_dataset.png "Visualization Images"
[image2]: ./writeup_images/barchart_dataset.png "Sign Type Distribution"
[image3]: ./writeup_images/image_greyscale.png "Example Image Greyscale"
[image4]: ./writeup_images/barchart_augmented_dataset.png "Sign Type Distribution After Data Augmentation"
[image5]: ./traffic-signs-for-classification/IMG_20170308_120539.png "Traffic Sign 1"
[image6]: ./traffic-signs-for-classification/IMG_20170308_120554.png "Traffic Sign 2"
[image7]: ./traffic-signs-for-classification/IMG_20170308_120903.png "Traffic Sign 3"
[image8]: ./traffic-signs-for-classification/IMG_20170308_121151.png "Traffic Sign 4"
[image9]: ./traffic-signs-for-classification/IMG_20170308_121403.png "Traffic Sign 5"
[image10]: ./traffic-signs-for-classification/IMG_20170308_122420.png "Traffic Sign 6"
[image11]: ./writeup_images/bar_chart_my_signs.png "Classification Result with Softmax"
[image12]: ./writeup_images/vis_conv1.png "Activations of Conv1"

## Rubric Points
---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

The code for this step is contained in the 3rd code cell of the IPython notebook.  

I used python and numpy libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

The code for this step is contained in the 5th and 6th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set and a bar chart showing the distribution of sign types

![alt text][image1]![alt text][image2]

### Design and Test of the Model Architecture

#### 1. Description of Preprocessing of the Image Data.

The code for this step is contained in the 20th code cell of the IPython notebook.

As a first step, I decided to convert the images to greyscale because this would speed up training time without decreasing prediction quality.

Here is an example of a traffic sign image after greyscaling.

![alt text][image3]

As a last step, I normalized the image. So that all values are between 0. and 1.

#### 2. Augmentation, Training, Validation and Testing Data.

I used the training, validation and test data as it was provided.

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630

to increase the number of training data and to balance out the vastly uneven distribution of sign types in the training data, I generated additional data from the existent images.

I generated new data (especially for the signs with low occurance) by
* rotation by -1, -2, 1, and 2 degrees around the center of the images
* zoom in by 5% and 10%

The data augmentation is realized in the 11th code cell of the IPython notebook. 

This step increased the size of the training set from 34799 to 302864 (�lmost 10x!) images and balanced out the distribution of different sign type quite good

![alt text][image4]

The images in the validation set and test set remained unchanged.

#### 3. Model Architecture

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
| Fully connected FC1	| 1024       									|
| RELU          		|            									|
| Dropout				| 0.7											|
| Fully connected FC2	| 512       									|
| RELU          		|            									|
| Dropout				| 0.7											|
| Fully connected FC3	| 43       		    							|
| Softmax				|             									|
 

#### 4. Hyperparameters

The code for training the model is located in the 27th cell of the ipython notebook. 

To train the model, I used:

* Adam Optimizer
* Learning Rate of 0.0001
* batch size of 128
* 50 Epochs

#### 5. Approach for Finding a Solution.

The code for calculating the accuracy of the model is located in the 27th and 28th cell of the Ipython notebook.

I first tried the architecture like it is given in the lecture notes.
Validation accuracy maxed at 0.91 and test accuracy was at 0.83.

To increase the over all accuracy I introduced an additional convolutional layer and an additional fully connected layer.
The gap between validation accuracy and test accuracy indicates overfitting. To reduce this gap I introduced dropouts (0.7) after conv2, con3, fc1 and fc2

The learning rate of 0.001 and 0.00001 didn't work - with these rates the validation accuracy remained at 0.05 through all epochs.

I generated as much new training data as possible that threw no memory errors when I started the training.

The most important measures were to introduce dropout and to augment the training data to increase training samples and to balance the occurance of sign types.

My final model results were:
* validation set accuracy: 0.958
* test set accuracy: 0.941
* accuracy on my traffic signs: 1.0

The test accuray (0.941) is very near the validation accuracy (0.958). That proves that the model works well, generalizes well and does not overfit.

### Testing the Model on New Images

#### 1. New Images

Here are five German traffic signs that I took in Munich:

##### Image 1 - Yield ![alt text][image5]

This image might be difficult to classify, because...
* There is a building on the lower left side corner in the background. This might obscure the shape of the sign.
* The perspective of the photo is not perfect. This results in a distortion and a rotation of the shape.
* The contrast inside the sign is low. This could make the distinction between the white center and the dark border line harder.

This image might be good to classify, because...
* The edge of the sign is clearly recognizable against the sky most of the time, making the shape good recognizable

##### Image 2 - No entry ![alt text][image6]

This image might be difficult to classify, because...
* The the photo is shot from the side. This results in a distortion of the shape. It appears more like an oval than a circle.

This image might be good to classify, because...
* The contrast inside the sign is very good.
* The edge of the sign is clearly recognizable against the uniformly dark background.

##### Image 3 - Turn right ahead ![alt text][image7] 

This image might be difficult to classify, because...
* The branches of the tree in the background obscuring the edge of the sign and possibly weakening the recognition of the shape of the sign.

This image might be good to classify, because...
* The perspective is very good, strengthening the recognition of the shape.
* The contrast inside the sign is very good.

##### Image 4 - Road narrows on the right ![alt text][image8]

This image might be difficult to classify, because...
* The background is very alternating in brightness, color, and shapes. This could make the recognition of the shape of the sign harder.

This image might be good to classify, because...
* The perspective is very good. This strengthens the recognition of the shape.
* The contrast inside the sign is very good.

##### Image 5 - Yield ![alt text][image9]

This image might be difficult to classify, because...
* The contrast inside the sign is low. This could make the distinction between the white center and the dark border line harder.
* The background is alternating in brighness, which could make the recognition of the shape of the sign harder.

This image might be good to classify, because...
* The perspective is very good. This strengthens the recognition of the shape.

##### Image 6 - Speed limit (60km/h) ![alt text][image10]

This image might be difficult to classify, because...
* The background is alternating in brighness and shapes, which could make the recognition of the shape of the sign harder.

This image might be good to classify, because...
* The perspective is very good.
* The contrast inside the sign is very good. This makes the content of the sign easier to recognize.

#### 2. Model Predictions on new Traffic Signs

The code for making predictions on my final model is located in the 33th cell of the Ipython notebook.

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

#### 3. Certainty of the Model when Predicting new Images

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
| 1.00000000e+00        | Speed limit (60km/h)				    		| 
| 1.79725904e-13       	| Speed limit (80km/h) 							|
| 1.41033235e-13    	| Speed limit (50km/h)							|
| 2.86833238e-21        | Turn left ahead   					 		|
| 9.85496641e-22        | Ahead only                  					|

Visualization of the predictions with a bar charts:
![alt text][image11]

### Visualization Activations of Convolutional Layers

The conv1 layer activation visualization shows that filters activate because of the shape of the sign (filter #9, #12, #14) and the arrow #4, #16, #23 and #26). There are filters which activate also on the background (#23, #26). That shows that the convnet learns on different aspects of the content and in later fully connected layers is able to compose these filter impressions into patterns that all images of a certain sign share. The more unique these pattern are, the more confident the network is in its predictions.

![alt text][image12]
