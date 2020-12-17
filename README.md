# **Traffic Sign Recognition** 

The goal of this project was to build a traffic sign image classifier. This problem has been solved before, and a lot of the ideas in this project are based on a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Sermanet and LeCun.

[//]: # (Image References)

[image1]: ./writeupimages/visualization.png "Visualization"
[image2]: ./writeupimages/exampleofdata.png "Example of dataset"
[image3]: ./writeupimages/exampleofaugmenteddata.png "Example of augmented dataset"
[image4]: ./writeupimages/internetimages.png "Internet images"
[image5]: ./writeupimages/internetimagespredictions.png "Internet image predictions"

### Data Set Summary & Exploration

The dataset I used for this project was 51839 German traffic signs. I split the dataset into three buckets for training, validation and testing.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here we can see the number of examples we have for each type of traffic sign. Each sign id is a different type of traffic sign. We can see that the image set has many examples of some types of images, and very few of some others.

![alt text][image1]

Here are some example images from the dataset:

![alt text][image2]

### Design and Test a Model Architecture

#### Preprocessed the image data.

It was noted in the paper by Sermanet and LeCun that converting the images to grayscale improved the accuracy of their model, so I did the same. 

After that, I doubled the size of the dataset by creating slightly augmented copies of each image. The images were rotated, shifted, or zoomed in. 

Lastly, I normalized the pixel values by applying `(pixel - 128)/ 128` to each pixel. The standard formula for normalizing is usually `(pixel - mean)/ standard deviation`, but 128 is close enough. Using a constant allows us to apply the same normalization later without keeping track of the mean and standard deviation.

Here is a sample of the augmented dataset:

![alt text][image3]

#### Model architecture

My final model consisted of the following layers:

| Layer        		    |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x6     | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5x16    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Convolution 5x5x400   | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout       		|                                               |
| Fully connected		| Input from layer 2 and 3, outputs 100			|
| RELU			    	|                                               |
| Dropout       		|                                               |
| Fully connected		| Input 100, outputs 43   						|
 
#### Training the model

To train the model, I used the Adam optimization algorithm. I figured out the values for the hyperparameters empirically. When I trained my algorithm, I could notice when the accuracy was hitting an asymptote, and I would know that adding more epochs wasn't going to help. So I played around with the batch size and learning rate until I got above 93% accuracy. 

```
EPOCHS = 20
BATCH_SIZE = 32
rate = 0.0005
```
#### Developing the model

My final model results were:
* training set accuracy of 95.2%
* validation set accuracy of 94.7%
* test set accuracy of 93.83%

I started with the LeNet architecture because that architecture was developed to solve a similar problem. I then modified it to be similar to the Traffic Sign classifier architecture discussed in the paper by Sermanet and LeCun. I took their idea of feeding two separate layers into the the final fully connected layers at the end of the network.  I do this by flattening both the second and third layers and then feeding both of them into a fully connected layer. 

Because LeNet was designed to work on simpler images with fewer classes, I knew I needed by network to be more complicated. So I tried adding more parameters where I could and then added dropout layers to help with any overfitting. 

After this, training was still not achieving the accuracy I needed. I could see that the accuracy was hitting an asymptote below 93%, so more epochs would not help. I lowered the batch size and played with the learning rate to try to get the accuracy to improve. Eventually, I got the accuracy to be above 93%. 
 

### Test a Model on New Images

#### Finding new images from the internet

Here are five German traffic signs that I found on the web:

![alt text][image4]

Notice that the 60km/h sign and one of the caution signs watermarks, this could make them more difficult to classify.

#### Models performance on the new images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| 60 km/h    			| 60 km/h 										|
| General Caution		| General Caution								|
| General Caution		| General Caution								|
| Priority Road			| Priority Road      							|

The model guessed all of them right! That's 100% accuracy. 

I calculated the softmax probabilities for each of the images. We can see that this algorithm is very confident about its predictions. The best second guess for any of the images is a 0.48% chance that the general caution sign is a traffic signal.

![alt text][image5]