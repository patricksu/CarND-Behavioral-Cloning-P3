#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/hist_U.png "Udacity data histogram"
[image2]: ./examples/hist_peng_recovery.png "Self collected recovery data histogram"
[image3]: ./examples/modelArch.png "Model Architecture"
[image4]: ./examples/UdacityImgShow.png "Udacity images show"
[image5]: ./examples/leftRightImages.png "flipping images"
[image6]: ./examples/brightnessAug.png "brightness augmentation"
[image7]: ./examples/Shadowing_Visualization.png "Shadowing augmentation"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 167-178). It has a total of 7 convolution layers followed by 3 fully connected layers. Each convolution and fully connected layer is followed by ELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 165). A 1x1 filter with depth 3 is used to transform the color space of the images. Research has shown that different color spaces are better suited for different applications.

####2. Attempts to reduce overfitting in the model

To reduce overfitting, the model contains three dropout layers which follow sets of convolution layers (model.py lines 170, 175, 180). I attempted to use Udacity's provided data as well as my own recovery data to train the model. The Udacity data is randomly split into training and validation data (20% validation). Each model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 188). I used the default starting learning rate of 0.001 (also tested other rates but they do not make much difference.) The Dropout rate is 0.3, less than normally used 0.5, as this dataset is not big. Using 0.2 did not help addressing overfitting problem. So 0.3 is a good tradeoff point. Batch_size is picked as 32: the biggest allowed by my GPU memory. The steering angle offset I used on the left and right camera images is 0.25. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I collected recovery and reverse driving and added these data to the Udacity dataset. However the combined dataset did not work well. As shown by the steering angle histograms of the Udacity data and the recovery data, these two driving behaviors are quite different. I.E. the Udacity data has a much wider range of steering angles, while my recovery data's steering angle is mostly between -0.2 and 0.2. It should explain why combining them did not work well. 
![alt text][image1]
![alt text][image2]

As a result, I did not use the recovery data, instead I used the Udacity data's left and right camera's images with added steering offsets to expand the angle range. I also tried to augment the data using translation (shifting the images left or right by a random amount) with angle offset, which is proportional to the translated pixels. However this did not work well, as the loss oscillated quite a lot. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a model, test it on the simulator, see where it fails, and collect more data for the failing section. At the same, I tuned the parameters, including dropout rate, learning rate, left and right images' angle offsets. 

My first step was to use a convolution neural network model. I adopted the one from [Vivek] (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9), which was proven to work on this problem. This model has a total of 23.9 Million parameters. I fed three images to the model and found that it did not always converge if Adam optimizer. When using SGD optimizer, it converged with 5 steps (loss becoming ZERO). This proved that the model actually worked. It might be because Adam adjusted the learning rate and overshot the minimal point. 

I used the left image (angle + .25), center image, and right image (angle - 0.25) in the training. In the validation, I only used center images, since the driving evaluation is based solely on the center image. Using left and right images with arbitrary angle shifts actually added more nosie to the data. As a result, the traing loss is much larger than the validation loss in the first two epochs. So in this project, I did not treat this as overfitting. As the training went on, the traing and validation losses both reduced, and traing loss became closer to the validation loss. At the end, traing and validation loss were both around 0.013. 

The final step was to run the simulator to see how well the car was driving around track one. My initial model run off track right after the bridge. So I collected more "afterBridge" data, and tuned the model parameters using the new data (like transfer-learning). After this, the model was able to safely pass the bridge, but failed again where a small section of the right-side road mark disappeared. This implied that the model relied on the road marks. To combat this problem, I tried multiple approaches, including collecting more data from that section, augmenting the existing data, and adjusting parameters. It turned that collecting more data did not help much, but data augmentation worked magically. I augmented the data by randomly adjusting the brightness and adding random shadows. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 164-186) consisted of a convolution neural network with the following layers and layer sizes. Here is a visualization of the architecture.

![alt text][image3]

####3. Creation of the Training Set & Training Process

I used the Udacity provide data together with a dataset of "afterBridge". To collectthe "afterBridge" data, I drove the vehicle back-and-forth around the muddy area for more than five laps. I also collected reverse and recovery image sets, although they were not used in building the final model. Instead I flipped the Udacity images to mimic reverse driving, and also used the left and right images to mimic recovery driving. 
The figure below shows a few images from the Udacity dataset. It only shows image with over 0.5 steering angles. The data has some nosise. E.g. the road in image at image(4,2) is straight but the angle is 0.626. However, as long as the model can generate larger angles to get the vehicle back when it is almost off-track, it should work. So the situation shown in the iamge (4,2) should not matter that much. 
![alt text][image4]

The figure below illustrates the left and right camera images. 
![alt text][image5]

After the collection process, I had 11493 data points, each with three images. I then preprocessed this data by some agumentations. The figure below illustrates brightness augmentation and random shadows being added. 
![alt text][image6]
![alt text][image7]

I randomly split the 11493 data points into training and validation dataset, with 20% as validaiton. Then I implemented a generator that randomly sample from the training data points. It firstly randomly picked one from three (left, center, and right images), then added random brightness and shadowing augmentations. Finally, under 50% it flipped the image and reverse the steering angle. Line 87 to 113 in model.py implemented this. In this way, the generator can generate unlimited sets of batch training samples. For the validation data, I did not use any augmentation, instead I picked only the central images. 

I used this training data generator for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
