# Udacity Self-Driving Car Nanodegree

_**Special Notes:  Please don't consider running through the projects as a substitute for watching the videos and doing the activities that precede the projects.  In some cases, the time spent on the projects may actually be LESS than the time spent absorbing the material Udacity provides as a precursor.  In that sense, what is presented here might seem disconnected or lacking in the background material necessary to complete the projects with a full appreciation of the techniques or algorithms used.  Additionally, the projects were expected to be run in the confines of an particular Anaconda environment.  Both Machine Learning (especially neural networks) AND the libraries written to implement them are evolving rapidly and you're likely to have some issues running the code in these projects with modern versions of the Python modules used.**_

The Self-Driving Car Nanodegree offered by Udacity in 2017/2018 was one of their flagship nanodegrees, covering a wide range of topics with some of the most thoroughly developed projects.  Sebastian Thrun was the head Stanford's winning entry in the DARPA Grand Challenge and personally appeared in many videos.  I have to admit, this was a very fun nanodegree that I truly looked forward to.  Often, Udacity will teach the same concepts in different nanodegrees, but luckily they always do it in a different way and always seem to introduce some new perspective on the problems.  The simulators provided by Udacity were top-notch and made the learning so much more fun.  I was happy that I had recently built a good machine with a nice GPU to run everything on.  I have many favorite projects in this course and some top ones are Behavioral Cloning (term 1), Semantic Segmentation (term 3), Path Planning (term 3), and Model Predictive Control (term 2).

## Projects
Below is the list of projects from the course, along with links to my solutions.  In each directory, you'll find a README.md written by Udacity, along with some of their helper code.  Generally, the student's view of the project is the Jupyter Notebook files with an .ipynb extension (viewable in GitHub), which is where you'll find the string of guidance, activities, and Q&A that make up the project.  Sometimes, I had to fill in TODO sections in a separate code file.

## Term 1
### [Lab: LeNet](./LeNet-Lab)
[Jupyter Notebook](./LeNet-Lab/LeNet-Lab-Solution.ipynb)

As I stated, topics are reintroduced in various nanodegrees.  In the Artificial Intelligence Nanodegree, Keras was used when dealing with neural networks, however "raw" TensorFlow is used in the Self-Driving Car Nanodegree.  This project introduces the LeNet architecture - one of the first forays of Neural Networks that shocked the world.  This neural network designed by Yan LeCun is famous for excelling at the MNIST challenge, outperforming all other solutions by a wide margin.  In this project, the LeNet-5 architecture is built.
![LeNet-5 Architecture](LeNet-Lab/lenet.png)

### [Lab: AlexNet Feature Extraction](./AlexNet-Feature-Extraction)
[Feature Extraction Python Script](./AlexNet-Feature-Extraction/feature_extraction_solution.py)

[Training Python Script](./AlexNet-Feature-Extraction/train_feature_extraction_solution.py)

[Traffic Sign Inference Python Script](./AlexNet-Feature-Extraction/traffic_sign_inference_solution.py)


This lab appears to be missing both an extensive Readme.md and any sort of Jupyter Notebook, so I've linked to the python solutions involved instead.  The job here is similar to the dog classifier project in the AIND - use a trained model that's been proven to do a similar task, take the first half of the network (including weights) and tack on new, appropriate layers, and retrain it to classify a different set of classes.  The first few layers may be frozen so that they don't change - this also speeds up the training.  In this example, we'll be taking [AlexNet](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html), which is meant to work with the [ImageNet](https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/) dataset, and using it to classify traffic signs.

A similar lab was done using the Inception, ResNet and VGG networks.

### [Traffic Sign Classifier](./Traffic-Sign-Classifier)

[Jupyter Notebook](./Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

[Report](./Traffic-Sign-Classifier/report.pdf)

![Classification 1](./Traffic-Sign-Classifier/output_images/classification_1.png)
![Classification 2](./Traffic-Sign-Classifier/output_images/classification_2.png)
![Classification 3](./Traffic-Sign-Classifier/output_images/classification_3.png)


A training accuracy of 99% and a test accuracy of 97.4%?  Pretty good!  I'll let the [report](./Traffic-Sign-Classifier/report.pdf) do most of the talking about this one.  In general, the purpose of this project was to identify European traffic signs using a Convolutional Neural Network (CNN).  The project uses a modified LeNet CNN trained from scratch.  Skip Layers were used to improve accuracy.  Feature extraction and visualization were incorporated to give some feedback about what the neural network is attempting to identify in the images.

Topics:
* Preprocessing
    * Image scaling
    * White balancing
    * Normalization
    * Greyscale
* Data Augmentation
    * Obfuscation
    * Shifting
    * Rotating
* Neural Network Design
    * Layer skipping
    * Feature Map Visualization

### [Behavioral Cloning](./Behavioral-Cloning)
[Jupyter Notebook](./Behavioral-Cloning/Behavioral_Cloning_Final.ipynb)

[Report](./Behavioral-Cloning/writeup_report.pdf)

[Stand Out Task (MP4 VIDEO)](./Behavioral-Cloning/video_mountain_right_lane_only.mp4)
![Mountain Pass](./Behavioral-Cloning/images/mountain_pass.png)

[Basic Pass (MP4 VIDEO)](./Behavioral-Cloning/video.mp4)
![Flat Loop](./Behavioral-Cloning/images/flat_loop.png)

I started to get a bit cocky at this point, due to how powerful I was finding these neural networks to be, especially on this task.  The task was to drive a car in a simulator around a track manually, recording steering angles and screenshots as you went, then use that saved data to train a neural network to steer based on screenshots.  The required "mph" for the basic run was 9mph.  Additionally, there was a "stand-out" task of a mountain course, which I not only got to run, but got to run at 20mph.... and kept the car in the right lane (ok, most of the time)!  A fun part of this project was hooking up my Xbox 360 controller to the PC to avoid having to use a keyboard to keep the car between the lines.

Topics:
* Preprocessing
    * Performing PreProc on the GPU
    * Clipping
    * Normalization
* Neural Network Design
   * Dropout Layers

### [Lane Lines](./Lane-Lines)

[Jupyter Notebook](./Lane-Lines/P1.ipynb)

[Report](./Lane-Lines/report.pdf)

[Video 1](./Lane-Lines/solidWhiteRight.mp4)

[Video 2](./Lane-Lines/solidYellowLeft.mp4)

[Challenge Video](./Lane-Lines/challenge.mp4)
![Challenge Screenshot](./Lane-Lines/challenge.png)

This project using OpenCV operations to detect lane lines in images.  As mentioned in the [report](./Lane-Lines/report.pdf), the first step is Canny Edge Detection, which is followed by probabilistic Hough transform.  A region of interest is also enforced, which helps avoid detecting edges in traffic signs and the like.  At this point, equations for the lines are derived and lines are overlayed on to the original image as a feedback mechanism.

### [Advanced Lane Lines](./Advanced-Lane-Lines)

[Jupyter Notebook](./Advanced-Lane-Lines/Advanced_Lane_Finding_Final.ipynb)

[Report](./Advanced-Lane-Lines/report.pdf)

[Video 1](./Advanced-Lane-Lines/project_video_out.mp4)

[Challenge Video](./Advanced-Lane-Lines/challenge_video_out.mp4)

![Debug Snapshot](./Advanced-Lane-Lines/output_images/debug7.png =640x360)
![Overlayed Snapshot](./Advanced-Lane-Lines/output_images/outimg7.png =640x360)

This lane finder project incorporates camera lens distortion correction, colorspace conversion and thresholding, (coolest of all) perspective transformation, curvature calculation, overlaying lines and finally perspective transformation reversal.  Important takeaways are that thresholding can be extremely temperamental and so can curvature parameters made in an expanded space where errors may be magnified.  In the report, I indicate that the challenge video was a failure, but looking at it now I see that the performance is pretty good (although not perfect).

### [Vehicle Detection](./Vehicle-Detection)

[Jupyter Notebook](./Vehicle-Detection/Vehicle_Detection_v1.ipynb)

[Report](./Vehicle-Detection/report.pdf)

[Video](./Vehicle-Detection/project_video_out.mp4)

The project starts off by training an SVM classifier on the Histogram of Oriented Gradients (HOG) features calculated from a dataset of images of rear views of cars and not-cars.  This means we now have something that, given a 64x64 image, can fairly quickly/cheaply tell us whether or not a car is in the image.  In order to leverage this classifiers, windows of various sizes are slid around the car camera image.  The windowed sample is converted to 64x64 and a car detection is performed.  Because the results can be noisy, heatmapping is used to overlay multiple, consistent detections, and a threshold is used to trigger an "overall" detection.  The report has some good discussion about the tradeoffs involved and possible improvements.

## Term 2
### [Unscented Kalman Filter](./)
### [Extended Kalman Filter](./)
### [Model Predictive Control](./)
### [PID Controller](./)
### [Kidnapped Vehicle Project](./)

## Term 3
### [Path Planning](./)
### [Semantic Segmentation](./)
### [Capstone](./)
