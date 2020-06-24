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
[Report](./Traffic_Sign_Classifier.ipynb/report.pdf)

I'll let the [report](./Traffic_Sign_Classifier.ipynb/report.pdf) do most of the talking about this one.  In general, the purpose of this project is to identify European traffic signs using a Convolutional Neural Network (CNN).  The project incorporates transfer learning and also feature extraction and visualization, which aides the designer by giving some feedback about what the neural network is attempting to identify in the images.

### [Behavior Cloning](./)
### [Lane Lines](./)
### [Advanced Lane Lines](./)
### [Vehicle Detection](./)

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
