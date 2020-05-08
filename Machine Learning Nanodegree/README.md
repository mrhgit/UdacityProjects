# Udacity Machine Learning Nanodegree

_**Special Notes:  Please don't consider running through the projects as a substitute for watching the videos and doing the activities that precede the projects.  In some cases, the time spent on the projects may actually be LESS than the time spent absorbing the material Udacity provides as a precursor.  In that sense, what is presented here might seem disconnected or lacking in the background material necessary to complete the projects with a full appreciation of the techniques or algorithms used.  Additionally, the projects were expected to be run in the confines of an particular Anaconda environment.  Both Machine Learning (especially neural networks) AND the libraries written to implement them are evolving rapidly and you're likely to have some issues running the code in these projects with modern versions of the Python modules used.**_

The Machine Learning Nanodegree offered by Udacity in 2017 was a great course to follow my previous run-through of Kirill Eremenko's fantastic "Machine Learning A-Z" course on Udemy.

It was my first Nanodegree.  My reasons for choosing a Udacity Nanodegree were that it was supposed to be something graded (read "earned") and was a chance to learn more while practicing ML and getting feedback from those who knew better than I did.  Generally, I'd say my expectations were met.

The best part of the course are the video instruction and all the legwork that was done to set up project frameworks such that I could focus on the "meat" of the problems.  It was a luxurious experience.  The downside of this approach is that sometimes it can feel like a simplistic, "fill in the blank" problem, however I felt truly engaged and wanting for an enriching experienced which kept me for just trying to find the simplest answer (OK, most of the time anyway).

## Projects
Speaking of projects, here is the list of projects from the course, along with links to my solutions.  In each directory, you'll find a README.md written by Udacity, along with some of their helper code.  Generally, the student's view of the project is the Jupyter Notebook files with an .ipynb extension (viewable in GitHub), which is where you'll find the string of guidance, activities, and Q&A that make up the project.  Sometimes, I had to fill in TODO sections in a separate code file.  Most of it is in Python 2.7, using primarily numpy, pandas, sklearn and matplotlib.

### [Titanic Survival Exploration](./titanic_survival_exploration)
What a great opening project!  It's fun, easy to imagine and comes with lots of interesting data.  What's a budding data scientist not to love?  The goals, as the project progresses, are to 1) understand the data and 2) come up with a model to predict survival.  In the end, a score of 80% does the trick.
Important concepts:
 - data exploration
 - identifying which field to predict ("target variable")
 - identifying which fields from which to derive a prediction ("features")
 - picking a basic model to beat (in this case, simply predicting no one survives)
 - model evaluation
 - intro to decision trees

### [Boston Housing Project](./boston_housing)
This is referred to as the "first project of Machine Learning Nanodegree," which it technically was, because the Titanic Survival Exploration was an optional project.  As I recall, I had to complete that one to get accepted... maybe...  Anyway, the Boston Housing project is another very relatable, very engaging project (as pretty much ALL Udacity projects are - call me a sycophant).
Important concepts:
 - supervised learning algorithms
 - data shuffling and splitting into training and testing datasets *very important*
 - performance metric:  in this case, coefficient of determination, R<sup>2</sup>
 - data statistics: e.g. min, max, mean, median, variance...
 - latent features:  features that aren't directly observable, but can be measured by their impact on observable features
 - decision tree model learning curves (intro to under/overfitting - *very important*)
 - complexity curves (basically convergence values for the learning curves vs. depth)
 - bias-variance tradeoff:  is the model not nuanced enough (high bias) or too nuanced (high variance)?
 - grid search: training multiple models with different hyperparameter values to find the best ones
 - k-fold cross-validation:  splitting the data into k chunks and using k-1 chunks for training and 1 for testing, rotating through the chunks such that each one is the testing set once
 - shuffled set validation:  shuffling N times, taking 20% as test data
 - sensitivity:  in this case, predicting the price multiple times with the same model, but different training/test datasets and observing the range in prices
 - applicability:  judging how well your model might work on other data, considering the data that was used to train it and the type of model chosen


 
