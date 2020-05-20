# Udacity Machine Learning Nanodegree

_**Special Notes:  Please don't consider running through the projects as a substitute for watching the videos and doing the activities that precede the projects.  In some cases, the time spent on the projects may actually be LESS than the time spent absorbing the material Udacity provides as a precursor.  In that sense, what is presented here might seem disconnected or lacking in the background material necessary to complete the projects with a full appreciation of the techniques or algorithms used.  Additionally, the projects were expected to be run in the confines of an particular Anaconda environment.  Both Machine Learning (especially neural networks) AND the libraries written to implement them are evolving rapidly and you're likely to have some issues running the code in these projects with modern versions of the Python modules used.**_

The Machine Learning Nanodegree offered by Udacity in 2017 was a great course to follow my previous run-through of Kirill Eremenko's fantastic "Machine Learning A-Z" course on Udemy.  One of the great things about "classic" Machine Learning is that it's understandable from a human perspective, in contrast to the obfuscated/tedious logic of Neural Networks.

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

### [Finding Donors for CharityML](./finding_donors)
The student is "hired" to produce a model for a fictitious charity using census data to find people who make over $50k (i.e. where to send solicitations for charity most effectively).  This is the first project where "size matters."  That is, we have concerns about how much time training and predicting might take.  Also, the data preprocessing is performed by the student.
 - data preprocessing
   - logarithmic transformation for skewed features:  this can help when dealing with outliers that are many orders of magnitude higher or lower than the average
   - normalizing features:  this can be scaling and sometimes centering a feature to zero
   - one-hot encoding categorical features - this works best when there's some dimensionality to the feature (e.g. "small", "medium", "large")
 - setting a benchmark for model performance
 - the F-Beta score:  ![F-Beta Score Equation](https://render.githubusercontent.com/render/math?math=F_%7B%5Cbeta%7D%20%3D%20%281%20%2B%20%5Cbeta%5E2%29%20%5Ccdot%20%5Cfrac%7Bprecision%20%5Ccdot%20recall%7D%7B%5Cleft%28%20%5Cbeta%5E2%20%5Ccdot%20precision%20%5Cright%29%20%2B%20recall%7D&mode=display)
   - When beta==0.5, you have F<sub>0.5</sub> aka the F-score, which will emphasize precision.  This is easier to intuit from expression of this equation in terms Type I and Type II errors, as seen on the Wikipedia page for the [F1 score](https://en.wikipedia.org/wiki/F1_score)
   - Precision is the ratio of true positive guesses to all positive guesses.  Recall is how many of the actual positives in the data were correctly identified as positive.  So you could always guess "positive" and have perfect Recall, but risk a low Precision.
   - Why we may choose F-Beta score:  when we value Type I and Type II errors differently, we will want to weight them differently.  For an accuracy score, they are waited the same.
 - (some of the) Supervised Learning Models available in sklearn (see the notebook for a list)
 - Feature Importance Ranking - some alogrithsm (e.g. RandomForests) will allow you to get a ranked feature list based on importance to the algorithm
 - Feature Selection - retraining with only the top N most important features

### [Creating Customer Segments](./customer_segments)
In this project, the student is presented with a list of grocery purchase orders.  The goal is a *very* fun one:  discover customer segments - that is, identify categories/types of customers based on purchases.  It's an upsupervised learning task, which means we don't have a target variable (at least one that's given to us). Obviously this requires identifying clusters/patterns in the data in some way.  The clever approach taught by Udacity is [Principal Component Analysis](https://www.youtube.com/watch?v=kw9R0nD69OU) (be sure to watch all three).  This is a Linear Algebra approach of finding eigenvectors and eigenvalues for the data.  The point is to represent the data in its most important dimensions (one dimension per feature).  As mentioned in the videos, the excellent outcome is that the eigenvalues can be sorted by magnitude, which results in a sort by importance.  The other interesting point is that each dimension does its best to represent the data with minimal error - and that's because the first dimension is along the maximum vector of variance and subsequent dimensions do the same, but are orthogonal to any that preceded it.  This is such an ideal algorithm, it's kind of ridiculous.  New topics:
 - Feature distribution analysis.
 - Scatter Matrix.  A great tool to check for cross-correlation in feature distributions.
 - Feature distribution normalization:  how can we transform the data to have something closer to a Normal distribution?
   - Box-Cox test: mentioned in passing, but unused. (See [Power Transforms](https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation))
   - Logarithmic Scaling:  useful in this case because it stats off high to the left and exponentially fades off to the right.  Applying a log expands smaller values and compresses larger values, thereby stretching the distribution out (unevenly).
 - Tukey's Method for identifying outliers:  calculate the interquartile range IQR = (Q3 - Q1), calculate a step = IQR * 1.5 (or 3.0 for extreme), then see which feature samples are above (Q3 + step) or below (Q1 - step).
 - Dimensionality Reduction:  by using an explained variance ratio, we can trim features that are far less important than others, thereby simplifying the analysis.
 - Biplot - projects an N-dimensional set of features onto a two-dimensional graph
 - Unsupervised Learning
   - Principal Component Analysis: as described above, systematically identify dimensions of maximum variance
     - both forward and inverse!  we use inverse when it's time to convert a clustered mean back into the original units
   - Clustering
     - K-Means clustering:  richness and consistnet (not scale-invariant)
     - Gaussian Mixture Model:  scale-invariant and consistent
     - silhouette coefficient:  how similar (up to 1)/dissimilar (down to -1) a given data point is from a given cluster.  The mean silhouette coefficient is used to score a clustering.
 - A/B testing

### [Smart Cab](./smartcab)
This project departs from datasets and features and enters into Reinforcement Learning.  No, not **Deep** Reinforcement Learning, just the regular type - Q-Learning, actually.  It also introduces the use of a simulator environment, a concept that's used heavily by Udacity in the Self-Driving Car and Flying Car Nanodegrees (those are rich, 3D worlds, whereas this project uses a super-basic 2D pygame realm).  Q-learning uses the known state and a list of ranked options to decide what to do.  How it ranks those options is based on experience - i.e. learning the "hard way," with rewards and punishments meted out by the simulation (critically, doing nothing is penalized).  That means there's an exploration-to-exploitation transition (and maybe back again) that occurs.  It's really cool to see the little cab start to get wise!  Note that the "gamma" factor used to include future values in the current reward/punishment tables is not used in this project (and can't be either, read the bottom of the ipynb notebook to see why).  If you'd like another pocket-sized example of reinforcement learning, you can look up people training the snake game on youtube.  Some good techniques:
 - Tracking performance across iterations during training:
   - Plotting average reward / action
   - Plotting frequency of "bad" actions by type
 - Defining and Sizing the State Space
   - Analyzing average length of runs and using that to determine if there's enough time to cover the state space
 - Analyzing the Policy Table
   - can see the current state of rewards and decide if more training is needed
   - identifying which ones are suboptimal (if possible)
