# Udacity Artificial Intelligence Nanodegree

_**Special Notes:  Please don't consider running through the projects as a substitute for watching the videos and doing the activities that precede the projects.  In some cases, the time spent on the projects may actually be LESS than the time spent absorbing the material Udacity provides as a precursor.  In that sense, what is presented here might seem disconnected or lacking in the background material necessary to complete the projects with a full appreciation of the techniques or algorithms used.  Additionally, the projects were expected to be run in the confines of an particular Anaconda environment.  Both Machine Learning (especially neural networks) AND the libraries written to implement them are evolving rapidly and you're likely to have some issues running the code in these projects with modern versions of the Python modules used.**_

The Artificial Intelligence Nanodegree offered by Udacity in 2017 was next on my list of courses to take, after having previously completed Machine Learning Nanodegree.  The MLND covered classical marchine learning and introduced neural networks.  The AIND did something similar in that it covered classical artificial intelligence and ventured more heavily into neural networks (such as using RNNs and GANs).

An important note is that many of the projects in the AIND were automatically graded, so there may only be filled-out, TODO sections instead of a report.

## Projects
Below is the list of projects from the course, along with links to my solutions.  In each directory, you'll find a README.md written by Udacity, along with some of their helper code.  Generally, the student's view of the project is the Jupyter Notebook files with an .ipynb extension (viewable in GitHub), which is where you'll find the string of guidance, activities, and Q&A that make up the project.  Sometimes, I had to fill in TODO sections in a separate code file.

### [Sudoku Solver](./Sudoku)
The sudoku solver project required me to solve a sudoku puzzle using a depth-first, recursive search:

0. Given a grid state,
1. Simplify (repeat until no change occurs)
    - Look for "naked twins," which involves finding two peer tiles with matching pairs of candidate values and then proceeding to remove those candidate values from peers of either tile (not including the original peer).
    - Look for tiles with only one candidate and remove that candidate from peer tiles.
    - Assign single candidates as true solutions
2. If no unsolved tiles remain, return the complete puzzle
3. Loop over the candidates of the tile with the least candidate values
    - set the value in the grid and recursively re-enter step 0.
    - if the result is False, continue, otherwise it's a complete puzzle - return it
4. If none of the candidates worked, return False so the recursion can continue

It was a fun project with lots of pythonic one-liners.  I might have gone a bit heavy on them, but they just look so good.

### [Planning and Search](./Planning)
This project required two products:  a [research paper](./Planning/research_review.pdf) that briefly covered the history of three planning and/or search algorithms and a [report](./Planning/heuristic_analysis.pdf) on the work done to solve three planning problems.  The planning problems involved moving N pieces of cargo from their starting airports to their destination airports with M planes at their own starting airports in as few flights as possible.

Many different algorithms were used (as mentioned in the report):
* Non-heuristic
    * breadth-first tree search
    * depth-first graph search
    * breadth-first search
    * depth-limited search
    * uniform cost search
* Heuristic
    * greedy best first graph search
    * recursive best first search with h1
    * a-star search with three different heuristics:
        * h1
        * ignore preconditions
        * planning graph level sum

The report includes descriptions of these, as well as graphs and analysis of their relative performances.

### [The Game of Isolation](./Isolation)
This project also required two products:  a [research paper](./Isolation/research_review.pdf) that reviewed a technical article in artificial intelligence and a [report](./Isolation/heuristic_analysis.pdf) on the results of an algorithmic competition in the game of Isolation.

This was a foray into alpha-beta pruning, which is a systematic branching technique used to evaluate the possible maximum and minimum rewards of making a specific move, given a game state.  An interesting twist was that there was a time cutoff, which was addressed in this project by progressively searching all options deeper iteratively.  This is a brute-force technique and in games where the number of moves is very high, it's important to prioritize which moves you should investigate and be able to estimate the consequential rewards quickly.  The paper explains how Google mangaged to build Alpha Go so it could use neural networks to perform the move prioritization and reward estimation in order to win at the game of Go.

### [Dog Breed Classifier](./dog-project)
[Jupyter Notebook](./dog-project/dog_app.ipynb)

In this project, images are fed to a dog breed classifier based on a neural network.  Actually, there are two stages of classification:  is it a human, a dog, or something else?  If it's a human or a dog, then determine the closest dog breed that resembles the subject in the photo.  The way in which it completes the preliminary dog/human/??? step is first trying to detect a dog using the ResNet50 model included with Keras.  If that fails, it attempts to detect a human using the Haar frontal-face detection cascade classifier.  If that fails, it prints a message saying it's stumped.  

Used in the project:
* Some light OpenCV code
* Cascade Classifiers
* Using the Store Neural Networks (and weights) included with Keras
* CNN Design
* CNN Training from Scratch
* Transfer Learning
