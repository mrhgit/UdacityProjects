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

### [Planning and Search](./AIND-Planning)
This project required two products:  a [research paper](./AIND-Planning/research_review.pdf) that briefly covered the history of three planning and/or search algorithms and a [report](./AIND-Planning/heuristic_analysis.pdf) on the work done to solve three planning problems.  The planning problems involved moving N pieces of cargo from their starting airports to their destination airports with M planes at their own starting airports in as few flights as possible.

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

The report includes descriptions of these, as well as graphs of their relative performances!
