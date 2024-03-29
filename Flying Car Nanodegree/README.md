# Udacity Flying Car Nanodegree

_**Special Notes:  Please don't consider running through the projects as a substitute for watching the videos and doing the activities that precede the projects.  In some cases, the time spent on the projects may actually be LESS than the time spent absorbing the material Udacity provides as a precursor.  In that sense, what is presented here might seem disconnected or lacking in the background material necessary to complete the projects with a full appreciation of the techniques or algorithms used.  Additionally, the projects were expected to be run in the confines of an particular Anaconda environment.  Both Machine Learning (especially neural networks) AND the libraries written to implement them are evolving rapidly and you're likely to have some issues running the code in these projects with modern versions of the Python modules used.**_

The Flying Car Nanodegree offered by Udacity in 2018 was a whimsical pursuit to satisfy my craving for learning more from Udacity's wonderful programs.  Unfortunately, this program was cut short after a single term.  I'm not sure where it stands now.  Another drawback was the heavy borrowing of content from other nanodegrees, with the exception of drone control, which was neat to learn about and code myself from fundamental equations.  The projects focused around controller quadcopter drones inside a nicely-built simulator designed by Udacity.  It was a lot of fun and I learned a few tricks along the way.

## Projects
Below is the list of projects from the course, along with links to my solutions.  In each directory, you'll find a README.md written by Udacity, along with some of their helper code.  Generally, the student's view of the project is the Jupyter Notebook files with an .ipynb extension (viewable in GitHub), which is where you'll find the string of guidance, activities, and Q&A that make up the project.  Sometimes, I had to fill in TODO sections in a separate code file.
<hr>

### [Backyard Flyer Project](./Backyard-Flyer)

This project was an introductory project in controlling the drone.  The concept was to use a Finite State Machine to handle the logic of progressing through a series of steps necessary to complete a planned flight.  Things like take-off, altitude check and progressing through a series of waypoints were steps that had to be managed.

Here's a photo of the drone zooming off to the next waypoint.  I definitely tuned it to fly aggressively through the waypoints, so it's banking hard!

![Backyard-Flyer](./Backyard-Flyer/backyard_flyer.png)

<hr>

### [Motion Planning](./Motion-Planning)

[Report](./Motion-Planning/report.pdf)

In this project, AI, graph-based solutions are used to find a path for the drone through a skyscraper-rich cityscape.  A 3-D map of the city is provided by defining building locations and dimensions.  The drone is given a starting point and destination point with a task of finding the best path.  This is similar to the planning algorithm projects in the Artificial Intelligence Nanodegree.  While it's possible to consider every single possible point as a node in the graph, a stochastic method is used for efficiency.  Basically, a smattering of points are randomly added to open areas of the map and two node are added for the start and destination points.  An A* algorithm is run with that limited set of nodes.  One more step is performed, which is the elimination of points which are unnecessary intermediates - i.e. if I have to travel from A->B->C->D, but I can go directly from A->D without running into anything, then I should simply skip points B and C.

![Motion Planning](./Motion-Planning/motion_planning_1.png)

<hr>

### [Drone Controller](./Drone-Controller)

[Report](./Drone-Controller/report.pdf)

As can be seen in the report, this project was very formula-heavy, which was actually quite satisfying to work with.  It's really neat to see the mathematical models of physics come to life.  The concept is that there are only four motors (propellers, really) that can be turned which are meant to control all aspects of the drone.  The consequential equations are listed neatly in the [report](./Drone-Controller/report.pdf).  PID controllers are used to keep the performance working under non-ideal circumstances, such as a lopsided drone, motors with different torque, etc.  The nicest thing about the project was of course the simulator which saved tons of time and hassle, but another nice thing was that we were forced to use real physics units (meters, Newtons, etc.), which offers a feeling of accomplishment that you could make something nearly real.

![Drone Controller](./Drone-Controller/drone_controller.png)

<hr>

### [Orientation-Estimation with Extended Kalman Filters](./Orientation-Estimation)

[Report](./Orientation-Estimation/report.pdf)

In this project, all the nice, ideal sensors and state knowledge are taken away and replaced with noisy measurements and a DIY approach to determining state.  The first part of the project is a trivial exercise in determining the noise (standard deviation) of the GPS and accelerometer measurements coming in.  For the noisy measurements and in order to estimate our state, we reintroduce the Extended Kalman Filter (seen before in the Self-Driving Car Nanodegree).  The sensor update intervals are meant to be realistic, with the IMU, magnetometer and GPS updating at differing rates.  Many equations are provided in the [report](./Orientation-Estimation/report.pdf), including rotation matrices, measurement-space-to-state-space conversions, and the EKF functions.

Once the sensors can viably be used with the EKF to substitute for what used to be ideal information, a task was given to follow a trajectory using that information instead.  The graphs below show the consequences of ideal vs noisy/estimated.  Quite a difference!  As the report indicates, perhaps a bit more tuning is needed before this one's ready for the sky!

![Ideal Estimators and Sensors](./Orientation-Estimation/ideal_estimator_sensors.png)
![EKF Estimators and Noisy Sensors](./Orientation-Estimation/noisy_estimator_sensors.png)
