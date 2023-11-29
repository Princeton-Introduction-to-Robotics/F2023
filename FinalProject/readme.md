# MAE345/549: Final Project

For the final project in this class, you will write a program that guides the
Crazyflie from one end of the netted area to the other. To navigate, each Crazyflie is outfitted with a PoV camera capable of live streaming a view from the Crazyflie to your computer. This project is _open ended_, and you are free to use any approach to complete the task at hand.

## Test Scripts

The instructors have provided two example Jupyter notebooks and a Python script based on previous Lab notebooks. If you are new to Python, the script is more like a conventional program you may have written in the past, as compared to the Jupyter notebooks we used throughout the semester, in that all the code in the script is executed at once. The notebooks can be run like normal (i.e., launching Jupyter notebook or Jupyter Lab and running the notebooks cell-by-cell) and the Python script can be run with the command `python <scriptname>` in your terminal.

The script `test_camera.py` uses OpenCV to open a video device on your computer and plays frames from it in a computer; this is from Lab8. Depending on your computer, the video encoder may be video device 0, 1, 2,... By default, the script reads frames from video device 0, so you may need to change the number manually in a text editor before running. The notebook `sample-camera.ipynb` contains the exact same code as the `test_camera.py` script, but in Jupyter notebook form.

The notebook `sample-crazyflie.ipynb` is a minimal implementation of the full control loop of the Crazyflie based on Lab8. In this notebook, you will first perform red filtering on the live video feed from the camera. The instructors have provided several helper functions, which you should review the functionality of; some of these functions contain tunable parameters. The last cell of the notebook then flies the CrazyFlie and performs simple obstacle avoidance. The full process can be described, as follows:

1. Connects to the Crazyflie.

2. Reads frames from the video encoder for five seconds. This removes any black frames that may be received while the radio searches for the camera's frequency.

3. Ascends from the ground.

4. Reads a frame and processes with OpenCV. The processing is broken into the following steps:

    a. Converts the frame from BGR (OpenCV's default) to HSV, which is a more convenient color space for filtering by color.

    b. Applies a mask that creates an image where a pixel is white if the corresponding pixel in the frame has an HSV value in a specified range and black otherwise. The color range was roughly tuned by the instructors to match the red of the obstacles, but different lighting conditions and cameras may require further tuning.

    c. Applies OpenCV's contour detection algorithm to the masked image. This finds the edges between regions of color in an image.

    d. Checks the area (in pixels squared) of the largest contour. If it's greater than a threshold, the Crazyflie is instructed to move right and start Step 4 over. Otherwise, we move on to step 5.

5. The Crazyflie is instructed to land.

Thus, this script causes the drone to move right until it no longer sees a red object of significant size. Remember to set your drone/group number and select the correct camera number. Pay particular attention to the `check_contours` and `findGreatestContour` functions, which perform the red filtering and obstacle detection. The `adjust_position` function is responsible for maneuvering the CrazyFlie (i.e., what controls the side-stepping behavior). Additionally, this file contains a function `position_estimate` which you may fund useful; this function provides an example of retreiving an estiamte of the CrazyFlie's position.

## Lab Setup

The lab setup is similar to the RRT lab, with two main changes:
- The obstacle locations will not be revealed beforehand. Your drone will have to detect where they are. 
- There will be a target object (a book) placed on a table at the end of the obstacle course. A portion of your grade will be based on landing near the target object (see below for details on grading).

As with the RRT lab, your drone will start off on the spot marked with "X" on the floor.

You are free to move the obstacles around for testing purposes, but do not remove them from the lab spaces. 

## Demo Day and Grading

We will hold a Demo Day for evaluating final projects. This will be held on Dean's Date (Friday, December 15th). At the beginning of December, we will send out a sign-up sheet for the Demo Day. Each team will sign up for a time-slot (20 minutes) and will have three attempts at the obstacle course. Each team will also explain the technical approach that they took to the course staff and will have to submit code for the project. We will use the three netted zones for the Demo Day; you will be able to choose which zone to use for the demo (to ensure that the lighting conditions are similar to what you assumed when programming the drone). 

Your score on each of the three trials will be based on the following criteria:

1. (80 pts) Distance along the x (i.e., forward) direction your robot traversed before colliding. In particular, the score for a trial will be the fraction of the course your robot successfully traversed before colliding, e.g., (80 pts) * 70/100 if your robot covered 70 percent of the course before colliding ("colliding" is defined as the point at which your robot first touches/hits an obstacle or the ground, or the netting).
2. (20 pts) Landing near the target. The target object will be a book of your choice (that you will bring to the Demo Day). The only restriction on the book is that its length and width should be less than one foot (12 inches); any standard book should satisfy these criteria. The book will be placed by the instructors on the table/stand at the end of the course. The book may be placed "upright" (if you like) to make it easier for the drone to see the book when it is flying. Your goal is to make the drone land on the ground near the target object. If your drone crosses all the obstacles and lands within 15cm of the book (as measured from the closest point on the drone to the closest point on the book in the horizontal direction, i.e., y-direction), you will receive 20 points. If your drone crosses all the obstacles and lands within 30cm of the book in the horizontal direction, you will receive 10 points. No points will be awarded for this portion if your drone fails to cross all the obstacles or lands more than 30 cm away from the book. 

Each of the three trials will be scored based on the two criteria above. Note that the two criteria are not completely independent; reaching the target relies on successful navigation through the obstacle course. However, it is possible that your drone collides with an obstacle/ground/netting and somehow manages to keep going and land successfully near the target. In this case, you will receive full points for landing (assuming this was successful), but will receive points for navigation based on where your robot first experienced a collision or left the flying zone, e.g., (80 pts) * 70/100 if your robot covered 70 percent of the course before first touching/hitting an obstacle/ground or leaving the allowable flying zone.

**Your total score will be the average of the two best scores from the three trials.** The rationale for this is to evaluate the reliability of your system. Thus, **robustness is the main feature to strive for** in this project (as it has been throughout the course!).

## Some Suggested Approaches

### Identifying the target

For identifying the target (book), you can use the pretrained neural networks you used in Assignment 8 (e.g., detecting a person if you print out a cover for your book). You can also use the color of the book for detection.

### Obstacle avoidance and navigation

One option is to do what is known as reactive planning. In reactive planning, you compute a control action to take based on the current context of the system. Notably, reactive planners often do not need a complete map to function. You can use the fact that obstacles are colored red in order to identify them in the image. There are many strategies you could implement for obstacle avoidance (e.g., seeking a gap to navigate to, or maneuvering to avoid the obstacle right in front of you).

Another (more involved) option is to construct an estimate of obstacle locations in order to build a map, and then use a sampling based motion planner to navigate toward the goal region. You will need to periodically recompute your motion plan as you gather more information about obstacle locations.

## Advice From The Instructors

- Don't reinvent the wheel. Unlike previous assignments, you are not restricted in the libraries and techniques you may use to approach this challenge. We especially recommend that you use OpenCV to simplify things as much as possible. For example, if you want to use optical flow to compute the time to collision of your drone with an obstacle, OpenCV has a very good implementation of the optical flow algorithm. Similarly, the script `test_cf.py` makes good use of OpenCV's contour detection algorithm. You are welcome to look up documentation / information on the use of OpenCV, Numpy, etc.
- There are some Crazyflie functions that may be useful; see [here](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/positioning/) and [here](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/high_level_commander/).
- Go through the tips on working with the Crazyflie [here](https://github.com/Princeton-Introduction-to-Robotics/F2023/blob/main/crazyflie-tips.md).

- You can test your vision loop separately from your control code. If you think you are having trouble identifying the obstacles, you can disable your code to fly the drone and display what is happening at different points in your vision pipeline using OpenCV's imshow function. This way you can confirm your drone is seeing what you think it is seeing. You may also be able to do this while the drone is flying, but the instructors had mixed results due to some lag it introduced in our setup.

- You may want to consistently use one netted zone for testing. This will ensure that lighting conditions are consistent (note that you will be able to choose which zone you use for the demo). 

- Start simple. Begin with just moving around one obstacle. Once that is working reliably, add a second and a third.

- Start early. As you may have learned during the hardware labs, getting robots to work in the real world is tricky. Get started early to give yourself the best chance of success on the project! 
- As emphasized above, the grading rubric strongly encourages (and tests for) robustness. Getting something that works reliably is the key thing in this project (and in robotics as a whole)! 

We look forward to seeing the approaches you come up with!
