# Tips for working with the Crazyflie drones

This is a list of helpful tips for working with the drones for the course. You may want to refer to these at multiple points in the semester when working with the drones. 

1. Take a look at the ["Getting Started"](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/) page for the Crazyflie. In addition to build instructions, this page also contains a lot of helpful information (e.g., switching the drone on/off, the meaning of different LED light patterns, etc.).
2. Make sure that your drone's propellers are installed with the correct orientations; see the ["Getting Started"](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/) page. 
3. Make sure that the battery cable isn't interfering with the spinning propellers. 
4. In Lab 2, if you see your drone fly straight up and into the ceiling, you have almost certainly installed the optical flow board upside down (this only applies to teams who built their own drones; drones we provided have been tested and will have the optical flow boards installed correctly). 
5. Make sure your drone's batteries are not low! The Crazyflie doesn't always perform well when the battery is low. You can check the battery charge status using cfclient (see Lab 2 for instructions).
6. Turn the quadrotor on when it is plugged in to charge it. It will not charge when it is powered off so that the onboard processor can monitor the battery and prevent it from overcharging.
7. Before switching the drone on, make sure to keep it on a level surface and hold it still. The drone calibrates its sensors when you switch it on; this calibration can get messed up if it is not on a flat surface or the drone is being moved around. 
8. Occasionally, we have seen different drones interfere with each other. This seems to happen more often when there are a number of drones (e.g., 3-4) trying to fly at the same time in close proximity, or if two drones with similar radio channel numbers (set by your team's group number) are operating in close proximity. If you notice that your drone is behaving very weirdly, try to make sure that there aren't a large number of other teams flying at the same time. 
