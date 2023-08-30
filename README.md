# Introduction to Robotics Fall 2022 Repository

Source code assignments for MAE345 (COS346, ECE345, MAE549).

This repository contains Jupyter notebook assignments for Princeton's MAE345 class. It is organized in the following manner:

- All Jupyter notebooks are placed in the top level directory so they have access to all other Python modules and paths referenced in them are consistent.
- All data provided / collected for use in assignments resides in `data`.

## Install Instructions 

Included in this repository is a conda environment named `env-mae345.yml`. For the unfamiliar, [conda](https://docs.conda.io/en/latest/) (short for Anaconda) is a tool for managing Python environments --- collections of software and libraries for developing Python programs. Conda environments make it very easy to reproduce and share code with other developers (in this case between the students and AIs).

To install the environment, do the following:

1. Download and install [Anaconda](https://www.anaconda.com/products/individual).

2. On Mac and Linux, open the terminal. Navigate to where this repository has been downloaded (entering `ls` will list the files and directories accessible from your current directory and `cd <name>` will change you to the `<name>` directory) and run `conda env create -f env-mae345.yml`. Accept any of the prompted changes. On Windows, do the same, use the Anaconda Prompt application that should be present in your start menu (on Windows you need to use `dir` to list the contents of a directory instead of `ls`).

## Working on Assignments

To work on an assignment, open the terminal (on Windows you need to use the same Anaconda Prompt application you used to install the environment) and navigate to the directory containing this repository. Enter the command `conda activate mae345` to load the environment. Then run either `jupyter lab` or `jupyter notebook`. Both launch an interface for editing and running Python scripts in your browser. The former is a newer, more featureful interface while the latter is older and straightforward. Follow the instructions within the notebook to complete the assignment.

Submission instructions are included at the end of each notebook file. 

## Accessing New and Updated Assignments

As new assignments are released, either download the new files and place them in the folder containing the existing labs, or redownload this repository and replace the existing folder. You do not need to recreate the conda environment.

