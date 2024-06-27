# Fully-Connected Neural Network (FCNN) Inverse Kinematics (IK) - 2-Link Robot Manipulator

## Requirements

**Programming Language**

```bash
Python
```

**Import Libraries**
```bash
For additional information, please refer to the individual scripts (.py) or
the 'Installation Dependencies' section located at the bottom of the Readme.md file.
```

**Supported on the following operating systems**
```bash
Linux, macOS, Windows
```

## Project Description

Text ...

## Installation Dependencies

It will be useful for the project to create a virtual environment using Conda. Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs and updates packages and their dependencies.

**Set up a new virtual environment called {name} with python {version}**
```
$ ../user_name> conda create -n {name} python={version}
$ ../user_name> conda activate {name}
```

**Installation of packages needed for the project**
```
Matplotlib
$ ../user_name> conda install -c conda-forge matplotlib

SciencePlots
$ ../user_name> conda install -c conda-forge scienceplots

Pandas
$ ../user_name> conda install -c conda-forge pandas

Scikit-learn
$ ../user_name> conda install anaconda::scikit-learn

Keras-Tuner
$ ../user_name> conda install conda-forge::keras-tuner

Tensorflow
$ ../user_name> conda install conda-forge::tensorflow


**Other useful commands for working with the Conda environment**
```
Deactivate environment.
$ ../user_name> conda deactivate

Remove environment.
$ ../user_name> conda remove --name {name} --all

To verify that the environment was removed, in your terminal window or an Anaconda Prompt, run.
$ ../user_name> conda info --envs

Rename the environment from the old name to the new one.
$ ../user_name> conda rename -n {old_name} {name_name}
```
