# Partial_View_SMPL

This repository contains the code for ...


## Install

Set the environment:

 - update in the existing environment conda env update -f environment.yml in an existing environment, or

 - create a new environment "Partial_View" conda env create -f environment.yml, for a new smplpytorch environment

Install the following packages: 

  - human body prior from https://github.com/nghorbani/human_body_prior

  - body_visualizer from https://github.com/nghorbani/body_visualizer


## Tutorial

### Partial view of a demo shape
You can use the following command

 python demo.py

The demo file will show three shapes as an illustration.


### Changing the view

In the file demo_translation.py, 

it is an illustration showing that the object moving along the x-axis, y-axis and z-axis for 10 time steps and for 10 shapes respectively.

So, for each moving direction, we show 100 shapes.

In the file demo_rotation.py, 

it is an illustration showing that the object rotating along the x-axis, y-axis and z-axis for 10 time steps and for 10 shapes respectively.

So, for each moving direction, we show 100 shapes.
