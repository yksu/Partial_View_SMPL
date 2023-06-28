# Partial_View_SMPL

This repository contains the code for ...


## Install

Set the environment:

 - update in the existing environment:
   **conda env update -f environment.yml**
   in an existing environment, or

 - create a new environment "Partial_View":
   **conda env create -f environment.yml**
   for a new smplpytorch environment

Install the following packages: 
  - open3d by pip install open3d
  - human body prior from https://github.com/nghorbani/human_body_prior

  - body_visualizer from https://github.com/nghorbani/body_visualizer

  - pytorch3d from https://github.com/facebookresearch/pytorch3d
## Tutorial

### Partial view of a demo shape
You can use the following command

- python demo.py

The demo file will show three shapes as an illustration.


### Changing the view

In the file **demo_translation.py**, 

it is an illustration showing that the object moving along the x-axis, y-axis and z-axis for 10 time steps and for 10 shapes respectively.

So, for each moving direction, we show 100 shapes.

In the file **demo_rotation.py**, 

it is an illustration showing that the object rotating along the x-axis, y-axis and z-axis for 10 time steps and for 10 shapes respectively.

So, for each rotating direction, we show 100 shapes.

### Vertex faces mapping

In the file **demo_mapping.ipynb**, 

it is an illustration for vertex mapping method and faces mapping method.

### Trainin the Auto-Encoder Model:

In the notebook **demo_AE.ipynb**, 

it is an illustration on how to train the auto-encoder model for point clouds.
