# GCNeuro
### Developing a Graph Convolution-Based Analysis Pipeline for Multi-Modal Neuroimage Data: An Application to Parkinson’s Disease
#### Christian McDaniel and Shannon Quinn

This work contains the code involved in an end-to-end neuroimage analysis pipeline. The pipeline preprocesses anatomical (T1w) MRI and diffusion MRI and combines them in a single graph-based format. In this format, the nodes of the graph are defined by anatomical regions of interest (ROIs) and the diffusion data is depicted as signals, or vectors, defined on these nodes. 

The graphs are processed using a novel graph convolutional network (GCN) architecture.

The paper, titled *Developing a Graph Convolution-Based Analysis Pipeline for Multi-Modal Neuroimage Data: An Application to Parkinson’s Disease* and which explains the pipeline in detail, has been published in the 2019 SciPy Proceedings.

## Dependencies

Non-python libraries: 
  bzip2
  git
  libxml2-dev
  unzip
  tcsh
  bc
  docker.io

Python libraries: 
  time
  math
  argparse
  os
  sys
  shutil
  subprocess
  json
  builtins
  pickle
  datetime
  numpy
  scipy
  pandas
  sklearn
  matplotlib
  seaborn
  joblib
  nibabel
  dipy
  torch

Softwares:
  python 3.5 or greater
  Freesurfer
  FSL
  MATLAB
  BrainSuite
  DTK
