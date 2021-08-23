# PyTracking

A general python library for visual tracking algorithms. 
## Table of Contents

* [Running a tracker](#running-a-tracker)
* [Overview](#overview)
* [Trackers](#trackers)
   * [JOINT](#JOINT)
* [Libs](#libs)
* [Integrating a new tracker](#integrating-a-new-tracker)


## Running a tracker
The installation script will automatically generate a local configuration file  "evaluation/local.py". In case the file was not generated, run ```evaluation.environment.create_default_local_file()``` to generate it. Next, set the paths to the datasets you want
to use for evaluations. You can also change the path to the networks folder, and the path to the results folder, if you do not want to use the default paths. If all the dependencies have been correctly installed, you are set to run the trackers.  

**Run the tracker on some dataset sequence**  
This is done using the run_tracker script. 
```bash
python run_tracker.py tracker_name parameter_name --dataset_name dataset_name --sequence sequence --debug debug --threads threads
```  

Here, the dataset_name is the name of the dataset used for evaluation, e.g. ```dv2017_val```. See [evaluation.datasets.py](evaluation/datasets.py) for the list of datasets which are supported. The sequence can either be an integer denoting the index of the sequence in the dataset, or the name of the sequence, e.g. ```'blackswan'```.
The ```debug``` parameter can be used to control the level of debug visualizations. ```threads``` parameter can be used to run on multiple threads.

## Overview
The tookit consists of the following sub-modules.
 - [analysis](analysis): Contains scripts to analyse tracking performance, e.g. obtain success plots, compute AUC score. It also contains a [script](analysis/playback_results.py) to playback saved results for debugging.  
 - [evaluation](evaluation): Contains the necessary scripts for running a tracker on a dataset. It also contains integration of a number of standard tracking and video object segmentation datasets, namely  [OTB-100](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [NFS](http://ci2cv.net/nfs/index.html),
 [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx), [Temple128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [TrackingNet](https://tracking-net.org/), [GOT-10k](http://got-10k.aitestunion.com/), [LaSOT](https://cis.temple.edu/lasot/), [VOT](http://www.votchallenge.net), [Temple Color 128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [DAVIS](https://davischallenge.org), and [YouTube-VOS](https://youtube-vos.org).  
 - [features](features): Contains tools for feature extraction, data augmentation and wrapping networks.  
 - [libs](libs): Includes libraries for optimization, dcf, etc.  
 - [parameter](parameter): Contains the parameter settings for different trackers.  
 - [tracker](tracker): Contains the implementations of different trackers.  evaluation servers, downloading pre-computed results. 
 - [utils](utils): Some util functions.
 
## Trackers
 The toolkit contains the implementation of the following trackers.  

### JOINT
The official implementation for the JOINT tracker ([paper](https://arxiv.org/pdf/2108.03679.pdf)). 
The tracker implementation file can be found at [tracker.joint](tracker/joint). 

##### Parameter Files
Two parameter settings are provided. These can be used to reproduce the results or as a starting point for your exploration.  
* **[joint_ytvos](parameter/joint/joint_ytvos.py)**: The default parameter setting with ResNet-50 backbone which was used to generate YouTubeVOS results.
* **[joint_davis](parameter/joint/joint_davis.py)**: The default parameter setting with ResNet-50 backbone which was used to generate DAVIS results.

## Libs
The pytracking repository includes some general libraries for implementing and developing different kinds of visual trackers, including deep learning based, optimization based and correlation filter based. The following libs are included:

* [**Optimization**](libs/optimization.py): Efficient optimizers aimed for online learning, including the Gauss-Newton and Conjugate Gradient based optimizer used in ATOM.
* [**Complex**](libs/complex.py): Complex tensors and operations for PyTorch, which can be used for DCF trackers.
* [**Fourier**](libs/fourier.py): Fourier tools and operations, which can be used for implementing DCF trackers.
* [**DCF**](libs/dcf.py): Some general tools for DCF trackers.

## Integrating a new tracker  
 To implement a new tracker, create a new module in "tracker" folder with name your_tracker_name. This folder must contain the implementation of your tracker. Note that your tracker class must inherit from the base tracker class ```tracker.base.BaseTracker```.
 The "\_\_init\_\_.py" inside your tracker folder must contain the following lines,  
```python
from .tracker_file import TrackerClass

def get_tracker_class():
    return TrackerClass
```
Here, ```TrackerClass``` is the name of your tracker class. See the [file for DiMP](tracker/dimp/__init__.py) as reference.

Next, you need to create a folder "parameter/your_tracker_name", where the parameter settings for the tracker should be stored. The parameter file shall contain a ```parameters()``` function that returns a ```TrackerParams``` struct. See the [default parameter file for DiMP](parameter/dimp/dimp50.py) as an example.

 
 
