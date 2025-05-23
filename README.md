# VehicleReID

## Description
This projects represents algorithms with the goal to detect, track and reidentify vehicles in general traffic. 
Various approaches are explored, and the performance of each method is evaluated, with a detailed comparison highlighting the best-performing model using Wandb.

## Features
* **Vehicle Detection**: Detection of vehicles in traffic using advanced computer vision techniques (YOLOv8, Faster-RCNN).
* **Tracking**: Follows detected vehicles across multiple frames to maintain consistent identities.
* **Reidentification**: Recognizes and matches the same vehicle across different viewpoints or time intervals.

## Results
Using Wandb ([Weights & Biases](https://wandb.ai/site/)) a thorough evaluation of different techniques is provided, along with metrics demonstrating the effectiveness of the best-performing algorithm.
Use this [link](https://wandb.ai/krausm00/VehicleReID/reports/Vehicle-Re-Identification--VmlldzoxMjI1NTc0NQ) to see the report. 

Along with this repository, an interactive demo is available on [HuggingFace spaces](https://api.wandb.ai/links/krausm00/zcq7i8gk) allowing users to view the tracking results without any installation.


## Usage
In order to run the individual scripts, please follow next steps.
```
git clone https://github.com/BerassHaggy/VehicleReID.git
cd VehicleReID
pip3 install -r requirements.txt
```
