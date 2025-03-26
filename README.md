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

## Usage
In order to run the individual scripts, please follow next steps.
```
git clone https://github.com/BerassHaggy/VehicleReID.git
cd VehicleReID
pip3 install -r requirements.txt
```
## Future Work