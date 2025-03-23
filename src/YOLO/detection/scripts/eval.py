"""
This script evaluates YOLO pretrained model on a given dataset.
"""

from ultralytics import YOLO
import torch
import yaml
import wandb
import numpy as np
torch.cuda.empty_cache()
model = YOLO('yolov8m.pt') # Base pretrained model
model = YOLO('/Users/martinkraus/Downloads/best.pt')

# evaluation
config_file = "/storage/plzen1/home/krausm00/MPV/configs/config.yaml"
metrics = model.val(data=config_file, imgsz=640)

print("-------------- Evaluation metrics --------------")
print(metrics.mean_results())
# returns the following: [self.mp, self.mr, self.map50, self.map, self.map75, self.ap[0]] + ap_per_class
evaluation_results = metrics.mean_results()
print("------------------------------------------------")

# read the individual classes
with open(config_file, 'r') as file:
    data = yaml.safe_load(file)

nc = data['nc']  # number of classes
class_names = data['names']  # list of class names

metrics = {} # for wandb logging 
# print the evaluation metrics (same as in detectron2)
print("mAP50: " + str(evaluation_results[2]))
print("mAP50-95: " + str(evaluation_results[3]))
print("mAP75: " + str(evaluation_results[4]))

metrics["eval/mAP50"] = evaluation_results[2]
metrics["eval/mAP50-95"] = evaluation_results[3]
metrics["eval/mAP75"] = evaluation_results[4]

f1_score = metrics.box.f1
metrics["eval/f1_score"] = f1_score

# calculate the fitness metric
w = np.array([0.1, 0.9])  # weights for [mAP@0.5, mAP@0.5:0.95]
values = np.array([metrics["eval/mAP50"], metrics["eval/mAP50-95"]])
fitness = np.sum(values * w)
metrics["eval/eval_fitness"] = fitness

# log the values into wandb
wandb.init(project="PEOPLE_DETECTION_YOLO", entity="krausm00", config=model.cfg, name="YOLO_eval")
wandb.log(metrics, step=1) # step=1 for better visualization
