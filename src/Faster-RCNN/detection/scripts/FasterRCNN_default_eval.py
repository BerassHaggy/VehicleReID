"""
This script evaluates Detectron2's Faster R-CNN pretrained model on a given dataset.
"""

import sys, os, distutils.core
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import wandb
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.engine import HookBase, DefaultTrainer

import torch
torch.cuda.empty_cache()

annotations_path = "/storage/plzen1/home/krausm00/DP/FastRCNN/datasets/highway_vehicles"

train_path_images = annotations_path + "/train"
train_path_annotations = train_path_images + "/annotations.json"

test_path_images = annotations_path + "/test"
test_path_annotations = test_path_images + "/annotations.json"

val_path_images = annotations_path + "/valid"
val_path_annotations = val_path_images + "/annotations.json"


register_coco_instances('my_dataset_train', {}, train_path_annotations, train_path_images)
register_coco_instances('my_dataset_val', {}, val_path_annotations, val_path_images)

vehicles_metadata = MetadataCatalog.get('my_dataset_train')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ('my_dataset_train', )
cfg.DATASETS.TEST = ('my_dataset_test', )


cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo


# Load the annotations file
with open(train_path_annotations, "r") as f:
    coco_annotations = json.load(f)


"""
    Get the class names from the coco annotations

"""
class_names = []
classes = coco_annotations["categories"]
for class_name in classes:
    id = class_name['id']
    if id == 0: # this is only for highway vehicles as they have additional class at 0
        continue
    name = class_name['name']
    class_names.append(name)

number_of_images = len(coco_annotations['images'])
number_of_epochs = 100
batch_size = 16
iterations_per_epoch = number_of_images / batch_size
total_iterations = iterations_per_epoch * number_of_epochs
print("total iterations: " + str(total_iterations))


cfg.SOLVER.IMS_PER_BATCH = batch_size  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = int(total_iterations)    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 200
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12 # (5 for custom dataset, 12 for highway vehicles)
"""
Define the image input size - 640 as in YOLOv8
"""
cfg.INPUT.MIN_SIZE_TRAIN = (640, )
cfg.INPUT.MAX_SIZE_TRAIN = 640
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 640

wandb.init(project="HIGHWAY_VEHICLES", entity="krausm00", config=cfg, name="RCNN_eval")

# Inference + evaluation
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01 # corresponds to "yolo conf" value which is set to 0.01 https://docs.ultralytics.com/modes/val/#usage-examples

"""
    Define the selected hyperparameter values for evaluation
"""


predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")

# Perform evaluation
evaluation_results = inference_on_dataset(predictor.model, val_loader, evaluator)


# Process the evaluation metrics for wandb
val_metrics = {}
# logging the evaluation metrics to wandb
evaluation_results_items = evaluation_results.items()
print(evaluation_results_items)
for key, value in evaluation_results_items: 

    # AP metrics (AP stands for mAP in coco evaluation)
    val_metrics["eval/mAP50-95"] = value["AP"]
    val_metrics["eval/mAP50"] = value["AP50"]
    val_metrics["eval/mAP75"] = value["AP75"]
    val_metrics["eval/mAPs"] = value["APs"]
    val_metrics["eval/mAPm"] = value["APm"]
    val_metrics["eval/mAPl"] = value["APl"]

    # AR metrics
    val_metrics["eval/mAR1"] = value["AR1"]
    val_metrics["eval/mAR10"] = value["AR10"]
    val_metrics["eval/mAR100"] = value["AR100"]
    val_metrics["eval/mARs"] = value["ARs"]
    val_metrics["eval/mARm"] = value["ARm"]
    val_metrics["eval/mARl"] = value["ARl"]

    # iterate through the class names to find class specific AP
    for class_name in class_names:
        val_metrics["eval/AP-" + class_name] = value["AP-" + class_name]

"""
    Calculate the fitness metric
"""
# calculate the fitness value
w = np.array([0.1, 0.9])  # weights for [mAP@0.5, mAP@0.5:0.95]
values = np.array([val_metrics["eval/mAP50"], val_metrics["eval/mAP50-95"]])
fitness = np.sum(values * w)
val_metrics["eval/eval_fitness"] = fitness

print("-------------- Evaluation metrics --------------")
print(val_metrics)
print("------------------------------------------------")


# Log evaluation results and cfg to wandb
wandb.log(val_metrics, step=1) # step=1 for better visualization