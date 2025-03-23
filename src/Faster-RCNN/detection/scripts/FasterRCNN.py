"""
    This script stands for a classic Faster R-CNN training with wandb logging.

    Some of the parameters are set in order to allign with YOLO configuration.

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
from detectron2.checkpoint import DetectionCheckpointer, Checkpointer

import torch
torch.cuda.empty_cache()

"""
    Define the folder with annotations.
"""
annotations_path = "/storage/plzen1/home/krausm00/DP/FastRCNN/datasets/custom_vehicles"

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
cfg.DATASETS.TEST = ('my_dataset_val', )


cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

# Load the annotations file
with open(train_path_annotations, 'r') as f:
    coco_annotations = json.load(f)


"""
    Get the class names from the coco annotations

"""
class_names = []
classes = coco_annotations["categories"]
for class_name in classes:
    id = class_name['id']
    # only for the highway vehicles, where the category 0 corresponds to all vehicles
    #if id == 0:
    ##   continue
    name = class_name['name']
    class_names.append(name)
    

number_of_images = len(coco_annotations['images'])
number_of_epochs = 100
batch_size = 16
iterations_per_epoch = number_of_images / batch_size
total_iterations = int(iterations_per_epoch) * number_of_epochs + 1
print("total iterations: " + str(total_iterations))
epoch = 0



class WandbHook(HookBase):
    def __init__(self, cfg, eval_period, evaluator=None):
        super().__init__()
        self.cfg = cfg
        # Calculate iterations per epoch
        images_per_batch = batch_size
        total_images = number_of_images  # This might need adjustment based on how you've set up your dataset
        self.iterations_per_epoch = total_images // images_per_batch
        self.evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=cfg.OUTPUT_DIR)
        self.eval_period = eval_period
        self.epoch = epoch
        self.bestFitness = -1
        self.bestModel = None

        print(self.iterations_per_epoch)

    # log training metrics after each step (iteration)
    def after_step(self):
        storage = get_event_storage()
    
        # Log the metrics to wandb, using the current iteration as the step
            # log only after each epoch
        if storage.iter % int(iterations_per_epoch) == 0 and storage.iter != 0:
            # Initialize an empty dictionary to store the metrics
            metrics_dict = {}

            # Loop through the items in the latest metrics
            for key, value in storage.latest().items():
                metric_value = value[0]
                if "/" in key:
                    key = key.replace("/", "_")
                    
                # adjust the names for wandb visualization
                if "loss_cls" in key:
                    key = "cls_loss"
                if "loss_box_reg" in key:
                    key = "box_loss"
                if "fast_rcnn_fg_cls_accuracy" in key:
                    key = "fg_cls_accuracy"
                if "fast_rcnn_cls_accuracy" in key:
                    key = "cls_accuracy"
                if "fast_rcnn_false_negative" in key:
                    key = "false_negative"
                
                # train section in wandb
                key = "train/" + key
                    
                metrics_dict[key] = metric_value
                #print(str(key) + ": " + str(metric_value))

            self.epoch += 1
            print("-------------- Training metrics ---------------")
            print(metrics_dict)
            print("------------------------------------------------")

            print("Epoch: {}, number of iterations: {}".format(self.epoch, storage.iter))
            wandb.log(metrics_dict, step=self.epoch)


            # call the evaluation after each epoch
            self.evaluate_and_log(storage.iter)

    # evaluate the model after each step (iteration)
    def evaluate_and_log(self, iteration):
        data_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        evaluation_results = inference_on_dataset(self.trainer.model, data_loader, self.evaluator)
        
        # for wandb visualization
        val_metrics = {}
        # logging the evaluation metrics to wandb
        evaluation_results_items = evaluation_results.items()
        for key, value in evaluation_results_items: 

            # AP metrics (AP stands for mAP in coco evaluation)
            val_metrics["val/mAP50-95"] = value["AP"]
            val_metrics["val/mAP50"] = value["AP50"]
            val_metrics["val/mAP75"] = value["AP75"]
            val_metrics["val/mAPs"] = value["APs"]
            val_metrics["val/mAPm"] = value["APm"]
            val_metrics["val/mAPl"] = value["APl"]

            # AR metrics
            val_metrics["val/mAR1"] = value["AR1"]
            val_metrics["val/mAR10"] = value["AR10"]
            val_metrics["val/mAR100"] = value["AR100"]
            val_metrics["val/mARs"] = value["ARs"]
            val_metrics["val/mARm"] = value["ARm"]
            val_metrics["val/mARl"] = value["ARl"]

            # iterate through the class names to find class specific AP
            for class_name in class_names:
                val_metrics["val/AP-" + class_name] = value["AP-" + class_name]

            """
            val_metrics["AP-big truck"] = value["AP-big truck"]
            val_metrics["AP-bus-l-"] = value["AP-bus-l-"]
            val_metrics["AP-bus-s-"] = value["AP-bus-s-"]
            val_metrics["AP-car"]= value["AP-car"]
            val_metrics["AP-mid truck"] = value["AP-mid truck"]
            val_metrics["AP-small bus"] = value["AP-small bus"]
            val_metrics["AP-small truck"] = value["AP-small truck"]
            val_metrics["AP-truck-l-"] = value["AP-truck-l-"]
            val_metrics["AP-truck-m-"] = value["AP-truck-m-"]
            val_metrics["AP-truck-s-"] = value["AP-truck-s-"]
            val_metrics["AP-truck-xl-"] = value["AP-truck-xl-"]
            """

        """
            Calculate the fitness metric
        """
        # calculate the fitness value
        w = np.array([0.1, 0.9])  # weights for [mAP@0.5, mAP@0.5:0.95]
        values = np.array([val_metrics["val/mAP50"], val_metrics["val/mAP50-95"]])
        fitness = np.sum(values * w)

        # log to wandb
        val_metrics["val/val_fitness"] = fitness
        
        print("-------------- Validation metrics --------------")
        print(val_metrics)
        print("------------------------------------------------")
            
        
        # log to wandb
        wandb.log(val_metrics, step=self.epoch)
        """
        # log to wandb
        wandb.log({"val/AP": val_metrics["AP"],
                   "val/AP50": val_metrics["AP50"],
                   "val/AP75": val_metrics["AP75"],
                   "val/APs": val_metrics["APs"],
                   "val/APm": val_metrics["APm"],
                   "val/APl": val_metrics["APl"],
                   "val/AP-small truck": val_metrics["AP-small truck"],
                   "val/AP-mid truck": val_metrics["AP-mid truck"],
                   "val/AP-big truck": val_metrics["AP-big truck"],
                   "val/AP-truck-s-": val_metrics["AP-truck-s-"],
                   "val/AP-truck-m-": val_metrics["AP-truck-m-"],
                   "val/AP-truck-l-": val_metrics["AP-truck-l-"],
                   "val/AP-truck-xl-": val_metrics["AP-truck-xl-"],
                   "val/AP-car": val_metrics["AP-car"],
                   "val/AP-small bus": val_metrics["AP-small bus"],
                   "val/AP-bus-l-": val_metrics["AP-bus-l-"],
                   "val/AP-bus-s-": val_metrics["AP-bus-s-"],
                   "val/AR1": val_metrics["AR1"],
                   "val/AR10": val_metrics["AR10"],
                   "val/AR100": val_metrics["AR100"],
                   "val/ARs": val_metrics["ARs"]
                   }, step=self.epoch)
        """

        # save the model based on the current fitness
        if fitness > self.bestFitness:
            self.bestFitness = fitness
            self.bestModel = self.trainer.model

            # save the new best model locally using the DetectionCheckpointer (torch save)
            checkpointer = DetectionCheckpointer(self.trainer.model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.save("best_model")  

        # if the current epoch is the final epoch then save the best model
        if self.epoch == number_of_epochs:
            # find the overall best model which is saved in the output dir
            best_model_pth = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
            best_model_artifact = wandb.Artifact('best_model', type='model', description='The best FastRCNN model checkpoint')
            best_model_artifact.add_file(best_model_pth)

            wandb.log_artifact(best_model_artifact)



class TrainerWithWandb(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        # Define how often you want to perform evaluation (e.g., every 1000 iterations)
        eval_period = number_of_images // batch_size
        # Create an evaluator instance (optional if you want to use a specific evaluator)
        evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
        # Insert the WandbHook with the evaluator and eval period
        hooks.append(WandbHook(self.cfg, eval_period=eval_period, evaluator=evaluator))
        return hooks
    

cfg.SOLVER.IMS_PER_BATCH = batch_size  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = int(total_iterations)    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

"""
Define the image input size - 640 as in YOLOv8
"""
cfg.INPUT.MIN_SIZE_TRAIN = (640, )
cfg.INPUT.MAX_SIZE_TRAIN = 640
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 640

wandb.init(project="CUSTOM_DATASET", entity="krausm00", config=cfg)

# Training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = TrainerWithWandb(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# Inference + evaluation
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")


"""
# save the final model to wandb
best_model_artifact = wandb.Artifact('best_model', type='model', description='The best Fast R-CNN model checkpoint')
best_model_artifact.add_file(cfg.MODEL.WEIGHTS)
wandb.log_artifact(best_model_artifact)
"""


# Perform evaluation
evaluation_results = inference_on_dataset(predictor.model, val_loader, evaluator)


# Process the evaluation metrics for wandb
val_metrics = {}
# logging the evaluation metrics to wandb
evaluation_results_items = evaluation_results.items()
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
print("Logging the evaluation metrics to wandb.")
wandb.log(val_metrics, step=number_of_epochs)
