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
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping

import torch
torch.cuda.empty_cache()
# device = torch.device('cpu')

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


cfg.DATALOADER.NUM_WORKERS = 1 # =0 for cpu || =1 or more for gpu
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
    # if id == 0:
    #   continue
    name = class_name['name']
    class_names.append(name)
    

number_of_images = len(coco_annotations['images'])
number_of_epochs = 40
batch_size = 16
iterations_per_epoch = number_of_images / batch_size
total_iterations = int(iterations_per_epoch) * number_of_epochs
# for custom datasets: total_iterations + 1
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
            try:
                val_metrics["val/mAR1"] = value["AR1"]
            except:
                print("AR1 metric not found.")
            try:
                val_metrics["val/mAR10"] = value["AR10"]
            except:
                print("AR10 metric not found.")
            try:
                val_metrics["val/mAR100"] = value["AR100"]
            except:
                print("AR100 metric not found.")
            try:
                val_metrics["val/mARs"] = value["ARs"]
            except:
                print("ARs metric not found.")
            try:
                val_metrics["val/mARm"] = value["ARm"]
            except:
                print("ARm metric not found.")
            try:
                val_metrics["val/mARl"] = value["ARl"]
            except:
                print("ARl metric not found.")
            

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
        # How often to perform the evaluation
        eval_period = number_of_images // batch_size
        # Create an evaluator instance 
        evaluator = COCOEvaluator(self.cfg.DATASETS.TEST[0], self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
        # Insert the WandbHook with the evaluator and eval period
        hooks.append(WandbHook(self.cfg, eval_period=eval_period, evaluator=evaluator))
        return hooks
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
            This method creates an optimizer based on the passed
            parameters. 
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

        optimizer_name = cfg.SOLVER.OPTIMIZER_NAME
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=0.9)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)
        elif optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR)
        elif optimizer_name == "NAdam":
            optimizer = torch.optim.NAdam(params, lr=cfg.SOLVER.BASE_LR)
        elif optimizer_name == "RAdam":
            optimizer = torch.optim.RAdam(params, lr=cfg.SOLVER.BASE_LR)
        elif optimizer_name == "RMSProp":
            optimizer = torch.optim.RMSprop(params, lr=cfg.SOLVER.BASE_LR)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer_name}")

        return maybe_add_gradient_clipping(cfg, optimizer)
    

# Default parameters for training
cfg.SOLVER.IMS_PER_BATCH = batch_size  
cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.MAX_ITER = int(total_iterations)    
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  

"""
    TRAINING

"""
def train():
    with wandb.init() as run:
        cfg.SOLVER.BASE_LR = wandb.config["learning_rate"]
        cfg.SOLVER.MOMENTUM = wandb.config["momentum"]
        cfg.SOLVER.WEIGHT_DECAY = wandb.config["weight_decay"]
        cfg.SOLVER.LR_SCHEDULER_NAME = wandb.config["lr_scheduler"]
        cfg.SOLVER.OPTIMIZER_NAME = wandb.config["optimizer"]
        # cfg.MODEL.DEVICE = 'cpu'

        # call the trainer
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = TrainerWithWandb(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

# Sweep configuration
sweep_config = {
    'method': 'random',  # or 'grid', 'bayesian'
    'metric': {
        'name': 'val/mAP50-95',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.01
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 0.99
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 0.00001,
            'max': 0.01
        },
        'lr_scheduler': {
            'values': ['WarmupMultiStepLR', 'WarmupCosineLR']
        },
        'optimizer': {
            'values': ['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="CD_Det2_sweep")
# Run the agent
wandb.agent(sweep_id, train, count=31)





