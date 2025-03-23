"""
    This script runs the YOLOv8 trainig and evaluation while specifying the individual
    parameters obtained by Wandb sweep.

"""
from ultralytics import YOLO
import torch
torch.cuda.empty_cache()
model = YOLO('yolov8m')

# train the model
model.train(
    data = "/storage/plzen1/home/krausm00/DP/configs/custom_dataset.yaml",
    # "/Users/martinkraus/Downloads/highway_vehicles.yaml"
    epochs = 100,
    workers=1,
    optimizer = 'SGD',
    device = 0, # gpu (=0) vs mac M1 (='mps')
    lr0 = 0.0016005571253559705,
    warmup_bias_lr = 0.005265379641707624,
    cos_lr = False,
    momentum = 0.6243754491708029,
    dropout = 0.4075112482615285,
    weight_decay = 0.0014969239725699678
)

# validate the model
metrics = model.val()