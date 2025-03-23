from ultralytics import YOLO
import torch
import wandb
torch.cuda.empty_cache()


# Define the function that initializes training
def train():
    with wandb.init() as run:
        # Create the model inside the function to ensure it's fresh for each sweep iteration
        model = YOLO('yolov8m.pt')
        # Using wandb.config to get the current configuration set by the sweep
        model.train(
            data = "/storage/plzen1/home/krausm00/DP/configs/highway_vehicles.yaml",
            # "/Users/martinkraus/Downloads/highway_vehicles.yaml"cd
            epochs = 40,
            workers=1,
            optimizer = wandb.config['optimizer'],
            device = 0, # gpu (=0) vs mac M1 (='mps')
            lr0 = wandb.config['lr0'],
            warmup_bias_lr = wandb.config['warmup_bias_lr'],
            cos_lr = wandb.config['cos_lr'],
            momentum = wandb.config['momentum'],
            dropout = wandb.config['dropout'],
            weight_decay = wandb.config['weight_decay']
        )

# Sweep configuration
sweep_config = {
    'method': 'random',  # or 'grid', 'bayesian'
    'metric': {
        'name': 'val/mAP50-95',
        'goal': 'maximize'
    },
    'parameters': {
        'optimizer': {
            'values': ['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']
        },
        'lr0': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.01
        },
        'warmup_bias_lr': {
            'min': 0.001,
            'max': 0.01
        },
        'cos_lr': {
            'values': [True, False]
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 0.99
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 0.00001,
            'max': 0.01
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="HV_YOLO_sweep")

# Run the agent
wandb.agent(sweep_id, train, count=31)
