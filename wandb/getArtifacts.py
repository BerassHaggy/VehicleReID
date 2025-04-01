import os
import wandb

# Authenticate if needed
wandb.login()
wandb.init()
# Artifact reference
artifact = wandb.use_artifact("krausm00/HIGHWAY_VEHICLES/best_model:v5", type="model")

# Download to current directory
artifact_dir = artifact.download(root=".")

# Locate the .pth file inside the downloaded artifact directory
model_path = None
for root, dirs, files in os.walk(artifact_dir):
    for file in files:
        if file.endswith(".pth"):
            model_path = os.path.join(root, file)
            break

print("Model saved at:", model_path)
