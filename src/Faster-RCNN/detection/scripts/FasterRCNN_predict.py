from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import cv2

"""
This script represents a Faster-RCNN prediction on a provided image.
"""

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "best_model.pth"
datasetType = "CD"  # CD or HV
if datasetType.startswith("CD"):
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    image_path = "CD_val_img.png"
    image = cv2.imread(image_path)
    label_map = {
        0: 'Car',
        1: 'Motorcycle',
        2: 'Truck',
        3: 'Bus',
        4: 'Bicycle'
    }
elif datasetType.startswith("HV"):
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12
    image_path = "HV_val_img.jpg"
    image = cv2.imread(image_path)
    label_map = {
        0: 'big bus',
        1: 'big truck',
        2: 'bus-l-',
        3: 'bus-s-',
        4: 'car',
        5: 'mid truck',
        6: 'small bus',
        7: 'small truck',
        8: 'truck-l-',
        9: 'truck-m-',
        10: 'truck-s-',
        11: 'truck-xl-'
    }
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25  # YOLO uses 0.25 based on the documentation: https://docs.ultralytics.com/modes/predict/#inference-arguments
classes = list(label_map.keys())

"""
    Define the folder with annotations.
"""
annotations_path = "/storage/plzen1/home/krausm00/MPV/dataFasterRCNN"
images_path = "/storage/plzen1/home/krausm00/MPV/dataYOLO"

train_path_images = images_path + "/train/images"
train_path_annotations = annotations_path + "/train/annotations.json"

test_path_images = images_path + "/test/images"
test_path_annotations = annotations_path + "/test/annotations.json"

val_path_images = images_path + "/val/images"
val_path_annotations = annotations_path + "/val/annotations.json"


register_coco_instances('my_dataset_train', {}, train_path_annotations, train_path_images)
register_coco_instances('my_dataset_val', {}, val_path_annotations, val_path_images)
register_coco_instances('my_dataset_test', {}, test_path_annotations, test_path_images)

metadata = MetadataCatalog.get('my_dataset_train')
metadata.set(thing_classes=classes)

predictor = DefaultPredictor(cfg)
outputs = predictor(image)

v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("FRCNN_prediction.jpg", out.get_image()[:, :, ::-1])
