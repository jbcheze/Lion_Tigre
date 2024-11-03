import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = r"C:/M1 MASERTI/GPIA/IA/environnement_virtuel/Mask_RCNN-master"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = r"C:/M1 MASERTI/GPIA/IA/environnement_virtuel/Mask_RCNN-master/logs"

class CustomConfig(Config):
    """Configuration for training on the custom dataset."""
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + tiger and lion
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        # Ajouter les classes
        self.add_class("object", 1, "lion")
        self.add_class("object", 2, "tigre")

        # Dossier contenant les fichiers JSON individuels (train/json ou val/json)
        json_dir = os.path.join(dataset_dir, subset, "json")

        # Parcourt chaque fichier JSON dans le dossier json
        for filename in os.listdir(json_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(json_dir, filename)
                with open(file_path) as f:
                    data = json.load(f)

                    # Extrait uniquement le nom du fichier image, en ignorant les "..\"
                    image_filename = os.path.basename(data['imagePath'])
                    image_path = os.path.normpath(
                        os.path.join(dataset_dir, subset, image_filename)
                    )
                    
                    height = data['imageHeight']
                    width = data['imageWidth']

                    # Obtenez les polygones et les labels des objets
                    polygons = [shape['points'] for shape in data['shapes']]
                    objects = [shape['label'] for shape in data['shapes']]
                    
                    # Convertir les labels en numéros de classe
                    name_dict = {"lion": 1, "tigre": 2}
                    num_ids = [name_dict[obj] for obj in objects]

                    # Ajouter l'image avec ses annotations
                    self.add_image(
                        "object",
                        image_id=image_filename,  # utilise uniquement le nom du fichier
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        num_ids=num_ids
                    )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        # Créer le masque pour chaque polygone
        for i, p in enumerate(info["polygons"]):
            all_points_y = [point[1] for point in p]
            all_points_x = [point[0] for point in p]
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(r"C:/M1 MASERTI/GPIA/IA/environnement_virtuel/Mask_RCNN-master/dataset", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(r"C:/M1 MASERTI/GPIA/IA/environnement_virtuel/Mask_RCNN-master/dataset", "val")
    dataset_val.prepare()

    # Training network heads
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
# Download weights file if not available
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

train(model)
