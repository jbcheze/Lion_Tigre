import os
import json
import numpy as np
import cv2
from skimage.draw import polygon
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Root directory of the project
ROOT_DIR = r"C:/M1 MASERTI/GPIA/IA/environnement_virtuel/Mask_RCNN-master"

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Configuration
class CustomConfig(Config):
    """Configuration for training on the custom dataset."""
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 2  # Background + lion and tiger
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    VALIDATION_STEPS = 5

# Dataset
class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        self.add_class("object", 1, "lion")
        self.add_class("object", 2, "tigre")

        json_dir = os.path.join(dataset_dir, subset, "json")
        for filename in os.listdir(json_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(json_dir, filename)
                with open(file_path) as f:
                    data = json.load(f)

                    image_filename = os.path.basename(data['imagePath'])
                    image_path = os.path.join(dataset_dir, subset, image_filename)

                    if not os.path.exists(image_path):
                        print(f"[WARNING] Image not found: {image_path}")
                        continue

                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"[ERROR] Unable to read image: {image_path}")
                        continue
                    height, width, _ = image.shape

                    polygons = [shape['points'] for shape in data['shapes']]
                    objects = [shape['label'] for shape in data['shapes']]

                    name_dict = {"lion": 1, "tigre": 2}
                    num_ids = [name_dict[obj] for obj in objects]

                    self.add_image(
                        "object",
                        image_id=image_filename,
                        path=image_path,
                        width=width,
                        height=height,
                        polygons=polygons,
                        num_ids=num_ids
                    )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            all_points_y = [point[1] for point in p]
            all_points_x = [point[0] for point in p]
            rr, cc = polygon(all_points_y, all_points_x)
            mask[rr, cc, i] = 1

        return mask, np.array(num_ids, dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

# Entraînement avec enregistrement des logs lisibles
def train_and_save_logs(model):
    """Train the model and save readable logs."""
    dataset_train = CustomDataset()
    dataset_train.load_custom(os.path.join(ROOT_DIR, "dataset"), "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom(os.path.join(ROOT_DIR, "dataset"), "val")
    dataset_val.prepare()

    print("Entraînement des couches heads...")
    log_file = os.path.join(DEFAULT_LOGS_DIR, "training_readable_logs.txt")

    # Garder le fichier log ouvert pendant l'entraînement
    with open(log_file, "w") as f:
        f.write("Training Log - Readable Format\n")
        f.write("=" * 50 + "\n")

        for epoch in range(2):  # Changez 10 par le nombre d'époques souhaité
            print(f"Epoch {epoch + 1}/10")

            model.train(
                dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epoch + 1,  # Jusqu'à l'époque actuelle
                layers='heads'
            )

            f.write(f"Epoch {epoch + 1} - Logs written automatically.\n")
            f.write("-" * 50 + "\n")
            f.flush()  # Assurez-vous que les données sont écrites immédiatement

    print(f"Logs enregistrés dans : {log_file}")

# Configuration et initialisation du modèle
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

if not os.path.exists(COCO_WEIGHTS_PATH):
    utils.download_trained_weights(COCO_WEIGHTS_PATH)

model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"
])

train_and_save_logs(model)
