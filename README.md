# Lion vs Tiger Object Detection with Mask R-CNN

This project trains a Mask R-CNN model to distinguish between lions and tigers using a custom dataset. The following steps outline the dataset preparation, model configuration, training, and evaluation process.

## Project Overview

### Objective
- **Dataset**: Build a dataset of at least 150 images, labeled for lions and tigers. The images are sourced from [University of Oxford's Visual Geometry Group](https://www.robots.ox.ac.uk/).
- **Model**: Train a Mask R-CNN model, retraining the head layers using the labeled dataset.
- **Output**: The project aims to evaluate the model using key metrics and visualizations, including training and validation loss, a confusion matrix, and hyperparameter optimization.

## Project Structure

- `dataset/`: Contains training and validation images in separate folders, each with associated JSON annotations (`train/json` and `val/json`).
- `logs/`: Stores training logs and model checkpoints.
- `mask_rcnn_coco.h5`: Pre-trained COCO weights used as the starting point for transfer learning.
- `custom.py`: Code to load the dataset, define the model, and execute the training and testing.

## Steps

1. **Dataset Preparation**
   - Gathered a total of 150+ labeled images, with annotations for lions and tigers.
   - Each image is labeled with polygons marking instances of lions and tigers, using JSON format for compatibility with Mask R-CNN.
   - Data is split into training (`train/`) and validation (`val/`) sets for model evaluation.

2. **Setting Up the Environment**
   - An environment for Mask R-CNN was prepared, installing necessary dependencies (`numpy`, `scipy`, `Pillow`, `matplotlib`, `scikit-image`, `tensorflow`, `keras`, etc.).
   - Downloaded the COCO pre-trained weights (`mask_rcnn_coco.h5`).

3. **Training the Model**
   - Retrained the head layers of Mask R-CNN on the custom dataset, following transfer learning principles.
   - The training process was set up to output logs recording losses for each epoch.
   - Used various hyperparameters to optimize the model (see Hyperparameters section below).

4. **Evaluation and Results**
   - **Training and Validation Losses**: Plotted over epochs to analyze convergence.
   - **Confusion Matrix**: Created a confusion matrix to evaluate the model's accuracy on a separate test set, distinguishing between lions and tigers.
   - **Sample Predictions**: Displayed sample predictions from the model on test images.

## Results

### Training and Validation Losses

A graph visualizing the model's training and validation losses over each epoch is included to showcase convergence.

### Confusion Matrix

The confusion matrix below represents the model's accuracy on the test set, showing its ability to distinguish between lions and tigers.

### Hyperparameters

The following hyperparameters were used during training:

- **Learning Rate (`lr`)**: `0.001`
- **Optimizer**: `SGD`
- **Activation Function**: `ReLU`
- **Epochs**: `50`
- **Batch Size**: `1`

Several configurations of these hyperparameters were tested to improve the model’s accuracy, with the best results obtained by tuning the learning rate and number of epochs.

## Smart Human-Generated Insights

From the confusion matrix, we observe that the model accurately distinguishes between lions and tigers, but certain lighting and background conditions contribute to some misclassifications. The training and validation loss graphs indicate that the model converged well, suggesting that the chosen hyperparameters (notably the learning rate and epochs) were effective in achieving this performance.
