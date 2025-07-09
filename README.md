# Road Extraction from Satellite Imagery using U-Net

This project focuses on the task of semantic segmentation to extract roads from satellite images using a **U-Net** architecture. The model is built with PyTorch and trained on the DeepGlobe Road Extraction Dataset. 🛰️🛣️

---

## Overview

The goal of this project is to accurately identify and segment road networks from high-resolution satellite imagery. This task has various applications in urban planning, traffic management, and autonomous navigation. We employ a U-Net, a convolutional neural network architecture that is particularly well-suited for image segmentation tasks due to its ability to capture both context and precise location details.

---

## Dataset

This project utilizes the **DeepGlobe Road Extraction Dataset**, a large-scale dataset designed for road extraction challenges.

-   **Data Source:** The dataset is available on Kaggle.
-   **Content:** It contains a collection of satellite images and their corresponding binary masks where roads are labeled.
-   **Data Split:** For this project, the data was split into training and validation sets with an 80:20 ratio, resulting in **4,980 training images** and **1,246 validation images**.

---

## Methodology

The project is implemented in a Jupyter Notebook (`UAS.ipynb`) and follows these key steps:

### 1. Environment Setup

The environment requires Python and several key libraries, including:
-   `torch` & `torchvision`
-   `segmentation-models-pytorch`
-   `albumentations` for data augmentation
-   `opencv-python`
-   `numpy`
-   `matplotlib`

### 2. Data Preprocessing & Augmentation

To improve the model's robustness and ability to generalize, various data augmentation techniques were applied to the training images and masks using the `albumentations` library. The transformations included:
-   **Resizing:** Images and masks were resized to 512x512 pixels.
-   **Geometric Transformations:** Random rotation, horizontal flipping, and vertical flipping.
-   **Normalization:** Pixel values were normalized.
-   Finally, the images were converted to PyTorch tensors.

### 3. Model Architecture

A **U-Net** architecture was implemented from scratch using PyTorch. The U-Net consists of:
-   An **encoder** (down-sampling path) to capture contextual information using a series of convolutional and max-pooling layers.
-   A **decoder** (up-sampling path) to enable precise localization and reconstruct the segmentation map using up-convolutional layers.
-   **Skip connections** that concatenate feature maps from the encoder path to the corresponding layers in the decoder path, helping the network recover fine-grained details lost during down-sampling.

### 4. Training

The model was trained on a GPU using the following configuration:
-   **Optimizer:** Adam optimizer with a learning rate of `1e-4`.
-   **Loss Function:** A custom `DiceBCELoss`, which is a combination of Dice Loss and Binary Cross-Entropy Loss, suitable for imbalanced segmentation tasks.
-   **Epochs:** The model was trained for 10 epochs.
-   **Batch Size:** A batch size of 4 was used.
-   **Checkpointing:** The model with the best Dice Score on the validation set was saved during training.

---

## Results

After training, the model's performance was evaluated on the validation set, achieving strong results:

-   **Pixel Accuracy:** **97.55%**
-   **Dice Score:** **0.6925**

---

## How to Run

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision opencv-python numpy matplotlib segmentation-models-pytorch albumentations
    ```
3.  **Set up the dataset:** Download the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/deepglobe/deepglobe-road-extraction-dataset) and place it in the appropriate directory as specified in the notebook.
4.  **Run the Jupyter Notebook:** Open and run the cells in `UAS.ipynb`. Ensure you have a capable GPU for faster training.

---

## Conclusion

This project successfully demonstrates the application of a U-Net for road extraction from satellite imagery. The model achieves a high pixel accuracy and a solid Dice score, confirming its effectiveness for this semantic segmentation task. Future work could involve training for more epochs, hyperparameter tuning, or utilizing more advanced pre-trained encoders (like ResNet) within the U-Net architecture to potentially boost performance further.
