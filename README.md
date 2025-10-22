
# AI-Generated Image Detection: Fine-Tuning EfficientNet-B0 with Grad-CAM and LIME

## Project Overview

This project is focused on detecting AI-generated images using a fine-tuned **EfficientNet-B0** model. The model is trained to classify images as either **real** or **fake**, and incorporates explainability methods such as **Grad-CAM**, to interpret and visualize the model’s decision-making process.

We use a combination of a pre-trained model fine-tuned on a dataset of real and fake faces, along with advanced techniques to compare how humans and machines interpret AI-generated images.

### Datasets Used
1. **Real and Fake Faces Dataset**:  
   [Kaggle - 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
2. **Fine-Tuning Dataset**:  
   [Generated Pairs Full Archive](https://northeastern-my.sharepoint.com/:u:/r/personal/mahadevarao_s_northeastern_edu/Documents/generated_pairs_full_archive.zip?csf=1&web=1&e=5frkzF)
3. **Ground Truth Data**:  
   [Google Drive - Ground Truth Folder](https://drive.google.com/drive/folders/1Dlh392g0tmBnJ64JEHcLC_XJNSMSdMxP?usp=drive_link)

## Features

- **EfficientNet-B0 Architecture**: A lightweight convolutional neural network (CNN) model used for both base and fine-tuning.
- **Grad-CAM**: A technique for visualizing which parts of the image influence the model’s predictions.
- **Transfer Learning**: Fine-tuning a pre-trained model to adapt it to new, task-specific data.

## Code Overview

### `code.ipynb` - Model Training

This notebook demonstrates how to train the EfficientNet-B0 model on the **Real vs Fake Faces dataset**. The key steps are:

1. **Data Preprocessing**: 
   - Image transformations such as resizing, normalization, and augmentation for the training dataset.
   - The validation and test datasets only apply resizing and normalization.

2. **Model Training**: 
   - We load the pre-trained **EfficientNet-B0** and modify the final layer for binary classification (real/fake).
   - The training loop includes the loss function, optimizer, and accuracy calculation.
   - The model is saved after each epoch, and the best-performing model is saved based on validation accuracy.

### `code_finetune.ipynb` - Fine-Tuning the Model

This notebook extends the training process by fine-tuning the pre-trained EfficientNet-B0 model on a new dataset.

1. **Dataset Splitting**: 
   - The dataset is split into training, validation, and test subsets.

2. **Learning Rate Scheduler and Early Stopping**:
   - A learning rate scheduler is used to reduce the learning rate when the validation loss plateaus.
   - Early stopping is implemented to halt training if validation loss does not improve for a specified number of epochs.

3. **Model Saving**:
   - The best-performing model is saved during training to prevent overfitting and to ensure that the model with the best generalization capability is used.

### Visualization with Grad-CAM and LIME

The project also includes **Grad-CAM** and **LIME** to visualize and interpret the model’s predictions:

- **Grad-CAM** is used to highlight the regions of the image that were most influential in the model's decision.
- **LIME** provides a local interpretation by perturbing the input image and explaining the model's decision for specific regions.

## Requirements

To run the notebooks, you will need the following libraries:

- **PyTorch**
- **Torchvision**
- **timm** (for EfficientNet-B0 model)
- **SHAP** and **LIME** (for interpretability)
- **OpenCV** (for visualization)
- **Matplotlib** (for plotting)

You can install the required dependencies by running:

```bash
pip install torch torchvision timm shap lime opencv-python matplotlib
```

## How to Use

1. **Download the datasets**:
   - Follow the links above to download the **Real and Fake Faces Dataset**, **Fine-Tuning Dataset**, and **Ground Truth Data**.
   - Unzip and place the datasets in the appropriate directories.

2. **Train the Model**:
   - Run `code.ipynb` to start training the model on the real vs fake images.

3. **Fine-Tune the Model**:
   - After training, fine-tune the model by running `code_finetune.ipynb`. This will apply additional epochs, early stopping, and learning rate scheduling.

4. **Visualize Predictions**:
   - Use **Grad-CAM** and **LIME** to visualize the model’s predictions and gain insights into how the model interprets the images.

## Conclusion

This project bridges deep learning with human interpretability by comparing how humans and models detect fake images. By using state-of-the-art techniques like Grad-CAM and LIME, we ensure transparency in AI decisions, which is crucial for building trust in automated systems.
