# Weed Detection in Rangelands using Transfer Learning (ResNet50)

## Project Summary
This project utilizes **Transfer Learning** to identify weed species in the **DeepWeeds** dataset. The dataset contains 17,509 images representing 8 different weed species native to Australia and one "negative" class.

This project demonstrates:

- Practical transfer learning for real-world computer vision
- Efficient dataset restructuring for PyTorch
- Partial fine-tuning of pretrained CNN architectures
- Multi-metric evaluation (Accuracy, Precision, Recall, F1-score)
- Confusion matrix analysis

---

## Dataset

- Dataset: **DeepWeeds** (downloaded via TensorFlow Datasets)
- Number of Classes: **9**
- Images reorganized into PyTorch `ImageFolder` format
- Train/Validation Split: **80% / 20%**

The dataset is automatically downloaded and converted into a directory structure compatible with PyTorch.

---

## Data Preprocessing

### Image Transformations

Training Transformations:
- Resize to **224 × 224**
- Random Horizontal Flip (Data Augmentation)
- Convert to Tensor
- Normalize using ImageNet mean and std:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

Validation Transformations:
- Resize to **224 × 224**
- Convert to Tensor
- ImageNet normalization

### DataLoader Configuration

- Batch Size: **32**
- Shuffle: Enabled for training
- Device: GPU (if available)

---
## Model Architecture
* **Base Model:** **ResNet50** (Pre-trained on ImageNet).
* **Transfer Learning Strategy:** **Partial Fine-Tuning**.
    * The convolutional "backbone" was frozen to preserve pre-learned ImageNet features (`requires_grad = False`).
    * The original 1000-class fully connected head was replaced with a new linear layer tailored for the 9 DeepWeeds classes.
* **Optimizer:** Adam with a learning rate of 0.001, focusing exclusively on the new classification head.
* **Loss Function:** CrossEntropyLoss
* **Learning Rate:** 0.001
* **Epochs:** 5
---
## Model Choice Justification
* **ResNet50:** The DeepWeeds dataset involves complex natural backgrounds (soil, grass, debris). ResNet50's residual connections allow for deeper feature extraction which is necessary to differentiate weeds from their surroundings.
* **Freezing Weights:** Given that the dataset is moderately sized (~17k images), freezing the backbone prevents overfitting and significantly reduces training time while leveraging a model already optimized for object recognition.

## Results & Key Findings

* **Test Accuracy**  : 0.7964
* **Precision** : 0.8043
* **Recall**    : 0.7964
* **F1-score**  : 0.7963
* **Key Observations:** The model reached over 74% accuracy in the very first epoch, demonstrating the efficiency of transfer learning. The confusion matrix indicates that the "Other" (negative) class is the primary source of misclassification due to its high visual variance.

---
## Technical Highlights

- Transfer Learning with ResNet50
- Proper ImageNet normalization
- Controlled fine-tuning (feature extractor mode)
- Multi-metric statistical evaluation
- Structured dataset pipeline
- GPU-aware training loop

---

## Future Improvements

- Unfreeze deeper ResNet blocks for full fine-tuning
- Add learning rate scheduling
- Perform hyperparameter tuning
- Introduce stronger augmentation
- Perform k-fold cross-validation

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- TensorFlow Datasets
- Scikit-learn
- NumPy
- Matplotlib
