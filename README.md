
# 🗑️ Waste Classification using ResNet50 (PyTorch)

## 🎯 Project Objective

The goal of this project is to build an automated system capable of **classifying different types of waste** using computer vision.  
This supports:

- ♻️ Improving recycling efficiency  
- 🤖 Enabling automated waste-sorting machines  
- 🔍 Reducing sorting errors  
- 🚀 Supporting smart-city waste management  

The model predicts one of 6 classes:  
**cardboard, glass, metal, paper, plastic, trash**.

---

This project builds an image classification model to classify waste into 6 categories using a **pretrained ResNet50 model** fine-tuned in PyTorch.  
The goal is to help automate waste sorting using computer vision.

---

## 📁 Dataset

The dataset contains **6 waste categories**:

- 🟫 cardboard  
- 🟩 glass  
- 🟡 metal  
- 📄 paper  
- ♻️ plastic  
- 🗑️ trash  

Images were resized to **224×224** and loaded using `ImageFolder()`.

---

## 🧰 Used Technologies

| Technology | Usage |
|-----------|--------|
| **Python** | Main programming language |
| **PyTorch** | Model training, DataLoader, GPU usage |
| **Torchvision** | Pretrained ResNet50, transforms |
| **Matplotlib** | Visualizing samples & predictions |
| **CUDA (GPU)** | Accelerating training |
| **ImageFolder** | Dataset management |
| **Dropout Layers** | Reduce overfitting |
| **Checkpointing** | Save best model |
| **Early Stopping** | Stop training when val loss stops improving |

---

## 🧠 Model Architecture

We fine-tuned a pretrained **ResNet50**:

```python
self.network.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, 6)
)
