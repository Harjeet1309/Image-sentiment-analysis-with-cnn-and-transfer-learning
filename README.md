# 📷 Enhancing Image Sentiment Analysis with CNN and Transfer Learning

This project focuses on facial emotion recognition using deep learning techniques applied to the **Extended Cohn-Kanade (CK+) dataset**. It implements and compares two approaches:

- A **Custom Convolutional Neural Network (CNN)** trained from scratch.
- A **Transfer Learning** model using **ResNet50**, a pre-trained model on ImageNet.

The system classifies images into one of **seven emotional categories**: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## 📌 Table of Contents

- [Project Overview](#-enhancing-image-sentiment-analysis-with-cnn-and-transfer-learning)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Directory Structure](#-directory-structure)
- [Model Architectures](#-model-architectures)
- [Training Process](#-training-process)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Future Work](#-future-work)
- [Authors](#-authors)
- [References](#-references)

---

## 🎯 Project Overview

In today's visual world, understanding emotions from images is critical for applications like:

- Social media sentiment analysis
- Emotion-aware virtual assistants
- Digital marketing and advertising
- Mental health monitoring

Unlike textual sentiment analysis, **image sentiment analysis** captures subtle visual features like facial expressions, color tones, and spatial cues. This project enhances the emotion recognition process by combining:

- A **Custom CNN** for feature learning
- A **ResNet50-based model** leveraging transfer learning

---

## 📁 Dataset

### 🔗 CK+ (Extended Cohn-Kanade) Dataset

- **Description**: A benchmark dataset for facial expression analysis.
- **Official Website**: [http://www.jeffcohn.net/Resources/](http://www.jeffcohn.net/Resources/)
- **Image Format**: Grayscale or RGB (depending on preprocessing)
- **Classes**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

### 📦 Preprocessed Dataset Directory Structure

```
CK+/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── val/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

All images must be:
- Resized to **224x224**
- Normalized to pixel values in **[0, 1]**

---

## ⚙️ Installation

### 📌 Requirements

Install required packages via pip:

```bash
pip install tensorflow matplotlib scikit-learn seaborn
```

Make sure you are using Python 3.8+ and TensorFlow 2.x.

---

## 📂 Directory Structure

```
project-root/
│
├── Image sentiment analysis with cnn and transfer learning.ipynb
├── README.md
├── models/
│   ├── custom_cnn.h5
│   └── resnet50_model.h5
└── CK+/          # Dataset directory (as described above)
```

---

## 🧠 Model Architectures

### ✅ Custom CNN (Trained from Scratch)

- 3 Convolutional layers with ReLU
- MaxPooling layers
- Dense layer with 128 units
- Dropout (0.3)
- Softmax output layer (7 classes)

### 🔁 ResNet50 Transfer Learning

- Pretrained ResNet50 (ImageNet weights)
- Frozen convolutional base
- GlobalAveragePooling2D
- Dense layer with ReLU
- Softmax output layer (7 classes)

---

## 🏋️‍♀️ Training Process

- Data augmentation using `ImageDataGenerator`
  - Rotation, shift, zoom, flip
- Split: 80% training, 20% validation
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 10+ (can be increased)
- Metrics: Accuracy, Precision, Recall, F1-Score

---

## 📈 Evaluation

- **Confusion Matrix**: Visualize classification performance
- **Classification Report**: View precision, recall, F1 for each emotion
- **Per-image Prediction**: Display label and confidence on test images

---

## 📊 Results

| Model         | Accuracy | Macro F1-Score | Best Performance |
|---------------|----------|----------------|------------------|
| Custom CNN    | ~55%     | Varies         | Happy, Surprise  |
| ResNet50      | ~84%     | 0.66           | Anger, Sad, Surprise |

### 🔍 Observations

- Custom CNN is lightweight but underperforms due to limited data and depth.
- ResNet50 generalizes better thanks to pretrained weights.
- Misclassifications mainly occur on subtle emotions like *fear* and *contempt*.

---

## 🚀 Future Work

- Extend to color and high-res images
- Integrate attention mechanisms for feature importance
- Use temporal data (videos) for dynamic emotion tracking
- Add edge deployment options (e.g., TensorFlow Lite)
- Combine audio + visual emotion cues (multimodal sentiment analysis)

---

## 👨‍💻 Authors

- **Harjeet Gudde** — `guddeharjeet@gmail.com`
- **Sindhuja Jamboth** — `jambothsindhuja@gmail.com`
Department of Computer Science and Engineering (AI & ML),  
Sphoorthy Engineering College

---

## 📚 References

1. I. S and F. M. H. Fernandez, "Multi-Scale CNN Architectures for Enhanced Feature Extraction in Image Sentiment Analysis"
2. S. Jindal and S. Singh, "Image sentiment analysis using deep convolutional neural networks with domain specific fine tuning"
3. M. Katsurai and S. Satoh, "Latent correlations among visual, textual, and sentiment views"
4. H. L. Sharma and M. Sharma, "Mood Identification with Face Feature Learning using CNN and OpenCV"
5. I. Jaiswal et al., "Facial Emotion Recognition from Video using Transfer Learning"

---

> This project bridges the gap between artificial intelligence and emotional intelligence. By mimicking human emotional perception, we enable smarter, more empathetic digital systems.
