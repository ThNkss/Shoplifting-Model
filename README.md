# 🎥 Video Anomaly Detection with EfficientNet + LSTM

This project is a deep learning framework for video-based anomaly detection using a combination of CNN and RNN architectures. Specifically, we use [EfficientNet](https://arxiv.org/abs/1905.11946) as a feature extractor for each frame, followed by an LSTM to model the temporal relationships between frames. This architecture is designed to detect anomalies such as shoplifting, violence, or other unusual events in surveillance video data.

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Customization](#customization)
- [Directory Structure](#directory-structure)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## 📖 About the Project

This project addresses the task of **anomaly detection in videos** using deep learning. Anomalies include activities that deviate from normal behavior, such as violence, theft, or other suspicious actions in surveillance footage.

We use:
- **EfficientNet-B0** to extract visual features from individual frames.
- **LSTM** to model temporal dependencies across sequences of frames.
- A **Binary Classifier** head for final prediction.

---

## 🎬 Dataset

We use the **[UCF-Crime dataset](http://crcv.ucf.edu/projects/real-world/)**, which consists of long surveillance videos (normal and anomalous). You can preprocess the videos into frame sequences and create labels based on video category.

- Classes: Abnormal / Normal
- Data Format: Raw video files (MP4)
- Labels: Stored in a JSON file with video names and binary labels (0 or 1)

**Note:** The dataset is not included due to licensing. Please download it manually.

---

## 🧠 Model Architecture

### ➤ EfficientNet-B0 (CNN)
- Used as a **frame-level feature extractor**
- Pretrained on ImageNet
- The final FC layer is removed (`nn.Identity()`)

### ➤ LSTM
- Input size: 1280 (EfficientNet output)
- Hidden size: configurable (e.g. 1024)
- Layers: configurable (e.g. 1)
- Outputs temporal context over sequences

### ➤ Classifier Head
- Takes the last hidden state from the LSTM
- Outputs a binary prediction (anomaly / normal)
