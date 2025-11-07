# 🖐️ Real-Time Hand Sign Recognition using ANN

This project implements a **real-time hand sign recognition system** using an **Artificial Neural Network (ANN)** trained on hand landmark data captured via a webcam.
It leverages **MediaPipe Hands** for feature extraction and **scikit-learn’s MLPClassifier** for training and classification.

---

## 📘 Overview

The project captures live hand gestures (ASL-inspired letters and phrases), extracts 3D landmark coordinates using **MediaPipe**, trains a neural network (ANN) using these features, and then performs **real-time sign recognition** from webcam input.

---

## ⚙️ Features

* Real-time hand tracking using **MediaPipe Hands**
* 63-dimensional feature extraction (x, y, z for 21 landmarks)
* Live data collection for custom training
* Artificial Neural Network training via **scikit-learn**
* Real-time sign prediction from webcam
* Adjustable dataset size and model parameters

---

## 🧠 Classes Used

### Letters:

```
A, B, C, D, E
```

### Phrases:

```
THANK_YOU, PLEASE, YES, NO, STOP
```

> You can modify these lists in the code (`LETTER_CLASSES` and `PHRASE_CLASSES`) to train additional signs.

---

## 🧩 Requirements

Install dependencies with:

```bash
pip install opencv-python mediapipe scikit-learn numpy
```

---

## 🖥️ How It Works

1. **Data Collection Phase**

   * The script captures hand pose data for each sign using your webcam.
   * Each frame is processed to extract 63 feature points (x, y, z).
   * You’ll be prompted to show each sign sequentially.

2. **Model Training Phase**

   * The captured data is split into training (90%) and testing (10%) sets.
   * An **MLPClassifier** (multi-layer perceptron) is trained to classify the signs.

3. **Live Prediction Phase**

   * The trained model predicts hand signs in real time from webcam input.
   * Predictions are displayed directly on the video feed.

---

## ▶️ Usage

Run the project directly:

```bash
python ann\ project.py
```

### Controls:

* **Show each sign** when prompted during data collection.
* **Press ‘q’** to quit at any time.

---

## 📈 Model Configuration

* Hidden Layers: `(100, 50)`
* Activation: `ReLU`
* Solver: `Adam`
* Max Iterations: `600`
* Train/Test Split: `90% / 10%`

---

## 🧰 File Structure

```
ann project.py         # Main project file
```

You can extend this to save trained models (e.g., `.pkl`), store data, or add more signs.

---


