# PRODIGY_ML_03
# 🐾 Cat vs Dog Image Classification using SVM

This project implements a **Support Vector Machine (SVM)** to classify images of **cats and dogs** using the popular [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

---

## 📂 Dataset

- Download from Kaggle: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- Place all images from the `train/` folder into the same directory as the script.

---

## 💡 Features

- Uses grayscale + resized images (64x64)
- Extracts HOG (Histogram of Oriented Gradients) features
- Trains a linear SVM classifier
- Evaluates using accuracy and classification report

---

## 🚀 Run

```bash
python cats_and_dogs_svm.py
