# 🧠 Brain Tumor Classification using CNN (TensorFlow & Keras)

This project implements a *Convolutional Neural Network (CNN)* for classifying *four types of brain MRI scans*:

* *Glioma*
* *Meningioma*
* *Pituitary*
* *No Tumor*

The model achieves *98% accuracy* on the test dataset and includes visualization tools, model evaluation metrics, confusion matrix, and prediction functions for single-image diagnosis.

---

## 📌 Features

### ✔️ Data Handling

* Automatically loads dataset from directory structure
* Balanced dataset visualization
* Data augmentation using ImageDataGenerator
* Training & testing split handling

### ✔️ Model Architecture

A custom CNN with:

* 4 Convolutional layers
* MaxPooling
* Dense + Dropout layers
* Softmax output for multi-class classification

### ✔️ Training Enhancements

* Early Stopping
* ReduceLROnPlateau
* Batch training with augmentation

### ✔️ Evaluation Tools

* Accuracy and Loss curves
* Confusion Matrix
* Precision, Recall & F1-score per class
* Sample Prediction Visualization
* Single-image tumor prediction function

### ✔️ Final Results

* *Test Accuracy:* ~*98.54%*
* *Loss:* 0.0443
* Strong performance across all four tumor types

---

## 📂 Dataset Structure


/Dataset
    /Training
        /glioma
        /meningioma
        /notumor
        /pituitary
    /Testing
        /glioma
        /meningioma
        /notumor
        /pituitary


The model automatically scans and labels files based on folder names.

---

## 🚀 Technologies Used

| Component           | Library             |
| ------------------- | ------------------- |
| Deep Learning       | TensorFlow, Keras   |
| Visualization       | Matplotlib, Seaborn |
| Data Augmentation   | ImageDataGenerator  |
| Metrics             | NumPy, Scikit-learn |
| Model Visualization | VisualKeras         |

---

## 🧩 Model Architecture Summary


Conv2D (32 filters, 4x4)  
MaxPooling2D  
Conv2D (64 filters, 4x4)  
MaxPooling2D  
Conv2D (128 filters, 4x4)  
MaxPooling2D  
Conv2D (128 filters, 4x4)  
Flatten  
Dense (512) + Dropout  
Dense (4) Softmax  
Total Params: ~496K  


---

## 📊 Evaluation

### *Confusion Matrix*

Shows very high precision across all classes.

### *Metrics Summary*

| Class      | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| Glioma     | 0.979     | 0.977  | 0.978    |
| Meningioma | 0.973     | 0.968  | 0.970    |
| No Tumor   | 0.989     | 0.998  | 0.993    |
| Pituitary  | 0.999     | 0.996  | 0.997    |

### *Overall Accuracy:* *98.5%*

---

## 🖼️ Sample Prediction Function

The project includes 2 prediction utilities:

### 1️⃣ *Tumor Type Classifier*

python
prediction = predict_tumor_type(model, "/path/to/image")
print(prediction)


Output examples:

* "Benign – Meningioma"
* "Malignant – Glioma"
* "No Tumor Detected"

### 2️⃣ *Prediction With Image Display*

python
predict_and_display_tumor(model, "/path/to/image")


Displays the MRI and prints:

* Tumor status (Benign/Malignant/No Tumor)
* Classified type (Glioma / Pituitary / Meningioma / No Tumor)

---

## 📈 Training Curve

Both accuracy and loss curves are plotted for:

* Training
* Validation

---

## 🛠️ Installation

bash
pip install tensorflow
pip install visualkeras
pip install seaborn


Dataset must be placed inside the project directory before running the notebook.

---

## ▶️ How to Run

1. Upload dataset to /content (if running on Google Colab)
2. Run all cells in the notebook
3. After training, evaluate or test with custom MRI images

---

## 📁 Files in Project

| File               | Description                         |
| ------------------ | ----------------------------------- |
| braintumor.ipynb | Full training + evaluation notebook |
| README.md        | Project documentation               |
| /archive.zip     | Dataset zip (used in notebook)      |

---

## 🧑‍💻 Author

*Aryan Kamble*
Email: [aryan.carrer@gmail.com](mailto:aryan.carrer@gmail.com)
Project built with TensorFlow, Keras & Google Colab.

---

## ⭐ Contributions

Feel free to fork, improve the model, add Grad-CAM, or convert to a web app using Flask/Streamlit.

---
