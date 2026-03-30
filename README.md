# BrainTD
Brain Tumor Detection using CNN
# 🧠 Brain Tumor Detection Using CNN

> A hybrid deep learning approach combining **Morphological Filtering** and **Convolutional Neural Networks (CNN)** for accurate, efficient brain tumor detection from MRI scans.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20Paper-brightgreen?style=flat-square)

---

## 👥 Authors

| Name | Degree | Current Role |
|------|--------|--------------|
| **Annarapu Divya Sri** | B.Tech CSE – Internet of Things | 💼 Accenture · Data Analyst |
| **SriKeerthana Elle** | B.Tech CSE – Internet of Things | 💼 Accenture · Data Engg. & Management Governance |
| **Yadlapalli Venakata Pavan Kumar** | B.Tech CSE – Internet of Things | 🎓 MSCS · University of Central Missouri, USA |
| **R. Annapurna** *(Faculty Guide)* | Assistant Professor | 🏛️ Dept. of CSE (IoT) · MRCET, Hyderabad |

**Institution:** Malla Reddy College of Engineering & Technology (MRCET), Hyderabad, India
**Department:** CSE – Internet of Things (IoT)

---

## 📌 Abstract

Brain tumor detection is a hazardous task in medical imaging that plays a key role in patient outcomes. Brain tumors have continued to increase for the last decade in several countries. Medical images play a very important role in making the right diagnosis for the doctor and in the patient's treatment process.

This study proposes an approach which is a combination of both **morphological filtering** and **convolutional neural networks (CNNs)**. First, the image is preprocessed using morphological filtering to enhance the contrast of the brain MRI image and remove any unwanted noise. The filtered images are then fed into a CNN model with a unique architecture that includes multiple convolutional and pooling layers. The CNN model learns and extracts the important features from the filtered images and predicts the presence of a tumor.

---

## 🔬 Key Contributions

- ✅ **Hybrid Detection Method** — Merges morphological filtering and CNNs, resulting in substantial improvement in brain tumor diagnosis accuracy and speed
- ✅ **Advanced Segmentation** — Focuses on challenging brain tumor segmentation, addressing the need for precise and efficient diagnosis
- ✅ **Multiclass Classification** — Unlike binary systems, introduces multiclass classification for specific tumor type identification
- ✅ **Traditional + Deep Learning** — Demonstrates the potential of combining classical image processing with modern deep learning methods

---

## 🏗️ System Architecture

```
MRI Input
    │
    ▼
Morphological Filtering
(Erosion → Dilation → Contrast Enhancement → Noise Removal)
    │
    ▼
CNN Feature Extraction
┌─────────────────────────────────────────────────────────┐
│  Conv1 (32) → Pool1 → Conv2 (64) → Pool2               │
│  → Conv3 (128) → Pool3 → FC1 (512) → FC2 (256)         │
│  → Softmax Output                                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Classification Output
(Tumor Present / No Tumor / Tumor Type)
```

### CNN Layer Details

| Layer | Type | Filters / Units | Activation |
|-------|------|-----------------|------------|
| Layer 1 | Conv2D | 32 filters, 3×3 | ReLU |
| Layer 2 | MaxPooling2D | 2×2 | — |
| Layer 3 | Conv2D | 64 filters, 3×3 | ReLU |
| Layer 4 | MaxPooling2D | 2×2 | — |
| Layer 5 | Conv2D | 128 filters, 3×3 | ReLU |
| Layer 6 | MaxPooling2D | 2×2 | — |
| Layer 7 | Dense (FC1) | 512 units | ReLU |
| Layer 8 | Dropout | 0.5 rate | — |
| Layer 9 | Dense (FC2) | 256 units | ReLU |
| Output | Dense | N classes | Softmax |

---

## 🛠️ Tech Stack

| Category | Tools / Frameworks |
|----------|--------------------|
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow, Keras |
| **Image Processing** | OpenCV |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | Scikit-learn |
| **Input Modality** | Brain MRI Images (JPG/PNG) |

---

## 📁 Project Structure

```
brain-tumor-detection-cnn/
│
├── 📂 dataset/
│   ├── tumor/              # MRI images with tumor
│   └── no_tumor/           # MRI images without tumor
│
├── 📂 preprocessing/
│   ├── morphological.py    # Erosion, dilation, opening, closing
│   └── augmentation.py     # Data augmentation utilities
│
├── 📂 model/
│   ├── cnn_model.py        # CNN architecture definition
│   ├── train.py            # Model training script
│   └── evaluate.py         # Evaluation & metrics
│
├── 📂 outputs/
│   ├── saved_model/        # Trained model weights
│   └── results/            # Prediction outputs
│
├── 📂 web/
│   └── brain-tumor-cnn.html  # Offline interactive demo webpage
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn
```

### Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection-cnn.git
cd brain-tumor-detection-cnn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python model/train.py --dataset ./dataset --epochs 50 --batch_size 32
```

### Run Prediction

```bash
python model/evaluate.py --image path/to/mri_scan.jpg
```

### Open the Demo Webpage

Simply open `web/brain-tumor-cnn.html` in any browser — **no internet required**.

---

## 📊 Datasets Used

- [Brain MRI Images for Brain Tumor Detection — Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- [BRATS — Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/cbica/brats/)
- Custom labeled MRI scan collection

---

## 🧪 Methodology

### Step 1 — Morphological Preprocessing
- Load raw brain MRI scan
- Apply **erosion** to remove small noise artifacts
- Apply **dilation** to restore tumor region boundaries
- **Histogram equalization** for contrast enhancement
- **Gaussian blur** for smooth noise reduction
- **Morphological opening** to extract the tumor candidate region

### Step 2 — CNN Feature Extraction & Classification
- Feed preprocessed image into CNN
- Convolutional layers extract spatial and texture features
- Pooling layers reduce dimensionality
- Fully connected layers combine features
- **Softmax** outputs class probabilities
- **Backpropagation + Adam optimizer** minimizes classification error

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Classification Type | Multiclass |
| Input Size | 224 × 224 px |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Dropout Rate | 0.5 |
| Improvement over binary-only CNN | ↑ Accuracy & Speed |

---

## 🔒 Web Demo Features

The included `brain-tumor-cnn.html` is a **fully offline** interactive demo:

- Upload 1–5 MRI images
- Step-by-step animated preprocessing pipeline
- CNN classification with confidence scores
- Visual tumor region highlighting on canvas
- Results: 🟢 green circles (Tumor Present) / 🔴 red circles (No Tumor)
- Dark / Light mode toggle
- Right-click content protection

---

## 🏛️ Institution

**Malla Reddy College of Engineering & Technology (MRCET)**
Hyderabad, Telangana, India
🌐 [www.mrcet.com](https://mrcet.com)
✉️ mrcet2004@gmail.com

- AICTE Approved · JNTUH Affiliated · NBA Tier-I · NAAC Certified · NIRF Ranked

---

## 📄 Keywords

`Brain Tumor Detection` · `Convolutional Neural Networks` · `Morphological Filtering` · `Medical Imaging` · `MRI Images` · `Feature Extraction` · `Deep Learning` · `Image Preprocessing` · `Multiclass Classification` · `Computer-Aided Diagnosis` · `Backpropagation`

---

## ⚠️ Disclaimer

This project is developed for **academic and research purposes only**. It is not intended to replace professional medical diagnosis. Always consult a qualified medical professional for clinical decisions.

---

*B.Tech Research Paper · Dept. of CSE (Internet of Things) · MRCET, Hyderabad, India*
