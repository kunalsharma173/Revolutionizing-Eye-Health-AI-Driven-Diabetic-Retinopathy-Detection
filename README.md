# 🏥 Revolutionizing-Eye-Health-AI-Driven-Diabetic-Retinopathy-Detection

## 📌 Project Overview  
Diabetic Retinopathy (DR) is a **leading cause of blindness** globally, affecting over **191 million people**. **Early detection** is critical for preventing vision loss, but **manual diagnosis is time-consuming and prone to errors**. This project leverages **Deep Learning** to **automate and enhance DR detection** using **CNNs** and **Transfer Learning (InceptionV3)** on **high-resolution retinal images**.  

## 🎯 Key Contributions  
✅ **Built a Custom CNN** for DR classification  
✅ **Implemented Transfer Learning with InceptionV3** for superior accuracy  
✅ **Performed Extensive EDA** (Image Quality Metrics, Aspect Ratios, Label Distribution)  
✅ **Optimized Model Performance** (Precision, Recall, AUC, F1-score)  
✅ **Applied Advanced Image Preprocessing** (Resizing, Normalization, Augmentation)  

## 🛠 Tech Stack  
🔹 **Language**: Python 🐍  
🔹 **Libraries**: `TensorFlow`, `Keras`, `OpenCV`, `Matplotlib`, `Pandas`, `NumPy`  
🔹 **Dataset**: [Kaggle - Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)  

## 🔬 Methodology  

### 1️⃣ Data Preprocessing  
📌 **Challenges:** High-resolution images, class imbalance, varying image quality  
- **Resized images** to standard dimensions (`256x256`)  
- **Normalized pixel values** for better feature extraction  
- **Data Augmentation** (Flips, Rotations, Zooming) to enhance diversity  

### 2️⃣ Model Development  
#### 🏗 **Custom CNN Architecture**  
- **Feature Extraction**: Multiple **Convolutional + Pooling Layers**  
- **Dense Layers** for final classification  
- **Softmax Activation** to predict severity levels  

#### 🔄 **InceptionV3 with Transfer Learning**  
- **Pre-trained on ImageNet**, adapted for DR detection  
- **Fine-tuned deeper layers** for domain-specific learning  
- **Reduced training time**, improved convergence  

### 3️⃣ Model Training & Evaluation  
📌 **Evaluation Metrics:**  
🔹 **Accuracy** 🔹 **Precision** 🔹 **Recall** 🔹 **F1-Score** 🔹 **ROC-AUC**  

| Model        | Accuracy | Precision | Recall | F1 Score | AUC  |
|-------------|----------|-----------|--------|----------|------|
| **CNN**       | 82%      | **99%**    | 73%    | 84%      | 86%  |
| **InceptionV3** | **89%**  | 84%       | **84%**  | 91%      | **91%**  |

📊 **Results:**  
✅ **InceptionV3 outperformed CNN**, achieving **89% accuracy**  
✅ **ROC-AUC score of 91%**, ensuring strong classification capabilities  
✅ **Faster convergence with Transfer Learning**, reducing training time  

## 📡 Future Scope  
🚀 **Expand dataset** with real-world clinical data  
📈 **Improve model generalization** using ensemble learning  
⚕ **Deploy model as a web-based diagnostic tool** for healthcare applications  

## 💻 How to Run  

1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/your-username/diabetic-retinopathy-detection.git
```

2️⃣ **Install dependencies:**  
```bash
pip install -r requirements.txt
```
3️⃣ **Run Preprocessing Scripts:**
```bash
python preprocessing/resize.py
python preprocessing/rotation.py
python preprocessing/update_labels.py
```

4️⃣ **Build Models:**
```bash
python models_build/cnn.py
python models_build/inception.py
```

5️⃣ **Train the Models:**
```bash
python training/train_cnn.py
python training/train_inception.py
```



💡 If you find this project useful, give it a ⭐ on GitHub!

