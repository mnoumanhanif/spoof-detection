## 🧠 Spoof Detection using Vision Transformers (ViT & Swin Transformer)

### 🔍 Overview

This project implements a **Spoof Detection System** using **Vision Transformer (ViT)** and **Swin Transformer** architectures. The goal is to classify facial images as **Real** or **Spoofed** to enhance the security of AI-powered facial authentication systems.

The project was developed as part of my **Generative AI course (Fall 2025)**. It explores the application of modern transformer-based models in **biometric security** and **anti-spoofing** tasks.

---

### 🚀 Objectives

* Build and train **binary classification models** (Real vs. Spoof).
* Compare the performance of **ViT** and **Swin Transformer** architectures.
* Evaluate the models using **Accuracy, Precision, Recall, and F1-score**.
* Test the system with **personal real and spoofed face samples** to validate performance.

---

### 📊 Dataset

* **Dataset Used:** [`nguyenkhoa/celeba-spoof-for-face-antispoofing-test`](https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test) (Hugging Face)
* The dataset contains **real and spoofed facial images** captured under various lighting and presentation attack conditions.

---

### 🧩 Models Implemented

1. **Vision Transformer (ViT)**

   * Uses global self-attention to capture high-level visual representations.
2. **Swin Transformer**

   * Employs hierarchical attention windows for localized and efficient feature extraction.

---

### ⚙️ Tech Stack

* **Framework:** PyTorch
* **Pretrained Models:** ViT, Swin Transformer (from `torchvision` and `transformers` libraries)
* **Dataset Source:** Hugging Face Datasets
* **Tools:** NumPy, Pandas, Matplotlib, Scikit-learn

---

### 🧪 Key Learnings

* Swin Transformer showed superior accuracy and faster convergence compared to ViT.
* Transformers effectively capture micro-level spoofing cues like **texture inconsistencies**, **reflections**, and **illumination artifacts**.
* Importance of dataset balance and preprocessing in anti-spoofing tasks.
* Improved understanding of **attention visualization** and **model interpretability**.

---

### 🧾 Citation

If you find this work helpful, please consider citing or referencing this repository.

---

### 🤝 Acknowledgment

This project was completed as part of the **Generative AI (Fall 2025)** course.
Special thanks to the open-source contributors of **Hugging Face** and **PyTorch** for enabling this research.

Would you like me to **add a short project tagline and GitHub “About” text** (for the top of your repository page, under the title)? That makes the repo look much more professional and LinkedIn-ready.
