# VisAIble: Explainable Deepfake Detection with EfficientNet-B0, Grad-CAM & LIME
![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red) ![Explainability](https://img.shields.io/badge/XAI-GradCAM%2FLIME-yellow)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Av1352/VisAIble/blob/main/notebooks/code.ipynb)

> 🧠 *Built an explainable deepfake detector using EfficientNet-B0, Grad-CAM, and LIME — visualizing what the model "sees" when distinguishing real vs. fake faces.*

**Detect and interpret AI-generated images using fine-tuned EfficientNet-B0 with full XAI workflow. Transparent trust-building for real vs fake image classification.**

---

## 🔎 Highlights & Demo

- **Live Demo:**  
  [![Open in Streamlit](https://static.streamlit.io/badges/streamlit.svg)](https://visaible.streamlit.app/)  
  _Try tumor prediction and XAI heatmaps interactively!_

**Full pipeline demo:**  
![](assets/demo.gif)

>*Test accuracy:* See results in [notebooks/code.ipynb](notebooks/code.ipynb), typically >95% on Kaggle validation split.

---

## 🥇 TL;DR Results

| Metric         | Value                    | Visual Example                    |
|----------------|--------------------------|-----------------------------------|
| Model          | EfficientNet-B0 (timm)   | ![](explanations/gradcam_real.png) |
| Test Accuracy  | **97.75%**               | ![](explanations/lime_fake.png)    |
| Dataset Size   | 140k images (2GB total)  |                                   |
| XAI Methods    | Grad-CAM, LIME           |                                   |

---

## 🚀 Why This Matters

Deepfakes are eroding digital trust. **VisAIble** provides explainable detection using state-of-the-art XAI techniques, showing *why* a prediction was made.
  
This approach supports **AI accountability** and is relevant for industries like **media forensics, cybersecurity, and content verification**.

---

## 🛠️ How to Run

> 💻 **Quick Start:** Open [VisAIble on Colab](https://colab.research.google.com/github/Av1352/VisAIble/blob/main/notebooks/code.ipynb) — no setup needed.
