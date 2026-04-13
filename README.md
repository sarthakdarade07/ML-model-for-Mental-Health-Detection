# 🧠 Mental Health Detection from Social Media Text

> Comparing Classical ML Models vs. DistilBERT Transformer for 5-Class Mental Health Classification

---

## 📌 Overview

This project builds and compares multiple NLP-based models to detect mental health conditions from Reddit social media posts. The system classifies text into 5 categories: **Anxiety, Depression, Loneliness, Suicidal Ideation, and Normal**. A total of 21,880 balanced samples were used across all experiments.

The core finding: **data quality improvement (+0.047 macro F1) outperformed all architectural upgrades combined.**

---

## 📄 Research Paper

This project is backed by a peer-reviewed research paper.

📥 [Read the Paper](./research_paper.pdf)

> **"Transformers vs. Classical Models for Mental Health Detection"**
> Suyash Jaiswal, Harshada Deshmukh, Aman Jaiswal, Sarthak Darade, Rushikesh Kokate, Dr. Pratvina Talele
> Dr. Vishwanath Karad MIT World Peace University, Pune

---

## 📁 Project Structure

```
Mental Health Detection/
│
├── dataset/
│   ├── anxiaug22.csv               # Anxiety Reddit posts
│   ├── depaug22.csv                # Depression Reddit posts
│   ├── loneaug22.csv               # Loneliness Reddit posts
│   ├── swaug22.csv                 # Suicidal Reddit posts
│   ├── normal.csv                  # Normal class (multi-source)
│   └── balanced_mental_health_dataset.csv
│
├── models/                         # V1 - LR Baseline
├── models_v2/                      # V2 - Improved Normal Class
├── models_v3/                      # V3 - Extended Normal + Splits
├── models_v5/                      # V5 - Final Ensemble (LR+SVM+NB)
│
├── evaluations/
│   ├── fig1_class_distribution.png
│   ├── fig3_confusion_matrices.png
│   ├── fig4_roc_curves.png
│   └── fig5_model_comparison.png
│
├── Architecture/
│   └── architecture_diagram.png
│
├── Notebook1.ipynb                 # Classical ML experiments
├── Notebook2.ipynb                 # Extended iterations & ensemble
├── Transformer.ipynb               # DistilBERT fine-tuning
├── improvements.ipynb              # Iteration tracking
├── research_paper.pdf              # Published research paper
└── README.md
```

---

## 🤖 Models & Results

| Model | Accuracy | Macro F1 | Mean AUC | Train Time |
|-------|----------|----------|----------|------------|
| V1 — LR Baseline (TF-IDF) | 72% | 0.7184 | 0.920 | < 1 min |
| V2 — Data Fix (Amazon + Emotion Tweets) | 77% | 0.7651 | — | < 1 min |
| V3 — Extended Normal Class | 76% | 0.7547 | — | < 1 min |
| V4 — Trigrams + Lemmatization (Reverted) | 73% | 0.7327 | — | < 1 min |
| V5 — Soft Voting Ensemble (LR+SVM+NB) | 76% | 0.7601 | 0.935 | ~5 min |
| V6 — Stacking Classifier (Evaluated) | 76% | 0.7573 | — | ~5 min |
| **DistilBERT V1 (Best Overall)** | **78%** | **0.7788** | **0.944** | ~40 min |

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)

**Models:** Logistic Regression · Linear SVM · Naive Bayes · Random Forest · Soft-Voting Ensemble · Stacking Classifier · DistilBERT

**NLP:** TF-IDF Vectorization · WordPiece Tokenization · TextBlob · NLTK · HuggingFace Transformers

---

## 🚀 How to Run

### Classical ML (Notebooks 1 & 2)
```bash
pip install scikit-learn pandas numpy nltk textblob scipy
jupyter notebook Notebook1.ipynb
```

### DistilBERT Transformer
```bash
pip install transformers torch datasets
jupyter notebook Transformer.ipynb
```
> Recommended: Run on **Google Colab** with T4 GPU for ~40 min training time.

---

## 🔑 Key Findings

- **Data quality > Model complexity** — fixing the normal class gave +0.047 macro F1, the single biggest gain
- **DistilBERT** outperforms best classical ensemble across all 5 classes
- **Depression–Suicidal confusion** is the hardest boundary (similar vocabulary)
- A **confidence threshold of 0.40** flags uncertain predictions for human review — critical for safe deployment

---

## ⚠️ Ethical Note

This system is intended as a **support tool only**. All high-risk predictions must be reviewed by a human. The model should assist clinical judgment, never replace it. Passive monitoring without user consent is unethical.

---

## 👥 Authors

| Name | Institution |
|------|-------------|
| Suyash Jaiswal | MIT-WPU, Pune |
| Harshada Deshmukh | MIT-WPU, Pune |
| Aman Jaiswal | MIT-WPU, Pune |
| Sarthak Darade | MIT-WPU, Pune |
| Rushikesh Kokate | MIT-WPU, Pune |
| Dr. Pratvina Talele (Guide) | MIT-WPU, Pune |
