# 🛡️ Sentiguard — Suicidal Ideation Detection Framework

> **Detecting suicidal ideation in social media text using Machine Learning and NLP.**  
> A web-based tool that analyzes CSV-uploaded posts and classifies them as **Suicidal** or **Non-Suicidal**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model Training](#model-training)
- [Pipeline Architecture](#pipeline-architecture)
- [Output Files](#output-files)

---

## Overview

**Sentiguard** is a web application built with Flask that allows users to upload a `.csv` file containing social media text and receive instant predictions on whether each entry shows signs of suicidal ideation.

The backend pipeline includes:
- A robust **NLP preprocessing** module (`preprocess.py`) for cleaning and normalizing large raw datasets
- A **Logistic Regression classifier** trained using TF-IDF features (`model.py`)
- A **Flask web server** (`app.py`) serving predictions through a clean browser interface

---

## Project Structure

```
Sentiguard_final_project/
│
├── app.py                   # Flask web server & prediction logic
├── model.py                 # ML model training & serialization
├── preprocess.py            # Full NLP preprocessing pipeline
│
├── model.pkl                # Trained model (vectorizer + classifier)
├── cleaned_dataset.csv      # Cleaned output from preprocessing
├── test.csv                 # Sample test input file
├── test.xlsx                # Sample test input (Excel format)
│
├── templates/
│   └── index.html           # Jinja2 HTML template for the web UI
│
├── style.css                # Stylesheet (standalone)
└── index.html               # Standalone HTML (root-level copy)
```

---

## Features

- 📁 **CSV Upload** — Upload any `.csv` file; the first column is used as text input
- 🔍 **Real-time Prediction** — Each row is classified as *Suicidal Ideation Detected* or *Non-Suicidal*
- 📊 **Confidence Score** — Each result includes a confidence percentage
- 🎨 **Color-coded Results** — Red for suicidal, green for non-suicidal
- 🧹 **Production-grade Preprocessing** — Memory-safe, two-pass streaming pipeline with atomic file writes
- 🔒 **Data Safeguards** — Original dataset is strictly read-only; outputs are written to separate files

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| ML Model | Scikit-learn (Logistic Regression + TF-IDF) |
| NLP Processing | NLTK (WordNet Lemmatizer, Porter Stemmer, Stopwords) |
| Data Handling | Pandas |
| Frontend | HTML5, CSS3, Jinja2 |
| Serialization | Pickle |

---

## Setup & Installation

### 1. Clone / Download the Project

```bash
git clone <your-repo-url>
cd Sentiguard_final_project
```

### 2. Install Dependencies

```bash
pip install flask pandas scikit-learn nltk
```

### 3. Download NLTK Resources

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

> **Note:** `preprocess.py` handles NLTK downloads automatically if they are missing.

### 4. Train the Model (if `model.pkl` is not present)

```bash
python model.py
```

This generates `model.pkl` containing the serialized TF-IDF vectorizer and Logistic Regression classifier.

### 5. Run the Web App

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## Usage

1. Go to **http://127.0.0.1:5000** in your browser
2. Click **Choose File** and select a `.csv` file
   - The CSV must have text data in its **first column**
3. Click **Check**
4. View the results table with each text, its classification, and confidence score

### Example CSV Format

```
text
I feel hopeless and don't know what to do anymore
Life is great and I'm excited for the future
I want to end everything
```

---

## Dataset & Preprocessing

The preprocessing pipeline (`preprocess.py`) is designed for large-scale datasets (~233K rows) and performs a **two-pass streaming** process:

### Pass 1 — Structural Scan & Duplicate Discovery
- Detects rows with **missing class labels**
- Identifies **malformed data** (text shifted into the class column)
- Finds **same-class duplicate** records (keeps one, drops the rest)
- Finds **conflicting-class duplicates** (same text, different labels — drops all)

### Pass 2 — NLP Preprocessing & Atomic Output
For each valid row:
1. **HTML entity decoding** (`&amp;` → `&`)
2. **URL removal**
3. **Whitespace normalization**
4. **Lowercasing**
5. **Tokenization** (alphabetical words only)
6. **Negation-preserving stopword removal** (keeps `not`, `never`, `can't`, etc.)
7. **Lemmatization** (WordNet)
8. **Stemming** (Porter Stemmer)

> **Atomic Safety:** Outputs are written to `.tmp` files first and only renamed on full success — preventing partial/corrupt output files.

### Dataset Statistics (Reference)

| Metric | Count |
|---|---|
| Raw rows in dataset.csv | 233,337 |
| Structurally corrupt rows removed | 1,292 |
| Same-class duplicates dropped | 1 |
| Conflicting-class duplicates dropped | 5 |
| **Final cleaned rows** | **232,039** |

---

## Model Training

`model.py` trains a lightweight binary classifier:

```
Texts  →  TF-IDF Vectorizer  →  Logistic Regression  →  model.pkl
```

- **Labels:** `1` = Suicidal, `0` = Non-Suicidal
- **Saved as:** `model.pkl` (dictionary with keys `vectorizer` and `classifier`)

The prediction logic in `app.py` uses keyword matching as a rule-based fallback. To use the trained `model.pkl` for inference, load it via `pickle` and call `vectorizer.transform()` + `classifier.predict()`.

---

## Pipeline Architecture

```
dataset.csv (read-only)
      │
      ▼
preprocess.py
  ├── Pass 1: Structural scan & duplicate analysis
  └── Pass 2: Clean + NLP preprocess (atomic write)
      ├── cleaned_dataset.csv        (text, class)
      └── preprocessed_dataset.csv  (text, class, label, lemmatized_text, stemmed_text)
      
model.py
  └── Trains on sample data → model.pkl

app.py  (Flask server)
  ├── GET  /  → renders upload form
  └── POST /  → reads CSV, predicts each row, renders results table
```

---

## Output Files

| File | Description |
|---|---|
| `cleaned_dataset.csv` | Cleaned text + class label (2 columns) |
| `preprocessed_dataset.csv` | Cleaned text + class + numeric label + lemmatized + stemmed (5 columns) |
| `cleaning_report.txt` | Full audit report with row counts, duplicate analysis, and class distribution |
| `model.pkl` | Serialized TF-IDF vectorizer + Logistic Regression classifier |

---

## ⚠️ Disclaimer

This tool is intended for **academic and research purposes only**. It is **not a substitute** for professional mental health assessment. If you or someone you know is in crisis, please contact a qualified mental health professional or a crisis helpline immediately.

---

*Built as part of the Suicidal Ideation Detection Framework on Social Media project.*
