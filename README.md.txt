# Sports vs Politics Text Classifier
## CSL 7640: Natural Language Understanding – Assignment Problem 4

### Author
**Name:** Sharad Singh  
**Roll Number:** m25mac009  

---

## Project Description

This project implements a machine learning based text classification system that classifies documents into two categories:

- Sports  
- Politics  

The classifier is trained using text documents stored in separate folders and uses TF-IDF feature representation along with multiple machine learning algorithms. The purpose of this project is to compare different machine learning techniques for text classification.

---

## Machine Learning Algorithms Used

The following three algorithms were implemented and compared:

1. **Multinomial Naive Bayes**
   - Probabilistic classifier based on Bayes' theorem
   - Works well with text data and word frequencies

2. **Logistic Regression**
   - Linear classification model
   - Effective for high-dimensional text data

3. **Support Vector Machine (Linear SVM)**
   - Margin-based classifier
   - Performs well in sparse, high-dimensional spaces

---

## Feature Extraction Method

**TF-IDF (Term Frequency – Inverse Document Frequency)** was used to convert text documents into numerical feature vectors.

TF-IDF measures the importance of a word in a document relative to the entire dataset.

Advantages:
- Reduces importance of common words
- Highlights meaningful words
- Improves classification performance

---

## Dataset Structure

The dataset is organized as follows:

dataset/
│
├── sports/
│ ├── sport1.txt
│ ├── sport2.txt
│ └── ...
│
└── politics/
├── politics1.txt
├── politics2.txt
└── ...


Each file contains one document related to the respective category.

---

## How to Run the Program

### Step 1: Open Command Prompt

Navigate to project folder:

```bash
cd D:\nluassign\prob_4
```


Step 2: Run the classifier
```bash
python m25mac009_prob4.py
```