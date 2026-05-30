# Customer Category Classifier

A machine learning project that classifies customers into four segments (A, B, C, D) based on demographic data. Three models — K-Nearest Neighbors, Logistic Regression, and a Neural Network — are trained and compared to find the best classifier.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [How to Access & Run](#how-to-access--run)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

---

## Project Overview

Businesses often need to understand their customer base by grouping them into meaningful segments. This project uses supervised machine learning to classify customers into one of four categories (A, B, C, D) using demographic and behavioural features such as age, gender, marital status, profession, work experience, and family size.

The project was developed as a group assignment (Group F) and follows a full ML pipeline: data exploration, preprocessing, model training, and evaluation.

---

## Dataset

- **File:** `Customer_Category_Classifier_Dataset.csv`
- **Source:** Loaded from Google Colab's sample data path (`/content/sample_data/`)
- **Size:** 8,068 data points
- **Target Variable:** `Segmentation` — four classes: A, B, C, D
- **Feature Types:** Both quantitative (e.g., Age, Work Experience, Family Size) and categorical (e.g., Gender, Ever Married, Profession, Graduated)

> **Note:** To run this notebook, place the dataset at the path expected in the code or update the path in the `pd.read_csv()` call to match your local setup.

---

## Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Checked dataset shape, feature types, and class distribution
   - Visualised segmentation distribution via bar chart
   - Generated histograms, boxplots, and a correlation heatmap

2. **Data Preprocessing**
   - Removed the `ID` column and duplicate rows
   - Handled missing values: categorical columns filled with mode, numerical columns with median
   - Applied One-Hot Encoding on categorical features
   - Scaled all numerical features using MinMaxScaler

3. **Unsupervised Exploration**
   - Applied KMeans clustering (4 clusters) to explore natural groupings
   - Visualised clusters using PCA (2D projection)

4. **Supervised Classification (One-vs-Rest)**
   - Each segmentation class (B, C, D) was treated as a binary classification target
   - Applied SMOTE on training data to handle class imbalance
   - Split data: 70% training / 30% testing with stratification

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, and AUC score were computed for each model

---

## Models Used

| Model | Notes |
|---|---|
| **K-Nearest Neighbors (KNN)** | Trained on SMOTE-balanced data |
| **Logistic Regression** | Trained with `class_weight='balanced'` |
| **Neural Network (Keras)** | 2-layer dense network with Dropout, BatchNorm, EarlyStopping, trained on SMOTE-balanced data |

---

## How to Access & Run

### Option 1 — Clone the Repository

```bash
git clone https://github.com/misha832-ux/Customer-Catagory-Classifier.git
cd Customer-Catagory-Classifier
```

### Option 2 — Download as ZIP

Go to the repository page on GitHub → click the green **Code** button → select **Download ZIP**.

### Option 3 — Open in Google Colab (Recommended)

Since the notebook was built in Google Colab, the easiest way to run it is:

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → GitHub**
3. Paste the repo URL: `https://github.com/misha832-ux/Customer-Catagory-Classifier`
4. Upload the dataset file to Colab's `/content/sample_data/` directory, or update the file path in the notebook

### Running Locally

```bash
pip install -r requirements.txt
jupyter notebook Group_F.ipynb
```

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries used:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow
scipy
```

---

## Results

Each segmentation label (B, C, D) was evaluated separately as a binary classification task. All three models were compared using accuracy, precision, recall, AUC score, and ROC curves.

| Model | Strength |
|---|---|
| KNN | Good recall with SMOTE balancing |
| Logistic Regression | Fast, interpretable baseline |
| Neural Network | Best overall performance with balanced training |

> Exact accuracy figures depend on the random seed and dataset version used during training.

---

## License

This project is open for academic and educational use. Feel free to fork and build upon it.

---
