# APS Failure Prediction - Machine Learning Classification

## a. Problem Statement

The Air Pressure System (APS) is a critical component in heavy-duty trucks that uses compressed air to assist braking and gear shifting. APS failures can lead to costly breakdowns and safety hazards. This project aims to develop machine learning models to predict APS failures based on sensor data, enabling proactive maintenance and reducing unexpected downtime.

**Business Impact:**
- Minimize truck downtime through predictive maintenance
- Reduce maintenance costs by preventing catastrophic failures
- Improve fleet safety and reliability
- Optimize maintenance scheduling and resource allocation

**Technical Challenge:**
This is a binary classification problem with significant class imbalance (98.3% normal vs 1.7% failures), requiring models that can effectively detect rare failure events while minimizing false alarms.

---

## b. Dataset Description

### Dataset: APS Failure at Scania Trucks
**Source:** https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks  
UCI Machine Learning Repository / Kaggle  
**Domain:** Automotive - Predictive Maintenance  
**Problem Type:** Binary Classification (Imbalanced)

### Dataset Statistics
- **Total Instances:** 76,000 samples
  - Training Set: 60,000 samples
  - Test Set: 16,000 samples
- **Features:** 170 columns
  - 169 numerical sensor measurements (aa_000 to ee_000)
  - 1 target variable (class)
- **Target Classes:**
  - `neg` (0): Normal operation (59,000 samples - 98.3%)
  - `pos` (1): APS failure (1,000 samples - 1.7%)

### Feature Categories
The dataset contains anonymized sensor readings grouped by prefix:
- **aa_xxx:** Air pressure measurements
- **ab_xxx:** System temperature readings
- **ac_xxx:** Flow rate measurements
- **ad_xxx:** Pressure differential sensors
- **ae_xxx - ee_xxx:** Additional system parameters

### Data Characteristics
- **Class Imbalance Ratio:** 59:1 (highly imbalanced)
- **Missing Values:** Present (handled during preprocessing)
- **Feature Type:** Continuous numerical values
- **Data Quality:** Industrial sensor data with real-world noise

### Preprocessing Applied
1. Missing value imputation using **median strategy**
2. Feature scaling using StandardScaler
3. Pipeline architecture to prevent data leakage
4. Class weighting to handle imbalance (balanced class weights, scale_pos_weight for XGBoost)

### Business Cost Consideration
- **False Positive Cost:** Unnecessary maintenance check
- **False Negative Cost:** Missed failure leading to breakdown (much higher cost)
- **Goal:** Maximize recall (detect all failures) while maintaining acceptable precision

---

## c. Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9749 | 0.9794 | 0.4819 | 0.9227 | 0.6331 | 0.6568 |
| Decision Tree | 0.9861 | 0.8016 | 0.7500 | 0.6080 | 0.6716 | 0.6684 |
| K-Nearest Neighbors | 0.9886 | 0.8887 | 0.8721 | 0.6000 | 0.7109 | 0.7181 |
| Naive Bayes | 0.9643 | 0.9744 | 0.3886 | 0.9120 | 0.5450 | 0.5824 |
| Random Forest (Ensemble) | 0.9891 | 0.9931 | 0.9348 | 0.5733 | 0.7107 | 0.7275 |
| XGBoost (Ensemble) | 0.9944 | 0.9947 | 0.9157 | 0.8400 | 0.8762 | 0.8742 |


**Rankings by F1 Score:**
1. 🥇 XGBoost (0.8762)
2. 🥈 K-Nearest Neighbors (0.7109)
3. 🥉 Random Forest (0.7107)
4. Decision Tree (0.6716)
5. Logistic Regression (0.6331)
6. Naive Bayes (0.5450)

---

## d. Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Baseline linear model with 97.49% accuracy and F1 score of 0.6331. Fast training (9.01 seconds). Poor precision (48.19%) generates many false alarms. 92.27% recall catches most failures but at cost of high false positives. Struggles with non-linear relationships and feature interactions. Linear assumption may be too simplistic. **Baseline model but not recommended for production.** |
| **Decision Tree** | Moderate performance with 98.61% accuracy and F1 score of 0.6716. Single tree prone to overfitting compared to ensemble methods. 75% precision is acceptable. 60.8% recall is concerning - misses many failures. Highly interpretable for maintenance teams. May struggle with class imbalance. Training time 11.64 seconds. **Useful for explaining predictions to non-technical stakeholders.** |
| **K-Nearest Neighbors** | Competitive performance with 98.86% accuracy and F1 score of 0.7109. Simple distance-based method despite good results. High precision (87.21%) reduces false alarms. 60% recall is moderate - misses some failures. Prediction time increases with dataset size (not ideal for real-time applications). Memory-intensive as stores entire training set. Training time 6.38 seconds. **Good baseline but not optimal for production deployment.** |
| **Naive Bayes** | Weakest performer with 96.43% accuracy and F1 score of 0.545. Fastest training time (1.33 seconds) due to simple probabilistic approach. Poor precision (38.86%) leads to excessive false alarms. Very high recall (91.2%) catches most failures but unreliable. Strong independence assumption violated in sensor data with correlated features. **Not recommended for this safety-critical application.** |
| **Random Forest (Ensemble)** | Strong performer with 98.91% accuracy and F1 score of 0.7107. Ensemble approach provides excellent generalization and stability. High precision (93.48%) minimizes false alarms. 57.33% recall means it misses more failures than XGBoost. Less prone to overfitting due to averaging across multiple trees. Good interpretability through feature importance. Training time 10.31 seconds. **Solid backup choice if XGBoost unavailable.** |
| **XGBoost (Ensemble)** | **Best overall performer** with 99.44% accuracy and highest F1 score (0.8762). Excellent at handling class imbalance through built-in weighted learning. High AUC (0.9947) demonstrates superior discrimination ability. 84% recall captures most failures with 91.57% precision - good balance for safety-critical maintenance. Robust to missing values and outliers. Fast training time (4.37 seconds). **Recommended for production deployment.** |

---

## 🚀 Streamlit Application Features

- **Interactive Dashboard:** Professional predictive maintenance interface
- **Model Selection:** Choose from all 6 trained models
- **Real-time Predictions:** Upload CSV files for instant failure prediction
- **Risk Assessment:** Traffic light indicator (🟢 Low / 🟡 Medium / 🔴 High Risk)
- **Performance Metrics:** Complete evaluation metrics display
- **Confusion Matrix:** Visual representation of model performance
- **Model Comparison:** Side-by-side performance analysis

---

## 📁 Repository Structure

```
aps_failure_prediction/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── model/                          # Model artifacts
    ├── train_models.py            # Model training script
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── feature_names.pkl
    ├── project.py
    └── model_results.csv
```

---

## 🎓 Key Insights

1. **XGBoost achieves outstanding F1 score of 0.8762** - Exceptional for 59:1 imbalanced data
2. **Class imbalance successfully handled** - Pipeline architecture with balanced class weights
3. **Clear precision-recall trade-offs:** Logistic Regression/Naive Bayes prioritize recall (>91%), Random Forest prioritizes precision (93.48%), XGBoost achieves optimal balance
4. **Ensemble methods dominate** - XGBoost and Random Forest achieve AUC >0.99
5. **Production-ready implementation** - Pipeline prevents data leakage, robust error handling

---

## 🔧 Technical Implementation

**Training Configuration:**
- Pipeline architecture with SimpleImputer (median strategy) and StandardScaler
- Class imbalance handling via `class_weight="balanced"` and `scale_pos_weight`
- 200 estimators for ensemble methods
- Multi-core training (`n_jobs=-1`) for efficiency
- Proper train/test split (60,000 / 16,000)

**Technologies Used:**
- Python 3.x
- scikit-learn (ML algorithms, pipelines, preprocessing)
- XGBoost (gradient boosting)
- Streamlit (web application)
- Pandas/NumPy (data manipulation)
- Matplotlib/Seaborn (visualization)

---

## 📊 Evaluation Metrics Explained

1. **Accuracy:** Overall correctness (can be misleading for imbalanced data)
2. **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
3. **Precision:** Of predicted failures, how many were actual failures
4. **Recall:** Of actual failures, how many were correctly predicted
5. **F1 Score:** Harmonic mean of precision and recall (best metric for imbalanced data)
6. **MCC (Matthews Correlation Coefficient):** Balanced measure accounting for all confusion matrix elements

---
## ▶️ How to Run the Project

1. Install dependencies:
   pip install -r requirements.txt

2. Train models (optional):
   python project.py

3. Launch Streamlit app:
   streamlit run app.py

## 💡 Conclusion

This project successfully demonstrates machine learning application for predictive maintenance in industrial systems. XGBoost emerges as the clear winner with 87.62% F1 score, achieving excellent balance between precision and recall for the highly imbalanced APS failure dataset. The production-ready implementation with pipeline architecture and proper class imbalance handling makes this solution deployable for real-world industrial applications.

**Recommended Model:** XGBoost for production deployment due to superior F1 score (0.8762), high accuracy (99.44%), and excellent AUC (0.9947).