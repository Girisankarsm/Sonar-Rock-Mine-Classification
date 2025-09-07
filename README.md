# 🎯 Sonar Rock vs Mine Classification  

A complete **Machine Learning project** that classifies sonar signals as **Rock** or **Mine** using the **UCI Sonar Dataset**.  
This project demonstrates **data preprocessing, dimensionality reduction, model building, hyperparameter tuning, and evaluation** with clear visualizations.  

---

## 📌 Project Overview  
- 📊 **Dataset**: Sonar signals (60 features, 208 samples)  
- 🎯 **Goal**: Predict whether the object detected is a **Rock (R)** or a **Mine (M)**  
- 🛠️ **Tech Stack**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- 🔍 **Approach**:  
  - Data cleaning & preprocessing  
  - Feature scaling + **PCA** (95% variance retained)  
  - Training multiple ML models  
  - **Hyperparameter tuning** with GridSearchCV  
  - Evaluation with Accuracy, Precision, Recall, F1-score, ROC & PR curves  

---

## 🚀 Models Implemented  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- K-Nearest Neighbors (KNN)  

Each model is optimized using **GridSearchCV** and compared using cross-validation scores.  

---

## 📊 Results & Insights  
- ✅ Automated model comparison with evaluation metrics  
- 📈 Confusion Matrix, ROC Curve, and Precision-Recall Curve visualizations  
- 🏆 Best-performing model is saved as `best_model_advanced.pkl` for future predictions  
- 🖥️ Supports **custom user input** for live predictions  

---

## ⚡ Applications  
- 🔎 **Defense Systems**: Mine detection in naval operations  
- 🌊 **Underwater Robotics**: Object classification in sonar-based navigation  
- 📡 **Signal Processing**: Demonstrates ML on high-dimensional frequency data  

---

## 📂 Repository Structure  
