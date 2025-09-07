# ---------------------------
# Advanced Sonar Classification Project (9.5/10)
# ---------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import joblib

# ---------------------------
# 1. Load Dataset
# ---------------------------
def load_data(path="sonar_data.csv"):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print("✅ Dataset loaded. Shape:", df.shape)
    return X, y, df

# ---------------------------
# 2. Exploratory Data Analysis
# ---------------------------
def eda(df, y):
    print("\n📊 Dataset Info:")
    print(df.info())
    print("\n🔍 Class Distribution:")
    print(y.value_counts())

    # Plot class distribution
    sns.countplot(x=y)
    plt.title("Class Distribution (R=Rock, M=Mine)")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12,6))
    sns.heatmap(df.iloc[:,:-1].corr(), cmap="coolwarm", cbar=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# ---------------------------
# 3. Model Training and Tuning
# ---------------------------
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000),
            "params": {"model__C":[0.01,0.1,1,10]}
        },
        "SVM": {
            "model": SVC(kernel="rbf", probability=True),
            "params": {"model__C":[0.1,1,10], "model__gamma":["scale","auto"]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"model__n_estimators":[50,100,200], "model__max_depth":[None,5,10]}
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"model__n_neighbors":[3,5,7,9]}
        }
    }

    best_estimators = {}
    for name, mp in models.items():
        print(f"\n🚀 Training & tuning {name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('model', mp["model"])
        ])
        grid = GridSearchCV(pipeline, mp["params"], cv=5, scoring="accuracy")
        grid.fit(X_train, y_train)
        best_estimators[name] = grid.best_estimator_
        print(f"✅ Best params for {name}: {grid.best_params_}")
        print(f"{name} CV Accuracy: {grid.best_score_:.3f}")
    return best_estimators

# ---------------------------
# 4. Evaluate Models
# ---------------------------
def evaluate_models(models_dict, X_test, y_test):
    results = {}
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"\n📌 {name} Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))
    return results

# ---------------------------
# 5. Confusion Matrix & ROC/PR Curve
# ---------------------------
def plot_evaluation(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rock","Mine"], yticklabels=["Rock","Mine"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC & AUC
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve((y_test=="M").astype(int), y_prob)
        roc_auc = roc_auc_score((y_test=="M").astype(int), y_prob)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve((y_test=="M").astype(int), y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()

# ---------------------------
# 6. User Input Prediction
# ---------------------------
def predict_custom_input(best_model):
    print("\n📝 Enter 60 sonar values separated by spaces:")
    values = input().split()
    if len(values)!=60:
        print("❌ Error: Exactly 60 values required")
        return
    values = np.array(values, dtype=float).reshape(1,-1)
    values_scaled = best_model.named_steps['scaler'].transform(values)
    values_pca = best_model.named_steps['pca'].transform(values_scaled)
    prediction = best_model.named_steps['model'].predict(values_pca)[0]
    print("🔍 Prediction:", "Mine" if prediction=="M" else "Rock")

# ---------------------------
# 7. Main Execution
# ---------------------------
if __name__ == "__main__":
    X, y, df = load_data()
    eda(df, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_models = train_models(X_train, y_train)
    results = evaluate_models(best_models, X_test, y_test)

    # Pick best model
    best_model_name = max(results, key=results.get)
    best_model = best_models[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name} with Accuracy {results[best_model_name]:.3f}")
    
    # Save best model
    joblib.dump(best_model, "best_model_advanced.pkl")
    print("💾 Best model saved as best_model_advanced.pkl")

    # Evaluation plots
    plot_evaluation(best_model, X_test, y_test)

    # Uncomment below to allow manual input prediction
    # predict_custom_input(best_model)
